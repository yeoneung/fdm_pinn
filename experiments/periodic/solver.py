"""Fixed-original-PDE experiments; original data never depend on artificial viscosity."""
from __future__ import annotations
import argparse, copy, gc, hashlib, json, math, os, sys, time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
import operators as base

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config(base.Config):
    batch_size: int = 256
    evaluation_batch: int = 256
    monitor_samples: int = 1024
    final_samples: int = 16384
    phase_features: bool = True


def directions(cfg, like):
    matrix = np.linalg.qr(np.random.default_rng(8421).normal(size=(cfg.dimension,min(4,cfg.dimension))))[0]
    return like.new_tensor(matrix)


def original_fields(p,cfg,Q):
    # FIXED PDE DATA: artificial viscosity is excluded from the source.
    return base.exact_fields(p,replace(cfg,viscosity=0.),Q)


class Network(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg
        sizes=[2*cfg.dimension+1+2*int(cfg.phase_features)]+[cfg.width]*cfg.depth+[1]
        self.layers=nn.ModuleList([nn.Linear(a,b) for a,b in zip(sizes[:-1],sizes[1:])])
        for layer in self.layers: nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias)
    def forward(self,p):
        t=p[:,:1]; x=p[:,1:]; phase=x.sum(-1,keepdim=True)
        features=[2*t/self.cfg.horizon-1,torch.sin(x),torch.cos(x)]
        if self.cfg.phase_features: features.extend([torch.sin(phase),torch.cos(phase)])
        z=torch.cat(features,-1)
        for layer in self.layers[:-1]: z=torch.sin(layer(z))
        g=.3*torch.cos(x).mean(-1)+.1*torch.cos(phase[:,0])
        return g+(self.cfg.horizon-t[:,0])*self.layers[-1](z).squeeze(-1)


def residual(model,p,method,cfg,Q,frozen=None,training=True):
    kind,scheme=method.split('_')
    if kind=='AD':
        query=p.detach().clone().requires_grad_(True); value=model(query)
        first=torch.autograd.grad(value.sum(),query,create_graph=True)[0]
        active=range(cfg.dimension) if cfg.viscosity>0 else range(0,cfg.dimension,2)
        diffusion=0.
        for j in active:
            second_j=torch.autograd.grad(first[:,j+1].sum(),query,retain_graph=True,create_graph=training)[0][:,j+1]
            diffusion=diffusion+.5*(cfg.viscosity+(.01 if j%2==0 else 0.))*second_j
    else:
        _,first,second=base.fd(model,p,cfg)
        diffusion=.5*(base.diffusion(cfg,p)*second).sum(-1)
    control=None if scheme=='direct' else base.policy(frozen,p,kind,cfg,Q)
    with torch.no_grad(): cost=original_fields(p,cfg,Q)['cost']
    return first[:,0]+base.hamiltonian(p,first[:,1:],cfg,Q,control)+diffusion+cost


def points(n,seed,cfg):
    # Independent CPU generator keeps diagnostic arrays out of training GPU memory.
    p=torch.rand((n,cfg.dimension+1),generator=torch.Generator().manual_seed(seed),dtype=getattr(torch,cfg.dtype))
    p[:,0]*=cfg.horizon; p[:,1:]*=2*math.pi
    return p


def evaluate(model,p_cpu,cfg,Q,derivatives=True):
    sums=dict(error=0.,ref=0.,gradient_error=0.,gradient_ref=0.,original_residual=0.,modified_residual=0.,fd_residual=0.)
    maximum=0.
    for raw in p_cpu.split(cfg.evaluation_batch):
        p=raw.to(cfg.device)
        if derivatives: value,first,second=base.ad(model,p,False)
        else:
            with torch.no_grad(): value=model(p)
        with torch.no_grad():
            ref=original_fields(p,cfg,Q); error=value-ref['value']
            sums['error']+=float(error.square().sum()); sums['ref']+=float(ref['value'].square().sum())
            maximum=max(maximum,float(error.abs().max()))
            if derivatives:
                sums['gradient_error']+=float((first[:,1:]-ref['gradient']).square().sum())
                sums['gradient_ref']+=float(ref['gradient'].square().sum())
                r0=first[:,0]+base.hamiltonian(p,first[:,1:],cfg,Q)+.5*(base.diffusion(replace(cfg,viscosity=0),p)*second).sum(-1)+ref['cost']
                rn=r0+.5*cfg.viscosity*second.sum(-1)
                _,fd1,fd2=base.fd(model,p,cfg)
                rh=fd1[:,0]+base.hamiltonian(p,fd1[:,1:],cfg,Q)+.5*(base.diffusion(cfg,p)*fd2).sum(-1)+ref['cost']
                for k,v in [('original_residual',r0),('modified_residual',rn),('fd_residual',rh)]: sums[k]+=float(v.square().sum())
    result=dict(relative_l2=math.sqrt(sums['error']/sums['ref']),error_rms=math.sqrt(sums['error']/len(p_cpu)),max_sampled_error=maximum)
    if derivatives:
        result['gradient_relative_l2']=math.sqrt(sums['gradient_error']/sums['gradient_ref'])
        for k in ['original_residual','modified_residual','fd_residual']: result[k+'_rms']=math.sqrt(sums[k]/len(p_cpu))
    with torch.no_grad():
        p=p_cpu[:127].to(cfg.device); terminal=p.clone(); terminal[:,0]=cfg.horizon
        result['terminal_max_error']=float((model(terminal)-original_fields(terminal,cfg,Q)['g']).abs().max())
        original=model(p); periodic=0.
        for j in range(cfg.dimension):
            shifted=p.clone(); shifted[:,j+1]+=2*math.pi
            periodic=max(periodic,float((model(shifted)-original).abs().max()))
        result['periodic_max_error']=periodic
    return result


def checks(cfg,Q):
    p=points(67,990031,cfg).to(cfg.device)
    class Exact(nn.Module):
        def forward(self,p): return original_fields(p,cfg,Q)['value']
    exact=Exact(); ref=original_fields(p,cfg,Q); v,first,second=base.ad(exact,p,False)
    derivative=max(float((v-ref['value']).detach().abs().max()),float((first[:,0]-ref['vt']).detach().abs().max()),float((first[:,1:]-ref['gradient']).detach().abs().max()),float((second-ref['second']).detach().abs().max()))
    rad=residual(exact,p,'AD_direct',replace(cfg,viscosity=0),Q,training=False).detach()
    source_delta=float((original_fields(p,replace(cfg,viscosity=.2),Q)['cost']-ref['cost']).abs().max())
    bias_residual=residual(exact,p,'AD_direct',cfg,Q,training=False).detach()
    bias_error=float((bias_residual-.5*cfg.viscosity*ref['second'].sum(-1)).abs().max())
    coarse=replace(cfg,h=.06,tau=.001); fine=replace(coarse,h=.03,tau=.00025)
    r1=residual(exact,p,'FD_direct',coarse,Q,training=False).detach()-bias_residual
    r2=residual(exact,p,'FD_direct',fine,Q,training=False).detach()-bias_residual
    ratio=float(r1.square().mean().sqrt()/r2.square().mean().sqrt())
    drift_bound=.2+cfg.minimizing_speed*Q[:,:2].abs().sum(-1)+cfg.maximizing_speed*Q[:,2:].abs().sum(-1)
    variance=base.diffusion(cfg,p)
    report=dict(analytic_derivative_max_error=derivative,original_pde_residual_max=float(rad.abs().max()),source_change_with_viscosity=source_delta,
                artificial_viscosity_residual_identity_error=bias_error,stencil_consistency_refinement_ratio=ratio,
                sufficient_spatial_margin=float((variance-cfg.h*drift_bound).min()),time_cfl=float(cfg.tau*variance.sum()/cfg.h**2),
                physical_diffusion_rank=(cfg.dimension+1)//2,dimension=cfg.dimension)
    # Skipping zero coefficients must preserve residuals and parameter gradients.
    with torch.random.fork_rng(devices=[0] if cfg.device=='cuda' else []):
        torch.manual_seed(871); probe=Network(cfg).to(device=cfg.device,dtype=getattr(torch,cfg.dtype))
        small=p[:7]; _,first_full,second_full=base.ad(probe,small,True)
        full=first_full[:,0]+base.hamiltonian(small,first_full[:,1:],cfg,Q)+.5*(base.diffusion(cfg,small)*second_full).sum(-1)+original_fields(small,cfg,Q)['cost']
        fast=residual(probe,small,'AD_direct',cfg,Q,training=True)
        g1=torch.autograd.grad(full.square().mean(),tuple(probe.parameters()))
        g2=torch.autograd.grad(fast.square().mean(),tuple(probe.parameters()))
        report['active_diffusion_residual_max_difference']=float((full-fast).detach().abs().max())
        report['active_diffusion_parameter_gradient_max_difference']=max(float((a-b).abs().max()) for a,b in zip(g1,g2))
        assert report['active_diffusion_residual_max_difference']<1e-11 and report['active_diffusion_parameter_gradient_max_difference']<1e-10
    assert derivative<1e-10 and float(rad.abs().max())<1e-10 and source_delta==0 and bias_error<1e-10 and 3.7<ratio<4.3,report
    return report


def source_hashes():
    return {name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ['solver.py','operators.py']}


def train(job,root):
    root.mkdir(parents=True,exist_ok=False)
    cfg=Config(**job['config']); method=job['method']; seed=job['seed']
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(seed)
    Q=directions(cfg,torch.empty((),device=cfg.device,dtype=getattr(torch,cfg.dtype)))
    started=time.perf_counter(); hashes=source_hashes()
    record=dict(job=job,status='running',started_utc=datetime.now(timezone.utc).isoformat(),source_sha256=hashes,
                gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
                target='Original PDE with source computed using physical diffusion only; fixed g,f,c,Sigma.',
                timing='Synchronized training blocks include sampling, policy refresh, residual/backprop and optimizer; diagnostics and I/O excluded.',
                memory='Maximum PyTorch CUDA allocation/reservation during training blocks only; diagnostic allocations excluded.',
                selection='Final fixed-update or elapsed-training-time budget; no error-based checkpoint selection.')
    def save_record(): (root/'run.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    save_record()
    try:
        record['self_checks']=checks(cfg,Q)
        if method.startswith('FD'): assert record['self_checks']['sufficient_spatial_margin']>=0 and record['self_checks']['time_cfl']<=1
        model=Network(cfg).to(device=cfg.device,dtype=getattr(torch,cfg.dtype))
        initial=base.state_hash(model); frozen=copy.deepcopy(model).requires_grad_(False) if method.endswith('PI') else None
        monitor=points(cfg.monitor_samples,701501,cfg); holdout=points(cfg.final_samples,701502,cfg)
        warm=copy.deepcopy(model); warm_opt=torch.optim.Adam(warm.parameters(),lr=cfg.learning_rate)
        for _ in range(3):
            p=monitor[:min(cfg.batch_size,len(monitor))].to(cfg.device)
            warm_opt.zero_grad(set_to_none=True); residual(warm,p,method,cfg,Q,frozen).square().mean().backward(); warm_opt.step()
        base.sync(cfg); del warm,warm_opt,p; gc.collect(); torch.cuda.empty_cache()
        opt=torch.optim.Adam(model.parameters(),lr=cfg.learning_rate)
        generator=torch.Generator(device=cfg.device).manual_seed(10000+seed)
        elapsed=0.; step=0; history=[]; per_step=0.; peak=0; reserved=0
        budget=job.get('budget_seconds'); target_steps=job.get('steps')
        next_log=budget/6 if budget else target_steps/5
        log_spacing=next_log
        while (elapsed<budget if budget else step<target_steps):
            block_size=min(25,target_steps-step) if not budget else 25
            torch.cuda.reset_peak_memory_stats(); base.sync(cfg); clock=time.perf_counter()
            for local in range(block_size):
                if frozen is not None and step%cfg.policy_interval==0: frozen.load_state_dict(model.state_dict())
                fraction=min(1.,(elapsed+local*per_step)/budget) if budget else step/max(1,target_steps-1)
                lr=cfg.learning_rate*(cfg.final_lr_ratio+(1-cfg.final_lr_ratio)*(1+math.cos(math.pi*fraction))/2)
                for group in opt.param_groups: group['lr']=lr
                p=base.sample(cfg.batch_size,generator,cfg)
                opt.zero_grad(set_to_none=True); loss=residual(model,p,method,cfg,Q,frozen).square().mean(); loss.backward(); opt.step(); step+=1
            base.sync(cfg); block_seconds=time.perf_counter()-clock
            elapsed+=block_seconds; per_step=block_seconds/block_size
            peak=max(peak,torch.cuda.max_memory_allocated()); reserved=max(reserved,torch.cuda.max_memory_reserved())
            progress=elapsed if budget else step
            if progress>=next_log or (elapsed>=budget if budget else step>=target_steps):
                loss_value=float(loss.detach()); assert math.isfinite(loss_value)
                del loss,p
                row=dict(step=step,training_seconds=elapsed,loss=loss_value,**evaluate(model,monitor,cfg,Q,False))
                history.append(row)
                with (root/'progress.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row)+'\n')
                print(f"{job['name']} step={step} train={elapsed:.2f}s error={100*row['relative_l2']:.3f}% peak={peak/2**20:.1f}MiB",flush=True)
                while next_log<=progress: next_log+=log_spacing
                gc.collect(); torch.cuda.empty_cache()
        final=evaluate(model,holdout,cfg,Q,True)
        checkpoint=dict(model=model.state_dict(),frozen=frozen.state_dict() if frozen is not None else None,
                        config=asdict(cfg),method=method,seed=seed,step=step,training_seconds=elapsed,
                        optimizer=opt.state_dict(),generator=generator.get_state(),Q=Q,job=job)
        torch.save(checkpoint,root/'checkpoint.pt')
        report=dict(method=method,seed=seed,config=asdict(cfg),initialization_sha256=initial,
                    steps=step,training_seconds=elapsed,budget_seconds=budget,overshoot_seconds=elapsed-budget if budget else 0,
                    training_peak_allocated_bytes=peak,training_peak_reserved_bytes=reserved,
                    parameters=sum(p.numel() for p in model.parameters()),history=history,final=final)
        (root/'metrics.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        record.update(status='complete',metrics_sha256=hashlib.sha256((root/'metrics.json').read_bytes()).hexdigest(),checkpoint_sha256=hashlib.sha256((root/'checkpoint.pt').read_bytes()).hexdigest())
    except BaseException as exc:
        record.update(status='failed',exception=repr(exc)); raise
    finally:
        record.update(finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started,source_unchanged=hashes==source_hashes())
        save_record()


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--job',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); train(json.loads(args.job.read_text(encoding='utf-8')),args.output)
