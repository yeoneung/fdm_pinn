"""Paired AD/FD x direct/PI on an explicitly smoothed Air3D problem."""
from __future__ import annotations
import argparse,copy,hashlib,json,math,os,platform,sys,time
from dataclasses import dataclass,asdict,replace
from datetime import datetime,timezone
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch import nn
HERE=Path(__file__).resolve().parent
METHODS=('AD_direct','FD_direct','AD_PI','FD_PI')

@dataclass(frozen=True)
class Config:
    steps:int=20000
    batch_size:int=2048
    boundary_batch:int=256
    width:int=128
    depth:int=3
    policy_interval:int=500
    log_interval:int=1000
    learning_rate:float=1e-3
    final_lr_ratio:float=.05
    h:float=.05
    tau:float=.0025
    horizon:float=.5
    extent:float=3.
    viscosity:float=.2
    epsilon:float=.1
    omega:float=1.5
    boundary_weight:float=1.
    monitor_samples:int=4096
    final_samples:int=32768
    evaluation_batch:int=1024
    dtype:str='float32'
    device:str='cuda'

def terminal(p,cfg):return torch.sqrt(p[:,1]**2+p[:,2]**2+cfg.epsilon**2)-.5

class Network(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg
        sizes=[5]+[cfg.width]*cfg.depth+[1]
        self.layers=nn.ModuleList([nn.Linear(a,b) for a,b in zip(sizes[:-1],sizes[1:])])
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight);nn.init.zeros_(layer.bias)
    def forward(self,p):
        t,x,y,angle=p.unbind(-1)
        z=torch.stack([2*t/self.cfg.horizon-1,x/self.cfg.extent,y/self.cfg.extent,torch.sin(angle),torch.cos(angle)],-1)
        for layer in self.layers[:-1]:z=torch.sin(layer(z))
        return terminal(p,self.cfg)+(self.cfg.horizon-t)*self.layers[-1](z).squeeze(-1)

def ad_derivatives(model,p,training):
    q=p.detach().clone().requires_grad_(True);value=model(q)
    d=torch.autograd.grad(value.sum(),q,create_graph=True)[0]
    second=[torch.autograd.grad(d[:,i].sum(),q,create_graph=training,retain_graph=True)[0][:,i] for i in [1,2,3]]
    return value,d,torch.stack(second,-1)

def fd_derivatives(model,p,cfg):
    offsets=p.new_zeros((8,4));offsets[1,0]=-cfg.tau
    for i in range(3):offsets[2+2*i,1+i]=cfg.h;offsets[3+2*i,1+i]=-cfg.h
    vals=model((p[None]+offsets[:,None]).reshape(-1,4)).reshape(8,-1)
    d=[(vals[0]-vals[1])/cfg.tau];second=[]
    for i in range(3):
        a,b=vals[2+2*i],vals[3+2*i];d.append((a-b)/(2*cfg.h));second.append((a-2*vals[0]+b)/cfg.h**2)
    return vals[0],torch.stack(d,-1),torch.stack(second,-1)

def policy(model,p,kind,cfg):
    if kind=='AD':
        q=p.detach().clone().requires_grad_(True)
        d=torch.autograd.grad(model(q).sum(),q)[0].detach()
    else:
        with torch.no_grad():
            offsets=p.new_zeros((6,4))
            for i in range(3):offsets[2*i,1+i]=cfg.h;offsets[2*i+1,1+i]=-cfg.h
            vals=model((p[None]+offsets[:,None]).reshape(-1,4)).reshape(6,-1)
            d=p.new_zeros(p.shape)
            for i in range(3):d[:,1+i]=(vals[2*i]-vals[2*i+1])/(2*cfg.h)
    return cfg.omega*torch.sign(p[:,2]*d[:,1]-p[:,1]*d[:,2]-d[:,3]),-cfg.omega*torch.sign(d[:,3])

def hamiltonian(p,d,cfg,control=None):
    px,py,pa=d[:,1],d[:,2],d[:,3]
    drift=(-.5+.5*torch.cos(p[:,3]))*px+.5*torch.sin(p[:,3])*py
    switching=p[:,2]*px-p[:,1]*py-pa
    return drift+(cfg.omega*switching.abs()-cfg.omega*pa.abs() if control is None else control[0]*switching+control[1]*pa)

def residual(model,p,method,cfg,frozen=None,training=True):
    kind,scheme=method.split('_')
    value,d,second=ad_derivatives(model,p,training) if kind=='AD' else fd_derivatives(model,p,cfg)
    controls=None if scheme=='direct' else policy(frozen,p,kind,cfg)
    return d[:,0]+hamiltonian(p,d,cfg,controls)+.5*cfg.viscosity*second.sum(-1)

def scale_points(raw,cfg):
    return raw*raw.new_tensor([cfg.horizon,2*cfg.extent,2*cfg.extent,2*math.pi])+raw.new_tensor([0,-cfg.extent,-cfg.extent,-math.pi])

def points(n,seed,cfg,dtype=None):
    g=torch.Generator(device=cfg.device).manual_seed(seed)
    return scale_points(torch.rand((n,4),generator=g,device=cfg.device,dtype=getattr(torch,dtype or cfg.dtype)),cfg)

def boundary_points(raw,cfg):
    p=scale_points(raw,cfg)
    ids=torch.arange(len(p),device=p.device)%4
    p[:,1]=torch.where(ids<2,torch.where(ids==0,-cfg.extent,cfg.extent),p[:,1])
    p[:,2]=torch.where(ids>=2,torch.where(ids==2,-cfg.extent,cfg.extent),p[:,2])
    return p,torch.where(ids<2,1,2)

def boundary_residual(model,p,axes,training):
    q=p.detach().clone().requires_grad_(True)
    d=torch.autograd.grad(model(q).sum(),q,create_graph=training)[0]
    return d.gather(1,axes[:,None]).squeeze(1)

def state_hash(model):
    h=hashlib.sha256()
    for name,v in model.state_dict().items():h.update(name.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def evaluate(model,p,cfg):
    # Convert checkpoint coefficients to float64 before both AD and FD diagnostics.
    m=copy.deepcopy(model).to(dtype=torch.float64);m.requires_grad_(False)
    totals={'continuum_residual_squared':0.,'fd_residual_squared':0.,'consistency_squared':0.,'fd_roundoff_squared':0.}
    core={k:0. for k in totals};ncore=0
    for q in p.to(torch.float64).split(cfg.evaluation_batch):
        v,d,second=ad_derivatives(m,q,False)
        with torch.no_grad():
            rad=d[:,0]+hamiltonian(q,d,cfg)+.5*cfg.viscosity*second.sum(-1)
            _,df,sf=fd_derivatives(m,q,cfg)
            rfd=df[:,0]+hamiltonian(q,df,cfg)+.5*cfg.viscosity*sf.sum(-1)
            original_q=q.to(dtype=next(model.parameters()).dtype)
            _,do,so=fd_derivatives(model,original_q,cfg)
            original_fd=do[:,0]+hamiltonian(original_q,do,cfg)+.5*cfg.viscosity*so.sum(-1)
            mask=(q[:,1].abs()<=1.5)&(q[:,2].abs()<=1.5);ncore+=int(mask.sum())
            for k,a in zip(totals,[rad*rad,rfd*rfd,(rad-rfd)**2,(original_fd.double()-rfd)**2]):
                totals[k]+=float(a.sum());core[k]+=float(a[mask].sum())
    raw=torch.rand((2048,4),generator=torch.Generator(device=cfg.device).manual_seed(808080),device=cfg.device,dtype=torch.float64)
    bp,axes=boundary_points(raw,cfg);bc=boundary_residual(m,bp,axes,False).detach()
    with torch.no_grad():
        q=p[:1024].double();q[:,0]=cfg.horizon
        ter=float((m(q)-terminal(q,cfg)).abs().max())
        q=p[:1024].double();shift=q.clone();shift[:,3]+=2*math.pi
        per=float((m(q)-m(shift)).abs().max())
    result={k.replace('_squared','_rms'):math.sqrt(a/len(p)) for k,a in totals.items()}
    result.update({'core_'+k.replace('_squared','_rms'):math.sqrt(a/max(ncore,1)) for k,a in core.items()})
    result.update(boundary_normal_rms=float((bc*bc).mean().sqrt()),boundary_normal_max=float(bc.abs().max()),
        terminal_max=ter,periodic_value_max=per,samples=len(p),core_samples=ncore)
    del m
    return result

def self_checks(cfg):
    double=replace(cfg,dtype='float64')
    torch.manual_seed(826);m=Network(double).to(device=cfg.device,dtype=torch.float64)
    p=points(257,7612,double)
    v,d,dd=ad_derivatives(m,p,False)
    fd=fd_derivatives(m,p,double);half=fd_derivatives(m,p,replace(double,h=cfg.h/2,tau=cfg.tau/2))
    # Spatial differences on smooth model: second order; time quotient: first order.
    err=float((fd[1][:,1:]-d[:,1:]).detach().square().mean().sqrt());errhalf=float((half[1][:,1:]-d[:,1:]).detach().square().mean().sqrt())
    controls=policy(m,p,'AD',cfg)
    attained=hamiltonian(p,d,cfg,controls);exact=hamiltonian(p,d,cfg)
    attainment=float((attained-exact).detach().abs().max())
    frozen=copy.deepcopy(m).requires_grad_(False);initial=state_hash(frozen)
    opt=torch.optim.Adam(m.parameters(),lr=.001)
    residual(m,p,'AD_PI',double,frozen).square().mean().backward();opt.step()
    independent=state_hash(frozen)==initial and state_hash(m)!=initial
    checks=dict(spatial_fd_error=err,spatial_fd_half_error=errhalf,spatial_fd_ratio=err/errhalf,
        hamiltonian_attainment_error=attainment,frozen_independent=independent,
        terminal_smoothing_uniform_bound=cfg.epsilon)
    assert attainment<1e-12 and independent and 3.5<err/errhalf<4.5,checks
    return checks

def run_one(method,seed,cfg,root,monitor,holdout):
    out=root/f'seed{seed}_{method}';out.mkdir()
    torch.manual_seed(seed);m=Network(cfg).to(device=cfg.device,dtype=getattr(torch,cfg.dtype))
    initial=state_hash(m);frozen=copy.deepcopy(m).requires_grad_(False) if method.endswith('PI') else None
    # Identical disposable warmup; live initialization never updated.
    warm=copy.deepcopy(m);wo=torch.optim.Adam(warm.parameters(),lr=cfg.learning_rate)
    for _ in range(3):
        wo.zero_grad(set_to_none=True);residual(warm,monitor[:cfg.batch_size],method,cfg,frozen).square().mean().backward();wo.step()
    torch.cuda.synchronize();del warm,wo
    opt=torch.optim.Adam(m.parameters(),lr=cfg.learning_rate)
    sampler=torch.Generator(device=cfg.device).manual_seed(10000+seed)
    bcsampler=torch.Generator(device=cfg.device).manual_seed(20000+seed)
    elapsed=0.;started=time.perf_counter();history=[];torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize();block_start=time.perf_counter()
    for step in range(1,cfg.steps+1):
        if frozen is not None and (step-1)%cfg.policy_interval==0:frozen.load_state_dict(m.state_dict())
        lr=cfg.learning_rate*(cfg.final_lr_ratio+(1-cfg.final_lr_ratio)*(1+math.cos(math.pi*(step-1)/max(cfg.steps-1,1)))/2)
        for group in opt.param_groups:group['lr']=lr
        p=scale_points(torch.rand((cfg.batch_size,4),generator=sampler,device=cfg.device,dtype=getattr(torch,cfg.dtype)),cfg)
        bp,axes=boundary_points(torch.rand((cfg.boundary_batch,4),generator=bcsampler,device=cfg.device,dtype=getattr(torch,cfg.dtype)),cfg)
        opt.zero_grad(set_to_none=True)
        r=residual(m,p,method,cfg,frozen);bn=boundary_residual(m,bp,axes,True)
        loss=r.square().mean()+cfg.boundary_weight*bn.square().mean()
        loss.backward();opt.step()
        if step%cfg.log_interval==0 or step==cfg.steps:
            torch.cuda.synchronize();elapsed+=time.perf_counter()-block_start
            lv=float(loss.detach());assert math.isfinite(lv),(method,seed,step)
            metrics=evaluate(m,monitor,cfg)
            row=dict(step=step,training_seconds=elapsed,loss=lv,**metrics);history.append(row)
            with (out/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            print(f'{method} seed={seed} step={step} time={elapsed:.1f}s residual={metrics["continuum_residual_rms"]:.4f} bc={metrics["boundary_normal_rms"]:.4f}',flush=True)
            # Recovery checkpoint; final step is the predeclared endpoint, not error-selected.
            torch.save(dict(model=m.state_dict(),frozen=frozen.state_dict() if frozen is not None else None,
                config=asdict(cfg),method=method,seed=seed,step=step,optimizer=opt.state_dict(),
                sampler=sampler.get_state(),boundary_sampler=bcsampler.get_state()),out/'checkpoint.pt')
            torch.cuda.synchronize();block_start=time.perf_counter()
    final=evaluate(m,holdout,cfg)
    record=dict(method=method,seed=seed,config=asdict(cfg),initialization_sha256=initial,
        training_seconds=elapsed,total_elapsed_seconds=time.perf_counter()-started,history=history,final=final,
        parameters=sum(p.numel() for p in m.parameters()),peak_memory_bytes=torch.cuda.max_memory_allocated())
    (out/'metrics.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    return record

def suite(cfg,seeds,methods,output):
    output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=Path(__file__);start=time.perf_counter()
    record=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),command=sys.argv,
        config=asdict(cfg),seeds=seeds,methods=methods,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),python=sys.version,
        monitor_seed=700101,holdout_seed=700102,training_generator='10000+seed',boundary_generator='20000+seed',
        checkpoint_selection='fixed final step, no error-based selection',
        timing='synchronized blocks including sampling, boundary loss, policy updates and optimization; excludes diagnostics/checkpoint I/O',
        problem='smoothed terminal sqrt(x^2+y^2+epsilon^2)-0.5; fixed positive viscosity, exact angular periodicity; soft Neumann penalty',
        limitation='terminal data do not satisfy Neumann compatibility; boundary mismatch is measured, not assumed zero')
    def save():(output/'run.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    save()
    try:
        record['self_checks']=self_checks(cfg);save()
        monitor=points(cfg.monitor_samples,700101,cfg);holdout=points(cfg.final_samples,700102,cfg)
        reports=[]
        for i,seed in enumerate(seeds):
            order=methods[i%len(methods):]+methods[:i%len(methods)]
            for method in order:
                r=run_one(method,seed,cfg,output,monitor,holdout);reports.append(r)
                record['completed']=[dict(method=r['method'],seed=r['seed']) for r in reports];save()
        paired=all(len({r['initialization_sha256'] for r in reports if r['seed']==seed})==1 for seed in seeds)
        assert paired;record.update(status='complete',paired_initialization_verified=paired)
    except BaseException as exc:
        record.update(status='failed',exception=repr(exc));raise
    finally:
        record.update(elapsed_seconds=time.perf_counter()-start,finished_utc=datetime.now(timezone.utc).isoformat(),
            source_unchanged=hashlib.sha256(source.read_bytes()).hexdigest()==record['source_sha256'],
            output_sha256={str(p.relative_to(output)):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for pattern in ['*/metrics.json','*/checkpoint.pt'] for p in output.glob(pattern)})
        save()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key,value in asdict(Config()).items():p.add_argument('--'+key.replace('_','-'),type=type(value),default=value)
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    p.add_argument('--output',type=Path,required=True)
    args=vars(p.parse_args());output=args.pop('output');seeds=args.pop('seeds');methods=args.pop('methods')
    suite(Config(**args),seeds,methods,output)
