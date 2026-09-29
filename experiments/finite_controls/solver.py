"""Finite-control degenerate HJI: matched direct and cached-policy campaigns."""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, shutil, time
from dataclasses import asdict, dataclass
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent

@dataclass
class Config:
    dimension: int = 20
    actions: int = 512
    width: int = 64
    depth: int = 3
    batch: int = 256
    interval: int = 5
    steps: int = 10000
    log_every: int = 500
    learning_rate: float = .001
    horizon: float = .5
    viscosity: float = .005
    dtype: str = 'float64'
    device: str = 'cuda'
    monitor: int = 2048
    holdout: int = 16384

def sync():
    torch.cuda.synchronize()

def configure():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def points(n, seed, cfg):
    gen = torch.Generator(device=cfg.device).manual_seed(seed)
    p = torch.rand(n, cfg.dimension+1, generator=gen, device=cfg.device,
                   dtype=getattr(torch, cfg.dtype))
    p[:, 0] *= cfg.horizon
    p[:, 1:] *= 2*math.pi
    return p

def drift(p):
    x = p[:, 1:]
    return .15*torch.sin(torch.roll(x, -1, -1)) + .05*torch.sin(torch.roll(x, -2, -1)-x)

class Problem:
    def __init__(self, cfg):
        self.cfg = cfg
        dtype = getattr(torch, cfg.dtype)
        rng = np.random.default_rng(8421)
        self.Q = torch.tensor(np.linalg.qr(rng.normal(size=(cfg.dimension, 6)))[0],
                              device=cfg.device, dtype=dtype)
        # Generic finite catalogs with an irregular joint-control interaction.
        # There is no analytic replacement of the finite min-max in this model.
        n = cfg.actions
        a = np.arange(n)*2*np.pi/n
        b = np.arange(n)*2*np.pi/n
        table = np.zeros((n,n,8))
        table[:,:,0] = -.4*np.cos(a)[:,None]
        table[:,:,1] = -.4*np.sin(a)[:,None]
        table[:,:,2] = .3*np.cos(b)[None,:]
        table[:,:,3] = .3*np.sin(b)[None,:]
        table[:,:,4:6] = rng.uniform(-.15,.15,(n,n,2))
        table[:,:,6:] = rng.uniform(-1,1,(n,n,2))
        self.table = torch.tensor(table.reshape(-1,8), device=cfg.device, dtype=dtype)
        self.table_t = self.table.T.contiguous()
        self.sigma = torch.zeros(cfg.dimension, device=cfg.device, dtype=dtype)
        self.sigma[::2] = .01
        self.variance = self.sigma+cfg.viscosity
        self.bound = .2 + self.Q.abs() @ self.table[:,:6].abs().amax(0)
        self.h = float(.8*(self.variance/self.bound).min())
        self.tau = float(.8*self.h**2/self.variance.sum())
        offsets = torch.zeros(2*cfg.dimension+2,cfg.dimension+1,device=cfg.device,dtype=dtype)
        offsets[1,0] = -self.tau
        for j in range(cfg.dimension):
            offsets[2+2*j,1+j] = self.h
            offsets[3+2*j,1+j] = -self.h
        self.offsets = offsets

    def features(self, p, grad):
        return torch.cat((grad@self.Q, .02*torch.sin(p[:,1:2]),
                          .015*torch.cos(p[:,2:3])), -1)

    @torch.no_grad()
    def oracle(self, features):
        scores = (features @ self.table_t).reshape(-1,self.cfg.actions,self.cfg.actions)
        inner, js = scores.max(-1)
        val, ii = inner.min(-1)
        jj = js.gather(1,ii[:,None])[:,0]
        index = ii*self.cfg.actions+jj
        return self.table[index], index, val

    def ham(self, p, grad, selected=None):
        z = self.features(p,grad)
        if selected is None:
            selected = self.oracle(z.detach())[0]
        return (drift(p)*grad).sum(-1)+(selected*z).sum(-1)

    def fields(self, p):
        t,x = p[:,0],p[:,1:]
        s = self.cfg.horizon-t
        sx,cx = torch.sin(x),torch.cos(x)
        phase=x.sum(-1); ss,cs=torch.sin(phase),torch.cos(phase)
        d=self.cfg.dimension
        g=.3*cx.mean(-1)+.1*cs
        adjacent=torch.roll(sx,-1,-1)+torch.roll(sx,1,-1)
        psi=(sx*torch.roll(sx,-1,-1)).mean(-1)+.4*ss
        gx=-.3*sx/d-.1*ss[:,None]; gxx=-.3*cx/d-.1*cs[:,None]
        px=cx*adjacent/d+.4*cs[:,None]; pxx=-sx*adjacent/d-.4*ss[:,None]
        rate=2*math.pi/self.cfg.horizon
        amplitude=s+.05*(1-torch.cos(rate*s))
        amplitude_s=1+.05*rate*torch.sin(rate*s)
        return dict(value=g+amplitude*psi,vt=-amplitude_s*psi,
                    gradient=gx+amplitude[:,None]*px,second=gxx+amplitude[:,None]*pxx,g=g)

    @torch.no_grad()
    def source(self,p):
        f=self.fields(p)
        return -f['vt']-self.ham(p,f['gradient'])-.5*(self.sigma*f['second']).sum(-1)

    def fd(self,model,p):
        vals=model((p[None]+self.offsets[:,None]).reshape(-1,p.shape[1])).reshape(len(self.offsets),-1)
        vp,vm=vals[2::2].T,vals[3::2].T
        grad=(vp-vm)/(2*self.h)
        dt=(vals[0]-vals[1])/self.tau
        diffusion=.5*((vp-2*vals[0,:,None]+vm)/self.h**2*self.variance).sum(-1)
        return dt,grad,diffusion

class Network(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg
        dims=[2*cfg.dimension+3]+[cfg.width]*cfg.depth+[1]
        self.layers=nn.ModuleList([nn.Linear(a,b) for a,b in zip(dims[:-1],dims[1:])])
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias)
    def forward(self,p):
        t,x=p[:,:1],p[:,1:]; phase=x.sum(-1,keepdim=True)
        z=torch.cat((2*t/self.cfg.horizon-1,torch.sin(x),torch.cos(x),torch.sin(phase),torch.cos(phase)),-1)
        for layer in self.layers[:-1]: z=torch.sin(layer(z))
        g=.3*torch.cos(x).mean(-1)+.1*torch.cos(phase[:,0])
        return g+(self.cfg.horizon-t[:,0])*self.layers[-1](z)[:,0]

@torch.no_grad()
def evaluate(model,pr,p,nonlinear=False):
    err=den=rr=0.
    for q in p.split(pr.cfg.batch):
        ref=pr.fields(q)['value']; pred=model(q)
        err+=float((pred-ref).square().sum()); den+=float(ref.square().sum())
        if nonlinear:
            dt,g,diff=pr.fd(model,q)
            r=dt+diff+pr.ham(q,g)+pr.source(q)
            rr+=float(r.square().sum())
    out={'relative_l2':math.sqrt(err/den)}
    if nonlinear:out['nonlinear_residual_rms']=math.sqrt(rr/len(p))
    return out

def check(pr):
    cfg=pr.cfg; p=points(7,93001,cfg).requires_grad_(True)
    v=pr.fields(p)['value']; f=pr.fields(p)
    dv=torch.autograd.grad(v.sum(),p,create_graph=True)[0]
    second=torch.stack([torch.autograd.grad(dv[:,j+1].sum(),p,retain_graph=True)[0][:,j+1] for j in range(cfg.dimension)],-1)
    derr=max(float((dv[:,0]-f['vt']).detach().abs().max()),float((dv[:,1:]-f['gradient']).detach().abs().max()),float((second-f['second']).detach().abs().max()))
    g=f['gradient'].detach().clone().requires_grad_(True)
    z=pr.features(p.detach(),g)
    literal=(z@pr.table_t).reshape(-1,cfg.actions,cfg.actions).max(-1).values.min(-1).values
    sel,idx,answer=pr.oracle(z)
    branch=(sel*z).sum(-1)
    ga=torch.autograd.grad(literal.sum(),g,retain_graph=True)[0]
    gb=torch.autograd.grad(branch.sum(),g)[0]
    # Verify extrema independently without the vectorized reduction, on CPU.
    scores=(z.detach()@pr.table_t).cpu().numpy().reshape(-1,cfg.actions,cfg.actions)
    nested=np.array([min(max(row) for row in mat) for mat in scores])
    residual=f['vt'].detach()+pr.ham(p.detach(),f['gradient'].detach())+.5*(pr.sigma*f['second'].detach()).sum(-1)+pr.source(p.detach())
    report=dict(analytic_derivative_max=derr,oracle_literal_max=float((literal.detach()-answer).abs().max()),
                oracle_python_max=float(np.max(np.abs(nested-answer.cpu().numpy()))),
                envelope_gradient_max=float((ga-gb).abs().max()),manufactured_original_residual_max=float(residual.abs().max()),
                stencil_margin=float((pr.variance-pr.h*pr.bound).min()),cfl=float(pr.tau*pr.variance.sum()/pr.h**2),
                physical_rank=int((pr.sigma>0).sum()),dimension=cfg.dimension)
    assert derr<1e-10 and report['envelope_gradient_max']<1e-10
    assert report['oracle_python_max']<1e-10 and report['manufactured_original_residual_max']<1e-10
    assert report['stencil_margin']>=0 and report['cfl']<=1
    # A witness of both convexity and concavity failure in momentum.
    with torch.no_grad():
        r=points(256,8309,cfg); gen=torch.Generator(device=cfg.device).manual_seed(8245)
        p1=torch.randn(256,cfg.dimension,generator=gen,device=cfg.device,dtype=pr.Q.dtype)
        p2=torch.randn(256,cfg.dimension,generator=gen,device=cfg.device,dtype=pr.Q.dtype)
        gap=pr.ham(r,(p1+p2)/2)-.5*(pr.ham(r,p1)+pr.ham(r,p2))
        report['jensen_gap_min']=float(gap.min()); report['jensen_gap_max']=float(gap.max())
    return report



def paired(cfg,seed,out,fresh_baseline=False):
    """Alternate method order each policy block to reduce external-load bias."""
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,out/'search_snapshot.py')
    begin=time.perf_counter(); pr=Problem(cfg); checks=check(pr)
    monitor=points(cfg.monitor,999901,cfg); holdout=points(cfg.holdout,999902,cfg)
    np.savez_compressed(out/'evaluation_points.npz',points=holdout.detach().cpu().numpy())
    states={}
    method_names=['FD_direct','FD_PI_cached']+(['FD_direct_fresh'] if fresh_baseline else [])
    for method in method_names:
        root=out/method;root.mkdir()
        torch.manual_seed(seed)
        model=Network(cfg).to(device=cfg.device,dtype=getattr(torch,cfg.dtype))
        init_hash=hashlib.sha256(b''.join(x.detach().cpu().numpy().tobytes() for x in model.parameters())).hexdigest()
        states[method]=dict(model=model,opt=torch.optim.Adam(model.parameters(),lr=cfg.learning_rate),
                            time=0.,history=[],peak=0,root=root,thresholds={})
        record=dict(config=asdict(cfg),method=method,seed=seed,self_checks=checks,
                    initialization_sha256=init_hash,source_sha256=sha(__file__),h=pr.h,tau=pr.tau,
                    gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
                    timing='separate synchronized training blocks, alternating method order; each includes its own sample/source preparation, policy search, backward, optimizer; diagnostics excluded',
                    effective_batch_reuse=1 if method=='FD_direct_fresh' else cfg.interval,
                    concurrent_process='External GPU load is not measured by this runner; document availability separately. Alternating blocks mitigate changing load.',status='running')
        states[method]['record']=record
        (root/'run.json').write_text(json.dumps(record,indent=2)+'\n')
    # Disposable warmup leaves both matched initializations unchanged.
    warm=copy.deepcopy(states['FD_direct']['model']); opt=torch.optim.Adam(warm.parameters(),lr=cfg.learning_rate)
    p=points(cfg.batch,701,cfg); src=pr.source(p)
    for _ in range(5):
        opt.zero_grad(set_to_none=True);dt,g,df=pr.fd(warm,p)
        (dt+df+pr.ham(p,g)+src).square().mean().backward();opt.step()
    sync();del warm,opt,p,src;torch.cuda.empty_cache()
    for block,start in enumerate(range(0,cfg.steps,cfg.interval)):
        methods=list(states)
        offset=(block+seed)%len(methods)
        methods=methods[offset:]+methods[:offset]
        if ((block+seed)//len(methods))%2:methods.reverse()
        end=min(start+cfg.interval,cfg.steps)
        for method in methods:
            st=states[method];model=st['model'];opt=st['opt']
            sync();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
            selected=None
            if method!='FD_direct_fresh':
                p=points(cfg.batch,100000+seed*10000+block,cfg);src=pr.source(p)
            for step in range(start,end):
                if method=='FD_direct_fresh':
                    p=points(cfg.batch,100000+seed*10000+step,cfg);src=pr.source(p)
                fraction=step/max(1,cfg.steps-1)
                lr=cfg.learning_rate*(.05+.95*(1+math.cos(math.pi*fraction))/2)
                for group in opt.param_groups:group['lr']=lr
                opt.zero_grad(set_to_none=True);dt,g,df=pr.fd(model,p)
                if method=='FD_PI_cached':
                    if selected is None:selected=pr.oracle(pr.features(p,g.detach()))[0]
                    hh=pr.ham(p,g,selected)
                else:hh=pr.ham(p,g)
                loss=(dt+df+hh+src).square().mean();loss.backward();opt.step()
            sync();st['time']+=time.perf_counter()-tick
            st['peak']=max(st['peak'],torch.cuda.max_memory_allocated())
            if end%cfg.log_every==0 or end==cfg.steps:
                row=dict(step=end,training_seconds=st['time'],loss=float(loss.detach()),**evaluate(model,pr,monitor))
                st['history'].append(row)
                with (st['root']/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(method,seed,json.dumps(row),flush=True)
                checkpoint=dict(model=model.state_dict(),config=asdict(cfg),step=end,training_seconds=st['time'])
                for threshold in [.10,.05,.03]:
                    if row['relative_l2']<=threshold and str(threshold) not in st['thresholds']:
                        file=st['root']/('threshold_'+str(threshold)+'.pt')
                        torch.save(checkpoint,file)
                        st['thresholds'][str(threshold)]=dict(step=end,training_seconds=st['time'],monitor_error=row['relative_l2'])
                torch.save(checkpoint,st['root']/'latest.pt')
            del loss,dt,g,df,hh,p,src,selected
    for method,st in states.items():
        final=evaluate(st['model'],pr,holdout,True)
        # Independent holdout confirmation for every first monitor threshold.
        for threshold,entry in st['thresholds'].items():
            model=Network(cfg).to(device=cfg.device,dtype=getattr(torch,cfg.dtype))
            cp=torch.load(st['root']/('threshold_'+threshold+'.pt'),map_location=cfg.device,weights_only=False)
            model.load_state_dict(cp['model']);entry['holdout']=evaluate(model,pr,holdout)
            del model,cp
        rec=st['record']
        rec.update(status='complete',steps=cfg.steps,training_seconds=st['time'],paired_campaign_elapsed_seconds=time.perf_counter()-begin,
                   history=st['history'],final=final,threshold_times={str(t):st['thresholds'].get(str(t)) for t in [.10,.05,.03]},
                   peak_allocated_bytes_in_shared_process=st['peak'],
                   training_policy_search_points=cfg.batch*(math.ceil(cfg.steps/cfg.interval) if method=='FD_PI_cached' else cfg.steps),
                   source_search_points=cfg.batch*(cfg.steps if method=='FD_direct_fresh' else math.ceil(cfg.steps/cfg.interval)),checkpoint_sha256=sha(st['root']/'latest.pt'))
        (st['root']/'run.json').write_text(json.dumps(rec,indent=2)+'\n')
        print('FINAL',json.dumps({k:rec[k] for k in ['method','seed','training_seconds','final','threshold_times']}),flush=True)
