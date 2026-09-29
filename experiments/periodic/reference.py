"""Independent periodic 2D FDM references for the fixed original HJB data."""
from dataclasses import replace
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,math,time
import numpy as np
import torch
import solver

HERE=Path(__file__).resolve().parent


def interp(u,ix,iy,wx,wy):
    n=u.shape[0]; jx=(ix+1)%n; jy=(iy+1)%n
    return ((1-wx)*(1-wy)*u[ix,iy]+wx*(1-wy)*u[jx,iy]+(1-wx)*wy*u[ix,jy]+wx*wy*u[jx,jy])


def solve(n,nu,out,dt_cap=.001):
    out.mkdir(parents=True,exist_ok=False); start=time.perf_counter()
    cfg=replace(solver.Config(),dimension=2,viscosity=nu,maximizing_speed=0.,phase_features=False)
    dtype=torch.float64; device='cuda'; h=2*math.pi/n; T=cfg.horizon
    q=solver.directions(cfg,torch.empty((),device=device,dtype=dtype))
    axis=torch.arange(n,device=device,dtype=dtype)*h
    x,y=torch.meshgrid(axis,axis,indexing='ij'); sx,sy=torch.sin(x),torch.sin(y); cx,cy=torch.cos(x),torch.cos(y)
    ss,cs=torch.sin(x+y),torch.cos(x+y)
    g=.15*(cx+cy)+.1*cs; psi=sx*sy+.4*ss
    gx,gy=-.15*sx-.1*ss,-.15*sy-.1*ss
    gxx=-.15*cx-.1*cs
    px,py=cx*sy+.4*cs,sx*cy+.4*cs; pxx=-sx*sy-.4*ss
    fx,fy=.15*sy,.15*sx
    bounds=.2+.4*q.abs().sum(-1)
    physical=torch.tensor([.01,0.],device=device,dtype=dtype)
    requested=physical+nu
    a=torch.maximum(requested,h*bounds)
    steps=math.ceil(T/min(dt_cap,float(.8*h*h/a.sum())))
    dt=T/steps
    def cost(s):
        A=s+.05*(1-math.cos(2*math.pi*s/T)); As=1+.05*2*math.pi/T*math.sin(2*math.pi*s/T)
        vx,vy=gx+A*px,gy+A*py
        ham=fx*vx+fy*vy-.4*(torch.abs(q[0,0]*vx+q[1,0]*vy)+torch.abs(q[0,1]*vx+q[1,1]*vy))
        return As*psi-ham-.005*(gxx+A*pxx)
    def rhs(u,s):
        xp,xm=torch.roll(u,-1,0),torch.roll(u,1,0)
        yp,ym=torch.roll(u,-1,1),torch.roll(u,1,1)
        ux,uy=(xp-xm)/(2*h),(yp-ym)/(2*h)
        return fx*ux+fy*uy-.4*(torch.abs(q[0,0]*ux+q[1,0]*uy)+torch.abs(q[0,1]*ux+q[1,1]*uy))+cost(s)+.5*(a[0]*(xp-2*u+xm)+a[1]*(yp-2*u+ym))/h**2
    # Independent source expression agrees with the analytic source used in training.
    pick=torch.arange(0,n,max(1,n//11),device=device)
    ix0,iy0=torch.meshgrid(pick,pick,indexing='ij'); sample=torch.stack([torch.full_like(x[ix0,iy0],.17),x[ix0,iy0],y[ix0,iy0]],-1).reshape(-1,3)
    source_error=float((cost(T-.17)[ix0,iy0].reshape(-1)-solver.original_fields(sample,cfg,q)['cost']).abs().max())
    assert source_error<1e-12
    query=solver.points(cfg.final_samples,701502,cfg)
    # Extra common points at five reported time slices include t=0.
    grid_query=solver.points(4096,701503,cfg)
    slices=[]
    for t in [0.,.1,.2,.3,.4]:
        p=grid_query.clone();p[:,0]=t;slices.append(p)
    query=torch.cat([query,*slices],0)
    order=np.argsort(T-query[:,0].numpy()); sr=(T-query[:,0].numpy())[order]
    p=query[order].to(device); position=p[:,1:]/h; indices=torch.floor(position).long()%n
    weights=position-torch.floor(position); values=torch.empty(len(p),device=device,dtype=dtype)
    u=g.clone(); cursor=0; torch.cuda.synchronize()
    for k in range(steps):
        s=k*dt; predictor=u+dt*rhs(u,s); following=.5*u+.5*(predictor+dt*rhs(predictor,s+dt))
        stop=int(np.searchsorted(sr,(k+1)*dt+1e-14,side='right'))
        if stop>cursor:
            sl=slice(cursor,stop); ix,iy=indices[sl,0],indices[sl,1]; wx,wy=weights[sl,0],weights[sl,1]
            lo=interp(u,ix,iy,wx,wy); hi=interp(following,ix,iy,wx,wy)
            fraction=torch.as_tensor((sr[sl]-s)/dt,device=device,dtype=dtype)
            values[sl]=lo+(hi-lo)*fraction; cursor=stop
        u=following
    assert cursor==len(query)
    unsorted=torch.empty_like(values);unsorted[torch.as_tensor(order,device=device)]=values
    exact=solver.original_fields(query.to(device),cfg,q)['value']
    err=float((unsorted-exact).square().sum().sqrt()/exact.square().sum().sqrt())
    torch.cuda.synchronize(); elapsed=time.perf_counter()-start
    np.savez_compressed(out/'queries.npz',points=query.numpy(),values=unsorted.cpu().numpy(),exact=exact.cpu().numpy(),final_grid=u.cpu().numpy())
    record=dict(status='complete',n=n,h=h,viscosity=nu,steps=steps,dt=dt,
                variance=requested.cpu().tolist(),actual_variance=a.cpu().tolist(),extra_stabilization=(a-requested).cpu().tolist(),
                time_cfl=float(dt*a.sum()/h**2),source_identity_max_error=source_error,
                relative_l2_to_original_exact=err,total_seconds=elapsed,query_holdout_count=cfg.final_samples,
                fixed_time_slices=[0,.1,.2,.3,.4],points_per_slice=4096,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),solver_source_sha256=solver.source_hashes(),
                query_sha256=hashlib.sha256((out/'queries.npz').read_bytes()).hexdigest(),finished_utc=datetime.now(timezone.utc).isoformat())
    (out/'run.json').write_text(json.dumps(record,indent=2)+'\n')
    print(f'FDM n={n} nu={nu:g} dt={dt:.6g} error-original={100*err:.5f}% time={elapsed:.1f}s',flush=True)
