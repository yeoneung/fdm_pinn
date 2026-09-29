"""Separate momentum, Hamiltonian-attainment and terminal defect histories.

Uses the local Fourier operators and exact manufactured target.
The practical policy uses a perturbed momentum and/or an admissible suboptimal
control. All continuum defect terms are independently recomputed on a shifted
validation grid.
"""
import argparse,csv,hashlib,importlib.util,json,sys,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
HERE=Path(__file__).resolve().parent
BASE=HERE/'fourier.py'
spec=importlib.util.spec_from_file_location('periodic_base_channels',BASE)
base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)

def controls(px,py,sp,momentum,attainment):
    qx=px+momentum*np.cos(2*sp.x+sp.y)
    qy=py-momentum*np.sin(sp.x-sp.y)
    aa=-sp.config.control_speed*np.sign(qx)*(1-2*attainment)
    bb=sp.config.disturbance_speed*np.sign(qy)
    return aa,bb,qx,qy

def evaluate(previous,times,sp,momentum,attainment,terminal):
    cfg=sp.config;oldx,oldy,_=sp.differential(previous)
    values=np.empty_like(previous);values[0]=sp.g+terminal*sp.forcing
    dt=times[1]-times[0]
    def rhs(v,px,py,s):
        vx,vy,diff=sp.differential(v);a,b,_,_=controls(px,py,sp,momentum,attainment)
        return diff+a*vx+b*vy+sp.cost(s)
    for j in range(cfg.steps):
        sx,sy=oldx[j],oldy[j];ex,ey=oldx[j+1],oldy[j+1];mx,my=(sx+ex)/2,(sy+ey)/2
        v=values[j];s=times[j]
        k1=rhs(v,sx,sy,s);k2=rhs(v+dt*k1/2,mx,my,s+dt/2)
        k3=rhs(v+dt*k2/2,mx,my,s+dt/2);k4=rhs(v+dt*k3,ex,ey,s+dt)
        values[j+1]=v+dt*(k1+2*k2+2*k3+k4)/6
    assert np.isfinite(values).all()
    return values

def diagnostic(current,previous,times,sp,momentum,attainment,factor=2,order=4):
    cfg=sp.config;size=factor*cfg.grid+1;target=base.Spatial(size,cfg)
    shift=np.pi/size;target.x+=shift;target.y+=shift
    for name in ['g','psi','gx','gy','px','py','dg','dp','forcing']:
        field=getattr(target,name);k=np.fft.fftfreq(size,d=1/size)
        setattr(target,name,np.fft.ifft2(np.fft.fft2(field)*np.exp(1j*(k[:,None]+k[None,:])*shift)).real)
    phase=np.exp(1j*(sp.kx+sp.ky)*shift)
    def lift(a):
        return base.interpolate_space(np.fft.ifft2(np.fft.fft2(a,axes=(-2,-1))*phase,axes=(-2,-1)).real,size)
    cx,cy,cd=sp.differential(current);px,py,_=sp.differential(previous)
    nodes,weights=np.polynomial.legendre.leggauss(order);fractions=(nodes+1)/2;weights=weights/2
    dt=times[1]-times[0];rsq=msq=asq=gsq=0.;vsq=0.;terminal=None
    for start in range(0,cfg.steps,8):
        stop=min(start+8,cfg.steps)
        vv=[lift(a[start:stop+1]) for a in [current,cx,cy,cd,px,py]]
        value,vx,vy,diff,oldx,oldy=vv
        error=value-target.exact(times[start:stop+1]);errors=(error**2).mean(axis=(-2,-1))
        vsq=max(vsq,float(errors.max()))
        if start==0:terminal=float(errors[0])
        slope=(value[1:]-value[:-1])/dt
        for q,w in zip(fractions,weights):
            s=times[start:stop]+q*dt
            dx,dy,d,opx,opy=[(1-q)*a[:-1]+q*a[1:] for a in vv[1:]]
            a,b,qx,qy=controls(opx,opy,target,momentum,attainment)
            r=-slope+a*dx+b*dy+d+target.cost(s)
            opt=np.abs(a*qx+b*qy-(-cfg.control_speed*np.abs(qx)+cfg.disturbance_speed*np.abs(qy)))
            eta=(qx-opx)**2+(qy-opy)**2
            ex=dx-target.gx-s[:,None,None]*target.px;ey=dy-target.gy-s[:,None,None]*target.py
            integral=lambda arr:dt*w*float(arr.mean(axis=(-2,-1)).sum())
            rsq+=integral(r*r);msq+=integral(eta);asq+=integral(opt*opt);gsq+=integral(ex*ex+ey*ey)
    return dict(residual_l2_squared=rsq,momentum_defect_l2_squared=msq,
        attainment_defect_l2_squared=asq,terminal_mismatch_squared=terminal,
        energy_error=vsq+cfg.ellipticity*gsq,value_error_sup_squared=vsq,gradient_error_integral=gsq)

def run(cfg,out):
    out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    record=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),config=base.asdict(cfg),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),base_source_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),
        neural_training=False,continuum_certificate=False,command=sys.argv,
        amplitudes=dict(momentum=.12,attainment=.15,terminal=.04),schedules=['persistent','decaying','early_pulse'])
    def save():(out/'run.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    save();rows=[];refinements=[]
    try:
        record['base_checks']=base.self_checks(cfg)
        sp=base.Spatial(cfg.grid,cfg);times=np.linspace(0,cfg.horizon,cfg.steps+1)
        prefactor=7/3*np.exp(cfg.kappa*cfg.horizon);initial=base.initial_values(sp,times)
        for channel in ['momentum','attainment','terminal','combined']:
            for schedule in record['schedules']:
                old=initial.copy();hist=base.initial_energy(cfg);reduced=hist
                for n in range(1,cfg.iterations+1):
                    scale=1. if schedule=='persistent' else (cfg.decay**(n-1) if schedule=='decaying' else float(n==2))
                    amps=[record['amplitudes'][key]*scale if channel in [key,'combined'] else 0. for key in ['momentum','attainment','terminal']]
                    current=evaluate(old,times,sp,*amps)
                    met=diagnostic(current,old,times,sp,*amps[:2])
                    full=met['residual_l2_squared']+4*cfg.f_squared*met['momentum_defect_l2_squared']+met['attainment_defect_l2_squared']+met['terminal_mismatch_squared']
                    hist=cfg.rho*hist+full;reduced=cfg.rho*reduced+met['residual_l2_squared']
                    row=dict(channel=channel,schedule=schedule,iteration=n,scale=scale,**met,
                        full_history_diagnostic=prefactor*hist,evaluation_only_diagnostic=prefactor*reduced)
                    row['energy_to_full_history_ratio']=met['energy_error']/(prefactor*hist)
                    rows.append(row)
                    if n==cfg.iterations:
                        fine=diagnostic(current,old,times,sp,*amps[:2],factor=4,order=8)
                        refinements.append(dict(channel=channel,schedule=schedule,standard=met,finer=fine))
                        np.savez_compressed(out/f'{channel}_{schedule}_final.npz',current=current,previous=old,times=times)
                    old=current
                print(f'{channel} {schedule}: sqrt(E)={np.sqrt(met["energy_error"]):.6g} ratio={row["energy_to_full_history_ratio"]:.5g}',flush=True)
                record['completed']=list({r['channel']+'_'+r['schedule'] for r in rows});save()
        with (out/'metrics.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        (out/'diagnostics.json').write_text(json.dumps(dict(rows=rows,refinements=refinements,prefactor=prefactor),indent=2)+'\n')
        record.update(status='complete',max_energy_to_full_history_ratio=max(r['energy_to_full_history_ratio'] for r in rows))
    except BaseException as exc:record.update(status='failed',exception=repr(exc));raise
    finally:
        record.update(elapsed_seconds=time.perf_counter()-started,finished_utc=datetime.now(timezone.utc).isoformat(),
            source_unchanged=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==record['source_sha256']);save()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--grid',type=int,default=33);p.add_argument('--steps',type=int,default=128);p.add_argument('--iterations',type=int,default=12);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(base.Config(grid=a.grid,steps=a.steps,iterations=a.iterations),a.output)
