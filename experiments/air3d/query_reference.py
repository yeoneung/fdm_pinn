"""Evaluate the Air3D GPU reference at reproducible common query states."""
import argparse,hashlib,json,math,sys,time
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import gpu_reference as base
cp=base.cp
ROOT=Path(__file__).resolve().parent

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def interpolate_gpu(u,xmin,h,hz,states):
    q=cp.asarray(states,dtype=cp.float64)
    r=(q[:,:2]-xmin)/h;ij=cp.floor(r).astype(cp.int64);xy=r-ij
    angle=(q[:,2]+math.pi)/hz;k=cp.floor(angle).astype(cp.int64);zz=angle-k
    nz=u.shape[2];k=k%nz;out=cp.zeros(len(q),dtype=cp.float64)
    for a in [0,1]:
        for b in [0,1]:
            for c in [0,1]:
                weight=(xy[:,0] if a else 1-xy[:,0])*(xy[:,1] if b else 1-xy[:,1])*(zz if c else 1-zz)
                out+=weight*u[ij[:,0]+a,ij[:,1]+b,(k+c)%nz]
    return cp.asnumpy(out)

def interpolation_check():
    x=np.linspace(-3,3,17);z=np.linspace(-math.pi,math.pi,16,endpoint=False)
    u=x[:,None,None]+2*x[None,:,None]+.1*np.sin(z[None,None,:])
    points=np.random.default_rng(98631).uniform([-1.5,-1.5,-math.pi],[1.5,1.5,math.pi],(1000,3))
    reference=RegularGridInterpolator((x,x,np.append(z,math.pi)),np.concatenate([u,u[:,:,:1]],2))(points)
    gpu=interpolate_gpu(cp.asarray(u),-3,x[1]-x[0],2*math.pi/16,points)
    error=float(abs(reference-gpu).max());assert error<1e-12
    return error

def evaluation_points(samples=16384):
    rng=np.random.default_rng(900302)
    states=rng.uniform([-1.5,-1.5,-np.pi],[1.5,1.5,np.pi],(samples,3))
    return states,np.linspace(0,.5,6)

def main(cells,viscosity,epsilon,output,extent=3.,angle_cells=None,query_file=None):
    output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    cfg=base.Config(problem='air',cells=cells,angle_cells=angle_cells or cells,extent=extent,crop=1.5,viscosity=viscosity,epsilon=epsilon)
    if query_file is None:
        states,times=evaluation_points()
    else:
        query_file=Path(query_file)
        with np.load(query_file) as query:
            states=query['states'].copy();times=query['times'].copy()
    assert times.shape==(6,) and np.allclose(times,np.linspace(0,cfg.horizon,6))
    record=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),command=sys.argv,
        config=asdict(cfg),source_sha256=sha(Path(__file__)),stencil_source_sha256=sha(Path(base.__file__)),
        evaluation_input_sha256=sha(query_file) if query_file else None,evaluation_states_sha256=hashlib.sha256(states.tobytes()).hexdigest(),
        samples=len(states),note='Numerical fixed-point queries on the finite Neumann box; no continuum or zero-viscosity error certificate.')
    def save():(output/'run.json').write_text(json.dumps(record,indent=2)+'\n')
    save()
    try:
        record['stencil_checks']=base.self_checks();record['interpolation_max_error']=interpolation_check();cp.get_default_memory_pool().free_all_blocks()
        x,z,trig,initial,diag=base.arrays(cfg);record['scheme']=diag
        u=cp.asarray(initial);v=cp.empty_like(u);del initial
        cx=cp.asarray(x);ct=cp.asarray(trig);kernel=base.get_kernel('float64');grid=((u.size+255)//256,)
        steps_each=math.ceil((times[1]-times[0])/(cfg.cfl*diag['dt_limit']));dt=(times[1]-times[0])/steps_each
        record.update(dt=dt,steps=5*steps_each,shape=list(u.shape));save()
        values=np.empty((len(times),len(states)));ranges=[]
        def snapshot(index):
            values[index]=interpolate_gpu(u,-cfg.extent,diag['spacing'][0],diag['spacing'][2],states)
            assert np.isfinite(values[index]).all()
            ranges.append(dict(time=float(times[index]),minimum=float(cp.min(u)),maximum=float(cp.max(u))))
            print(f'n={cells} nu={viscosity} epsilon={epsilon} t={times[index]:.1f} elapsed={time.perf_counter()-started:.1f}',flush=True)
        snapshot(5)
        for index in reversed(range(5)):
            for _ in range(steps_each):kernel(grid,(256,),base.kernel_args(u,v,cx,ct,cfg,diag,dt));u,v=v,u
            cp.cuda.Stream.null.synchronize();snapshot(index)
        exact=np.sqrt(states[:,0]**2+states[:,1]**2+epsilon**2)-.5
        record['terminal_interpolation_max_difference']=float(abs(values[-1]-exact).max())
        record['ranges']=ranges
        np.savez_compressed(output/'query_values.npz',states=states,times=times,values=values)
        record.update(status='complete',output_sha256=sha(output/'query_values.npz'))
    except BaseException as exc:record.update(status='failed',exception=repr(exc));raise
    finally:
        record.update(finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started,
            source_unchanged=sha(Path(__file__))==record['source_sha256'] and sha(Path(base.__file__))==record['stencil_source_sha256']);save()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cells',type=int,required=True);p.add_argument('--viscosity',type=float,required=True)
    p.add_argument('--epsilon',type=float,default=.1);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--extent',type=float,default=3.);p.add_argument('--angle-cells',type=int)
    p.add_argument('--queries',type=Path,help='Optional NPZ containing states and times')
    a=p.parse_args();main(a.cells,a.viscosity,a.epsilon,a.output,a.extent,a.angle_cells,a.queries)
