"""Additional time/space refinement of the persistent momentum channel only."""
import hashlib,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import benchmark_channels as exp
HERE=Path(__file__).resolve().parent;base=exp.base

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def quadratic_integral(a,dt):
    return float(dt/3*np.mean(a[:-1]**2+a[:-1]*a[1:]+a[1:]**2,axis=(-2,-1)).sum())

def run(grid,steps,out):
    out=Path(out)
    if (out/'run.json').exists():
        r=json.loads((out/'run.json').read_text())
        if r['status']=='complete':return r
        raise RuntimeError('Existing unfinished output '+str(out))
    out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    cfg=base.Config(grid=grid,steps=steps,iterations=12);sp=base.Spatial(grid,cfg)
    times=np.linspace(0,cfg.horizon,steps+1);old=base.initial_values(sp,times)
    sources=[Path(__file__),Path(exp.__file__),Path(base.__file__)]
    record=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),config=base.asdict(cfg),
        source_hashes={p.name:sha(p) for p in sources},channel='momentum',schedule='persistent',
        selection='Targeted post-matrix numerical refinement of the channel with the largest observed grid change; no training changes.',
        base_checks=base.self_checks(cfg),rows=[])
    def save():(out/'run.json').write_text(json.dumps(record,indent=2)+'\n')
    save();history=base.initial_energy(cfg);reduced=history;prefactor=7/3*np.exp(cfg.kappa*cfg.horizon)
    try:
        for n in range(1,13):
            current=exp.evaluate(old,times,sp,.12,0.,0.)
            met=exp.diagnostic(current,old,times,sp,.12,0.)
            total=met['residual_l2_squared']+4*cfg.f_squared*met['momentum_defect_l2_squared']+met['attainment_defect_l2_squared']+met['terminal_mismatch_squared']
            history=cfg.rho*history+total;reduced=cfg.rho*reduced+met['residual_l2_squared']
            record['rows'].append(dict(iteration=n,**met,full_history_diagnostic=prefactor*history,evaluation_only_diagnostic=prefactor*reduced))
            print(f'n={grid} t={steps} iteration={n} sqrtE={np.sqrt(met["energy_error"]):.9g} elapsed={time.perf_counter()-started:.1f}',flush=True);save()
            if n==12:
                fine=exp.diagnostic(current,old,times,sp,.12,0.,factor=4,order=8)
                record['quadrature_refinement']=dict(standard=met,finer=fine)
                np.savez_compressed(out/'final.npz',current=current,previous=old,times=times)
                # Independent resolved-grid norm check; squared time fields are quadratic.
                kk=np.fft.fftfreq(grid,d=1/grid);coeff=np.fft.fft2(current,axes=(-2,-1))
                dx=np.fft.ifft2(1j*kk[:,None]*coeff,axes=(-2,-1)).real-sp.gx-times[:,None,None]*sp.px
                dy=np.fft.ifft2(1j*kk[None,:]*coeff,axes=(-2,-1)).real-sp.gy-times[:,None,None]*sp.py
                value=float(((current-sp.exact(times))**2).mean(axis=(-2,-1)).max())
                energy=value+cfg.ellipticity*(quadratic_integral(dx,times[1]-times[0])+quadratic_integral(dy,times[1]-times[0]))
                record['independent_norm_check']=dict(energy=energy,absolute_difference=abs(energy-met['energy_error']),
                    analytic_momentum_l2_squared=cfg.horizon*.12**2,
                    momentum_difference=abs(met['momentum_defect_l2_squared']-cfg.horizon*.12**2))
                assert abs(energy-met['energy_error'])<1e-12 and record['independent_norm_check']['momentum_difference']<1e-12
            old=current
        record.update(status='complete',output_sha256=sha(out/'final.npz'))
    except BaseException as exc:record.update(status='failed',exception=repr(exc));raise
    finally:
        record.update(elapsed_seconds=time.perf_counter()-started,finished_utc=datetime.now(timezone.utc).isoformat(),
            source_unchanged=all(sha(p)==record['source_hashes'][p.name] for p in sources));save()
    return record
