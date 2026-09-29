"""Independent history bookkeeping and analytic norms of saved Fourier/time-linear lifts."""
import argparse,hashlib,json,math
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def source_hash(name):
    return sha(HERE/Path(name).name)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def close(a,b):return abs(a-b)<=1e-20+1e-8*abs(b)

def squared_time_integral(values,dt):
    # Integral of a squared linear interpolant, evaluated exactly per interval.
    return float(dt/3*np.mean(values[:-1]**2+values[:-1]*values[1:]+values[1:]**2,axis=(-2,-1)).sum())

def main(source, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    root=Path(source);record=json.loads((root/'run.json').read_text());assert record['status']=='complete'
    data=json.loads((root/'diagnostics.json').read_text());cfg=record['config'];rows=data['rows'];grid=cfg['grid']
    f2=cfg['control_speed']**2+cfg['disturbance_speed']**2;lam=cfg['viscosity'];rho=cfg['rho'];T=cfg['horizon']
    kappa=3+4*f2/lam+16*f2/(3*rho*lam);prefactor=7/3*math.exp(kappa*T)
    initial=lam*(2-math.exp(-kappa*T)*((kappa*T)**2+2*kappa*T+2))/kappa**3
    x=2*math.pi*np.arange(grid)[:,None]/grid;y=2*math.pi*np.arange(grid)[None,:]/grid
    g=np.cos(x)+.5*np.sin(y)+.25*np.cos(x+y);psi=.4*np.sin(x)*np.sin(y)+.2*np.cos(2*x)
    gx=-np.sin(x)-.25*np.sin(x+y);gy=.5*np.cos(y)-.25*np.sin(x+y)
    px=.4*np.cos(x)*np.sin(y)-.4*np.sin(2*x);py=.4*np.sin(x)*np.cos(y)
    k=np.fft.fftfreq(grid,d=1/grid);kx=k[:,None];ky=k[None,:]
    checks=dict(source_unchanged=source_hash('benchmark_channels.py')==record['source_sha256'],
        base_source_unchanged=source_hash('fourier.py')==record['base_source_sha256'],
        all_iterations_present=len(rows)==4*3*cfg['iterations'],prefactor=close(prefactor,data['prefactor']))
    reports=[]
    for channel in ['momentum','attainment','terminal','combined']:
        for schedule in ['persistent','decaying','early_pulse']:
            rr=sorted([r for r in rows if r['channel']==channel and r['schedule']==schedule],key=lambda r:r['iteration'])
            history=initial;reduced=initial;history_ok=True;analytic_injection_ok=True
            for row in rr:
                scale=row['scale'];m=.12*scale if channel in ['momentum','combined'] else 0.;z=.04*scale if channel in ['terminal','combined'] else 0.
                full=row['residual_l2_squared']+4*f2*row['momentum_defect_l2_squared']+row['attainment_defect_l2_squared']+row['terminal_mismatch_squared']
                history=rho*history+full;reduced=rho*reduced+row['residual_l2_squared']
                history_ok &= close(prefactor*history,row['full_history_diagnostic']) and close(prefactor*reduced,row['evaluation_only_diagnostic'])
                analytic_injection_ok &= close(T*m*m,row['momentum_defect_l2_squared']) and close(.5*z*z,row['terminal_mismatch_squared'])
            last=rr[-1];scale=last['scale'];m=.12*scale if channel in ['momentum','combined'] else 0.;a=.15*scale if channel in ['attainment','combined'] else 0.
            with np.load(root/f'{channel}_{schedule}_final.npz') as saved:
                current=saved['current'];previous=saved['previous'];s=saved['times'];dt=float(s[1]-s[0])
                coeff=np.fft.fft2(current,axes=(-2,-1))
                dx=np.fft.ifft2(1j*kx*coeff,axes=(-2,-1)).real-gx-s[:,None,None]*px
                dy=np.fft.ifft2(1j*ky*coeff,axes=(-2,-1)).real-gy-s[:,None,None]*py
                error=current-g-s[:,None,None]*psi
                sup=float(np.mean(error**2,axis=(-2,-1)).max())
                energy=sup+lam*(squared_time_integral(dx,dt)+squared_time_integral(dy,dt))
                del coeff,dx,dy,error,current
                if a:
                    oldx=np.fft.ifft2(1j*kx*np.fft.fft2(previous,axes=(-2,-1)),axes=(-2,-1)).real
                    qx=oldx+m*np.cos(2*x+y)
                    attainment=(2*cfg['control_speed']*a)**2*squared_time_integral(qx,dt)
                    del oldx,qx
                else:attainment=0.
                del previous
            cc=dict(history_reconstructed=bool(history_ok),analytic_momentum_and_terminal=bool(analytic_injection_ok),
                saved_lift_energy=close(energy,last['energy_error']),analytic_attainment_norm=close(attainment,last['attainment_defect_l2_squared']))
            reports.append(dict(channel=channel,schedule=schedule,checks=cc,
                energy_difference=abs(energy-last['energy_error']),attainment_norm_difference=abs(attainment-last['attainment_defect_l2_squared'])))
            print(channel,schedule,all(cc.values()),flush=True)
    checks['all_case_checks']=all(all(r['checks'].values()) for r in reports)
    result=dict(checked_utc=datetime.now(timezone.utc).isoformat(),grid=grid,checks=checks,cases=reports,
        note='Exact polynomial time integration and resolved trigonometric norms in floating point; sign-dependent residual quadrature remains uncertified.')
    (output/'FINAL_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
    assert all(checks.values()),result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.input,a.output)
