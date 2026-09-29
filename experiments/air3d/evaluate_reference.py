"""Independent full-angle core-box reference evaluation and sensitivity checks."""
import argparse,hashlib,json,math,sys,time
from dataclasses import replace
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import torch
from train_air3d import Config,Network
HERE=Path(__file__).resolve().parent

def interpolate(path,index,states):
    axes=np.load(path/'axes.npz');x=axes['x'];z=axes['z']
    values=np.load(path/f'value_{index:02d}.npy',mmap_mode='r')
    extended=np.concatenate([values,values[:,:,:1]],axis=2)
    zz=np.append(z,np.pi)
    return RegularGridInterpolator((x,x,zz),extended,bounds_error=True)(states)

def metrics(pred,ref):
    err=pred-ref;e2=np.mean(err**2);r2=np.mean(ref**2);ratio=np.sqrt(e2/r2)
    influence=(err**2-ratio**2*ref**2)/(2*max(ratio,1e-300)*r2)
    return dict(relative_l2=float(ratio),rms=float(np.sqrt(e2)),mae=float(abs(err).mean()),
        max_sampled_error=float(abs(err).max()),reference_rms=float(np.sqrt(r2)),
        relative_l2_mc_se=float(influence.std(ddof=1)/np.sqrt(len(err))))

def pooled_metrics(pred,ref):
    # Each state is reused across snapshot times: cluster the influence by state.
    result=metrics(pred.ravel(),ref.ravel())
    e2=np.mean((pred-ref)**2,0);r2=np.mean(ref**2,0);ratio=result['relative_l2']
    influence=(e2-ratio**2*r2)/(2*max(ratio,1e-300)*r2.mean())
    result['relative_l2_mc_se']=float(influence.std(ddof=1)/np.sqrt(pred.shape[1]))
    return result

def main(runs,output,samples,references):
    output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    rng=np.random.default_rng(900302)
    states=rng.uniform([-1.5,-1.5,-np.pi],[1.5,1.5,np.pi],(samples,3));times=np.linspace(0,.5,6)
    root=Path(references)
    reference_values={}
    for path in sorted(root.glob('air_*')):
        if not path.is_dir():continue
        record=json.loads((path/'run.json').read_text());assert record['status']=='complete',path
        reference_values[path.name]=np.array([interpolate(path,i,states) for i in range(6)])
    fine=reference_values['air_n384_nu02_eps01'];rows=[]
    for directory in sorted(runs.glob('seed*')):
        if not (directory/'metrics.json').exists():continue
        report=json.loads((directory/'metrics.json').read_text())
        state=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=False)
        cfg=Config(**state['config']);model=Network(replace(cfg,device='cpu')).double()
        model.load_state_dict(state['model']);pred=[]
        with torch.no_grad():
            for t in times:
                q=torch.tensor(np.column_stack([np.full(samples,t),states]),dtype=torch.float64)
                pred.append(torch.cat([model(chunk) for chunk in q.split(4096)]).numpy())
        pred=np.array(pred)
        snapshots=[dict(time=float(t),**metrics(a,b)) for t,a,b in zip(times,pred,fine)]
        pool=pooled_metrics(pred[:5],fine[:5])
        exact=np.sqrt(states[:,0]**2+states[:,1]**2+cfg.epsilon**2)-.5
        terminal_error=float(abs(pred[5]-exact).max());assert terminal_error<1e-12
        row=dict(method=report['method'],seed=report['seed'],snapshots=snapshots,pooled_nonterminal=pool,
            terminal_max_error=terminal_error,training_seconds=report['training_seconds'],final=report['final'],
            checkpoint_sha256=hashlib.sha256((directory/'checkpoint.pt').read_bytes()).hexdigest())
        rows.append(row);np.save(output/(directory.name+'_prediction.npy'),pred)
        print(f'{directory.name}: t0={snapshots[0]["relative_l2"]:.5f} pooled={pool["relative_l2"]:.5f}',flush=True)
    refinements={name:dict(initial=metrics(v[0],fine[0]),pooled_nonterminal=pooled_metrics(v[:5],fine[:5]),
        terminal_max=float(abs(v[-1]-fine[-1]).max())) for name,v in reference_values.items() if name!='air_n384_nu02_eps01'}
    raw=reference_values.get('air_n384_nu02_eps00')
    smooth_check=dict(sampled_max_difference=float(abs(fine-raw).max()),bound=.1,passed=bool(abs(fine-raw).max()<=.1+1e-10)) if raw is not None else None
    if smooth_check:assert smooth_check['passed'],smooth_check
    np.savez_compressed(output/'evaluation_points_and_references.npz',states=states,times=times,**reference_values)
    result=dict(status='complete',runs=rows,reference_differences=refinements,terminal_smoothing_comparison=smooth_check,
        sample_seed=900302,samples_per_time=samples,domain='[-1.5,1.5]^2 x S1',
        aggregation='five nonterminal times, t=0,...,0.4; state-cluster Monte Carlo SE',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),command=sys.argv)
    (output/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--samples',type=int,default=16384)
    p.add_argument('--references',type=Path,required=True)
    a=p.parse_args();main(a.runs,a.output,a.samples,a.references)
