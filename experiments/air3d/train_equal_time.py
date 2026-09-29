"""Actual equal-GPU-training-time Air3D ablation, with fixed stopping budgets.

Time counts sampling, BC/PDE optimization and policy updates; diagnostic and
checkpoint I/O time is excluded. Synchronized blocks of 50 updates give a small
recorded overshoot. The learning-rate schedule is parameterized by elapsed
training time, estimated within each block. No error-based checkpoint choice.
"""
import argparse,copy,hashlib,json,math,sys,time
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import torch
import train_air3d as base
HERE=Path(__file__).resolve().parent

def one(method,seed,cfg,budget,root,monitor,holdout):
    out=root/f'seed{seed}_{method}';out.mkdir()
    torch.manual_seed(seed);m=base.Network(cfg).to(device=cfg.device,dtype=getattr(torch,cfg.dtype))
    initial=base.state_hash(m);frozen=copy.deepcopy(m).requires_grad_(False) if method.endswith('PI') else None
    warm=copy.deepcopy(m);wo=torch.optim.Adam(warm.parameters(),lr=cfg.learning_rate)
    for _ in range(3):
        wo.zero_grad(set_to_none=True);base.residual(warm,monitor[:cfg.batch_size],method,cfg,frozen).square().mean().backward();wo.step()
    torch.cuda.synchronize();del warm,wo
    opt=torch.optim.Adam(m.parameters(),lr=cfg.learning_rate)
    sampler=torch.Generator(device=cfg.device).manual_seed(10000+seed);bcsampler=torch.Generator(device=cfg.device).manual_seed(20000+seed)
    elapsed=0.;step=0;history=[];next_log=budget/10;per_step_estimate=0.;started=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    while elapsed<budget:
        torch.cuda.synchronize();block_start=time.perf_counter()
        for local in range(50):
            if frozen is not None and step%cfg.policy_interval==0:frozen.load_state_dict(m.state_dict())
            progress=min(1.,(elapsed+local*per_step_estimate)/budget)
            lr=cfg.learning_rate*(cfg.final_lr_ratio+(1-cfg.final_lr_ratio)*(1+math.cos(math.pi*progress))/2)
            for group in opt.param_groups:group['lr']=lr
            p=base.scale_points(torch.rand((cfg.batch_size,4),generator=sampler,device=cfg.device,dtype=getattr(torch,cfg.dtype)),cfg)
            bp,axes=base.boundary_points(torch.rand((cfg.boundary_batch,4),generator=bcsampler,device=cfg.device,dtype=getattr(torch,cfg.dtype)),cfg)
            opt.zero_grad(set_to_none=True)
            rr=base.residual(m,p,method,cfg,frozen);bc=base.boundary_residual(m,bp,axes,True)
            loss=rr.square().mean()+cfg.boundary_weight*bc.square().mean();loss.backward();opt.step();step+=1
        torch.cuda.synchronize();block_seconds=time.perf_counter()-block_start
        elapsed+=block_seconds;per_step_estimate=block_seconds/50
        if elapsed>=next_log or elapsed>=budget:
            lv=float(loss.detach());assert math.isfinite(lv)
            row=dict(step=step,training_seconds=elapsed,loss=lv,**base.evaluate(m,monitor,cfg));history.append(row)
            with (out/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            print(f'{method} seed={seed} step={step} training_seconds={elapsed:.2f} residual={row["continuum_residual_rms"]:.4f}',flush=True)
            while next_log<=elapsed:next_log+=budget/10
            torch.save(dict(model=m.state_dict(),frozen=frozen.state_dict() if frozen is not None else None,
                config=asdict(cfg),method=method,seed=seed,step=step,optimizer=opt.state_dict(),
                sampler=sampler.get_state(),boundary_sampler=bcsampler.get_state(),training_seconds=elapsed,budget=budget),out/'checkpoint.pt')
    result=dict(method=method,seed=seed,config=asdict(cfg),initialization_sha256=initial,
        budget_seconds=budget,overshoot_seconds=elapsed-budget,steps=step,training_seconds=elapsed,
        total_elapsed_seconds=time.perf_counter()-started,history=history,final=base.evaluate(m,holdout,cfg),
        parameters=sum(p.numel() for p in m.parameters()),peak_memory_bytes=torch.cuda.max_memory_allocated())
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def main(cfg,budget,seeds,output):
    output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.perf_counter();source=Path(__file__);base_source=Path(base.__file__)
    record=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),command=sys.argv,
        config=asdict(cfg),seeds=seeds,methods=list(base.METHODS),budget_seconds=budget,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),base_source_sha256=hashlib.sha256(base_source.read_bytes()).hexdigest(),
        gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
        selection='final block at/after fixed training time, no error selection',
        timing='50-step synchronized blocks; excludes diagnostics and checkpoint I/O',
        learning_rate='cosine in elapsed training time; intra-block estimate from previous block')
    def save():(output/'run.json').write_text(json.dumps(record,indent=2)+'\n')
    save()
    try:
        record['self_checks']=base.self_checks(cfg);save();reports=[]
        monitor=base.points(cfg.monitor_samples,700101,cfg);holdout=base.points(cfg.final_samples,700102,cfg)
        for i,seed in enumerate(seeds):
            methods=base.METHODS[i%4:]+base.METHODS[:i%4]
            for method in methods:
                reports.append(one(method,seed,cfg,budget,output,monitor,holdout))
                record['completed']=[dict(method=r['method'],seed=r['seed']) for r in reports];save()
        paired=all(len({r['initialization_sha256'] for r in reports if r['seed']==seed})==1 for seed in seeds)
        assert paired;record.update(status='complete',paired_initialization_verified=paired)
    except BaseException as exc:record.update(status='failed',exception=repr(exc));raise
    finally:
        record.update(finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started,
            source_unchanged=hashlib.sha256(source.read_bytes()).hexdigest()==record['source_sha256'] and hashlib.sha256(base_source.read_bytes()).hexdigest()==record['base_source_sha256'],
            output_sha256={str(p.relative_to(output)):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for pattern in ['*/metrics.json','*/checkpoint.pt'] for p in output.glob(pattern)})
        save()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--budget',type=float,default=300.);p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config-json',type=Path,help='Completed fixed-step run.json, to match its physical/network configuration')
    args=p.parse_args();cfg=base.Config(**json.loads(args.config_json.read_text())['config']) if args.config_json else base.Config()
    if args.budget<=0:raise ValueError('Training budget must be positive')
    main(cfg,args.budget,args.seeds,args.output)
