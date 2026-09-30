"""Aggregate paired finite-control runs and verify saved checkpoints."""
from .common import read, write, stats
from experiments.finite_controls.results_tools import METHODS, evaluate_campaign, monitor_time_comparison


def summarize(inputs, registry, output):
    jobs = [j for j in registry if j['family'] == 'finite_controls']
    rows = []
    for job in jobs:
        source = inputs[job['id']]
        evaluate_campaign(source, output/'evaluations'/f"finite_seed{job['seed']}")
        records = [read(source/m/'run.json') for m in METHODS]
        if len({r['initialization_sha256'] for r in records}) != 1:
            raise ValueError(f"{job['id']}: initializations are not paired")
        for record in records:
            rows.append(dict(seed=record['seed'], method=record['method'],
                             relative_l2=record['final']['relative_l2'],
                             training_seconds=record['training_seconds'], history=record['history'],
                             thresholds=record['threshold_times']))
    groups = {}
    for method in METHODS:
        rr = [r for r in rows if r['method'] == method]
        thresholds = {}
        for threshold in rr[0]['thresholds']:
            reached = [r['thresholds'][threshold] for r in rr if r['thresholds'][threshold] is not None]
            thresholds[threshold] = dict(n_reached=len(reached), n_total=len(rr),
                n_holdout_confirmed=sum(r['holdout']['relative_l2'] <= float(threshold) for r in reached),
                time=stats([r['training_seconds'] for r in reached]) if len(reached) >= 2 else None)
        groups[method] = dict(relative_l2=stats([r['relative_l2'] for r in rr]),
                             training_seconds=stats([r['training_seconds'] for r in rr]), thresholds=thresholds)
    write(output/'finite_controls.json', dict(all_passed=True, seeds=[j['seed'] for j in jobs],
          runs=rows, groups=groups, monitor_time_comparison=monitor_time_comparison(rows)))
