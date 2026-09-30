"""Aggregate periodic runs and evaluate their spatial slices and policy gradients."""
from collections import defaultdict
from dataclasses import replace
import copy
import math
import numpy as np
import torch
from analysis.common import read, write, stats, relative, sha
from run import load


def network(path):
    solver = load('periodic', 'solver')
    record = read(path/'run.json')
    if sha(path/'checkpoint.pt') != record['checkpoint_sha256']:
        raise ValueError(f'Checkpoint hash mismatch: {path}')
    cp = torch.load(path/'checkpoint.pt', map_location='cpu', weights_only=False)
    cfg = replace(solver.Config(**cp['config']), device='cpu')
    model = solver.Network(cfg).double().eval()
    model.load_state_dict(cp['model'])
    return solver, cp, cfg, model, cp['Q'].double().cpu()


def infer(model, points):
    with torch.no_grad():
        return torch.cat([model(p) for p in points.split(512)]).numpy()


def policy_diagnostics(path):
    solver, cp, cfg, model, q = network(path)
    frozen = copy.deepcopy(model).requires_grad_(False)
    if cp['frozen'] is not None:
        frozen.load_state_dict(cp['frozen'])
    points = solver.points(2048, 880204, cfg)
    sums = dict(momentum=0., nonlinear_residual=0., gap=0.)
    bound = .2*math.sqrt(cfg.dimension)+math.sqrt(2*cfg.minimizing_speed**2+2*cfg.maximizing_speed**2)
    max_excess = -math.inf
    with torch.no_grad():
        for p in points.split(256):
            _, new, second = solver.base.fd(model, p, cfg)
            _, previous, _ = solver.base.fd(frozen, p, cfg)
            projection = previous[:, 1:]@q
            controls = torch.cat([-cfg.minimizing_speed*torch.sign(projection[:, :2]),
                                  cfg.maximizing_speed*torch.sign(projection[:, 2:])], -1)
            common = new[:, 0]+.5*(solver.base.diffusion(cfg, p)*second).sum(-1)
            common += solver.original_fields(p, cfg, q)['cost']
            nonlinear = common+solver.base.hamiltonian(p, new[:, 1:], cfg, q)
            fixed = common+solver.base.hamiltonian(p, new[:, 1:], cfg, q, controls)
            delta = torch.linalg.vector_norm(new[:, 1:]-previous[:, 1:], dim=-1)
            gap = nonlinear-fixed
            max_excess = max(max_excess, float((gap.abs()-2*bound*delta).max()))
            for key, value in [('momentum', delta), ('nonlinear_residual', nonlinear), ('gap', gap)]:
                sums[key] += float(value.square().sum())
    if max_excess > 1e-10:
        raise ValueError('The sampled frozen-policy inequality failed.')
    return {key+'_rms': math.sqrt(value/len(points)) for key, value in sums.items()}


def summarize(inputs, registry, output):
    torch.set_num_threads(2)
    jobs = [j for j in registry if j['family'] == 'periodic' and j['kind'] == 'neural']
    records = []
    locations = {}
    for job in jobs:
        path = inputs[job['id']]
        run, metrics = read(path/'run.json'), read(path/'metrics.json')
        if sha(path/'metrics.json') != run['metrics_sha256']:
            raise ValueError(f'Metric hash mismatch: {path}')
        row = dict(metrics, name=job['job']['name'], stage=job['job']['stage'], self_checks=run['self_checks'])
        records.append(row)
        locations[row['name']] = path
    # Check the initial weights across methods and regularization settings.
    paired = defaultdict(set)
    for r in records:
        paired[(r['config']['dimension'], r['seed'])].add(r['initialization_sha256'])
    if any(len(hashes) != 1 for hashes in paired.values()):
        raise ValueError('The paired initial weights do not agree.')
    refs, arrays = {}, {}
    for job in registry:
        if job['family'] == 'periodic' and job['kind'] == 'reference':
            path = inputs[job['id']]
            name = job['id'].split('/')[-1]
            refs[name] = read(path/'run.json')
            if sha(path/'queries.npz') != refs[name]['query_sha256']:
                raise ValueError(f'Reference hash mismatch: {path}')
            with np.load(path/'queries.npz') as z:
                arrays[name] = {k: z[k].copy() for k in ['points', 'values', 'exact']}
    reference_summary = {}
    for nu in [.08, .04, .02, .01]:
        fine, coarse = arrays[f'nu{nu:g}_n1024'], arrays[f'nu{nu:g}_n512']
        if not np.array_equal(fine['points'], coarse['points']):
            raise ValueError('The reference query coordinates differ.')
        reference_summary[f'{nu:g}'] = dict(relative_l2_bias=relative(fine['values'][:16384], fine['exact'][:16384]),
            relative_l2_grid_difference=relative(coarse['values'][:16384], fine['values'][:16384]))
    primary = [r for r in records if r['stage'] in ['fixed_target', 'dimension_scaling']]
    extended = [r for r in records if r['stage'] == 'extended_fifty']
    if len(primary) != 39 or len(extended) != 6:
        raise ValueError('Expected 39 refinement/dimension runs and six longer-budget runs.')
    slices, slice_bucket, bucket = [], defaultdict(list), defaultdict(list)
    times = [0., .1, .2, .3, .4]
    for r in primary+extended:
        print('Evaluating periodic slices:', r['name'], flush=True)
        solver, cp, cfg, model, q = network(locations[r['name']])
        stage = 'extended' if r in extended else r['stage']
        key = (f'd2_{r["method"]}_nu{cfg.viscosity:g}' if stage == 'fixed_target'
               else f'd{cfg.dimension}_{r["method"]}'+('_900s' if stage == 'extended' else ''))
        spatial = solver.points(4096, 701503, cfg)
        series = []
        reference = arrays[f'nu{cfg.viscosity:g}_n1024'] if cfg.dimension == 2 and r['method'] == 'FD_PI' else None
        if reference is not None:
            p = solver.points(16384, 701502, cfg)
            if not np.array_equal(p.numpy(), reference['points'][:16384]):
                raise ValueError('Neural and grid test coordinates differ.')
            r['neural_relative_l2_to_regularized_reference'] = relative(infer(model, p), reference['values'][:16384])
        for index, t in enumerate(times):
            p = spatial.clone()
            p[:, 0] = t
            prediction = infer(model, p)
            exact = solver.original_fields(p, cfg, q)['value'].numpy()
            error = prediction-exact
            item = dict(time=t, relative_l2=relative(prediction, exact),
                        absolute_rms=float(np.sqrt(np.mean(error**2))), max_sampled_error=float(np.max(np.abs(error))))
            if reference is not None:
                sl = slice(16384+index*4096, 16384+(index+1)*4096)
                if not np.array_equal(p.numpy(), reference['points'][sl]):
                    raise ValueError('Fixed-time reference coordinates differ.')
                item.update(to_fdm_relative_l2=relative(prediction, reference['values'][sl]),
                            fdm_bias_relative_l2=relative(reference['values'][sl], exact))
            series.append(item)
        sr = dict(group=key, stage=stage, method=r['method'], dimension=cfg.dimension, viscosity=cfg.viscosity,
                  seed=r['seed'], name=r['name'], slices=series)
        slices.append(sr)
        slice_bucket[key].append(sr)
        if r in primary:
            bucket[key].append(r)
    def grouped(runs):
        runs = sorted(runs, key=lambda r: r['seed'])
        if [r['seed'] for r in runs] != [0, 1, 2]:
            raise ValueError('Expected the three paired seeds 0, 1, 2.')
        result = dict(config=runs[0]['config'])
        for metric in ['relative_l2', 'gradient_relative_l2']:
            result[metric] = stats([r['final'][metric] for r in runs])
        for metric in ['steps', 'training_seconds']:
            result[metric] = stats([r[metric] for r in runs])
        result['memory_mib'] = stats([r['training_peak_allocated_bytes']/2**20 for r in runs])
        if 'neural_relative_l2_to_regularized_reference' in runs[0]:
            result['neural_relative_l2_to_regularized_reference'] = stats([r['neural_relative_l2_to_regularized_reference'] for r in runs])
        return result
    write(output/'periodic.json', dict(all_passed=True, configured_runs=len(primary), failed_runs=0,
          groups={k: grouped(v) for k, v in bucket.items()}, reference_summary=reference_summary,
          reference_runs=refs, individual_runs=primary))
    write(output/'extended.json', dict(all_passed=True, configured_runs=len(extended),
          groups={m: grouped([r for r in extended if r['method'] == m]) for m in ['AD_direct', 'FD_PI']}, individual_runs=extended))
    groups = {}
    for key, runs in slice_bucket.items():
        runs.sort(key=lambda r: r['seed'])
        series = []
        for index, t in enumerate(times):
            rows = [r['slices'][index] for r in runs]
            item = dict(time=t)
            for metric in ['relative_l2', 'absolute_rms', 'max_sampled_error', 'to_fdm_relative_l2']:
                if metric in rows[0]:
                    item[metric] = stats([v[metric] for v in rows])
            if 'fdm_bias_relative_l2' in rows[0]:
                item['fdm_bias_relative_l2'] = rows[0]['fdm_bias_relative_l2']
            series.append(item)
        groups[key] = {k: runs[0][k] for k in ['stage', 'method', 'dimension', 'viscosity']}
        groups[key]['series'] = series
    write(output/'time_slices.json', dict(all_passed=True, evaluated_checkpoints=len(slices), groups=groups,
          times=times, spatial_seed=701503, spatial_points_per_slice=4096, individual_runs=slices))
    selected = [r for r in records if r['name'].startswith('pi_') or (r['stage'] == 'dimension_scaling' and r['config']['dimension'] == 20 and r['method'] == 'FD_PI')]
    if len(selected) != 12:
        raise ValueError('Expected twelve policy-interval comparison runs.')
    buckets = defaultdict(list)
    for r in selected:
        r.update(policy_diagnostics(locations[r['name']]))
        key = 'FD_direct' if r['method'] == 'FD_direct' else f"K{r['config']['policy_interval']}"
        buckets[key].append(r)
    groups = {}
    for key, runs in buckets.items():
        groups[key] = grouped(runs)
        for metric in ['nonlinear_residual_rms', 'momentum_rms', 'gap_rms']:
            groups[key][metric] = stats([r[metric] for r in sorted(runs, key=lambda r: r['seed'])])
    write(output/'policy_intervals.json', dict(all_passed=True, additional_configured_runs=9,
          reused_primary_runs=3, groups=groups, individual_runs=selected))
