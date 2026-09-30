"""Evaluate Air3D checkpoints and compare numerical reference solutions."""
import sys
from pathlib import Path
import numpy as np
from .common import read, write, stats, sha, relative


def difference(a, b):
    return float(np.sqrt(np.mean((a-b)**2)/np.mean(b*b))), float(abs(a-b).max())


def latex_number(value):
    if value == 0:
        return '$0$'
    if abs(value) >= 1e-4:
        return f'${value:.5f}$'
    mantissa, exponent = f'{value:.3e}'.split('e')
    return f'${mantissa}\\times10^{{{int(exponent)}}}$'


def summarize(inputs, registry, output):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/air3d'))
    import evaluate_reference as evaluator
    references = {j['id'].split('/')[-1]: inputs[j['id']] for j in registry
                  if j['family'] == 'air3d' and j['kind'] == 'reference'}
    evaluated = {}
    cached = None
    for suite in ['main', 'equal_time', 'fd_h0025', 'fd_h010']:
        source = inputs['air3d/'+suite]
        destination = output/'evaluations'/suite
        evaluator.main(source, destination, 16384, references, cached)
        report = read(destination/'evaluation.json')
        record = read(source/'run.json')
        if cached is None:
            with np.load(destination/'evaluation_points_and_references.npz') as z:
                cached = {key: z[key].copy() for key in references}
                states, times = z['states'].copy(), z['times'].copy()
        rows = report['runs']
        expected = {(m, s) for m in record['methods'] for s in record['seeds']}
        if len(rows) != len(expected) or {(r['method'], r['seed']) for r in rows} != expected:
            raise ValueError(f'{suite}: missing or duplicate checkpoint evaluations')
        for row in rows:
            directory = source/f"seed{row['seed']}_{row['method']}"
            metrics = read(directory/'metrics.json')
            row['optimizer_updates'] = metrics.get('steps', metrics['config']['steps'])
            # The campaign hashes protect both training diagnostics and checkpoints.
            for filename in ['metrics.json', 'checkpoint.pt']:
                key = directory.name+'/'+filename
                hashes = {k.replace('\\', '/'): v for k, v in record.get('output_sha256', {}).items()}
                if hashes.get(key) not in (None, sha(directory/filename)):
                    raise ValueError(f'{suite}/{key}: checksum mismatch')
        evaluated[suite] = rows
        if suite not in ['main', 'equal_time']:
            continue
        groups = {}
        for method in record['methods']:
            rr = sorted((r for r in rows if r['method'] == method), key=lambda r: r['seed'])
            g = dict(initial_relative_l2=stats([r['snapshots'][0]['relative_l2'] for r in rr]),
                     pooled_relative_l2=stats([r['pooled_nonterminal']['relative_l2'] for r in rr]))
            for key in ['training_seconds', 'optimizer_updates']:
                g[key] = stats([r[key] for r in rr])
            for key in rr[0]['final']:
                if isinstance(rr[0]['final'][key], (float, int)):
                    g[key] = stats([r['final'][key] for r in rr])
            groups[method] = g
        write(output/('air_fixed.json' if suite == 'main' else 'air_time.json'),
              dict(config=record['config'], seeds=record['seeds'], methods=groups,
                   reference_differences=report['reference_differences']))
    sensitivity = []
    for suite in ['fd_h0025', 'main', 'fd_h010']:
        cfg = read(inputs['air3d/'+suite]/'run.json')['config']
        for method in ['FD_direct', 'FD_PI']:
            rr = sorted((r for r in evaluated[suite] if r['method'] == method and r['seed'] in [0, 1, 2]),
                        key=lambda r: r['seed'])
            sensitivity.append(dict(suite=suite, h=cfg['h'], tau=cfg['tau'], method=method,
                pooled_error=stats([r['pooled_nonterminal']['relative_l2'] for r in rr]),
                consistency_rms=stats([r['final']['consistency_rms'] for r in rr]),
                fd_roundoff_rms=stats([r['final']['fd_roundoff_rms'] for r in rr])))
    write(output/'fd_sensitivity.json', dict(status='complete', seeds=[0, 1, 2], groups=sensitivity))
    baseline = cached['air_n384_nu02_eps01']
    groups = [
        ('Numerical refinement', [('192 vs 384', 'air_n192_nu02_eps01'), ('288 vs 384', 'air_n288_nu02_eps01'), ('Half time step', 'air_n384_nu02_eps01_halfdt')]),
        ('Terminal smoothing', [(r'$\epsilon=0$', 'air_n384_nu02_eps00'), (r'$\epsilon=0.05$', 'air_n384_nu02_eps005'), (r'$\epsilon=0.2$', 'air_n384_nu02_eps02')]),
        ('Artificial viscosity', [(r'$\nu=0.1$', 'air_n384_nu01_eps01'), (r'$\nu=0.4$', 'air_n384_nu04_eps01')]),
    ]
    rows = []
    for title, entries in groups:
        for label, key in entries:
            error, maximum = difference(cached[key][0], baseline[0])
            rows.append(f'{title} & {label} & {100*error:.5f} & {maximum:.6f}')
    tables = dict(air_reference_audit=rows)

    def query(key):
        with np.load(inputs['air3d/'+key]/'query_values.npz') as z:
            if not np.array_equal(z['states'], states) or not np.array_equal(z['times'], times):
                raise ValueError(f'{key}: evaluation points or snapshot times differ')
            return z['values'].copy()

    zero = query('vanishing_viscosity_queries/nu0_n720')
    arrays = {0.05: query('vanishing_viscosity_queries/nu005_n720'),
              0.1: cached['air_n384_nu01_eps01'], 0.2: baseline, 0.4: cached['air_n384_nu04_eps01']}
    tables['air_zero_comparator'] = [
        f'{nu} & {720 if nu == .05 else 384} & {100*difference(v[0], zero[0])[0]:.3f} & {100*difference(v[:5], zero[:5])[0]:.3f}'
        for nu, v in arrays.items()]
    fine = query('domain_queries/extent_6')
    rows = []
    for extent, cells in [(3, 384), (4.5, 576), (6, 768)]:
        values = query(f'domain_queries/extent_{extent}')
        initial, pooled = relative(values[0], fine[0]), relative(values[:5], fine[:5])
        rows.append(f'{extent:g} & {cells} & {latex_number(100*initial)} & {latex_number(100*pooled)}')
    tables['air_domain_audit'] = rows
    write(output/'air_reference_tables.json', tables)
