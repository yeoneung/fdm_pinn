"""Recompute finite-control summaries and evaluate preserved checkpoints on CPU."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
METHODS = ('FD_direct', 'FD_PI_cached', 'FD_direct_fresh')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(points, horizon):
    t, x = points[:, 0], points[:, 1:]
    s = horizon-t
    g = .3*np.cos(x).mean(axis=1)+.1*np.cos(x.sum(axis=1))
    psi = (np.sin(x)*np.roll(np.sin(x), -1, axis=1)).mean(axis=1)+.4*np.sin(x.sum(axis=1))
    return g+(s+.05*(1-np.cos(2*np.pi*s/horizon)))*psi


def group(values):
    return dict(mean=float(np.mean(values)), std=float(np.std(values, ddof=1)), values=values)


def monitor_time_comparison(rows, target_seconds=140.0):
    """Describe saved monitor histories near a common time, without interpolation."""
    methods = {}
    for method in METHODS:
        selected = []
        for run in sorted((r for r in rows if r['method'] == method), key=lambda r: r['seed']):
            h = min(run['history'], key=lambda h: (abs(h['training_seconds']-target_seconds), h['step']))
            selected.append(dict(seed=run['seed'], step=h['step'],
                                 training_seconds=h['training_seconds'], relative_l2=h['relative_l2']))
        methods[method] = dict(rows=selected,
            training_seconds=group([r['training_seconds'] for r in selected]),
            relative_l2=group([r['relative_l2'] for r in selected]))
    return dict(target_seconds=target_seconds, methods=methods,
                selection='Nearest recorded training time per method and seed; no interpolation or error-based selection.',
                evaluation='Common 2048-point monitoring set, CUDA generator seed 999901; not the final independent test set.',
                scope='Descriptive comparison of stored histories, added after training; approximate common time, not a newly run fixed-budget experiment.')


def evaluate_campaign(source, output):
    """Independently evaluate final and threshold checkpoints; never train."""
    import torch
    from .solver import Config, Network, points
    torch.set_num_threads(2)
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = [read(source/m/'run.json') for m in METHODS]
    cfg = Config(**records[0]['config'])
    stored = source/'independent_predictions.npz'
    if stored.exists():
        with np.load(stored) as z:
            query = z['points'].copy()
    elif (source/'evaluation_points.npz').exists():
        with np.load(source/'evaluation_points.npz') as z:
            query = z['points'].copy()
    else:
        # Match the device-specific generator used during training.
        query = points(cfg.holdout, 999902, cfg).cpu().numpy()
    q = torch.from_numpy(query)
    ref = reference(query, cfg.horizon)
    arrays = dict(points=query, reference=ref)
    reports = []
    for method, record in zip(METHODS, records):
        model = Network(cfg).to(dtype=torch.float64).eval()
        def infer(path):
            cp = torch.load(path, map_location='cpu', weights_only=False)
            model.load_state_dict(cp['model'])
            with torch.no_grad():
                pred = torch.cat([model(p) for p in q.split(512)]).numpy()
            return pred, float(np.linalg.norm(pred-ref)/np.linalg.norm(ref)), cp
        latest = source/method/'latest.pt'
        assert sha(latest) == record['checkpoint_sha256']
        pred, error, cp = infer(latest)
        assert cp['step'] == record['steps'] and cp['training_seconds'] == record['training_seconds']
        delta = abs(error-record['final']['relative_l2'])
        assert delta < 1e-10
        arrays[method] = pred
        thresholds = {}
        for th, entry in record['threshold_times'].items():
            if entry is None:
                thresholds[th] = None
                continue
            _, e, cp = infer(source/method/('threshold_'+th+'.pt'))
            assert cp['step'] == entry['step'] and cp['training_seconds'] == entry['training_seconds']
            assert abs(e-entry['holdout']['relative_l2']) < 1e-10
            thresholds[th] = dict(relative_l2=e, meets_threshold=e <= float(th))
        reports.append(dict(method=method, seed=record['seed'], relative_l2=error,
                            difference=delta, thresholds=thresholds))
    np.savez_compressed(output/'independent_predictions.npz', **arrays)
    report = dict(all_passed=True, source=str(source), runs=reports,
                  method='CPU checkpoint inference against an independently implemented NumPy reference on the original held-out points.')
    (output/'evaluation.json').write_text(json.dumps(report, indent=2)+'\n')
    return report
