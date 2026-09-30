"""Utilities for aggregating numerical runs."""
from pathlib import Path
import hashlib
import json
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stats(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Sample mean/SD requires at least two finite observations.')
    return dict(mean=float(values.mean()), std=float(values.std(ddof=1)), values=values.tolist())


def relative(prediction, reference):
    return float(np.linalg.norm(prediction-reference)/np.linalg.norm(reference))


def validate_run(job, directory):
    """Reject incomplete runs and outputs belonging to another configuration."""
    records = ([directory/m/'run.json' for m in job['methods']]
               if job['family'] == 'finite_controls' else [directory/'run.json'])
    for file in records:
        record = read(file)
        if record.get('status') != 'complete':
            raise ValueError(f"{job['id']}: run is not complete")
        if job['family'] == 'periodic' and job['kind'] == 'neural':
            if record['job'] != job['job']:
                raise ValueError(f"{job['id']}: configuration differs from the registry")
        elif 'config' in job:
            for key, value in job['config'].items():
                if record['config'][key] != value:
                    raise ValueError(f"{job['id']}: configuration field {key} differs")
        elif job['family'] == 'periodic':
            if record['n'] != job['n'] or record['viscosity'] != job['nu']:
                raise ValueError(f"{job['id']}: reference configuration differs")
        if job['family'] == 'finite_controls' and record['seed'] != job['seed']:
            raise ValueError(f"{job['id']}: seed differs")
        if job['family'] == 'finite_controls' and record['method'] != file.parent.name:
            raise ValueError(f"{job['id']}: method differs")
    if job['family'] == 'air3d' and job['kind'] == 'neural_campaign':
        if record['seeds'] != job['seeds'] or record['methods'] != job['methods']:
            raise ValueError(f"{job['id']}: seeds or methods differ")
        if record.get('budget_seconds') != job['budget_seconds']:
            raise ValueError(f"{job['id']}: training-time budget differs")


def paths(manifest, registry):
    manifest = Path(manifest).resolve()
    entries = read(manifest)['runs']
    missing = [j['id'] for j in registry if j['id'] not in entries]
    if missing:
        raise ValueError('Missing input entries: '+', '.join(missing))
    result = {key: (manifest.parent / value).resolve() for key, value in entries.items()}
    for job in registry:
        validate_run(job, result[job['id']])
    return result
