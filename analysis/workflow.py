"""Build summaries from completed experiments, without training."""
from .common import paths, read, write, sha


def main(manifest, registry, output):
    from . import periodic, air, finite_controls
    import torch
    torch.set_num_threads(2)
    inputs = paths(manifest, registry)
    output.mkdir(parents=True, exist_ok=False)
    print('Evaluating periodic checkpoints and policy diagnostics...', flush=True)
    periodic.summarize(inputs, registry, output)
    print('Evaluating Air3D checkpoints and references...', flush=True)
    air.summarize(inputs, registry, output)
    print('Evaluating finite-control checkpoints...', flush=True)
    finite_controls.summarize(inputs, registry, output)
    matrices, targeted = {}, {}
    for job in registry:
        if job['family'] != 'defects':
            continue
        source = inputs[job['id']]
        record = read(source/'run.json')
        if job['kind'] == 'defect_matrix':
            data = read(source/'diagnostics.json')
            matrices[str(job['config']['grid'])] = data
        else:
            targeted[job['id'].split('/')[-1]] = dict(config=record['config'], rows=record['rows'],
                quadrature_refinement=record['quadrature_refinement'],
                independent_norm_check=record['independent_norm_check'])
    write(output/'perturbations.json', matrices['129'])
    write(output/'perturbation_refinement.json', dict(matrices=matrices, targeted=targeted))
    fingerprints = {}
    for key, source in inputs.items():
        files = [source/'run.json'] if (source/'run.json').exists() else sorted(source.glob('*/run.json'))
        fingerprints[key] = {str(f.relative_to(source)): sha(f) for f in files}
    report = dict(status='complete', configurations=len(registry), input_record_sha256=fingerprints,
                  summaries=sorted(p.name for p in output.glob('*.json')))
    write(output/'analysis_report.json', report)
    print(f'Analysis complete: {output}', flush=True)
