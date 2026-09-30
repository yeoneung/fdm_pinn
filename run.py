"""Run finite-difference and neural Hamilton-Jacobi experiments."""
from pathlib import Path
import argparse
import importlib
import json
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
FAMILIES = ['periodic', 'air3d', 'finite_controls', 'defects']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def fresh_output(path):
    path = Path(path).resolve()
    if path == ROOT or path.exists():
        raise ValueError(f'Choose a new output directory: {path}')
    for name in ['experiments', 'configs', 'analysis', 'plots', '.git']:
        if path.is_relative_to(ROOT / name):
            raise ValueError('Outputs must be outside source and configuration directories.')
    return path


def load(family, name):
    sys.path.insert(0, str(ROOT / 'experiments' / family))
    return importlib.import_module(name)


def run_job(job, output, queries=None):
    family, kind = job['family'], job['kind']
    if family == 'periodic':
        if kind == 'neural':
            load(family, 'solver').train(job['job'], output)
        else:
            import torch
            torch.set_num_threads(2)
            torch.backends.cuda.matmul.allow_tf32 = False
            load(family, 'reference').solve(job['n'], job['nu'], output,
                                           dt_cap=job['dt_cap'])
    elif family == 'air3d':
        if kind == 'neural_campaign':
            base = load(family, 'train_air3d')
            cfg = base.Config(**job['config'])
            if job['budget_seconds'] is None:
                base.suite(cfg, job['seeds'], job['methods'], output)
            else:
                load(family, 'train_equal_time').main(cfg, job['budget_seconds'],
                                                     job['seeds'], output)
        elif kind == 'reference':
            base = load(family, 'gpu_reference')
            base.solve(base.Config(**job['config']), output)
        else:
            cfg = job['config']
            load(family, 'query_reference').main(
                cfg['cells'], cfg['viscosity'], cfg['epsilon'], output,
                cfg['extent'], cfg['angle_cells'], queries)
    elif family == 'finite_controls':
        from experiments.finite_controls import solver
        solver.configure()
        solver.paired(solver.Config(**job['config']), job['seed'], output,
                      fresh_baseline=True)
    elif kind == 'targeted':
        load(family, 'targeted_refinement').run(
            job['config']['grid'], job['config']['steps'], output)
    else:
        base = load(family, 'benchmark_channels')
        base.run(base.base.Config(**job['config']), output)


def run_all(registry, output, resume=False, dry_run=False):
    from analysis.common import validate_run, write
    output = output.resolve()
    if output == ROOT or any(output.is_relative_to(ROOT/name) for name in
                             ['experiments', 'configs', 'analysis', 'plots', '.git']):
        raise ValueError('Outputs must be outside source and configuration directories.')
    manifest = {'runs': {j['id']: j['id'] for j in registry}}
    if output.exists():
        if not resume or not (output/'inputs.json').is_file():
            raise ValueError('Choose a new output directory, or use --resume with an existing inputs.json.')
        if read(output/'inputs.json') != manifest:
            raise ValueError('The existing input manifest differs from the configuration registry.')
    else:
        fresh_output(output)
    pending = []
    for job in registry:
        directory = output/job['id']
        if directory.exists():
            validate_run(job, directory)
        else:
            pending.append(job)
    print(f'{len(pending)} configurations pending; {len(registry)-len(pending)} complete.', flush=True)
    if dry_run:
        for job in pending:
            print(job['id'])
        return
    if not output.exists():
        output.mkdir(parents=True)
        write(output/'inputs.json', manifest)
    for index, job in enumerate(pending, 1):
        print(f"[{index}/{len(pending)}] {job['id']}", flush=True)
        # Isolate each job's imports and GPU allocations.
        subprocess.run([sys.executable, '-X', 'utf8', '-B', str(ROOT/'run.py'),
                        'run', '--id', job['id'], '--output', str(output/job['id'])], check=True)
        validate_run(job, output/job['id'])


def evaluate(family, source, output, references):
    if family == 'finite_controls':
        from experiments.finite_controls.results_tools import evaluate_campaign
        print(json.dumps(evaluate_campaign(source, output), indent=2))
    elif family == 'defects':
        load(family, 'verify_results').main(source, output)
    elif family == 'air3d':
        if references is None:
            raise ValueError('Air3D evaluation requires --references DIR.')
        load(family, 'evaluate_reference').main(source, output, 16384, references)
    else:
        import torch
        from dataclasses import replace
        torch.set_num_threads(2)
        base = load(family, 'solver')
        checkpoint = torch.load(source / 'checkpoint.pt', map_location='cpu',
                                weights_only=False)
        cfg = replace(base.Config(**checkpoint['config']), device='cpu')
        model = base.Network(cfg).to(dtype=torch.float64).eval()
        model.load_state_dict(checkpoint['model'])
        directions = checkpoint['Q'].to(device='cpu', dtype=torch.float64)
        result = base.evaluate(model, base.points(cfg.final_samples, 701502, cfg),
                               cfg, directions, True)
        output.mkdir(parents=True)
        (output / 'evaluation.json').write_text(json.dumps(result, indent=2) + '\n',
                                                 encoding='utf-8')
        print(json.dumps(result, indent=2))


def check(family):
    if family == 'defects':
        base = load(family, 'benchmark_channels').base
        return base.self_checks(base.Config(grid=33))
    import torch
    torch.set_num_threads(2)
    if family == 'periodic':
        base = load(family, 'solver')
        cfg = base.Config(device='cpu')
        directions = base.directions(cfg, torch.empty((), dtype=torch.float64))
        return base.checks(cfg, directions)
    if family == 'air3d':
        base = load(family, 'train_air3d')
        return base.self_checks(base.Config(device='cpu'))
    from experiments.finite_controls import solver
    solver.configure()
    return solver.check(solver.Problem(solver.Config(device='cpu')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('list', help='List available experiment configurations')
    p.add_argument('--family', choices=FAMILIES)
    p = commands.add_parser('run', help='Train networks or solve a reference equation')
    p.add_argument('--id', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--queries', type=Path, help='Optional states/times NPZ for Air3D queries')
    p = commands.add_parser('run-all', help='Run all configured experiments sequentially')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true', help='Skip complete runs with matching configurations')
    p.add_argument('--dry-run', action='store_true', help='List pending runs without executing or writing files')
    p = commands.add_parser('analyze', help='Evaluate and aggregate all completed experiment outputs')
    p.add_argument('--input', type=Path, required=True, help='JSON manifest mapping configuration IDs to run directories')
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('plot', help='Generate tables and figures from computed summaries')
    p.add_argument('--input', type=Path, required=True, help='Output directory from analyze')
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('evaluate', help='Evaluate outputs from a completed run')
    p.add_argument('--family', choices=FAMILIES, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--references', type=Path, help='Directory containing Air3D reference runs')
    p = commands.add_parser('check', help='Run analytic and operator checks on CPU')
    p.add_argument('--family', choices=FAMILIES, required=True)
    args = parser.parse_args()
    if args.command == 'check':
        print(json.dumps(check(args.family), indent=2))
        return
    if args.command == 'evaluate':
        evaluate(args.family, args.input.resolve(), fresh_output(args.output),
                 args.references.resolve() if args.references else None)
        return
    registry = read(ROOT / 'configs/experiments.json')
    if args.command == 'run-all':
        run_all(registry, args.output, args.resume, args.dry_run)
        return
    if args.command == 'analyze':
        from analysis.workflow import main as analyze
        analyze(args.input, registry, fresh_output(args.output))
        return
    if args.command == 'plot':
        from plots.render import main as plot
        plot(args.input.resolve(), fresh_output(args.output))
        return
    if args.command == 'list':
        for job in registry:
            if args.family is None or job['family'] == args.family:
                print(f"{job['id']}  [{job['kind']}]")
        return
    jobs = [job for job in registry if job['id'] == args.id]
    if len(jobs) != 1:
        parser.error('Unknown experiment ID; use "python run.py list".')
    if args.queries and not (jobs[0]['family'] == 'air3d'
                            and jobs[0]['kind'] == 'query_reference'):
        parser.error('--queries applies only to Air3D query configurations.')
    run_job(jobs[0], fresh_output(args.output), args.queries)


if __name__ == '__main__':
    main()
