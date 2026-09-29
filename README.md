# Finite-difference neural solvers

Numerical implementations for Hamilton-Jacobi-Bellman and Hamilton-Jacobi-Isaacs
equations, using direct residual minimization and policy iteration.

| Directory | Experiments |
|---|---|
| `experiments/periodic` | Periodic degenerate equations with exact solutions, neural solvers, and finite-difference references |
| `experiments/air3d` | Air3D neural solvers, monotone GPU references, and interpolation-based evaluation |
| `experiments/finite_controls` | Finite-action Isaacs games with direct and cached-policy training |
| `experiments/defects` | Fourier discretization and controlled policy-evaluation perturbations |
| `configs/experiments.json` | Dimensions, seeds, discretization parameters, network settings, and training budgets |

## Environment

Use Python 3.12. Install a CUDA-enabled PyTorch build for neural training and
periodic finite-difference reference solves, then install the dependencies:

```sh
python -m pip install -r requirements-training.txt
```

Air3D reference solves also require CuPy:

```sh
python -m pip install -r requirements-reference.txt
```

Fourier experiments and analytic checks run on CPU. On Windows, the commands
below enable UTF-8 explicitly.

## List and run experiments

```sh
python -X utf8 -B run.py list
python -X utf8 -B run.py list --family finite_controls
python -X utf8 -B run.py run --id periodic/fixed_d2_FD_PI_nu0.01_seed0 --output outputs/periodic_example
python -X utf8 -B run.py run --id finite_controls/seed1 --output outputs/finite_seed1
python -X utf8 -B run.py run --id air3d/main --output outputs/air3d
python -X utf8 -B run.py run --id defects/grid33 --output outputs/defects_grid33
```

A periodic neural configuration trains one network. Each Air3D configuration
specifies a campaign of methods and seeds. Each finite-control configuration
trains three methods for one seed. A Fourier configuration runs all prescribed
perturbation channels and schedules. Output directories must be new.

The configuration registry contains fixed update counts or elapsed-training-time
budgets. Runs save their configuration, diagnostics, timing history, and final
checkpoints. Time budgets include synchronized training operations and exclude
diagnostic evaluation and checkpoint writes.

## Evaluate generated outputs

```sh
python -X utf8 -B run.py evaluate --family periodic --input outputs/periodic_example --output outputs/periodic_eval
python -X utf8 -B run.py evaluate --family finite_controls --input outputs/finite_seed1 --output outputs/finite_eval
python -X utf8 -B run.py evaluate --family defects --input outputs/defects_grid33 --output outputs/defects_eval
```

Finite-control training saves the held-out coordinates so that subsequent
checkpoint evaluation can run on CPU with the same points.

Generate an Air3D reference before evaluating an Air3D campaign:

```sh
python -X utf8 -B run.py run --id air3d/reference/air_n384_nu02_eps01 --output outputs/references/air_n384_nu02_eps01
python -X utf8 -B run.py evaluate --family air3d --input outputs/air3d --references outputs/references --output outputs/air3d_eval
```

Additional reference configurations can be generated under `outputs/references`
using their configuration names as directory names. Air3D query-only solves
generate common states using NumPy seed 900302, or accept an explicit
`--queries` NPZ file with `states` and `times` arrays.

## Numerical checks

```sh
python -X utf8 -B run.py check --family periodic
python -X utf8 -B run.py check --family air3d
python -X utf8 -B run.py check --family finite_controls
python -X utf8 -B run.py check --family defects
```

These check analytic derivatives, equation identities, control selection,
finite-difference consistency, and Fourier operators. The `run` command starts
training or a reference solve; `list`, `check`, and `evaluate` have separate roles.
