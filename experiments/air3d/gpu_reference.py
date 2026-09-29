"""Fixed-viscosity monotone central/Lax--Friedrichs GPU reference solver.

The diagonal diffusion coefficient is max(Sigma_jj+nu, h_j*a_j), where a_j
bounds the coordinate Hamiltonian slope. This is a consistent monotone
stabilization; its excess diffusion is reported, not silently treated as nu.
Boundary ghosts replicate the endpoint.
Angular grids omit the duplicate endpoint and wrap exactly.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, sys, time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
_dll_handles = []
for base in [Path(sys.prefix) / 'Lib/site-packages', Path(sys.base_prefix) / 'Lib/site-packages']:
    for relative in ['nvidia/cuda_runtime/bin', 'nvidia/cuda_nvrtc/bin', 'torch/lib']:
        p = base / relative
        if os.name == 'nt' and p.exists():
            _dll_handles.append(os.add_dll_directory(str(p)))
            os.environ['PATH'] = str(p) + os.pathsep + os.environ.get('PATH', '')
import cupy as cp

@dataclass
class Config:
    problem: str = 'air'
    cells: int = 384
    angle_cells: int = 384
    horizon: float = 0.5
    extent: float = 3.0
    viscosity: float = 0.2
    sigma: float = 0.1
    epsilon: float = 0.1
    omega: float = 1.5
    cfl: float = 0.8
    dtype: str = 'float64'
    snapshots: int = 6
    crop: float = 1.5
SOURCE = '\nextern "C" __global__ void advance(const REAL* u, REAL* v,\n const REAL* coords, const REAL* trig, int nx, int nz, int problem,\n REAL h, REAL hz, REAL dt, REAL nu, REAL sig, REAL omega) {\n long long id=(long long)blockDim.x*blockIdx.x+threadIdx.x;\n long long total=(long long)nx*nx*nz;\n if(id>=total) return;\n int k=id%nz, j=(id/nz)%nx, i=id/(nz*nx);\n long long sx=(long long)nx*nz, sy=nz;\n REAL c=u[id], xp=u[id+(i<nx-1?sx:0)], xm=u[id-(i>0?sx:0)];\n REAL yp=u[id+(j<nx-1?sy:0)], ym=u[id-(j>0?sy:0)];\n long long kp=(k<nz-1?1:-(nz-1));\n long long km=(k>0?-1:nz-1);\n REAL zp=u[id+kp], zm=u[id+km];\n REAL px=(xp-xm)/(2*h),py=(yp-ym)/(2*h),pz=(zp-zm)/(2*hz);\n REAL x=coords[i],y=coords[j],H,ax,ay,az,d0,d1,d2;\n   REAL f0=(REAL)-.5+(REAL).5*trig[k], f1=(REAL).5*trig[nz+k];\n   H=f0*px+f1*py+omega*fabs(y*px-x*py-pz)-omega*fabs(pz);\n   ax=fabs(f0)+omega*fabs(y);ay=fabs(f1)+omega*fabs(x);az=2*omega;\n   d0=nu;d1=nu;d2=nu;\n d0=fmax(d0,h*ax);d1=fmax(d1,h*ay);d2=fmax(d2,hz*az);\n REAL diff=d0*(xp-2*c+xm)/(h*h)+d1*(yp-2*c+ym)/(h*h)+d2*(zp-2*c+zm)/(hz*hz);\n v[id]=c+dt*(H+(REAL).5*diff);\n}\n'

def arrays(cfg):
    x = np.linspace(-cfg.extent, cfg.extent, cfg.cells + 1)
    h = 2 * cfg.extent / cfg.cells
    z = np.linspace(-np.pi, np.pi, cfg.angle_cells, endpoint=False)
    hz = 2 * np.pi / cfg.angle_cells
    trig = np.concatenate([np.cos(z), np.sin(z)])
    xy = np.sqrt(x[:, None] ** 2 + x[None, :] ** 2 + cfg.epsilon ** 2) - 0.5
    initial = np.broadcast_to(xy[:, :, None], (len(x), len(x), len(z))).copy()
    a = np.array([1 + cfg.omega * cfg.extent, 0.5 + cfg.omega * cfg.extent, 2 * cfg.omega])
    phys = np.full(3, cfg.viscosity)
    spacing = np.array([h, h, hz])
    effective = np.maximum(phys, a * spacing)
    return (x, z, trig, initial, dict(spacing=spacing.tolist(), physical_diffusion=phys.tolist(), max_effective_diffusion=effective.tolist(), max_extra_diffusion=(effective - phys).tolist(), hamiltonian_slope_bounds=a.tolist(), dt_limit=float(1 / np.sum(effective / spacing ** 2))))

def numpy_step(u, cfg, dt):
    x, z, trig, _, diag = arrays(cfg)
    h, h2, hz = diag['spacing']
    xp = np.concatenate([u[1:], u[-1:]], 0)
    xm = np.concatenate([u[:1], u[:-1]], 0)
    yp = np.concatenate([u[:, 1:], u[:, -1:]], 1)
    ym = np.concatenate([u[:, :1], u[:, :-1]], 1)
    zp = np.roll(u, -1, 2)
    zm = np.roll(u, 1, 2)
    px = (xp - xm) / (2 * h)
    py = (yp - ym) / (2 * h)
    pz = (zp - zm) / (2 * hz)
    xx = x[:, None, None]
    yy = x[None, :, None]
    zz = z[None, None, :]
    f0 = -0.5 + 0.5 * np.cos(zz)
    f1 = 0.5 * np.sin(zz)
    H = f0 * px + f1 * py + cfg.omega * abs(yy * px - xx * py - pz) - cfg.omega * abs(pz)
    aa = (abs(f0) + cfg.omega * abs(yy), abs(f1) + cfg.omega * abs(xx), 2 * cfg.omega)
    second = ((xp - 2 * u + xm) / h ** 2, (yp - 2 * u + ym) / h ** 2, (zp - 2 * u + zm) / hz ** 2)
    diffusion = sum((np.maximum(d, a * hs) * dd for d, a, hs, dd in zip(diag['physical_diffusion'], aa, diag['spacing'], second)))
    return u + dt * (H + 0.5 * diffusion)

def get_kernel(dtype):
    return cp.RawKernel(SOURCE.replace('REAL', 'double' if dtype == 'float64' else 'float'), 'advance', options=('--std=c++11',))

def kernel_args(u, v, cx, ct, cfg, diag, dt):
    real = np.float64 if cfg.dtype == 'float64' else np.float32
    return (u, v, cx, ct, np.int32(u.shape[0]), np.int32(u.shape[2]), np.int32(cfg.problem == 'air'), real(diag['spacing'][0]), real(diag['spacing'][2]), real(dt), real(cfg.viscosity), real(cfg.sigma), real(cfg.omega))

def self_checks():
    reports = []
    for problem in ['air']:
        cfg = Config(problem=problem, cells=12, angle_cells=16, extent=3.0)
        x, z, trig, initial, diag = arrays(cfg)
        dt = 0.8 * diag['dt_limit']
        rng = np.random.default_rng(817)
        u0 = initial + 0.01 * rng.normal(size=initial.shape)
        u = cp.asarray(u0)
        v = cp.empty_like(u)
        cx = cp.asarray(x)
        ct = cp.asarray(trig)
        kernel = get_kernel(cfg.dtype)
        grid = ((u.size + 255) // 256,)
        kernel(grid, (256,), kernel_args(u, v, cx, ct, cfg, diag, dt))
        out = cp.asnumpy(v)
        error = float(abs(out - numpy_step(u0, cfg, dt)).max())
        lower = out.copy()
        upper = u0 + rng.uniform(0, 0.02, size=u0.shape)
        u.set(upper)
        kernel(grid, (256,), kernel_args(u, v, cx, ct, cfg, diag, dt))
        min_gap = float((cp.asnumpy(v) - lower).min())
        u.fill(1.25)
        kernel(grid, (256,), kernel_args(u, v, cx, ct, cfg, diag, dt))
        constant = float(abs(cp.asnumpy(v) - 1.25).max())
        assert error < 1e-12 and min_gap >= -1e-12 and (constant < 1e-12)
        reports.append(dict(problem=problem, numpy_gpu_max_error=error, order_min_gap=min_gap, constant_error=constant))
    return reports

def solve(cfg, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    source = Path(__file__)
    record = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(), config=asdict(cfg), command=sys.argv, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), cupy=cp.__version__, numpy=np.__version__, device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(), boundary='endpoint-replicating Neumann ghosts; unique periodic angular nodes', reference_claim='numerical reference with separately measured refinement differences; not certified')

    def save():
        (output / 'run.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    save()
    try:
        record['self_checks'] = self_checks()
        cp.get_default_memory_pool().free_all_blocks()
        x, z, trig, initial, diag = arrays(cfg)
        record['scheme'] = diag
        save()
        u = cp.asarray(initial, dtype=cfg.dtype)
        v = cp.empty_like(u)
        del initial
        cx = cp.asarray(x, dtype=cfg.dtype)
        ct = cp.asarray(trig, dtype=cfg.dtype)
        kernel = get_kernel(cfg.dtype)
        grid = ((u.size + 255) // 256,)
        keep = np.where(np.abs(x) <= cfg.crop + 1e-10)[0]
        if not len(keep):
            raise ValueError('Empty crop')
        sl = slice(int(keep[0]), int(keep[-1]) + 1)

        def snapshot(index, t):
            arr = cp.asnumpy(u[sl, sl, :])
            if not np.isfinite(arr).all():
                raise FloatingPointError('Nonfinite reference')
            np.save(output / f'value_{index:02d}.npy', arr)
            print(f'{cfg.problem} cells={cfg.cells} snapshot t={t:.3f} range=[{arr.min():.5f},{arr.max():.5f}] elapsed={time.perf_counter() - start:.1f}s', flush=True)
        times = np.linspace(0, cfg.horizon, cfg.snapshots)
        np.savez(output / 'axes.npz', x=x[sl], z=z, times=times)
        snapshot(cfg.snapshots - 1, cfg.horizon)
        steps_each = math.ceil((times[1] - times[0]) / (cfg.cfl * diag['dt_limit']))
        dt = (times[1] - times[0]) / steps_each
        record.update(dt=dt, steps=steps_each * (cfg.snapshots - 1), shape=list(u.shape), stored_shape=[len(keep), len(keep), len(z)])
        save()
        cp.cuda.Stream.null.synchronize()
        solve_start = time.perf_counter()
        for index in reversed(range(cfg.snapshots - 1)):
            for step in range(steps_each):
                kernel(grid, (256,), kernel_args(u, v, cx, ct, cfg, diag, dt))
                u, v = (v, u)
            cp.cuda.Stream.null.synchronize()
            snapshot(index, times[index])
            record['last_completed_t'] = float(times[index])
            save()
        record.update(status='complete', solve_seconds=time.perf_counter() - solve_start, output_sha256={p.name: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest() for p in output.glob('*.np*')})
    except BaseException as exc:
        record.update(status='failed', exception=repr(exc))
        raise
    finally:
        record.update(elapsed_seconds=time.perf_counter() - start, finished_utc=datetime.now(timezone.utc).isoformat(), source_unchanged=hashlib.sha256(source.read_bytes()).hexdigest() == record['source_sha256'])
        save()
    return record
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name, typ in [('problem', str), ('cells', int), ('angle_cells', int), ('horizon', float), ('extent', float), ('viscosity', float), ('sigma', float), ('epsilon', float), ('omega', float), ('cfl', float), ('dtype', str), ('snapshots', int), ('crop', float)]:
        p.add_argument('--' + name.replace('_', '-'), type=typ, default=getattr(Config(), name))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output
    del a.output
    solve(Config(**vars(a)), out)
