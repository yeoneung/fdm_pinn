"""Generate only the Air3D and prescribed-error exhibits in the numerical comparisons."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plots import common

HERE = Path(__file__).resolve().parent
METHODS = ['AD_direct', 'FD_direct', 'AD_PI', 'FD_PI']
NAMES = ['Direct PINN', 'FD-direct', 'AD-PINN-PI', 'FD-PINN-PI']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def plain_table(name, spec, header, rows):
    lines = [r'\begin{tabular}{' + spec + '}', r'\toprule', header + r' \\', r'\midrule']
    lines += [row + r' \\' for row in rows]
    lines += [r'\bottomrule', r'\end{tabular}']
    (HERE / 'tables' / (name + '.tex')).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    common.HERE = HERE
    fixed = read(HERE / 'data/air_fixed.json')
    timed = read(HERE / 'data/air_time.json')
    sensitivity = read(HERE / 'data/fd_sensitivity.json')
    for name, selected, digits, timed_tag in [
        ('air', METHODS, 2, '600 seconds'),
        ('primary_air', ['AD_direct', 'FD_PI'], 1, '600-second budget'),
    ]:
        rows = []
        for tag, data in [('30,000 updates', fixed), (timed_tag, timed)]:
            rows.append(r'\multicolumn{4}{l}{\textit{' + tag + r'}} \\')
            for key in selected:
                m = data['methods'][key]
                updates = r'$30\,000$' if data is fixed else common.pm(m['optimizer_updates'], digits=0)
                rows.append(f"{NAMES[METHODS.index(key)]} & {common.pm(m['pooled_relative_l2'], 100)} & "
                            f"{common.pm(m['training_seconds'], digits=digits)} & {updates} " + r'\\')
            if data is fixed:
                rows.append(r'\midrule')
        common.table(name, 'lrrr', r'Method & Value error (\%) & Time (s) & Updates', rows)
    rows = []
    for g in sensitivity['groups']:
        rows.append(f"{g['h']:.3f} & {g['tau']:.6f} & {NAMES[METHODS.index(g['method'])]} & "
                    f"{common.pm(g['pooled_error'], 100)} & "
                    f"{common.pm(g['consistency_rms'], 1000)} & "
                    f"{common.pm(g['fd_roundoff_rms'], 1000)} " + r'\\')
    common.table('fd_sensitivity', 'rrlrrr',
                 r'$h$ & $\tau$ & Method & Error (\%) & '
                 r'\shortstack{Consistency\\($10^{-3}$)} & '
                 r'\shortstack{Precision\\($10^{-3}$)}', rows)
    rows = []
    for method, name in zip(METHODS, NAMES):
        m = fixed['methods'][method]
        fields = ['continuum_residual_rms', 'consistency_rms', 'boundary_normal_rms']
        rows.append(name + ' & ' + ' & '.join(f"${m[k]['mean']:.4f}\\pm{m[k]['std']:.4f}$" for k in fields))
    plain_table('air_boundary_audit', 'lccc', 'Method & Domain residual RMS & FD--AD discrepancy & Boundary normal RMS', rows)

    reference_rows = read(HERE / 'data/air_reference_tables.json')
    for name, spec, header in [
        ('air_reference_audit', 'llrr', r'Channel & Variation & Relative difference [\%] & Max sampled difference'),
        ('air_zero_comparator', 'rrcc', r'$\nu$ & Cells & $t=0$ difference [\%] & Pooled difference [\%]'),
        ('air_domain_audit', 'rrcc', r'Half extent & Cells & $t=0$ difference [\%] & Pooled difference [\%]'),
    ]:
        plain_table(name, spec, header, reference_rows[name])

    data = read(HERE / 'data/perturbations.json')
    rows = data['rows']
    final = [r for r in rows if r['iteration'] == 12 and r['schedule'] == 'persistent']
    channel_labels = {'attainment': 'Control optimization'}
    lines = [r'\begin{tabular}{lrrr}', r'\toprule', r'Perturbed term & $\sqrt{\overline E_{12}}$ & $\overline E_{12}/B_{12}$ & $\overline E_{12}/B^r_{12}$\\', r'\midrule']
    for r in final:
        lines.append(f"{channel_labels.get(r['channel'], r['channel'].capitalize())} & ${np.sqrt(r['energy_error']):.5f}$ & ${r['energy_to_full_history_ratio']:.5f}$ & ${r['energy_error']/r['evaluation_only_diagnostic']:.3f}$" + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (HERE / 'tables/defect_channels.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    plt.rcdefaults()
    colors = {'persistent': '#ce563d', 'decaying': '#16789c', 'early_pulse': '#7d52a1'}
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.7), sharex=True, sharey=True)
    for ax, channel in zip(axes.ravel(), ['momentum', 'attainment', 'terminal', 'combined']):
        for schedule, color in colors.items():
            rr = [r for r in rows if r['channel'] == channel and r['schedule'] == schedule]
            nn = [r['iteration'] for r in rr]
            ax.semilogy(nn, np.maximum(1e-13, np.sqrt([r['energy_error'] for r in rr])), 'o-', color=color, label=schedule.replace('_', ' '), markersize=3)
            ax.semilogy(nn, np.sqrt([r['full_history_diagnostic'] for r in rr]), '--', color=color, alpha=.65)
        ax.set(title=channel_labels.get(channel, channel.capitalize()), xlabel='Policy iteration', ylabel=r'$\sqrt{\overline{E}_n}$ and $\sqrt{B_n}$')
        ax.grid(alpha=.2)
        ax.set_xticks([2, 4, 6, 8, 10, 12])
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / 'figures/defect_channels.pdf')
    fig.savefig(HERE / 'figures/defect_channels.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
