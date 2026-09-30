"""Finite-control comparison: three paired seeds, all three methods."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, ScalarFormatter
from plots.common import table, pm

HERE = Path(__file__).resolve().parent


def threshold_time(entry):
    if entry['n_reached'] == 0:
        return '---'
    if entry['n_reached'] < entry['n_total']:
        return f"{entry['n_reached']}/{entry['n_total']} reached"
    return pm(entry['time'], digits=1)


def main():
    data = json.loads((HERE/'data/finite_controls.json').read_text())
    labels = {'FD_direct': 'FD-direct, reused samples',
              'FD_PI_cached': 'FD-PINN-PI, cached policy',
              'FD_direct_fresh': 'FD-direct, fresh samples'}
    rows = []
    for method, label in labels.items():
        g = data['groups'][method]
        rows.append(f"{label} & {threshold_time(g['thresholds']['0.05'])} & "
                    f"{pm(g['relative_l2'], 100, digits=3)} & {pm(g['training_seconds'], digits=1)} "+r'\\')
    table('finite_controls', 'lrrr',
          r'Method & Time to $5\%$ (s) & Final error (\%) & Total time (s)', rows)
    with plt.rc_context({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42}):
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), constrained_layout=True)
        seen = set()
        colors = {'FD_direct': '#0072B2', 'FD_PI_cached': '#D55E00', 'FD_direct_fresh': '#009E73'}
        styles = {'FD_direct': '-', 'FD_PI_cached': '--', 'FD_direct_fresh': '-.'}
        for row in data['runs']:
            method, history = row['method'], row['history']
            for ax, key in zip(axes, ['training_seconds', 'step']):
                ax.plot([h[key] for h in history], [100*h['relative_l2'] for h in history],
                        color=colors[method], linestyle=styles[method], alpha=.8,
                        label=labels[method] if method not in seen else None)
            seen.add(method)
        for ax in axes:
            ax.set_ylabel(r'Relative $L^2$ error (%)')
            ax.set_yscale('log'); ax.grid(alpha=.2)
            ax.yaxis.set_major_locator(FixedLocator([2, 3, 5, 10, 20, 40]))
            ax.yaxis.set_major_formatter(ScalarFormatter()); ax.minorticks_off()
            ax.axhline(5, color='0.4', linewidth=.6, linestyle=':', alpha=.6)
        axes[0].set_xlabel('Accumulated training time (s)')
        axes[1].set_xlabel('Optimizer updates')
        axes[0].legend(fontsize=7)
        for ext in ['pdf', 'png']:
            fig.savefig(HERE/f'figures/finite_controls.{ext}', dpi=170)
        plt.close(fig)


if __name__ == '__main__':
    main()
