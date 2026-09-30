"""Generate fixed-time accuracy diagnostics without retraining any network."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plots.common import table,pm,check_summary

HERE=Path(__file__).resolve().parent


def main():
    d=json.loads((HERE/'data/time_slices.json').read_text());assert d['all_passed'] and d['evaluated_checkpoints']==45
    count=check_summary(d['groups']);g=d['groups'];names={'AD_direct':'Direct PINN','FD_PI':'FD-PINN-PI'}
    rows=[]
    for key in ['d2_AD_direct_nu0']+[f'd2_FD_PI_nu{nu:g}' for nu in [.08,.04,.02,.01]]:
        group=g[key];v=group['series'][0]
        reference=pm(v['to_fdm_relative_l2'],100) if 'to_fdm_relative_l2' in v else '---'
        bias=f"${100*v['fdm_bias_relative_l2']:.3f}$" if 'fdm_bias_relative_l2' in v else '---'
        rows.append(f"{names[group['method']]} & {group['viscosity']:.2f} & {pm(v['relative_l2'],100)} & {reference} & {bias} "+r'\\')
    table('initial_fixed_target','lrrrr',r'Method & $\nu$ & Original error (\%) & To FDM (\%) & FDM bias (\%)',rows)
    rows=[]
    for dim,budget in [(5,180),(10,180),(20,180),(50,180),(50,900)]:
        for method in ['AD_direct','FD_PI']:
            key=f'd{dim}_{method}'+('_900s' if budget==900 else '')
            v=g[key]['series'][0]
            rows.append(f"{dim} & {budget} & {names[method]} & {pm(v['relative_l2'],100)} & {pm(v['absolute_rms'],digits=5)} "+r'\\')
        if (dim,budget)!=(50,900):rows.append(r'\midrule')
    table('initial_dimension_study','rrlrr',r'$d$ & Budget (s) & Method & Original error (\%) & Absolute RMS',rows)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(8.2,3.25),constrained_layout=True)
    settings=[('d2_AD_direct_nu0','d2_FD_PI_nu0.01'),('d50_AD_direct_900s','d50_FD_PI_900s')]
    for ax,keys in zip(axes,settings):
        for key,color,marker in zip(keys,['#3567A8','#B15B30'],['o','D']):
            values=100*np.array([x['relative_l2']['values'] for x in g[key]['series']])
            for seed in range(3):ax.plot(d['times'],values[:,seed],color=color,alpha=.22,lw=1)
            ax.errorbar(d['times'],values.mean(1),yerr=values.std(1,ddof=1),fmt=marker+'-',color=color,capsize=3,label=names[g[key]['method']])
        ax.set(xlabel='Time',ylabel='Original-solution relative error (%)',xticks=d['times']);ax.grid(alpha=.2)
    axes[0].set_title('2D HJB: finest FD refinement');axes[0].legend(fontsize=9)
    axes[1].set_title('50D HJI: 900-second runs')
    for ext in ['pdf','png']:fig.savefig(HERE/f'figures/time_slice_accuracy.{ext}',dpi=170)
    plt.close(fig)


if __name__=='__main__':main()
