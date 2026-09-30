"""Generate the fixed-original-PDE and matched-time dimension-study results."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plots.common import table,pm,check_summary

HERE=Path(__file__).resolve().parent
COLORS=['#3567A8','#B15B30']
NAMES={'AD_direct':'Direct PINN','FD_PI':'FD-PINN-PI'}


def main():
    d=json.loads((HERE/'data/periodic.json').read_text(encoding='utf-8'))
    assert d['all_passed'] and d['configured_runs']==39 and d['failed_runs']==0
    checks=check_summary(d['groups']); g=d['groups']; refs=d['reference_summary']
    fine=g['d2_FD_PI_nu0.01']; coarse=g['d2_FD_PI_nu0.08']; direct=g['d2_AD_direct_nu0']
    rows=[f"Direct PINN & $0$ & {pm(direct['relative_l2'],100)} & --- & --- & {pm(direct['training_seconds'],digits=1)} "+r'\\']
    for nu in [.08,.04,.02,.01]:
        x=g[f'd2_FD_PI_nu{nu:g}']
        rows.append(f"FD-PINN-PI & ${nu:.2f}$ & {pm(x['relative_l2'],100)} & "
                    f"{pm(x['neural_relative_l2_to_regularized_reference'],100)} & "
                    f"${100*refs[f'{nu:g}']['relative_l2_bias']:.3f}$ & {pm(x['training_seconds'],digits=1)} "+r'\\')
    table('fixed_target','lrrrrr',r'Method & $\nu$ & Original error (\%) & To FDM (\%) & FDM bias (\%) & Time (s)',rows)
    rows=[]
    for dim in [5,10,20,50]:
        for method in ['AD_direct','FD_PI']:
            x=g[f'd{dim}_{method}']
            rows.append(f"{dim} & {NAMES[method]} & {pm(x['relative_l2'],100)} & {pm(x['steps'],digits=0)} & {pm(x['memory_mib'],digits=1)} "+r'\\')
        if dim!=50:rows.append(r'\midrule')
    table('dimension_scaling','rlrrr',r'$d$ & Method & Original error (\%) & Updates & Memory (MiB)',rows)
    rows=[]
    for dim in [5,10,20,50]:
        for method in ['AD_direct','FD_PI']:
            runs=[r for r in d['individual_runs'] if r['stage']=='dimension_scaling' and r['config']['dimension']==dim and r['method']==method]
            def diagnostic(key):
                values=np.array([r['final'][key] for r in runs])
                return {'mean':float(values.mean()),'std':float(values.std(ddof=1))}
            rows.append(f"{dim} & {NAMES[method]} & {pm(g[f'd{dim}_{method}']['gradient_relative_l2'],100)} & "
                        f"{pm(diagnostic('original_residual_rms'),digits=4)} & {pm(diagnostic('modified_residual_rms'),digits=4)} & "
                        f"{pm(diagnostic('fd_residual_rms'),digits=4)} "+r'\\')
    table('priority_derivative_diagnostics','rlrrrr',r'$d$ & Method & Gradient error (\%) & $R_0$ RMS & $R_\nu$ RMS & $R_h$ RMS',rows)
    rows=[]
    for name,ref in sorted(d['reference_runs'].items(),key=lambda x:(-x[1]['viscosity'],x[1]['n'],x[1]['dt'])):
        rows.append(f"{ref['viscosity']:.2f} & {ref['n']} & {ref['dt']:.6g} & {ref['steps']} & {max(ref['extra_stabilization']):.6g} "+r'\\')
    table('priority_reference_details','rrrrr',r'$\nu$ & Grid & $\Delta t$ & Steps & Extra variance',rows)
    rows=[]
    for dim in [2,5,10,20,50]:
        xs=[g[f'd2_FD_PI_nu{nu:g}'] for nu in [.08,.04,.02,.01]] if dim==2 else [g[f'd{dim}_FD_PI']]
        for x in xs:
            cfg=x['config']
            record=next(r for r in d['individual_runs'] if r['config']==cfg and r['method']=='FD_PI')
            c=record['self_checks']
            rows.append(f"{dim} & {cfg['viscosity']:.6g} & {cfg['h']:.6g} & {cfg['tau']:.6g} & {c['sufficient_spatial_margin']:.6g} & {c['time_cfl']:.3f} "+r'\\')
    table('priority_coefficients','rrrrrr',r'$d$ & $\nu$ & $h$ & $\tau$ & Spatial margin & Time CFL',rows)

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(8.2,3.25),constrained_layout=True)
    nus=[.01,.02,.04,.08]
    for i,nu in enumerate(nus):
        x=g[f'd2_FD_PI_nu{nu:g}']
        for ax,key in zip(axes,['relative_l2','neural_relative_l2_to_regularized_reference']):
            v=np.array(x[key]['values'])*100
            ax.scatter(np.full(len(v),nu),v,color=COLORS[1],alpha=.55,s=19,zorder=3)
            ax.errorbar(nu,v.mean(),yerr=v.std(ddof=1),fmt='D',color=COLORS[1],capsize=3,markersize=5)
    axes[0].plot(nus,[100*g[f'd2_FD_PI_nu{v:g}']['relative_l2']['mean'] for v in nus],color=COLORS[1],label='FD-PINN-PI: original error')
    axes[0].plot(nus,[100*refs[f'{v:g}']['relative_l2_bias'] for v in nus],color='#50836c',marker='s',label='FDM regularization bias')
    axes[0].axhline(100*direct['relative_l2']['mean'],color=COLORS[0],ls='--',label='Direct PINN: original error')
    axes[0].legend(fontsize=9)
    axes[1].plot(nus,[100*g[f'd2_FD_PI_nu{v:g}']['neural_relative_l2_to_regularized_reference']['mean'] for v in nus],color=COLORS[1])
    for ax in axes:
        ax.set_xscale('log',base=2);ax.set_xticks(nus,[f'{v:.2f}' for v in nus]);ax.set_xlabel('Artificial viscosity');ax.grid(alpha=.2)
    axes[0].set_ylabel('Relative error (%)');axes[0].set_title('Fixed original HJB problem')
    axes[1].set_ylabel('Relative error (%)');axes[1].set_title('Network vs regularized FDM')
    for ext in ['pdf','png']:fig.savefig(HERE/f'figures/fixed_target_refinement.{ext}',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(8.6,3.3),constrained_layout=True)
    dims=[5,10,20,50]
    for method,color,marker in zip(['AD_direct','FD_PI'],COLORS,['o','D']):
        for ax,key,factor in zip(axes,['relative_l2','steps','memory_mib'],[100,1,1]):
            y=np.array([g[f'd{dim}_{method}'][key]['mean'] for dim in dims])*factor
            sd=np.array([g[f'd{dim}_{method}'][key]['std'] for dim in dims])*factor
            ax.errorbar(dims,y,yerr=sd,fmt=marker+'-',color=color,capsize=3,markersize=5,label=NAMES[method])
            for dim in dims:ax.scatter(np.full(3,dim),factor*np.array(g[f'd{dim}_{method}'][key]['values']),color=color,s=12,alpha=.35)
    for ax in axes:ax.set(xlabel='Dimension',xticks=dims);ax.grid(alpha=.2)
    axes[0].set(ylabel='Original-equation error (%)',title='180-second accuracy');axes[0].legend(fontsize=9)
    axes[1].set(ylabel='Completed updates',title='Training work')
    axes[2].set(ylabel='Peak CUDA allocation (MiB)',title='Training memory')
    for ext in ['pdf','png']:fig.savefig(HERE/f'figures/dimension_scaling.{ext}',dpi=170)
    plt.close(fig)


if __name__=='__main__':main()
