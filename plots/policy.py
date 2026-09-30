"""Generate the focused policy interval comparison and its diagnostic table."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plots.common import table,pm,check_summary

HERE=Path(__file__).resolve().parent


def main():
    path=HERE/'data/policy_intervals.json'
    if not path.exists():
        raise FileNotFoundError('Run the policy-interval configurations and analyze their outputs first.')
    d=json.loads(path.read_text());assert d['all_passed'] and d['additional_configured_runs']==9 and d['reused_primary_runs']==3
    checks=check_summary(d['groups']);groups=d['groups']
    keys=['FD_direct','K50','K200','K1000'];labels=['FD-direct',r'$K=50$',r'$K=200$',r'$K=1000$']
    rows=[]
    for key,label in zip(keys,labels):
        x=groups[key]
        rows.append(f"{label} & {pm(x['relative_l2'],100)} & {pm(x['steps'],digits=0)} & "
                    f"{pm(x['nonlinear_residual_rms'],digits=5)} & {pm(x['momentum_rms'],digits=5)} "+r'\\')
    table('policy_interval','lrrrr',r'Method & Original error (\%) & Updates & Nonlinear residual & Gradient lag',rows)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(8.2,3.2),constrained_layout=True)
    for ax,metric,factor,title in zip(axes,['relative_l2','nonlinear_residual_rms'],[100,1],['Original-equation error (%)','Final nonlinear FD residual (RMS)']):
        for i,key in enumerate(keys):
            v=np.array(groups[key][metric]['values'])*factor; color='#3567A8' if i==0 else '#B15B30'
            ax.scatter(i+np.array([-.07,0,.07]),v,color=color,s=22,alpha=.6)
            ax.errorbar(i,v.mean(),yerr=v.std(ddof=1),fmt='s',color=color,capsize=4,markersize=6)
        ax.set(xticks=range(4),xticklabels=['FD-direct','PI: 50','PI: 200','PI: 1000'],ylabel=title)
        ax.grid(axis='y',alpha=.2)
    for ext in ['pdf','png']:fig.savefig(HERE/f'figures/policy_interval.{ext}',dpi=170)
    plt.close(fig)


if __name__=='__main__':main()
