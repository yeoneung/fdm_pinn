"""Generate the configured longer-budget fifty-dimensional accuracy comparison."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plots.common import table,pm,check_summary

HERE=Path(__file__).resolve().parent


def main():
    d=json.loads((HERE/'data/extended.json').read_text());assert d['all_passed'] and d['configured_runs']==6
    count=check_summary(d['groups']); names={'AD_direct':'Direct PINN','FD_PI':'FD-PINN-PI'}
    rows=[]
    for method in ['AD_direct','FD_PI']:
        g=d['groups'][method]
        rows.append(f"{names[method]} & {pm(g['relative_l2'],100)} & {pm(g['steps'],digits=0)} "+r'\\')
    table('extended_fifty','lrr',r'Method & Original error (\%) & Updates',rows)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(6.6,3.3),constrained_layout=True)
    for method,color,marker in [('AD_direct','#3567A8','o'),('FD_PI','#B15B30','D')]:
        runs=sorted([r for r in d['individual_runs'] if r['method']==method],key=lambda x:x['seed'])
        for r in runs:
            ax.plot([h['training_seconds'] for h in r['history']],[100*h['relative_l2'] for h in r['history']],color=color,alpha=.28,lw=1)
        if len({len(r['history']) for r in runs}) == 1:
            times=np.array([[h['training_seconds'] for h in r['history']] for r in runs]);values=100*np.array([[h['relative_l2'] for h in r['history']] for r in runs])
            ax.errorbar(times.mean(0),values.mean(0),yerr=values.std(0,ddof=1),fmt=marker+'-',color=color,capsize=3,label=names[method])
        else:
            ax.plot([], [], color=color, label=names[method]+' (individual runs)')
    ax.set(xlabel='Training time (s)',ylabel='Monitor error against original solution (%)',title='Fresh 900-second runs in 50 dimensions')
    ax.grid(alpha=.2);ax.legend()
    for ext in ['pdf','png']:fig.savefig(HERE/f'figures/extended_fifty_history.{ext}',dpi=170)
    plt.close(fig)


if __name__=='__main__':main()
