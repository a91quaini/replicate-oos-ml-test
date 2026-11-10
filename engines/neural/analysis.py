"""Learning-only pilot review and full-run size/power analysis."""
import numpy as np
import pandas as pd
from scipy.stats import binomtest

def records(out):
    from run import read
    for path in sorted((out/'draws').glob('*/*.json.gz')):
        from resample import check_saved,destination
        if not check_saved(out,path):raise RuntimeError('Missing bootstrap checkpoint')
        extras={(r['cell_id'],r['replication']):r['statistics'] for r in read(destination(out,path))['rows']}
        for r in read(path):
            r['statistics'].update(extras[r['cell_id'],r['replication']])
            yield r

def learning_review(out):
    from run import configuration
    cfg=configuration(out);cells={c['id']:c for c in cfg['cells']};groups={i:[] for i in cells}
    for r in records(out):groups[r['cell_id']].append(r)
    result=[]
    for i,c in cells.items():
        rr=groups[i];relative=[];boundary=[];epochs=[];gradient=[]
        for j in (0,1):
            per_draw=[np.mean([v['relative_forecast_error'] for v in r['learning'] if v['rule']==j]) for r in rr]
            relative.append(float(np.median(per_draw)))
        for r in rr:
            for v in r['learning']:
                boundary.append(v['fit']['selected_kappa']==3.)
                epochs.append(v['fit']['epochs']);gradient.append(v['fit']['projected_gradient'])
        drift=float(np.mean([abs(r['diagnostics']['normalized_corrected_drift']) for r in rr]))
        valid=float(np.mean([r['statistics']['orthogonal']['valid'] and r['statistics']['raw']['valid'] for r in rr]))
        ratios=[r['statistics']['orthogonal']['variance']/max(r['statistics']['oracle']['variance'],1e-24)
                for r in rr if r['statistics']['orthogonal']['valid'] and r['statistics']['oracle']['valid']]
        ratio=float(np.median(ratios)) if ratios else 0.
        passed=bool(max(relative)<=.95 and drift<=.5 and valid>=.95 and .1<=ratio<=5.)
        result.append(dict(cell_id=i,n=c['n'],T=c['T'],dgp=c['dgp'],draws=len(rr),
            median_relative_forecast_error_by_rule=relative,
            mean_absolute_normalized_corrected_drift=drift,valid_fraction=valid,
            median_corrected_to_oracle_hac_variance=ratio,
            strongest_penalty_fraction=float(np.mean(boundary)),mean_selected_epochs=float(np.mean(epochs)),
            mean_projected_gradient=float(np.mean(gradient)),learning_checks_passed=passed))
    return dict(ready_for_production=all(r['learning_checks_passed'] for r in result if r['dgp']=='structured'),
                criteria='Each rule: median across draws of mean-window RMSE/signal-SD <=0.95; mean absolute normalized corrected drift <=0.5; valid fraction >=0.95; median corrected/oracle HAC variance ratio in [0.1,5].',
                interpretation='Prespecified pilot screens, not sufficient conditions for calibrated inference. No rejection frequencies are used.',cells=result)

def analyze(out):
    from run import configuration
    cfg=configuration(out)
    if cfg['profile']!='production':raise ValueError('Pilot cannot enter production evidence')
    cells={c['id']:c for c in cfg['cells']};rows=[];learn=[]
    for r in records(out):
        for method,s in r['statistics'].items():
            score=(-s['p_value'] if 'bootstrap' in method else abs(s['estimate']/s['se'])) if s['valid'] else np.nan
            if s['valid'] and (bool(s['p_value']<=.05) if 'bootstrap' in method else bool(score>1.959963984540054))!=s['reject']:raise RuntimeError('Rejection rule mismatch')
            rows.append(dict(cell_id=r['cell_id'],replication=r['replication'],method=method,**s,
                             score=score,error=s['estimate']-r['theta'],**r['diagnostics']))
        for v in r['learning']:
            learn.append(dict(cell_id=r['cell_id'],replication=r['replication'],rule=v['rule'],block=v['block'],
                forecast_error=v['forecast_error'],representer_error=v['representer_error'],
                relative_forecast_error=v['relative_forecast_error'],population_mse=v['population_mse'],
                population_constant_mse=v['population_constant_mse'],
                **{k:v['fit'][k] for k in ('selected_kappa','penalty','epochs','training_mse','validation_mse','validation_constant_mse','projected_gradient')}))
    df=pd.DataFrame(rows)
    from baseline import COUNTS
    df['phase']=['reused' if r<=COUNTS.get(str(i),0) else 'new' for i,r in zip(df.cell_id,df.replication)]
    df.to_csv(out/'draw_statistics.csv.gz',index=False)
    phase=[]
    for (i,method,part),g in df.groupby(['cell_id','method','phase']):
        v=g[g.valid];N=len(v);rate=v.reject.mean() if N else np.nan
        phase.append(dict(cell_id=i,method=method,phase=part,total=len(g),valid=N,
                          rejection=rate,mcse=np.sqrt(rate*(1-rate)/N) if N else np.nan))
    pd.DataFrame(phase).to_csv(out/'phase_summary.csv',index=False)
    pd.DataFrame(learn).to_csv(out/'learning.csv.gz',index=False)
    groups={k:g for k,g in df.groupby(['cell_id','method'])}
    nulls={(c['n'],c['T'],c['dgp']):i for i,c in cells.items() if c['delta']==0}
    summary=[]
    for (i,method),g in groups.items():
        c=cells[i];v=g[g.valid];N=len(v);base=groups[nulls[c['n'],c['T'],c['dgp']],method]
        scores=base.loc[base.valid,'score'].to_numpy();cut=np.nan;gamma=np.nan;power=np.nan;mcse=np.nan
        if len(scores) and N:
            cut=float(np.quantile(scores,.95,method='inverted_cdf'))
            gamma=float(np.clip((.05-np.mean(scores>cut))/np.mean(scores==cut),0,1))
            weights=(v.score>cut).astype(float)+gamma*(v.score==cut)
            power=weights.mean();mcse=weights.std(ddof=1)/np.sqrt(N) if N>1 else np.nan
        rate=v.reject.mean() if N else np.nan
        row=dict(cell_id=i,n=c['n'],T=c['T'],delta=c['delta'],method=method,completed=len(g),valid=N,
            invalid=len(g)-N,rejection=rate,rejection_mcse=np.sqrt(rate*(1-rate)/N) if N else np.nan,
            coverage=v.covered.mean(),mean_interval_length=v.interval_length.mean(),bias=g.error.mean(),
            rmse=np.sqrt(np.mean(g.error**2)),null_valid=len(scores),critical=cut,boundary_weight=gamma,
            adjusted_power=power,conditional_power_mcse=mcse)
        for k in ('first_order_drift','quadratic_raw_drift','corrected_drift','absolute_raw_drift','absolute_corrected_drift',
                  'max_forecast_error','max_representer_error','normalized_raw_drift','normalized_corrected_drift'):
            row['mean_'+k]=g[k].mean()
        summary.append(row)
    paired=[]
    for i in cells:
        a=groups[i,'orthogonal'].set_index('replication')
        for method in ('raw','oracle_raw','oracle','oracle_alpha','raw_bootstrap','orthogonal_bootstrap'):
            b=groups[i,method].set_index('replication');mask=a.valid&b.valid
            aa=a.loc[mask,'reject'].astype(bool);bb=b.loc[mask,'reject'].astype(bool)
            difference=aa.astype(float)-bb.astype(float);x=int((aa&~bb).sum());y=int((bb&~aa).sum())
            paired.append(dict(cell_id=i,comparison='orthogonal minus '+method,paired_valid=len(difference),
                difference=difference.mean(),mcse=difference.std(ddof=1)/np.sqrt(len(difference)) if len(difference)>1 else np.nan,
                exact_p=binomtest(x,x+y,.5).pvalue if x+y else 1.))
    summary=pd.DataFrame(summary);summary.to_csv(out/'summary.csv',index=False)
    summary[summary.delta==0].to_csv(out/'size.csv',index=False);summary[summary.delta!=0].to_csv(out/'power.csv',index=False)
    pd.DataFrame(paired).to_csv(out/'paired_test_differences.csv',index=False)
    from diagnostics import summarize_diagnostics,power_uncertainty
    summarize_diagnostics(out,df,pd.DataFrame(learn),cells)
    power_uncertainty(out,df,cells)
    figures(out,summary)
    print('Saved size, power, diagnostics, paired comparisons and figures. Adjusted-power MCSE excludes cutoff uncertainty.')

def figures(out,df):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles={'raw':('#B65035','Raw'),'orthogonal':('#0072B2','Orthogonal'),
            'raw_bootstrap':('#D09832','Raw block bootstrap'),'oracle':('#59646B','Oracle orthogonal')}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    dest=out/'figures';dest.mkdir(exist_ok=True)
    def save(fig,name):
        fig.tight_layout()
        for suffix in ('pdf','png'):fig.savefig(dest/f'{name}.{suffix}',dpi=180)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));null=df[df.delta==0]
    for k,(method,(color,label)) in enumerate(styles.items()):
        q=null[null.method==method].sort_values('cell_id')
        rate=q.rejection.to_numpy();N=q.valid.to_numpy();z=1.96
        centre=(rate+z*z/(2*N))/(1+z*z/N);half=z*np.sqrt(rate*(1-rate)/N+z*z/(4*N*N))/(1+z*z/N)
        ax.errorbar(np.arange(len(q))+(k-1.5)*.1,100*rate,yerr=100*np.array([rate-centre+half,centre+half-rate]),fmt='o',color=color,label=label,capsize=3)
    labels=null[['cell_id','n','T']].drop_duplicates().sort_values('cell_id')
    ax.set_xticks(range(len(labels)),[f'n={r.n}, T={r.T}' for r in labels.itertuples()])
    ax.axhline(5,color='gray',ls='--');ax.set_ylabel('Null rejection (%)');ax.legend(fontsize=8);ax.set_ylim(bottom=0);save(fig,'size')
    fig,axes=plt.subplots(1,2,figsize=(9,4),sharey=True)
    for ax,col,title in zip(axes,['rejection','adjusted_power'],['Nominal rejection','Size-adjusted power']):
        for method,(color,label) in styles.items():
            q=df[(df.n==1000)&(df['T']==4000)&(df.method==method)].sort_values('delta')
            ax.plot(q.delta,100*q[col],'o-',color=color,label=label,ms=4)
        ax.axhline(5,color='gray',ls='--');ax.set_title(title);ax.set_xlabel('Local effect');ax.set_ylim(bottom=0)
    axes[0].set_ylabel('Rejection (%)');axes[0].legend(fontsize=8);save(fig,'power')
    q=null[null.method=='orthogonal'].sort_values('cell_id');fig,axes=plt.subplots(1,2,figsize=(9,4))
    for col,label in [('mean_first_order_drift','Raw first-order'),('mean_quadratic_raw_drift','Raw quadratic'),('mean_corrected_drift','Corrected')]:
        axes[0].plot(range(3),q[col],'o-',label=label)
    axes[0].axhline(0,color='gray',lw=.7);axes[0].set_ylabel('Mean population drift');axes[0].legend(fontsize=8)
    axes[1].plot(range(3),q.mean_max_forecast_error,'o-',color='#0072B2');axes[1].set_ylabel('Mean maximum forecast L2 error')
    for ax in axes:ax.set_xticks(range(3),[f'{r.n}/{r.T}' for r in labels.itertuples()]);ax.set_xlabel('Evaluation / training size')
    save(fig,'diagnostics')
