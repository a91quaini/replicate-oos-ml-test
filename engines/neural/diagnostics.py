"""Mechanism diagnostics and paired power uncertainty including null calibration."""
import numpy as np
import pandas as pd

def summarize_diagnostics(out,df,learning,cells):
    rows=[]
    for (i,method),g in df.groupby(['cell_id','method']):
        v=g[g.valid];oracle=df[(df.cell_id==i)&(df.method=='oracle')].set_index('replication')
        variance_ratio=v.set_index('replication').variance/oracle.variance
        rows.append(dict(cell_id=i,n=cells[i]['n'],T=cells[i]['T'],delta=cells[i]['delta'],method=method,
            valid=len(v),invalid=len(g)-len(v),bias=v.error.mean(),empirical_sd=v.estimate.std(ddof=1),
            mean_se=v.se.mean(),rms_se=np.sqrt((v.se**2).mean()),
            mean_standardized_error=(v.error/v.se).mean(),sd_standardized_error=(v.error/v.se).std(ddof=1),
            mean_abs_normalized_raw_drift=g.normalized_raw_drift.abs().mean(),
            mean_abs_normalized_corrected_drift=g.normalized_corrected_drift.abs().mean(),
            mean_raw_drift=g.first_order_drift.mean()+g.quadratic_raw_drift.mean(),
            mean_corrected_drift=g.corrected_drift.mean(),median_variance_to_oracle=variance_ratio.median()))
    table=pd.DataFrame(rows);table.to_csv(out/'diagnostic_summary.csv',index=False)
    learning.groupby(['cell_id','rule'])[['forecast_error','representer_error','relative_forecast_error','validation_mse','training_mse','selected_kappa','epochs']].agg(['mean','median']).to_csv(out/'learning_summary.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,4));null=table[(table.delta==0)&(table.method=='orthogonal')].sort_values('cell_id')
    x=np.arange(len(null));labels=[f'{r.n}/{r.T}' for r in null.itertuples()]
    for key,label,color in [('mean_abs_normalized_raw_drift','Raw','#B65035'),('mean_abs_normalized_corrected_drift','Orthogonal','#0072B2')]:
        axes[0].plot(x,null[key],'o-',label=label,color=color)
    axes[0].set_ylabel('Mean absolute normalized drift');axes[0].legend()
    for method,label,color in [('raw','Raw','#B65035'),('orthogonal','Orthogonal','#0072B2'),('oracle','Oracle orthogonal','#59646B')]:
        q=table[(table.delta==0)&(table.method==method)].sort_values('cell_id')
        axes[1].plot(x,q.rms_se/q.empirical_sd,'o-',label=label,color=color)
    axes[1].axhline(1,color='gray',ls='--');axes[1].set_ylabel('RMS standard error / empirical SD');axes[1].legend(fontsize=8)
    for ax in axes:ax.set_xticks(x,labels);ax.set_xlabel('Evaluation / training size')
    fig.tight_layout();(out/'figures').mkdir(exist_ok=True)
    for suffix in ('pdf','png'):fig.savefig(out/'figures'/f'mechanisms.{suffix}',dpi=180)
    plt.close(fig)

def adjusted(null,alt):
    null=np.asarray(null);alt=np.asarray(alt);null=null[np.isfinite(null)];alt=alt[np.isfinite(alt)]
    if not len(null) or not len(alt):return np.nan
    cutoff=np.quantile(null,.95,method='inverted_cdf')
    weight=np.clip((.05-np.mean(null>cutoff))/np.mean(null==cutoff),0,1)
    return np.mean((alt>cutoff).astype(float)+weight*(alt==cutoff))

def power_uncertainty(out,df,cells,repeats=1000):
    methods=('raw','orthogonal','oracle','raw_bootstrap','orthogonal_bootstrap')
    arrays={}
    for (i,m),g in df.groupby(['cell_id','method']):
        if m in methods:
            g=g.sort_values('replication');assert np.array_equal(g.replication,np.arange(1,len(g)+1))
            arrays[i,m]=g.score.to_numpy()
    rows=[];rng=np.random.default_rng(2026111419)
    for i,c in cells.items():
        if c['delta']==0:continue
        null=next(j for j,d in cells.items() if d['delta']==0 and (d['n'],d['T'],d['dgp'])==(c['n'],c['T'],c['dgp']))
        n=len(arrays[i,'raw']);N=len(arrays[null,'raw']);assert N>=n
        comparisons=('raw','oracle','raw_bootstrap','orthogonal_bootstrap');boot=np.empty((repeats,len(comparisons)))
        for b in range(repeats):
            # Preserve common data seeds between null and alternatives and between methods.
            j=rng.integers(n,size=n);k=rng.integers(n,N,size=N-n) if N>n else np.array([],dtype=int);index=np.r_[j,k]
            vals={m:adjusted(arrays[null,m][index],arrays[i,m][j]) for m in methods}
            boot[b]=[vals['orthogonal']-vals[m] for m in comparisons]
        for k,m in enumerate(comparisons):
            values=boot[:,k];good=values[np.isfinite(values)]
            ci=np.quantile(good,[.025,.975]) if len(good) else [np.nan,np.nan]
            rows.append(dict(cell_id=i,delta=c['delta'],comparison='orthogonal minus '+m,
                adjusted_power_difference=adjusted(arrays[null,'orthogonal'],arrays[i,'orthogonal'])-adjusted(arrays[null,m],arrays[i,m]),
                lower=ci[0],upper=ci[1],bootstrap_draws=repeats,valid_bootstrap_draws=len(good),
                interpretation='Pointwise paired stratified percentile interval, including estimated null cutoff; not multiplicity adjusted.'))
    pd.DataFrame(rows).to_csv(out/'power_comparisons.csv',index=False)
