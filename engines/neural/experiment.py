"""Chronological probability fitting and exact conditional-mean score diagnostics."""
import hashlib
import numpy as np
import design
from inference import statistic,bartlett_lrv


def simulate(c,reps,device='cuda',epochs=None,override=None):
    from network import fit
    meta=design.meta(c);n,T=c['n'],c['T'];origin,gap=meta['origin'],meta['gap'];B=len(reps)
    ds=[design.data(c,r) for r in reps] if override is None else override
    dg=[design.diagnostic(c,r) for r in reps]
    xx=[np.stack([d['x'][j] for d in ds]) for j in (0,1)];yy=np.stack([d['y'] for d in ds])
    dx=[np.stack([d['x'][j] for d in dg]) for j in (0,1)];dm=np.stack([d['m'] for d in dg])
    prediction=np.zeros((B,2,n));learning=[[] for _ in reps];drift=np.zeros((B,3))
    pop=design.population(n,c['delta'],c['dgp'])
    for block,dates in enumerate(np.array_split(np.arange(n),c['refits'])):
        end=origin+int(dates[0])-gap;start=end-T
        if start<0:raise ValueError('Insufficient history')
        for j in (0,1):
            predict,fitdiag=fit(xx[j][:,start:end],yy[:,start:end],c,reps,j,int(dates[0]),device,epochs)
            prediction[:,j,dates]=predict(xx[j][:,origin+dates]);f=predict(dx[j]);m=dm[:,j];e=f-m
            linear=design.alpha(m)*e;quadratic=(1+design.TAU*m)*e**2
            corrected=linear+quadratic-design.alpha(f)*e
            sign=1 if j==0 else -1;weight=sign*len(dates)/n
            drift[:,0]+=weight*linear.mean(1);drift[:,1]+=weight*quadratic.mean(1);drift[:,2]+=weight*corrected.mean(1)
            for k in range(B):
                error=float(np.sqrt(np.mean(e[k]**2)))
                learning[k].append(dict(block=block,rule=j,training_start=start,training_stop=end-1,
                    evaluation_start=origin+int(dates[0]),forecast_error=error,
                    relative_forecast_error=error/float(pop['signal_sd'][j]),
                    representer_error=float(np.sqrt(np.mean((design.alpha(f[k])-design.alpha(m[k]))**2))),
                    population_mse=float(np.mean(m[k]*(1-m[k])+e[k]**2)),
                    population_constant_mse=float(np.mean(m[k]*(1-m[k])+(design.MU-m[k])**2)),
                    fit=fitdiag[k]))
            del predict
    records=[]
    for k,r in enumerate(reps):
        d=ds[k];sl=slice(origin,origin+n);m=d['m'][:,sl];y=d['y'][sl]
        ss=design.scores(y,prediction[k],m)
        stats={method:statistic(values,pop['theta'],bandwidth=meta['bandwidth']) for method,values in ss.items()}
        # Diagnostic scale only; empirical HAC variability is retained rather than hidden.
        scale=np.sqrt(max(bartlett_lrv(ss['oracle'],meta['bandwidth']),1e-24))
        raw=drift[k,0]+drift[k,1]
        diagnostics=dict(max_forecast_error=max(v['forecast_error'] for v in learning[k]),
            max_representer_error=max(v['representer_error'] for v in learning[k]),
            first_order_drift=float(drift[k,0]),quadratic_raw_drift=float(drift[k,1]),corrected_drift=float(drift[k,2]),
            absolute_raw_drift=float(abs(raw)),absolute_corrected_drift=float(abs(drift[k,2])),
            normalized_raw_drift=float(np.sqrt(n)*raw/scale),
            normalized_corrected_drift=float(np.sqrt(n)*drift[k,2]/scale),oracle_hac_sd=float(scale))
        records.append(dict(cell_id=c['id'],replication=r,failed=False,theta=pop['theta'],statistics=stats,
            learning=learning[k],diagnostics=diagnostics,
            prediction_sha256=hashlib.sha256(prediction[k].tobytes()).hexdigest(),
            series=dict(y=y.tolist(),forecasts=prediction[k].tolist(),targets=m.tolist(),
                        scores={method:values.tolist() for method,values in ss.items()})))
    return records
