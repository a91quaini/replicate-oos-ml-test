"""Prespecified expanded study; historical engines/results remain unchanged."""
from functools import lru_cache
import hashlib
from pathlib import Path
import sys
import warnings

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(ROOT/'ideas/neural-networks/simulations'))
import numpy as np
from sklearn.linear_model import Lasso
import lasso_designs as lasso
import core as neural
import inference
from networks import new_network, fit_network
from bootstrap import bootstrap_test

VERSION = 'comprehensive-v1'
SEED = 2026110901
BOOTSTRAP_DRAWS = 499
SCHEDULES = ('fixed', 'infrequent', 'frequent')
CONSTANTS = (.03125, .0625, .125, .25, .5, 1., 2.)
_original_fit_lasso = lasso._fit_lasso


def refits(n, schedule):
    return dict(fixed=1, infrequent=5, frequent=int(np.ceil(n/25)), daily=n)[schedule]


def grid():
    cells, seen = [], {}
    def add(family, n=1000, ratio=1., complexity=None, tuning=None,
            schedule='infrequent', delta=0., group='core'):
        complexity = complexity or (32 if family=='lasso' else (8,))
        tuning = tuning or ('cv' if family=='lasso' else 'validation')
        key = (family, n, ratio, complexity, tuning, schedule, delta)
        if key in seen:
            cells[seen[key]]['groups'].append(group)
            return
        cell = dict(id=len(cells), family=family, n=n, ratio=ratio,
                    complexity=complexity, tuning=tuning, schedule=schedule,
                    delta=delta, groups=[group], rho=.5,
                    replications=250 if schedule=='daily' else (1000 if delta==0 else 500))
        if family == 'lasso':
            template = next(c for c in lasso.make_cells('pilot')
                            if c['scenario']=='theory_state_main' and c['n']==n and c['alternative']==0)
            cell = dict(template, **{k:v for k,v in cell.items() if k not in template},
                        **{k:v for k,v in cell.items() if k in template})
            cell.update(m=int(ratio*n), p1=complexity, p2=complexity,
                        tuning='theory' if tuning=='analytic' else tuning,
                        calibration='theory' if tuning=='analytic' else 'practical',
                        alternative=delta)
            cell['tuning_label'] = tuning
        else:
            cell.update(architecture=list(complexity), width=complexity[-1], tau=1.,
                        regularization=tuning, maxiter=600, starts=2, tuning_label=tuning)
        cell['refits'] = refits(n, schedule)
        seen[key] = len(cells)
        cells.append(cell)
    for n in (500, 1000):
        for family, tunings, magnitude in [('lasso', ('cv','aic','analytic'), .7),
                                           ('neural', ('validation','none'), .2)]:
            for tuning in tunings:
                for schedule in SCHEDULES:
                    for delta in (-magnitude, 0., magnitude):
                        add(family, n=n, tuning=tuning, schedule=schedule, delta=delta)
                for delta in (-magnitude/2, magnitude/2):
                    add(family, n=n, tuning=tuning, delta=delta, group='power_curve')
    for ratio in (.5, 1., 2.):
        for p in (32, 128, 256):
            for tuning in ('cv','aic','analytic'):
                add('lasso', ratio=ratio, complexity=p, tuning=tuning, group='complexity')
        for architecture in ((3,), (8,), (16,), (8,8)):
            for tuning in ('validation','none'):
                add('neural', ratio=ratio, complexity=architecture, tuning=tuning, group='complexity')
    for tuning in ('validation','none'):
        for schedule in SCHEDULES:
            for delta in (-.2, 0., .2):
                add('neural', complexity=(8,8), tuning=tuning, schedule=schedule,
                    delta=delta, group='deep_power')
    for family in ('lasso','neural'):
        add(family, n=500, schedule='daily', group='daily')
    for ratio in (10.,50.):
        add('lasso', ratio=ratio, tuning='analytic', group='analytic_history')
    # Power at the demanding end of each complexity sweep, paired with the
    # already included nulls; avoid a full complexity-by-power factorial.
    for ratio in (.5,1.,2.):
        for family,complexity,tunings,magnitude in (
                ('lasso',256,('cv','aic','analytic'),.7),
                ('neural',(16,),('validation','none'),.2)):
            for tuning in tunings:
                for delta in (-magnitude,0.,magnitude):
                    add(family,ratio=ratio,complexity=complexity,tuning=tuning,
                        delta=delta,group='complexity_power')
    return cells


def description(cell):
    n = cell['n']
    if cell['family']=='lasso':
        T, gap, theta = cell['m'], cell['gap'], lasso.population_parameters(cell)['theta']
        counts = [cell['p1'],cell['p2']]
    else:
        pop = neural.population_record(cell)
        T, gap, theta = pop['T'], pop['gap'], pop['theta']
        counts = [new_network(p, cell['architecture']).dimension for p in (4,6)]
    # The two very long analytic controls form a separate paired path group.
    history = 50*n if cell['ratio']>2 else 2*n
    return dict(T=T, gap=gap, theta=theta, parameters=counts,
        parameters_per_training_observation=[p/T for p in counts],
        evaluation_start=history+gap, bandwidth=max(int(np.ceil(n**.25)),gap),
        refits=cell['refits'], bootstrap_block_length=2*(max(int(np.ceil(n**.25)),gap)+1))


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a, dtype='<f8')
        h.update(str(a.shape).encode()+b'\0'+a.tobytes())
    return h.hexdigest()


@lru_cache(maxsize=2)
def lasso_base(n, replication, seed, long_history):
    template = next(c for c in grid() if c['family']=='lasso' and c['n']==n and c['delta']==0)
    template.update(m=(50 if long_history else 2)*n,
                    p1=32 if long_history else 256, p2=32 if long_history else 256)
    data = lasso._sample(template, np.random.default_rng(
        np.random.SeedSequence([seed,n,replication,1,int(long_history)])))
    return data


@lru_cache(maxsize=2)
def neural_base(n, replication, seed):
    gap = int(np.ceil(2*np.log(n)/abs(np.log(.5))))
    size, rho = 2*n+gap+n, .5
    rng = np.random.default_rng(np.random.SeedSequence([seed,n,replication,2]))
    z = np.empty((size,9))
    z[0] = rng.normal(size=9)
    innovations = rng.normal(size=(size-1,9))
    for t in range(1,size):
        z[t] = rho*z[t-1]+np.sqrt(1-rho*rho)*innovations[t-1]
    return np.tanh(z), rng.uniform(size=size)


def make_data(cell, replication, seed=SEED):
    if cell['family']=='lasso':
        base = lasso_base(cell['n'],replication,seed,cell['ratio']>2)
        p = cell['p1']
        beta = [b[:p].copy() for b in base['beta']]
        b2 = lasso.population_parameters(cell)['b2']
        y = base['y']+(b2-beta[1][1])*base['x'][1][:,1]
        beta[1][1] = b2
        return dict(x=[x[:,:p] for x in base['x']], y=y, q=base['q'], beta=beta,
                    alpha=[a[:p] for a in base['alpha']])
    z, uniforms = neural_base(cell['n'],replication,seed)
    probability, targets = neural.truth(z, neural.population_record(cell))
    return dict(x=neural.views(z), y=(uniforms<probability).astype(float), targets=targets)


def fit_lasso(x,y,cell,gap,response_builder=None,role='forecast',audit=False,training_start=0):
    if cell['tuning']!='aic':
        return _original_fit_lasso(x,y,cell,gap,response_builder,role,audit,training_start)
    # Working Gaussian likelihood, estimated residual variance, active-design
    # rank as df. No iid or generated-label AIC guarantee is asserted.
    scale = max(float(np.std(y,ddof=1)),1e-8)*np.sqrt(2*np.log(x.shape[1]+1)/len(y))
    model = Lasso(fit_intercept=False,max_iter=10000,tol=1e-8,selection='cyclic',warm_start=True)
    candidates, coefficients = [], []
    for constant in reversed(CONSTANTS):
        penalty = max(constant*scale,1e-12)
        model.set_params(alpha=penalty)
        model.fit(np.asfortranarray(x),y)
        support = np.flatnonzero(model.coef_)
        df = int(np.linalg.matrix_rank(x[:,support])) if len(support) else 0
        mse = float(np.mean((y-model.predict(x))**2))
        criterion = len(y)*np.log(max(mse,np.finfo(float).tiny))+2*df
        candidates.append(dict(multiplier=constant,penalty=penalty,aic=float(criterion),
            df=df,support=len(support),mse=mse,converged=bool(model.dual_gap_<1e-5),
            iterations=int(model.n_iter_),dual_gap=float(model.dual_gap_)))
        coefficients.append(model.coef_.copy())
    minimum = min(c['aic'] for c in candidates)
    chosen = next(i for i,c in enumerate(candidates) if c['aic']<=minimum+1e-12)
    meta = candidates[chosen].copy()
    # The prediction method uses coef_, not the last path point.
    model.coef_ = coefficients[chosen]
    model.set_params(alpha=meta['penalty'])
    model.n_iter_, model.dual_gap_ = meta['iterations'], meta['dual_gap']
    meta.update(training_start=training_start,training_stop=training_start+len(y)-1,
        coefficient_sha256=digest(model.coef_),cv_folds=[],all_validation_eligible=True,
        all_cv_fits_converged=all(c['converged'] for c in candidates),aic_candidates=candidates)
    return model,meta


def simulate_lasso(cell,replication,seed,data,collect_fits=False):
    meta = description(cell)
    old = lasso._fit_lasso
    lasso._fit_lasso = fit_lasso
    try:
        result = lasso.simulate(cell,replication,seed,data=data,
            evaluation_offset=meta['evaluation_start'],strict_nested_gap=True,
            audit_protocol=True,collect_fits=collect_fits)
    finally:
        lasso._fit_lasso = old
    learning = []
    for row in result['learning']:
        if (row['training_stop']+1 != row['evaluation_start']-meta['gap'] or
            row['training_stop']+1-row['training_start'] != meta['T'] or
            not row['all_nested_validation_eligible']):
            raise AssertionError('Invalid chronology')
        saved = {k:v for k,v in row.items() if k not in
                 ('forecast_fit_audit','representer_fit_audit','preliminary_forecast_audits')}
        for role in ('forecast','representer'):
            audit = row[role+'_fit_audit']
            saved[role+'_selection'] = {k:v for k,v in audit.items() if k in
                ('penalty','multiplier','coefficient_sha256','cv_constants','cv_squared_errors','aic_candidates')}
        learning.append(saved)
    result['learning'] = learning
    result['diagnostics']['all_nested_fits_converged'] = all(r['all_nested_fits_converged'] for r in learning)
    result['scores']['oracle_alpha'] = result['scores'].pop('known_representer')
    return result


def simulate_neural(cell,replication,seed,data,collect_fits=False):
    meta, pop = description(cell), neural.population_record(cell)
    n, origin, gap, T = cell['n'],meta['evaluation_start'],meta['gap'],meta['T']
    prediction = np.zeros((n,2))
    rng = np.random.default_rng(np.random.SeedSequence([seed,cell['n'],replication,101]))
    diagnostic_z = np.tanh(rng.normal(size=(1024,9)))
    xx = neural.views(diagnostic_z)
    _, truth = neural.truth(diagnostic_z,pop)
    learning, fits = [], []
    raw_drift = corrected_drift = 0.
    for block, dates in enumerate(np.array_split(np.arange(n),cell['refits'])):
        stop = origin+int(dates[0])-gap
        train = slice(stop-T,stop)
        for j in (0,1):
            fit_seed = np.random.SeedSequence([seed,n,replication,j,29,int(dates[0])])
            net = fit_network(data['x'][j][train],data['y'][train],cell['architecture'],
                cell['regularization'],fit_seed,gap,cell['maxiter'],cell['starts'])
            prediction[dates,j] = net.predict(data['x'][j][origin+dates])
            f,m = net.predict(xx[j]),truth[j]
            error = f-m
            da = neural.alpha(f,cell['tau'])-neural.alpha(m,cell['tau'])
            raw = neural.alpha(m,cell['tau'])*error+(1+cell['tau']*m)*error**2
            corrected = raw-neural.alpha(f,cell['tau'])*error
            weight = (1 if j==0 else -1)*len(dates)/n
            raw_drift += weight*float(np.mean(raw))
            corrected_drift += weight*float(np.mean(corrected))
            diagnostics = net.diagnostics
            learning.append(dict(block=block,rule=j+1,training_start=stop-T,
                training_stop=stop-1,evaluation_start=origin+int(dates[0]),
                block_length=len(dates),forecast_error=float(np.sqrt(np.mean(error**2))),
                representer_error=float(np.sqrt(np.mean(da**2))),
                coefficient_sha256=digest(net.theta),fit=diagnostics))
            if collect_fits:
                fits.append(dict(block=block,rule=j+1,theta=net.theta.copy()))
    e = slice(origin,origin+n)
    scores = neural.scores(data['y'][e],prediction.T,tuple(m[e] for m in data['targets']),cell['tau'])
    final_converged = all(r['fit']['converged'] for r in learning)
    nested_converged = final_converged and all(c['converged'] for r in learning
                                              for c in r['fit']['selection']['candidates'])
    forecast = max(r['forecast_error'] for r in learning)
    representer = max(r['representer_error'] for r in learning)
    result = dict(theta=pop['theta'],scores=scores,learning=learning,warnings=[],
        diagnostics=dict(all_fits_converged=final_converged,all_nested_fits_converged=nested_converged,
            max_forecast_error=forecast,max_representer_error=representer,
            uniform_remainder_scale=float(np.sqrt(n)*forecast*(forecast+representer)),
            scaled_population_raw_drift=float(np.sqrt(n)*raw_drift),
            scaled_population_corrected_drift=float(np.sqrt(n)*corrected_drift),
            diagnostics_sample_size=1024))
    if collect_fits:
        result['fitted_objects'] = dict(prediction=prediction,coefficients=fits)
    return result


def simulate(cell,replication,seed=SEED,bootstrap_draws=BOOTSTRAP_DRAWS,*,data=None,collect_fits=False):
    data = make_data(cell,replication,seed) if data is None else data
    meta = description(cell)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        engine = simulate_lasso if cell['family']=='lasso' else simulate_neural
        result = engine(cell,replication,seed,data,collect_fits)
    statistics = {}
    for method in ('raw','orthogonal','oracle_alpha','oracle'):
        statistics[method] = inference.statistic(result['scores'][method],meta['theta'],bandwidth=meta['bandwidth'])
    for method in ('raw','orthogonal'):
        statistics[method+'_bootstrap'] = bootstrap_test(result['scores'][method],meta['theta'],
            meta['bandwidth'],np.random.SeedSequence([seed,cell['n'],replication,80]),draws=bootstrap_draws)
    e = slice(meta['evaluation_start'],meta['evaluation_start']+cell['n'])
    out = dict(cell_id=cell['id'],replication=replication,theta=meta['theta'],statistics=statistics,
        diagnostics=result['diagnostics'],learning=result['learning'],
        warnings=sorted(set(result['warnings']+[str(w.message) for w in caught])),
        evaluation_data_sha256=digest(data['y'][e],data['x'][0][e,:3],data['x'][1][e,:3]),
        failed=False)
    if collect_fits:
        out['fitted_objects'] = result['fitted_objects']
    return out
