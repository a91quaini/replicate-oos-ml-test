"""Prespecified binary forecasting experiment: trained neural probability rules.

All oracle inputs are confined to calibration and diagnostics, never fitting.
"""
from functools import lru_cache
import hashlib
import numpy as np
from scipy.special import roots_hermitenorm
from network import Network
from selection import KAPPAS, fit_with_validation, split_record

SCHEMA = 2
SEED = 20261106
CRITICAL = 1.959963984540054
METHODS = ('raw', 'orthogonal', 'oracle_alpha', 'oracle')
LABELS = dict(raw='Raw HAC', orthogonal='Orthogonal', oracle_alpha='Oracle representer (diagnostic)', oracle='Population oracle (diagnostic)')
DELTAS = (-.2, -.1, 0., .1, .2)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(None)
def moments(order=320):
    x, w = roots_hermitenorm(order)
    w = w/np.sqrt(2*np.pi)
    bounded = np.tanh(x)
    weights = w[:, None]*w[None, :]
    v1 = np.sum(weights*np.tanh(1.2*bounded[:, None]+.4*bounded[None, :])**2)
    v2 = np.sum(weights*np.tanh(.9*bounded[:, None]-.6*bounded[None, :])**2)
    return float(v1), float(v2)


def grid():
    cells = []
    base = dict(ratio=1., width=8, maxiter=600, starts=2, rho=.5, tau=1.,
                regularization='validation')
    for n in (500, 1000):
        for delta in DELTAS:
            cells.append(dict(scenario='main', n=n, delta=delta, **base))
    controls = [('short_training', dict(ratio=.5)), ('long_training', dict(ratio=2.)),
                ('narrow_network', dict(width=3)), ('wide_network', dict(width=16)),
                ('minimal_network', dict(width=1)), ('optimization_budget', dict(maxiter=1200)),
                ('persistent_predictors', dict(rho=.8)), ('aligned_loss', dict(tau=0.))]
    for name, options in controls:
        cells.append(dict(scenario=name, n=1000, delta=0., **(base|options)))
    for n in (500, 1000):
        cells.append(dict(scenario='unregularized_null', n=n, delta=0.,
                          **(base|dict(regularization='none'))))
    return [dict(id=i, **c) for i, c in enumerate(cells)]


def population_record(cell):
    v1, v2 = moments()
    n, tau = cell['n'], cell['tau']
    a = .18
    b_squared = (a*a*v1+cell['delta']/((1+tau/2)*np.sqrt(n)))/v2
    if b_squared <= 0:
        raise ValueError('Alternative has nonpositive signal square')
    b = float(np.sqrt(b_squared))
    if a+b >= .48:
        raise ValueError('Full conditional probability leaves prespecified [0.02,0.98] bounds')
    gap = int(np.ceil(2*np.log(n)/abs(np.log(cell['rho']))))
    T = int(round(cell['ratio']*n))
    regularization = cell['regularization']
    return dict(cell_id=cell['id'], a=a, b=b, v1=v1, v2=v2,
        theta=float((1+tau/2)*(b*b*v2-a*a*v1)),
        risk1=float((1+tau/2)*(.25-a*a*v1)), risk2=float((1+tau/2)*(.25-b*b*v2)),
        probability_lower=.5-a-b, probability_upper=.5+a+b,
        T=T, gap=gap, regularization=regularization,
        validation_split=split_record(T, gap) if regularization=='validation' else None,
        candidate_kappas=list(KAPPAS) if regularization=='validation' else [0.],
        candidate_lambdas=[float(kappa/T) for kappa in KAPPAS] if regularization=='validation' else [0.],
        bandwidth=max(int(np.ceil(n**.25)), gap), original_predictors=[4, 6],
        fitted_parameters=[Network(4, cell['width']).dimension, Network(6, cell['width']).dimension])


def views(z):
    return np.column_stack((z[:, :3], z[:, 8])), np.column_stack((z[:, 3:8], z[:, 8]))


def truth(z, population):
    h1 = np.tanh(1.2*z[:, 0]+.4*z[:, 1])
    h2 = np.tanh(.9*z[:, 3]-.6*z[:, 4])
    p1 = .5+population['a']*h1
    p2 = .5+population['b']*h2
    return p1+p2-.5, (p1, p2)


def loss(y, p, tau):
    return (1+tau*y)*(y-p)**2


def alpha(p, tau):
    return -2*tau*p*(1-p)


def scores(y, fitted, targets, tau):
    f1, f2 = fitted
    m1, m2 = targets
    raw = loss(y, f1, tau)-loss(y, f2, tau)
    orthogonal = raw+alpha(f1, tau)*(y-f1)-alpha(f2, tau)*(y-f2)
    oracle_alpha = raw+alpha(m1, tau)*(y-f1)-alpha(m2, tau)*(y-f2)
    oracle = loss(y, m1, tau)-loss(y, m2, tau)+alpha(m1, tau)*(y-m1)-alpha(m2, tau)*(y-m2)
    return dict(raw=raw, orthogonal=orthogonal, oracle_alpha=oracle_alpha, oracle=oracle)


def hac(values, bandwidth):
    centered = values-values.mean()
    n = len(values)
    result = float(centered@centered/n)
    for lag in range(1, min(bandwidth, n-1)+1):
        result += 2*(1-lag/(bandwidth+1))*float(centered[lag:]@centered[:-lag]/n)
    return result


def statistic(values, theta, bandwidth):
    estimate = float(values.mean())
    variance = hac(values, bandwidth)
    if not np.isfinite(variance) or variance <= 1e-16:
        return dict(estimate=estimate, error=estimate-theta, variance=float(variance), valid=False,
                    invalid_reason='nonfinite_or_nonpositive_HAC_variance', se=None, rejection=None, coverage=None,
                    interval_length=None, t_zero=None)
    se = float(np.sqrt(variance/len(values)))
    return dict(estimate=estimate, error=estimate-theta, variance=variance, valid=True, invalid_reason=None,
                se=se, rejection=bool(abs(estimate/se)>CRITICAL), coverage=bool(abs((estimate-theta)/se)<=CRITICAL),
                interval_length=2*CRITICAL*se, t_zero=estimate/se)


def simulate(cell, replication):
    pop = population_record(cell)
    n, T, gap = cell['n'], pop['T'], pop['gap']
    rho, tau = cell['rho'], cell['tau']
    # Same length paths and fit starts pair alternatives/settings at equal n,T,rho.
    rng = np.random.default_rng(np.random.SeedSequence([SEED, n, replication, 11]))
    z = np.empty((T+gap+n, 9))
    z[0] = rng.normal(size=9)  # stationary initialization, no arbitrary burn-in
    innovations = rng.normal(size=(len(z)-1, 9))
    for t in range(1, len(z)):
        z[t] = rho*z[t-1]+np.sqrt(1-rho*rho)*innovations[t-1]
    z = np.tanh(z)  # bounded observed predictors, latent Gaussian AR dependence
    full_probability, targets = truth(z, pop)
    y = (rng.uniform(size=len(z))<full_probability).astype(float)
    xx = views(z)
    train, evaluation = slice(0, T), slice(T+gap, T+gap+n)
    networks = []
    for j in (0, 1):
        seed = np.random.SeedSequence([SEED, n, replication, j, 29])
        if cell['regularization']=='validation':
            network = fit_with_validation(xx[j][train], y[train], cell['width'],
                seed=seed, gap=gap, maxiter=cell['maxiter'], starts=cell['starts'])
        elif cell['regularization']=='none':
            network = Network(xx[j].shape[1], cell['width']).fit(xx[j][train], y[train],
                seed=seed, maxiter=cell['maxiter'], starts=cell['starts'])
            network.diagnostics['selection'] = dict(mode='none', split=None,
                selected_kappa=0., selected_lambda=0., validation_mse=None,
                minimum_validation_mse=None, candidates=[], candidate_count=0,
                final_refit_training_size=T, validation_used_for_final_refit=False)
        else:
            raise ValueError('Unknown regularization mode')
        networks.append(network)
    fitted = tuple(networks[j].predict(xx[j][evaluation]) for j in (0, 1))
    target_eval = tuple(m[evaluation] for m in targets)
    evaluated = scores(y[evaluation], fitted, target_eval, tau)
    statistics = {name: statistic(value, pop['theta'], pop['bandwidth']) for name, value in evaluated.items()}
    # Fresh stationary marginal sample; these inputs never enter training or inference.
    diagnostic_rng = np.random.default_rng(np.random.SeedSequence([SEED, n, replication, 101]))
    diagnostic_z = np.tanh(diagnostic_rng.normal(size=(4096, 9)))
    diagnostic_x = views(diagnostic_z)
    _, diagnostic_m = truth(diagnostic_z, pop)
    diagnostic_f = tuple(networks[j].predict(diagnostic_x[j]) for j in (0, 1))
    errors = [float(np.mean((diagnostic_f[j]-diagnostic_m[j])**2)) for j in (0, 1)]
    representer_errors = [float(np.mean((alpha(diagnostic_f[j], tau)-alpha(diagnostic_m[j], tau))**2)) for j in (0, 1)]
    bias_terms = []
    for f, m in zip(diagnostic_f, diagnostic_m):
        error = f-m
        raw_bias = alpha(m, tau)*error+(1+tau*m)*error**2
        orthogonal_bias = (alpha(m, tau)-alpha(f, tau))*error+(1+tau*m)*error**2
        bias_terms.append((float(raw_bias.mean()), float(orthogonal_bias.mean())))
    diagnostics = dict(conditional_raw_bias=float(bias_terms[0][0]-bias_terms[1][0]),
        conditional_orthogonal_bias=float(bias_terms[0][1]-bias_terms[1][1]),
        forecast_l2_squared_error=errors, representer_l2_squared_error=representer_errors,
        root_n_forecast_l2_squared_error=[float(np.sqrt(n)*v) for v in errors],
        rate_product=[float(np.sqrt(n)*np.sqrt(errors[j])*(np.sqrt(errors[j])+np.sqrt(representer_errors[j]))) for j in (0, 1)],
        raw_oracle_difference=float(np.sqrt(n)*(evaluated['raw']-evaluated['oracle']).mean()),
        orthogonal_oracle_difference=float(np.sqrt(n)*(evaluated['orthogonal']-evaluated['oracle']).mean()),
        correction_loss_covariance=float(np.cov(evaluated['raw'], evaluated['orthogonal']-evaluated['raw'], ddof=0)[0, 1]),
        forecast_target_l2_squared_signal=[float(np.mean((m-.5)**2)) for m in diagnostic_m],
        diagnostics_sample_size=len(diagnostic_z), fits=[net.diagnostics for net in networks])
    return dict(theta=pop['theta'], population=pop, statistics=statistics, diagnostics=diagnostics,
        seeds=dict(path=[SEED, n, replication, 11], fit=[[SEED, n, replication, j, 29] for j in (0, 1)],
                   diagnostic=[SEED, n, replication, 101]))
