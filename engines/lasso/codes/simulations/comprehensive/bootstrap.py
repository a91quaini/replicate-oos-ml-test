"""Circular block bootstrap-t of a frozen, centered score sequence.

This changes critical values, not the forecast/correction fits. It is a
finite-sample benchmark, not a validity claim for arbitrary learned scores.
"""
import numpy as np
from inference import statistic


def batch_lrv(values, bandwidth):
    centered = values-values.mean(axis=1, keepdims=True)
    n = values.shape[1]
    result = np.einsum('ij,ij->i', centered, centered)/n
    for h in range(1, min(bandwidth, n-1)+1):
        result += 2*(1-h/(bandwidth+1))*np.einsum(
            'ij,ij->i', centered[:, h:], centered[:, :-h])/n
    return np.maximum(result, 0.)


def bootstrap_test(values, theta, bandwidth, seed, draws=499, block_length=None):
    values = np.asarray(values)
    n = len(values)
    length = 2*(bandwidth+1) if block_length is None else int(block_length)
    if draws < 19 or not 1 <= length <= n:
        raise ValueError('Invalid bootstrap specification')
    out = statistic(values, theta, bandwidth=bandwidth)
    out.update(bootstrap_draws=draws, block_length=length,
               bootstrap_valid_draws=0, critical=None, p_value=None)
    if not out['valid']:
        return out
    rng = np.random.default_rng(seed)
    starts = rng.integers(n, size=(draws, int(np.ceil(n/length))))
    indices = ((starts[:, :, None]+np.arange(length)) % n).reshape(draws, -1)[:, :n]
    sample = (values-values.mean())[indices]
    variances = batch_lrv(sample, bandwidth)
    valid = np.isfinite(variances) & (variances > 1e-24)
    out['bootstrap_valid_draws'] = int(valid.sum())
    # Degenerate bootstrap draws invalidate the test; never discard them and
    # quietly replace B by a more favorable conditional reference distribution.
    if not valid.all():
        out.update(valid=False, reject=None, covered=None, interval_length=None)
        return out
    pivot = np.abs(sample.mean(axis=1)/np.sqrt(variances/n))
    # Order statistic implements the conservative (1 + exceedances)/(B + 1) test.
    rank = int(np.ceil(.95*(draws+1)))
    critical = float(np.sort(pivot)[rank-1])
    observed = abs(out['estimate']/out['se'])
    pvalue = float((1+np.count_nonzero(pivot >= observed))/(draws+1))
    out.update(critical=critical, p_value=pvalue, reject=pvalue <= .05,
        covered=bool(abs(out['estimate']-theta) <= critical*out['se']),
        interval_length=2*critical*out['se'])
    return out
