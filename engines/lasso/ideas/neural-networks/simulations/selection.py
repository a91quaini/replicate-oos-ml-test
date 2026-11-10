"""Chronological, training-only validation of vanishing neural weight decay.

This module receives only the advertised training sample. Validation predicts
the last quarter using the earlier prefix separated by a gap. After choosing
the minimum validation MSE, a newly initialized model refits all training data.
No evaluation outcome, target probability, or rejection statistic is available.
"""
import numpy as np
from network import Network

KAPPAS = (0., .05, .5, 5., 50.)
VALIDATION_FRACTION = .25
TIE_RTOL = 1e-10
TIE_ATOL = 1e-12


def split_record(T, gap):
    """Return zero-based, half-open intervals within the T training dates."""
    if int(T) != T or int(gap) != gap or T < 1 or gap < 0:
        raise ValueError('T must be a positive integer and gap a nonnegative integer')
    T, gap = int(T), int(gap)
    validation_size = max(1, int(np.ceil(VALIDATION_FRACTION*T)))
    validation_start = T-validation_size
    fit_stop = validation_start-gap
    if fit_stop < 2:
        raise ValueError('Training sample too short for the prescribed validation split and gap')
    return dict(fit_start=0, fit_stop=fit_stop, gap_start=fit_stop,
                gap_stop=validation_start, validation_start=validation_start,
                validation_stop=T, fitting_size=fit_stop,
                validation_size=validation_size, total_training_size=T,
                gap_size=gap)


def independent_seed_copy(seed):
    """Recreate a seed without consuming state, pairing candidate starts."""
    if isinstance(seed, np.random.SeedSequence):
        return np.random.SeedSequence(seed.entropy, spawn_key=seed.spawn_key,
                                      pool_size=seed.pool_size)
    if isinstance(seed, np.random.Generator):
        raise TypeError('Use a SeedSequence or integer seed, not a mutable Generator')
    return np.random.SeedSequence(seed)


def fit_with_validation(x, y, width, seed, gap, maxiter=600, starts=2):
    """Fit a forecast using only x,y (the complete eligible training sample)."""
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError('Expected a predictor matrix and same-length outcome vector')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Training inputs must be finite')
    T = len(y)
    split = split_record(T, gap)
    fitting = slice(split['fit_start'], split['fit_stop'])
    validation = slice(split['validation_start'], split['validation_stop'])
    candidates = []
    for kappa in KAPPAS:
        strength = kappa/T
        candidate = Network(x.shape[1], width).fit(
            x[fitting], y[fitting], seed=independent_seed_copy(seed),
            maxiter=maxiter, starts=starts, weight_decay=strength)
        mse = float(np.mean((candidate.predict(x[validation])-y[validation])**2))
        if not np.isfinite(mse):
            raise ArithmeticError('Nonfinite validation loss')
        candidates.append(dict(kappa=float(kappa), weight_decay=float(strength),
                               validation_mse=mse, fit=candidate.diagnostics.copy()))
    minimum = min(candidate['validation_mse'] for candidate in candidates)
    tied = [candidate for candidate in candidates
            if np.isclose(candidate['validation_mse'], minimum,
                          rtol=TIE_RTOL, atol=TIE_ATOL)]
    chosen = max(tied, key=lambda candidate: candidate['weight_decay'])
    # Reinitialize rather than warm-starting a model selected on validation data.
    # Validation and internal gap observations now join the final training fit.
    final = Network(x.shape[1], width).fit(
        x, y, seed=independent_seed_copy(seed), maxiter=maxiter,
        starts=starts, weight_decay=chosen['weight_decay'])
    final.diagnostics['selection'] = dict(
        mode='chronological_validation', split=split,
        selected_kappa=chosen['kappa'], selected_lambda=chosen['weight_decay'],
        validation_mse=chosen['validation_mse'], minimum_validation_mse=minimum,
        candidates=candidates, candidate_count=len(candidates),
        numerical_tie_count=len(tied), tie_rule='larger lambda within isclose tolerance',
        tie_rtol=TIE_RTOL, tie_atol=TIE_ATOL,
        validation_criterion='unweighted mean squared prediction error',
        penalty_scaling='lambda=kappa/T for candidate fitting and final refitting',
        initialization='same per-rule seed recreated for each candidate and final refit; no warm starts',
        final_refit_training_size=T, validation_used_for_final_refit=True)
    return final
