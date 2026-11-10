"""Simultaneous finite-sample score bounds for the declared refresh-chain design.

No fitted coefficient, simulated outcome, population risk contrast, or realized
refresh indicator is used. See CALIBRATION.md for the proof and its boundaries.
"""
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import product
import math

import numpy as np


@dataclass(frozen=True)
class StateScoreCalibration:
    penalty: float
    score_threshold: float
    failure_probability: float
    training_length: int
    maximum_dimension: int
    total_fits: int
    persistence: float
    signal_bound: float
    state_slope_bound: float
    role: str
    weighted: bool
    selected_tilt: float
    log_mgf_upper: float
    deterministic_score_bound: float
    deterministic_bound_used: bool

    def to_dict(self):
        return asdict(self)


def confidence_budget(n):
    """Finite failure budget tending to zero along the asymptotic sequence."""
    if isinstance(n, bool) or int(n) != n or n < 1:
        raise ValueError("n must be a positive integer")
    return 0.05 / math.log(math.e * int(n))


def _perron_upper(diagonal, rho):
    """Collatz upper bound for sqrt(D) P sqrt(D), without trusting an eigenvalue.

    Any z>rho*max(D) supplies a positive test vector. Bisection merely chooses
    a useful vector; its accuracy does not determine mathematical validity.
    """
    diagonal = np.asarray(diagonal, dtype=float)
    if rho == 0:
        return float(np.nextafter(np.mean(diagonal) * (1 + 64 * np.finfo(float).eps), np.inf))
    lower = rho * float(np.max(diagonal))
    upper = float(np.max(diagonal))
    for _ in range(48):
        middle = (lower + upper) / 2
        if middle <= lower or middle >= upper:
            break
        value = (1 - rho) * float(np.mean(diagonal / (middle - rho * diagonal)))
        if value > 1:
            lower = middle
        else:
            upper = middle
    z = np.nextafter(upper, np.inf)
    denominators = z - rho * diagonal
    if np.any(denominators <= 0):
        raise FloatingPointError("Perron test vector is not strictly positive")
    value = (1 - rho) * float(np.mean(diagonal / denominators))
    ratios = rho * diagonal + denominators * value
    bound = float(np.max(ratios))
    # All terms in the ratio are nonnegative. Leave an explicit rounding margin
    # and round upward; the statistical proof uses the exact Collatz expression.
    return float(np.nextafter(bound * (1 + 128 * np.finfo(float).eps), np.inf))


def _score_potentials(role, weighted, signal_bound, slope_bound):
    """Envelope vertices and coordinate types; rows are (conditional mean, noise scale)."""
    states = np.array(list(product([-1.0, 1.0], repeat=4)))
    s1, s2, state, irrelevant = states.T
    unique = {}
    for rule in [0, 1]:
        for x in [np.ones(16), s1 if rule == 0 else s2, state, irrelevant]:
            for signal, slope in product([-signal_bound, signal_bound],
                                         [-slope_bound, slope_bound]):
                residual_mean = (slope * state * s1 + signal * s2 if rule == 0
                                 else (signal + slope * state) * s1)
                if role == "forecast":
                    mean = x * residual_mean
                    noise = np.ones(16)
                else:
                    q2 = 1 + state if weighted else np.full(16, 2.0)
                    alpha = -slope * s1 if weighted and rule == 0 else np.zeros(16)
                    mean = x * (-q2 * residual_mean - alpha)
                    noise = q2
                for tail in [-1.0, 1.0]:
                    # P has uniform refresh distribution, so state permutations
                    # leave its MGF unchanged. Deduplicate only exact float pairs.
                    pairs = tuple(sorted(zip(tail * mean, noise)))
                    unique[pairs] = np.array(pairs, dtype=float)
    return tuple(unique.values())


@lru_cache(maxsize=64)
def _spectral_envelopes(role, weighted, rho, signal_bound, slope_bound, tilt_scale):
    potentials = _score_potentials(role, weighted, signal_bound, slope_bound)
    score_bound = max(float(np.max(np.abs(p[:, 0]) + p[:, 1])) for p in potentials)
    # Every grid point tends to zero at the root(log-budget/m) scale. A fixed
    # positive minimum tilt would otherwise produce a positive penalty floor.
    center = tilt_scale * math.sqrt((1 - rho) / (1 + rho)) / max(1.0, score_bound)
    tilts = np.unique(np.minimum(center * np.geomspace(0.01, 16.0, 160),
                                12.0 / max(1.0, score_bound)))
    log_means = np.zeros((len(tilts), len(potentials)))
    log_radii = np.zeros_like(log_means)
    for j, potential in enumerate(potentials):
        mean, noise = potential.T
        score_bound = max(score_bound, float(np.max(np.abs(mean) + noise)))
        for i, tilt in enumerate(tilts):
            argument = tilt * noise
            noise_mgf = np.ones(len(mean))
            np.divide(np.sinh(argument), argument, out=noise_mgf, where=argument != 0)
            diagonal = np.exp(tilt * mean) * noise_mgf
            # Upward margins on both factors protect against ordinary rounding;
            # no global optimizer or numerical eigenvalue is used as a bound.
            log_means[i, j] = math.log(float(np.mean(diagonal))) + 256 * np.finfo(float).eps
            log_radii[i, j] = math.log(_perron_upper(diagonal, rho))
    return tilts, log_means, log_radii, score_bound


@lru_cache(maxsize=256)
def calibrate_state_score(m, p, total_fits, rho, n, role="forecast", weighted=True,
                          signal_bound=0.8, state_slope_bound=0.6):
    """Return lambda with joint oracle-score failure at most delta_n across fits.

    The same maximum dimension and total_fits must be supplied for every fit in
    the family. For two rules, two roles and K windows, total_fits=4*K. A common
    forecast penalty is used across rules, and likewise for the correction.
    """
    for label, value in [("m", m), ("p", p), ("total_fits", total_fits), ("n", n)]:
        if isinstance(value, bool) or not np.isfinite(value) or int(value) != value or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    if role not in {"forecast", "correction"}:
        raise ValueError("role must be forecast or correction")
    if not np.isfinite(rho) or not 0 <= rho < 1:
        raise ValueError("rho must lie in [0,1)")
    if not all(np.isfinite(x) and x >= 0 for x in [signal_bound, state_slope_bound]):
        raise ValueError("Coefficient envelopes must be finite and nonnegative")
    m, p, total_fits, n = int(m), int(p), int(total_fits), int(n)
    delta = confidence_budget(n)
    log_tail = math.log(2 * total_fits * p / delta)
    tilts, log_means, log_radii, score_bound = _spectral_envelopes(
        role, bool(weighted), float(rho), float(signal_bound), float(state_slope_bound),
        math.sqrt(log_tail / m))
    log_mgf = np.max(log_means + (m - 1) * log_radii, axis=1)
    thresholds = (log_mgf + log_tail) / (m * tilts)
    chosen = int(np.argmin(thresholds))
    stochastic = float(thresholds[chosen])
    deterministic = stochastic >= score_bound
    threshold = score_bound if deterministic else stochastic
    threshold = float(np.nextafter(threshold, np.inf))
    return StateScoreCalibration(
        penalty=2 * threshold, score_threshold=threshold, failure_probability=delta,
        training_length=m, maximum_dimension=p, total_fits=total_fits,
        persistence=float(rho), signal_bound=float(signal_bound),
        state_slope_bound=float(state_slope_bound), role=role, weighted=bool(weighted),
        selected_tilt=float(tilts[chosen]), log_mgf_upper=float(log_mgf[chosen]),
        deterministic_score_bound=score_bound, deterministic_bound_used=deterministic)


def verify_calibration():
    """Check the MGF bound by direct short-path enumeration, both tails and roles."""
    maximum_excess = 0.0
    for rho in [0.0, 0.5, 0.8]:
        for role in ["forecast", "correction"]:
            for weighted in [False, True]:
                potentials = _score_potentials(role, weighted, 0.8, 0.6)
                for potential in potentials:
                    mean, noise = potential.T
                    transition = rho * np.eye(len(mean)) + (1 - rho) * np.ones((len(mean), len(mean))) / len(mean)
                    for tilt in [0.003, 0.05, 0.5]:
                        argument = tilt * noise
                        noise_mgf = np.ones(len(mean))
                        np.divide(np.sinh(argument), argument, out=noise_mgf, where=argument != 0)
                        diagonal = np.exp(tilt * mean) * noise_mgf
                        radius = _perron_upper(diagonal, rho)
                        matrix = np.sqrt(diagonal[:, None]) * transition * np.sqrt(diagonal[None, :])
                        assert radius + 1e-13 >= np.linalg.eigvalsh(matrix)[-1]
                        distribution = np.ones(len(mean)) / len(mean)
                        for length in range(1, 7):
                            distribution *= diagonal
                            direct = float(np.sum(distribution))
                            bound = float(np.mean(diagonal) * radius ** (length - 1))
                            maximum_excess = max(maximum_excess, direct - bound)
                            assert direct <= bound * (1 + 1e-12)
                            distribution = distribution @ transition
    for role in ["forecast", "correction"]:
        calibrated = calibrate_state_score(50000, 32, 20, 0.5, 1000, role)
        longer = calibrate_state_score(200000, 32, 20, 0.5, 1000, role)
        assert calibrated.penalty == 2 * calibrated.score_threshold
        assert calibrated.failure_probability < 0.01
        assert calibrated.penalty > 0
        assert longer.penalty < 0.6 * calibrated.penalty
    return f"Calibration: exact finite-state MGFs and Collatz bounds agree (max excess {maximum_excess:.2g})."
