"""Bounded sparse projection experiments with exact population risk contrasts.

The asymmetric reference retains the DGP and chronological tuning protocol in
scripts/simulations-new. The state experiment adapts the mechanism of old Design B
to bounded observations. Feasible fits never use population targets or evaluation
observations. Population quantities below are exclusively simulation diagnostics.
"""
from itertools import product
from functools import lru_cache
import hashlib

import numpy as np
from numpy.polynomial.legendre import leggauss
from sklearn.linear_model import Lasso
from calibration import calibrate_state_score, confidence_budget


def _practical_cells(profile="pilot"):
    """Original practical comparisons, retained only as an explicit option."""
    if profile not in {"pilot", "smoke"}:
        raise ValueError("profile must be pilot or smoke")
    sizes = [120] if profile == "smoke" else [500, 1000]
    alternatives = [0.0] if profile == "smoke" else [0.0, -0.7, 0.7]
    out = []

    def add(scenario, n, alternative=0.0, **changes):
        mechanism = "state" if scenario.startswith("state_") else "asymmetric"
        cell = dict(family="lasso", scenario=scenario, mechanism=mechanism,
                    calibration="practical",
                    n=n, m=n, rho=0.5, alternative=alternative, kappa=1.0,
                    state_weight=True, state_slope=0.6, signal=0.3,
                    tuning="cv", fixed_constant=1.0 / 3.0,
                    dimension_ratio=2, high_dimension=False,
                    refits=max(2, int(n ** 0.25)), control=False)
        cell.update(changes)
        cell["gap"] = (int(np.ceil(2 * np.log(np.e * n) / abs(np.log(cell["rho"]))))
                       if cell["rho"] else 1)
        # Freeze the dictionary at the evaluation-size specification when a
        # training-history control changes m; this isolates training information.
        p1 = 2 * n if cell["high_dimension"] else int(np.ceil(np.sqrt(n)))
        cell["p1"] = p1
        cell["p2"] = (p1 if mechanism == "state"
                      else p1 * cell["dimension_ratio"])
        out.append(cell)

    for scenario in ["asymmetric_reference", "asymmetric_stronger",
                     "state_weighted_cv", "state_weighted_fixed", "state_high_dimension"]:
        for n in sizes:
            for alternative in alternatives:
                changes = {}
                if scenario == "asymmetric_stronger":
                    changes["kappa"] = 3.0
                if scenario in {"state_weighted_fixed", "state_high_dimension"}:
                    changes["tuning"] = "fixed_rate"
                if scenario == "state_high_dimension":
                    changes["high_dimension"] = True
                add(scenario, n, alternative, **changes)
    n = sizes[-1]
    controls = [
        ("asymmetric_aligned", dict(kappa=0.0)),
        ("state_unweighted", dict(state_weight=False, tuning="fixed_rate", high_dimension=True)),
        ("state_strong_dependence", dict(rho=0.8, tuning="fixed_rate", high_dimension=True)),
        ("state_more_training", dict(m=2 * n, tuning="fixed_rate", high_dimension=True)),
    ]
    for scenario, changes in controls:
        if profile == "smoke" and scenario not in {"state_unweighted", "asymmetric_aligned"}:
            continue
        add(scenario, n, control=True, **changes)
    return out


def make_cells(profile="pilot", include_practical=False):
    """Default: finite-confidence score calibration; practical tuning is optional."""
    templates = _practical_cells(profile)
    out = []
    sizes = [120] if profile == "smoke" else [500, 1000]
    alternatives = [0.0] if profile == "smoke" else [0.0, -0.7, 0.7]

    def add(scenario, n, ratio, alternative=0.0, **changes):
        template = next(c for c in templates if c["scenario"] == "state_weighted_fixed"
                        and c["n"] == n and c["alternative"] == 0)
        cell = dict(template)
        cell.update(scenario=scenario, calibration="theory", tuning="theory",
                    alternative=alternative, m=int(ratio * n),
                    signal_bound=0.8, state_slope_bound=0.6,
                    score_failure_probability=confidence_budget(n),
                    control=scenario not in {"theory_state_main", "theory_state_training20"})
        cell.update(changes)
        cell["gap"] = (int(np.ceil(2 * np.log(np.e * n) / abs(np.log(cell["rho"]))))
                       if cell["rho"] else 1)
        out.append(cell)

    for n in sizes:
        for alternative in alternatives:
            add("theory_state_main", n, 50, alternative)
    n = sizes[-1]
    if profile != "smoke":
        for alternative in alternatives:
            add("theory_state_training20", n, 20, alternative)
        add("theory_state_dependence", n, 50, rho=0.8)
        add("theory_state_dimension256", n, 50, p1=256, p2=256)
    add("theory_state_short_history", n, 1)
    add("theory_state_unweighted", n, 50, state_weight=False)
    return out + (templates if include_practical else [])


def population_parameters(cell):
    """Set E[evaluation loss 1 - evaluation loss 2] exactly to delta/sqrt(n)."""
    theta = float(cell["alternative"] / np.sqrt(cell["n"]))
    b1 = float(cell["signal"])
    if cell["mechanism"] == "asymmetric":
        b2_squared = b1 * b1 + theta / (1 + cell["kappa"] / 2)
    elif cell["state_weight"]:
        b2_squared = b1 * b1 + 2 * b1 * cell["state_slope"] + 2 * theta
    else:
        b2_squared = b1 * b1 + theta
    if b2_squared <= 0:
        raise ValueError("Local alternative exceeds the prespecified signal range")
    b2 = float(np.sqrt(b2_squared))
    if cell["mechanism"] == "asymmetric" and max(b1, b2) >= 0.6:
        raise ValueError("The closed-form asymmetric representer requires signal < min sigma")
    return dict(b1=b1, b2=b2, theta=theta)


def _conditional_population(cell, signs, innovation):
    """Active-sign population values; accepts vector innovation for quadrature."""
    par = population_parameters(cell)
    s1, s2, state = signs
    if cell["mechanism"] == "asymmetric":
        sigma = 1 + 0.4 * state
        y = 0.5 + par["b1"] * s1 + par["b2"] * s2 + sigma * innovation
        f1, f2 = 0.5 + par["b1"] * s1, 0.5 + par["b2"] * s2
        a1 = cell["kappa"] * (sigma ** 2 + par["b2"] ** 2) / (2 * sigma)
        a2 = cell["kappa"] * (sigma ** 2 + par["b1"] ** 2) / (2 * sigma)
        q = 1.0
    else:
        y = ((par["b1"] + cell["state_slope"] * state) * s1
             + par["b2"] * s2 + innovation)
        f1, f2 = par["b1"] * s1, par["b2"] * s2
        q = float(state == 1) if cell["state_weight"] else 1.0
        # E[2 I(W=1) S1 {f1-Y}] = -c because E[I(W=1)W]=1/2.
        a1 = -cell["state_slope"] * s1 if cell["state_weight"] else 0.0
        a2 = 0.0
    return y, f1, f2, a1, a2, q


def _loss(y, f, cell):
    residual = y - f
    if cell["mechanism"] == "asymmetric":
        return residual ** 2 + cell["kappa"] * np.maximum(-residual, 0) ** 2
    return residual ** 2


def _derivative(y, f, q, cell):
    error = f - y
    if cell["mechanism"] == "asymmetric":
        return 2 * (error + cell["kappa"] * np.maximum(error, 0))
    return 2 * q * error


@lru_cache(maxsize=256)
def _oracle_moments_cached(mechanism, n, alternative, signal, kappa,
                           state_slope, state_weight, rho):
    cell = dict(mechanism=mechanism, n=n, alternative=alternative, signal=signal,
                kappa=kappa, state_slope=state_slope, state_weight=state_weight)
    nodes, weights = leggauss(8)
    raw_means, corrected_means, raw_seconds, corrected_seconds = [], [], [], []
    par = population_parameters(cell)
    for signs in product([-1.0, 1.0], repeat=3):
        cuts = [-1.0, 1.0]
        if mechanism == "asymmetric":
            sigma = 1 + 0.4 * signs[2]
            cuts += [-par["b2"] * signs[1] / sigma,
                     -par["b1"] * signs[0] / sigma]
        cuts = sorted(set(v for v in cuts if -1 <= v <= 1))
        totals = np.zeros(4)
        for left, right in zip(cuts[:-1], cuts[1:]):
            epsilon = (right + left) / 2 + (right - left) * nodes / 2
            ww = weights * (right - left) / 4  # Uniform[-1,1] probability weights.
            y, f1, f2, a1, a2, q = _conditional_population(cell, signs, epsilon)
            raw = q * (_loss(y, f1, cell) - _loss(y, f2, cell))
            corrected = raw + a1 * (y - f1) - a2 * (y - f2)
            totals += [ww @ raw, ww @ corrected,
                       ww @ (raw ** 2), ww @ (corrected ** 2)]
        raw_means.append(totals[0]); corrected_means.append(totals[1])
        raw_seconds.append(totals[2]); corrected_seconds.append(totals[3])
    mean_raw = float(np.mean(raw_means))
    mean_corrected = float(np.mean(corrected_means))
    # The active signs share a refresh indicator. Independent outcome innovations
    # imply Cov(g_t,g_{t+h})=rho^h Var(E[g_t|active signs]).
    multiplier = 2 * rho / (1 - rho)
    raw_lrv = float(np.mean(raw_seconds) - mean_raw ** 2 + multiplier * np.var(raw_means))
    corrected_lrv = float(np.mean(corrected_seconds) - mean_corrected ** 2
                          + multiplier * np.var(corrected_means))
    return dict(theta_raw=mean_raw, theta_corrected=mean_corrected,
                oracle_lrv=corrected_lrv, oracle_raw_lrv=raw_lrv,
                oracle_variance_lag0=float(np.mean(corrected_seconds) - mean_corrected ** 2),
                oracle_conditional_mean_variance=float(np.var(corrected_means)),
                oracle_raw_variance_lag0=float(np.mean(raw_seconds) - mean_raw ** 2),
                oracle_raw_conditional_mean_variance=float(np.var(raw_means)))


def population_moments(cell):
    """Exact finite-state / piecewise-polynomial integration, not estimated nulls."""
    names = ["mechanism", "n", "alternative", "signal", "kappa", "state_slope",
             "state_weight", "rho"]
    return dict(_oracle_moments_cached(*(cell[name] for name in names)))


def _sample(cell, rng):
    size = cell["m"] + cell["gap"] + cell["n"]
    p1, p2 = cell["p1"], cell["p2"]
    z = rng.choice([-1.0, 1.0], size=(size, p1 + p2 - 3))
    retain = rng.random(size) < cell["rho"]
    for t in range(1, size):
        if retain[t]:
            z[t] = z[t - 1]
    epsilon = rng.uniform(-1, 1, size)
    par = population_parameters(cell)
    x = [np.column_stack([np.ones(size), z[:, 0], z[:, 2], z[:, 3:p1]]),
         np.column_stack([np.ones(size), z[:, 1], z[:, 2], z[:, p1:p1 + p2 - 3]])]
    assert x[0].shape[1] == p1 and x[1].shape[1] == p2
    if cell["mechanism"] == "asymmetric":
        sigma = 1 + 0.4 * z[:, 2]
        y = 0.5 + par["b1"] * z[:, 0] + par["b2"] * z[:, 1] + sigma * epsilon
        q = np.ones(size)
    else:
        y = ((par["b1"] + cell["state_slope"] * z[:, 2]) * z[:, 0]
             + par["b2"] * z[:, 1] + epsilon)
        q = (z[:, 2] == 1).astype(float) if cell["state_weight"] else np.ones(size)
    betas, alphas = [], []
    for j, p in enumerate([p1, p2]):
        beta = np.zeros(p)
        beta[1] = par["b1"] if j == 0 else par["b2"]
        alpha = np.zeros(p)
        if cell["mechanism"] == "asymmetric":
            beta[0] = 0.5
            omitted = par["b2"] if j == 0 else par["b1"]
            values = cell["kappa"] * (np.array([0.6, 1.4]) ** 2 + omitted ** 2) / (2 * np.array([0.6, 1.4]))
            alpha[0] = np.mean(values)
            alpha[2] = (values[1] - values[0]) / 2
        elif cell["state_weight"] and j == 0:
            alpha[1] = -cell["state_slope"]
        betas.append(beta); alphas.append(alpha)
    return dict(x=x, y=y, q=q, beta=betas, alpha=alphas)


def _fit_lasso(x, y, cell, gap, response_builder=None, role="forecast",
               audit=False, training_start=0):
    """Finite-confidence state calibration, or explicitly optional practical tuning.

    The state theory rule guarantees simultaneous population-target score bounds.
    The retained CV and fixed-rate comparisons do not inherit that guarantee.
    """
    def fit(xx, yy, penalty):
        model = Lasso(alpha=max(float(penalty), 1e-12), fit_intercept=False,
                      max_iter=10000, tol=1e-8, selection="cyclic")
        model.fit(xx, yy)
        return model

    calibration = None
    fold_audits = []
    if cell["tuning"] == "theory":
        if cell["mechanism"] != "state":
            raise ValueError("The finite-state bound is proved for the state design only")
        calibration = calibrate_state_score(
            len(y), max(cell["p1"], cell["p2"]), 4 * cell["refits"],
            cell["rho"], cell["n"], "forecast" if role == "forecast" else "correction",
            cell["state_weight"], cell["signal_bound"], cell["state_slope_bound"])
        penalty = calibration.penalty
        selected = 1.0
    elif cell["tuning"] == "fixed_rate":
        m, p, n, refits = len(y), x.shape[1], cell["n"], cell["refits"]
        penalty = cell["fixed_constant"] * np.sqrt(
            np.log(np.e * n * refits * m) * np.log(np.e * p * refits * n) / m)
        selected = cell["fixed_constant"]
    elif cell["tuning"] == "cv":
        constants = np.array([0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0])
        errors = np.zeros(len(constants))
        for start, end_fraction in [(0.5, 2 / 3), (2 / 3, 5 / 6), (5 / 6, 1.0)]:
            first, end = int(len(y) * start), int(len(y) * end_fraction)
            last = first - gap
            if last < 8:
                raise ValueError("Insufficient chronological training history")
            train_y, validation_y = ((y[:last], y[first:end]) if response_builder is None
                                      else response_builder(last, first, end))
            scale = max(np.std(train_y, ddof=1), 1e-8) * np.sqrt(
                2 * np.log(x.shape[1] + 1) / last)
            # A decreasing penalty path reuses coefficients, not validation
            # outcomes. The grid, objective, tolerance and tie rule are unchanged.
            path_model = Lasso(fit_intercept=False, max_iter=10000, tol=1e-8,
                               selection="cyclic", warm_start=True)
            xx = np.asfortranarray(x[:last])
            validation_x = x[first:end]
            fold_iterations, fold_gaps, fold_convergence, fold_hashes = [], [], [], []
            for k in range(len(constants) - 1, -1, -1):
                path_model.set_params(alpha=max(float(constants[k] * scale), 1e-12))
                path_model.fit(xx, train_y)
                predicted = path_model.predict(validation_x)
                errors[k] += np.sum((validation_y - predicted) ** 2)
                if audit:
                    fold_iterations.append(int(path_model.n_iter_))
                    fold_gaps.append(float(path_model.dual_gap_))
                    fold_convergence.append(bool(path_model.dual_gap_ < 1e-5))
                    fold_hashes.append(hashlib.sha256(np.asarray(path_model.coef_, dtype="<f8").tobytes()).hexdigest())
            if audit:
                fold_audits.append(dict(
                    training_start=int(training_start), training_stop=int(training_start+last-1),
                    validation_start=int(training_start+first), validation_stop=int(training_start+end-1),
                    declared_gap=int(gap), omitted_dates=int(first-last),
                    eligible=bool(first-last >= gap), candidate_fits=len(constants),
                    all_candidates_converged=all(fold_convergence),
                    candidate_coefficient_sha256=fold_hashes,
                    max_iterations=max(fold_iterations), max_dual_gap=max(fold_gaps)))
        selected = constants[np.flatnonzero(errors <= errors.min() + 1e-12)[-1]]
        penalty = selected * max(np.std(y, ddof=1), 1e-8) * np.sqrt(
            2 * np.log(x.shape[1] + 1) / len(y))
    else:
        raise ValueError("Unknown lasso tuning protocol")
    fitted = fit(x, y, penalty)
    if not np.all(np.isfinite(fitted.coef_)):
        raise FloatingPointError("Lasso returned nonfinite coefficients")
    metadata = dict(penalty=float(penalty), multiplier=float(selected),
                    converged=bool(fitted.dual_gap_ < 1e-5),
                    iterations=int(fitted.n_iter_), dual_gap=float(fitted.dual_gap_))
    if calibration is not None:
        metadata["score_calibration"] = calibration.to_dict()
    if audit:
        metadata.update(training_start=int(training_start), training_stop=int(training_start+len(y)-1),
                        coefficient_sha256=hashlib.sha256(np.asarray(fitted.coef_, dtype="<f8").tobytes()).hexdigest(),
                        support=int(np.count_nonzero(fitted.coef_)), cv_folds=fold_audits,
                        all_cv_fits_converged=all(f["all_candidates_converged"] for f in fold_audits),
                        all_validation_eligible=all(f["eligible"] for f in fold_audits))
        if cell["tuning"] == "cv":
            metadata["cv_constants"] = constants.tolist()
            metadata["cv_squared_errors"] = errors.tolist()
    return fitted, metadata


def simulate(cell, replication, seed, collect_diagnostics=False, *, data=None,
             evaluation_offset=None, strict_nested_gap=False, audit_protocol=False,
             collect_fits=False):
    """One draw. Return scores; the common runner performs every inference test."""
    mechanism_seed = 0 if cell["mechanism"] == "asymmetric" else 400000009
    draw_seed = int(seed + 100003 * replication + mechanism_seed + cell["n"])
    if data is None:
        data = _sample(cell, np.random.default_rng(draw_seed))
    n, m, gap = cell["n"], cell["m"], cell["gap"]
    offset = m + gap if evaluation_offset is None else int(evaluation_offset)
    if offset < m+gap or len(data["y"]) < offset+n:
        raise ValueError("Supplied path does not contain the eligible history and evaluation dates")
    prediction, representer = np.zeros((n, 2)), np.zeros((n, 2))
    records, warning_messages = [], []
    fitted_coefficients = []
    population_raw_drift = population_corrected_drift = first_order_drift = 0.0
    if collect_diagnostics and cell["mechanism"] != "state":
        raise ValueError("Exact drift diagnostics are implemented for the state design only")
    drift_series = ({name: np.zeros(n) for name in ["linear", "quadratic", "product"]}
                    if collect_diagnostics else None)
    blocks = np.array_split(np.arange(n), cell["refits"])
    zero_correction = ((cell["mechanism"] == "asymmetric" and cell["kappa"] == 0)
                       or (cell["mechanism"] == "state" and not cell["state_weight"]))
    for block_index, evaluation in enumerate(blocks):
        stop = offset + int(evaluation[0]) - gap
        training = np.arange(stop - m, stop)
        dates = offset + evaluation
        assert training[-1] < dates[0] - gap
        for j in range(2):
            x, y, q = data["x"][j][training], data["y"][training], data["q"][training]
            fit_options = (dict(audit=True, training_start=int(training[0])) if audit_protocol else {})
            model, forecast_meta = _fit_lasso(x, y, cell, gap, **fit_options)
            ftrain = model.predict(x)
            prediction[evaluation, j] = model.predict(data["x"][j][dates])
            generated = _derivative(y, ftrain, q, cell)
            preliminary_audits = []
            if zero_correction:
                alpha_hat = np.zeros(x.shape[1])
                correction_meta = dict(penalty=0.0, multiplier=0.0, converged=True,
                                       iterations=0, dual_gap=0.0)
            else:
                def build_response(last, first, end):
                    # Validation derivative labels are generated by a preliminary
                    # forecast fitted/tuned only before the inner validation block.
                    prefix_gap = gap if strict_nested_gap else min(gap, max(1, last // 12))
                    preliminary, preliminary_meta = _fit_lasso(x[:last], y[:last], cell, prefix_gap,
                                                              **fit_options)
                    if audit_protocol:
                        preliminary_meta.update(derivative_validation_start=int(training[0]+first),
                                                derivative_validation_stop=int(training[0]+end-1),
                                                derivative_gap=int(first-last))
                        preliminary_audits.append(preliminary_meta)
                    train_label = _derivative(y[:last], preliminary.predict(x[:last]), q[:last], cell)
                    validation_label = _derivative(y[first:end], preliminary.predict(x[first:end]), q[first:end], cell)
                    return train_label, validation_label

                correction, correction_meta = _fit_lasso(x, generated, cell, gap,
                                                         response_builder=build_response,
                                                         role="representer", **fit_options)
                alpha_hat = correction.coef_
            representer[evaluation, j] = data["x"][j][dates] @ alpha_hat
            beta0, alpha0 = data["beta"][j], data["alpha"][j]
            forecast_error = float(np.linalg.norm(model.coef_ - beta0))
            representer_error = float(np.linalg.norm(alpha_hat - alpha0))
            f0 = x @ beta0
            h0 = _derivative(y, f0, q, cell)
            forecast_score = float(np.max(np.abs(x.T @ (y - f0) / m)))
            correction_score = float(np.max(np.abs(x.T @ (h0 - x @ alpha0) / m)))
            row = dict(block=block_index, rule=j + 1, training_start=int(training[0]),
                       training_stop=int(training[-1]), evaluation_start=int(dates[0]),
                       forecast_error=forecast_error, representer_error=representer_error,
                       remainder_scale=float(np.sqrt(n) * (forecast_error ** 2
                                             + forecast_error * representer_error)),
                       forecast_penalty=forecast_meta["penalty"],
                       representer_penalty=correction_meta["penalty"],
                       forecast_score=forecast_score, representer_score=correction_score,
                       forecast_score_dominated=forecast_meta["penalty"] >= 2 * forecast_score,
                       representer_score_dominated=(zero_correction or correction_meta["penalty"] >= 2 * correction_score),
                       forecast_converged=forecast_meta["converged"],
                       representer_converged=correction_meta["converged"],
                       forecast_nonzero=int(np.count_nonzero(model.coef_)),
                       representer_nonzero=int(np.count_nonzero(alpha_hat)))
            if collect_fits:
                fitted_coefficients.append(dict(block=block_index, rule=j+1,
                                                forecast=model.coef_.copy(), representer=alpha_hat.copy()))
            if audit_protocol:
                audits = [forecast_meta, correction_meta, *preliminary_audits]
                nested_converged = all(a.get("converged", True) and a.get("all_cv_fits_converged", True)
                                       for a in audits)
                row.update(block_length=len(evaluation), block_weight=float(len(evaluation)/n),
                           forecast_fit_audit=forecast_meta, representer_fit_audit=correction_meta,
                           preliminary_forecast_audits=preliminary_audits,
                           all_nested_fits_converged=nested_converged,
                           all_nested_validation_eligible=all(a.get("all_validation_eligible", True) for a in audits),
                           squared_error_remainder_scale=float(np.sqrt(n)*forecast_error**2),
                           product_error_remainder_scale=float(np.sqrt(n)*forecast_error*representer_error))
                if not nested_converged:
                    warning_messages.append(f"Nested lasso KKT tolerance not reached at block {block_index}, rule {j+1}")
            if x.shape[1] <= 256:
                row["min_gram_eigenvalue"] = float(np.linalg.eigvalsh(x.T @ x / m)[0])
            if cell["calibration"] == "theory":
                forecast_calibration = forecast_meta["score_calibration"]
                correction_calibration = calibrate_state_score(
                    m, max(cell["p1"], cell["p2"]), 4 * cell["refits"],
                    cell["rho"], n, "correction", cell["state_weight"],
                    cell["signal_bound"], cell["state_slope_bound"]).to_dict()
                row.update(calibration_failure_probability=forecast_calibration["failure_probability"],
                           forecast_score_threshold=forecast_calibration["score_threshold"],
                           representer_score_threshold=correction_calibration["score_threshold"],
                           guaranteed_forecast_penalty=forecast_calibration["penalty"],
                           guaranteed_representer_penalty=correction_calibration["penalty"],
                           forecast_bound_tilt=forecast_calibration["selected_tilt"],
                           representer_bound_tilt=correction_calibration["selected_tilt"],
                           forecast_bound_log_mgf=forecast_calibration["log_mgf_upper"],
                           representer_bound_log_mgf=correction_calibration["log_mgf_upper"])
            records.append(row)
            if not forecast_meta["converged"] or not correction_meta["converged"]:
                warning_messages.append(f"Lasso KKT tolerance not reached at block {block_index}, rule {j+1}")
            if cell["mechanism"] == "state":
                u, b = model.coef_ - beta0, alpha_hat - alpha0
                quadratic = (0.5 * np.dot(u, u) + u[0] * u[2]
                             if cell["state_weight"] else np.dot(u, u))
                first = float(alpha0 @ u)
                sign = 1 if j == 0 else -1
                weight = sign * len(evaluation) / n
                population_raw_drift += weight * (first + quadratic)
                population_corrected_drift += weight * (quadratic - b @ u)
                first_order_drift += weight * first
                if collect_diagnostics:
                    # Independent-population expectations with fitted objects
                    # held fixed; these are not conditional expectations of
                    # the dependent, possibly overlapping evaluation sample.
                    drift_series["linear"][evaluation] += sign * first
                    drift_series["quadratic"][evaluation] += sign * quadratic
                    drift_series["product"][evaluation] -= sign * float(b @ u)
                    row.update(population_linear_drift=sign * first,
                               population_quadratic_drift=sign * quadratic,
                               population_product_drift=-sign * float(b @ u),
                               forecast_signal_coefficient=float(model.coef_[1]),
                               representer_signal_coefficient=float(alpha_hat[1]))
    evaluation_dates = np.arange(offset, offset + n)
    y, q = data["y"][evaluation_dates], data["q"][evaluation_dates]
    f0 = np.column_stack([data["x"][j][evaluation_dates] @ data["beta"][j] for j in range(2)])
    a0 = np.column_stack([data["x"][j][evaluation_dates] @ data["alpha"][j] for j in range(2)])

    def corrected(f, a):
        risk = q * (_loss(y, f[:, 0], cell) - _loss(y, f[:, 1], cell))
        return risk + a[:, 0] * (y - f[:, 0]) - a[:, 1] * (y - f[:, 1])

    scores = dict(raw=q * (_loss(y, prediction[:, 0], cell) - _loss(y, prediction[:, 1], cell)),
                  orthogonal=corrected(prediction, representer),
                  known_representer=corrected(prediction, a0),
                  oracle=corrected(f0, a0),
                  oracle_raw=q * (_loss(y, f0[:, 0], cell) - _loss(y, f0[:, 1], cell)))
    max_forecast = max(row["forecast_error"] for row in records)
    max_representer = max(row["representer_error"] for row in records)
    diagnostic = dict(max_forecast_error=max_forecast, max_representer_error=max_representer,
                      uniform_remainder_scale=float(np.sqrt(n) * (max_forecast ** 2 + max_forecast * max_representer)),
                      simultaneous_forecast_score_event=all(row["forecast_score_dominated"] for row in records),
                      simultaneous_representer_score_event=all(row["representer_score_dominated"] for row in records),
                      score_mean_gap=float(np.mean(scores["orthogonal"] - scores["oracle"])),
                      score_rms_gap=float(np.sqrt(np.mean((scores["orthogonal"] - scores["oracle"]) ** 2))),
                      all_fits_converged=all(row["forecast_converged"] and row["representer_converged"] for row in records))
    diagnostic["simultaneous_score_event"] = bool(
        diagnostic["simultaneous_forecast_score_event"]
        and diagnostic["simultaneous_representer_score_event"])
    eigenvalues = [row["min_gram_eigenvalue"] for row in records if "min_gram_eigenvalue" in row]
    if eigenvalues:
        diagnostic["min_gram_eigenvalue"] = min(eigenvalues)
        diagnostic["all_gram_eigenvalues_above_half"] = min(eigenvalues) >= 0.5
    if cell["calibration"] == "theory":
        diagnostic.update(calibration_failure_probability=records[0]["calibration_failure_probability"],
                          guaranteed_forecast_penalty=records[0]["guaranteed_forecast_penalty"],
                          guaranteed_representer_penalty=records[0]["guaranteed_representer_penalty"],
                          representer_structurally_zero=zero_correction)
    if cell["mechanism"] == "state":
        diagnostic.update(scaled_population_raw_drift=float(np.sqrt(n) * population_raw_drift),
                          scaled_population_corrected_drift=float(np.sqrt(n) * population_corrected_drift),
                          scaled_first_order_raw_drift=float(np.sqrt(n) * first_order_drift))
    moments = population_moments(cell)
    result = dict(theta=population_parameters(cell)["theta"], scores=scores,
                diagnostics=diagnostic, learning=records, warnings=warning_messages,
                oracle_lrv=moments["oracle_lrv"], oracle_raw_lrv=moments["oracle_raw_lrv"])
    if collect_diagnostics:
        drift_series["raw"] = drift_series["linear"] + drift_series["quadratic"]
        drift_series["orthogonal"] = drift_series["quadratic"] + drift_series["product"]
        drift_series["known_representer"] = drift_series["quadratic"].copy()
        scores.update(
            raw_centered=scores["raw"] - drift_series["raw"],
            orthogonal_centered=scores["orthogonal"] - drift_series["orthogonal"],
            orthogonal_no_quadratic=scores["orthogonal"] - drift_series["quadratic"],
            orthogonal_no_product=scores["orthogonal"] - drift_series["product"],
            known_forecasts=corrected(f0, representer),
            known_representer_centered=scores["known_representer"] - drift_series["known_representer"])
        for name in ["linear", "quadratic", "product"]:
            diagnostic[f"population_{name}_drift"] = float(drift_series[name].mean())
            diagnostic[f"scaled_population_{name}_drift"] = float(np.sqrt(n) * drift_series[name].mean())
        result["drift_series"] = drift_series
        result["oracle_moments"] = moments
    if collect_fits:
        result["fitted_objects"] = dict(prediction=prediction, representer=representer,
                                         coefficients=fitted_coefficients)
    return result


def verify():
    """Independent integration checks for target, representer, and directions."""
    nodes, weights = leggauss(24)
    checked = 0
    for cell in make_cells("pilot", include_practical=True):
        moment = population_moments(cell)
        theta = population_parameters(cell)["theta"]
        assert abs(moment["theta_raw"] - theta) < 1e-12
        assert abs(moment["theta_corrected"] - theta) < 1e-12
        assert moment["oracle_lrv"] > 0
        # Check E[X dot ell] = alpha coefficients for the active dictionary.
        derivative_moments = np.zeros((2, 3))
        training_moments = np.zeros((2, 3))
        alpha_moments = np.zeros((2, 3))
        par = population_parameters(cell)
        for signs in product([-1.0, 1.0], repeat=3):
            cuts = [-1.0, 1.0]
            if cell["mechanism"] == "asymmetric":
                sigma = 1 + 0.4 * signs[2]
                cuts += [-par["b2"] * signs[1] / sigma, -par["b1"] * signs[0] / sigma]
            cuts = sorted(set(v for v in cuts if -1 <= v <= 1))
            for left, right in zip(cuts[:-1], cuts[1:]):
                epsilon = (right + left) / 2 + (right - left) * nodes / 2
                ww = weights * (right - left) / 32  # probability includes 1/8 sign mass
                y, f1, f2, a1, a2, q = _conditional_population(cell, signs, epsilon)
                for j, (f, a) in enumerate([(f1, a1), (f2, a2)]):
                    x = np.array([1.0, signs[j], signs[2]])
                    derivative_moments[j] += x * float(ww @ _derivative(y, f, q, cell))
                    training_moments[j] += x * float(ww @ (f - y))
                    alpha_moments[j] += x * a * np.sum(ww)
        assert np.max(np.abs(training_moments)) < 1e-12
        # Sigma=I, so equality verifies the entire linear derivative identity,
        # including all directions (noise coordinates have zero expectations).
        assert np.max(np.abs(derivative_moments - alpha_moments)) < 1e-12
        checked += 1
    return f"Lasso: {checked} cells passed exact population target, mean-zero training score, and representer checks."


def verify_no_future_leakage():
    """Perturb every evaluation outcome and compare the initial fitted objects' diagnostics."""
    cell = next(c for c in make_cells("smoke") if c["scenario"] == "theory_state_main")
    original_sample = globals()["_sample"]
    baseline = simulate(cell, 91, 20261104)

    def changed_future(c, rng):
        data = original_sample(c, rng)
        data["y"][c["m"] + c["gap"]:] += 10
        return data

    globals()["_sample"] = changed_future
    try:
        altered = simulate(cell, 91, 20261104)
    finally:
        globals()["_sample"] = original_sample
    for initial, modified in zip(baseline["learning"][:2], altered["learning"][:2]):
        assert initial == modified
    return "Lasso: future outcomes leave both initial forecast/correction fits unchanged."
