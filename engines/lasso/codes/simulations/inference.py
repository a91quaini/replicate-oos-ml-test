"""Inference and diagnostic summaries shared by the exploratory designs."""
import numpy as np
from scipy.stats import norm


def bartlett_lrv(values, bandwidth=None):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Scores must be a finite one-dimensional array")
    n = len(values)
    if n < 2:
        raise ValueError("At least two evaluation observations are required")
    lag = min(n - 1, int(np.ceil(n ** 0.25)) if bandwidth is None else int(bandwidth))
    if lag < 0:
        raise ValueError("Bandwidth must be nonnegative")
    centered = values - values.mean()
    variance = float(centered @ centered / n)
    for h in range(1, lag + 1):
        variance += 2 * (1 - h / (lag + 1)) * float(centered[h:] @ centered[:-h] / n)
    return max(0.0, variance)


def statistic(values, theta, variance=None, bandwidth=None):
    values = np.asarray(values, dtype=float)
    estimate = float(values.mean())
    variance = bartlett_lrv(values, bandwidth) if variance is None else float(variance)
    valid = bool(np.isfinite(variance) and variance > 1e-24)
    se = float(np.sqrt(variance / len(values))) if valid else None
    critical = float(norm.ppf(0.975))
    return dict(
        estimate=estimate, variance=variance if np.isfinite(variance) else None,
        se=se, reject=bool(abs(estimate) > critical * se) if valid else None,
        covered=bool(abs(estimate - theta) <= critical * se) if valid else None,
        interval_length=2 * critical * se if valid else None, valid=valid,
    )


def summarize(result, n, bandwidth=None):
    """Known-variance statistics are explicitly infeasible diagnostics."""
    if n < 2:
        raise ValueError("At least two evaluation observations are required")
    theta = float(result["theta"])
    if not np.isfinite(theta):
        raise ValueError("The population risk contrast must be finite")
    scores = result["scores"]
    required = {"raw", "orthogonal", "known_representer", "oracle"}
    if not required.issubset(scores):
        raise ValueError(f"Missing score methods: {sorted(required - set(scores))}")
    for method, values in scores.items():
        if np.asarray(values).shape != (n,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid scores for {method}")
    bandwidth = int(np.ceil(n ** 0.25)) if bandwidth is None else int(bandwidth)
    statistics = {method: statistic(values, theta, bandwidth=bandwidth) for method, values in scores.items()}
    # Prespecified sensitivity: use exactly the shorter bandwidth of the older pilot.
    for method in ["raw", "orthogonal"]:
        statistics[method + "_short_hac"] = statistic(scores[method], theta)
    if result.get("oracle_lrv") is not None:
        oracle_lrv = float(result["oracle_lrv"])
        if not np.isfinite(oracle_lrv) or oracle_lrv <= 1e-12:
            raise ValueError("The design's oracle long-run variance is not positive")
        for method in ["orthogonal", "oracle"]:
            statistics[method + "_known_variance"] = statistic(scores[method], theta, oracle_lrv)
    diagnostics = dict(result.get("diagnostics", {}))
    gap = np.asarray(scores["orthogonal"]) - np.asarray(scores["oracle"])
    diagnostics.update(
        score_mean_gap=float(gap.mean()),
        sqrt_n_score_mean_gap=float(np.sqrt(n) * gap.mean()),
        score_rms_gap=float(np.sqrt(np.mean(gap ** 2))),
        hac_bandwidth=bandwidth,
        short_hac_bandwidth=int(np.ceil(n ** 0.25)),
    )
    if result.get("oracle_lrv") is not None:
        diagnostics["oracle_lrv"] = float(result["oracle_lrv"])
        diagnostics["normalized_score_mean_gap"] = float(np.sqrt(n) * gap.mean() / np.sqrt(result["oracle_lrv"]))
    learning = result.get("learning", [])
    for column, destination in [("forecast_error", "max_forecast_error"),
                                ("representer_error", "max_representer_error"),
                                ("remainder_scale", "max_remainder_scale")]:
        values = [float(row[column]) for row in learning if column in row]
        if values:
            diagnostics[destination] = max(values)
    return dict(theta=theta, statistics=statistics, diagnostics=diagnostics,
                learning=learning, warnings=result.get("warnings", []))
