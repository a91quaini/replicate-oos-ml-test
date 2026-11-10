"""Audited publication assets for the completed five-input ReLU experiment.

Reads only the frozen expanded-study results. It never fits a network, changes
the simulation protocol, or edits TeX. Output tables are fragments to embed in
the self-contained manuscript. Run with the project's scientific Python.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/neural"
OUT = ROOT / "output/neural"
FIGURES = ROOT / "output/figures"
os.environ.setdefault("MPLCONFIGDIR", str(OUT / "plot-cache"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

METHODS = ("raw", "raw_bootstrap", "orthogonal", "orthogonal_bootstrap")
LABELS = dict(zip(METHODS, ("Raw", "Bootstrap Raw", "Orthogonal", "Bootstrap Orthogonal")))
RAW, ORTH = "#B65035", "#0072B2"
STYLES = {
    "raw": (RAW, "o", "-", True),
    "raw_bootstrap": (RAW, "^", "--", True),
    "orthogonal": (ORTH, "o", "-", True),
    "orthogonal_bootstrap": (ORTH, "^", "--", True),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    if path.suffix == ".gz":
        return json.loads(gzip.decompress(path.read_bytes()))
    return json.loads(path.read_text())


def adjusted(null, alternative):
    cutoff = np.quantile(null, .95, method="inverted_cdf")
    weight = np.clip((.05 - np.mean(null > cutoff)) / np.mean(null == cutoff), 0, 1)
    power = np.mean((alternative > cutoff).astype(float) + weight * (alternative == cutoff))
    return float(power), float(cutoff), float(weight)


def audit():
    """Check compact saved statistics, replication IDs, and all summaries."""
    cfg = read(DATA / "configuration.json")
    assert cfg["profile"] == "production"
    cells = {c["id"]: c for c in cfg["cells"]}
    counts = {int(k): v for k, v in cfg["counts"].items()}
    assert len(cells) == 11 and sum(counts.values()) == 7168
    saved = pd.read_csv(DATA / "draw_statistics.csv.gz")
    assert not saved.duplicated(["cell_id", "replication", "method"]).any()
    summary = pd.read_csv(DATA / "summary.csv")
    diagnostic = pd.read_csv(DATA / "diagnostic_summary.csv")
    assert saved.valid.all()
    seen = set(map(tuple, saved[["cell_id", "replication"]].drop_duplicates().to_numpy()))
    expected = {(i, r) for i, n in counts.items() for r in range(1, n + 1)}
    assert seen == expected
    for method, group in saved.groupby("method"):
        rejection = group.score >= -.05 if "bootstrap" in method else group.score > 1.959963984540054
        assert (rejection == group.reject).all(), ("rejection rule", method)
        if "bootstrap" not in method:
            assert np.allclose(group.score, abs(group.estimate / group.se))
    groups = {key: g.sort_values("replication") for key, g in saved.groupby(["cell_id", "method"])}
    nulls = {(c["n"], c["T"], c["dgp"]): i for i, c in cells.items() if c["delta"] == 0}
    for row in summary.itertuples():
        g = groups[row.cell_id, row.method]
        c = cells[row.cell_id]
        null = groups[nulls[c["n"], c["T"], c["dgp"]], row.method]
        power, cutoff, weight = adjusted(null.score.to_numpy(), g.score.to_numpy())
        assert len(g) == row.completed == row.valid == counts[row.cell_id]
        for a, b in [(g.reject.mean(), row.rejection), (g.error.mean(), row.bias),
                     (power, row.adjusted_power), (cutoff, row.critical), (weight, row.boundary_weight)]:
            assert np.isclose(a, b, rtol=1e-11, atol=1e-13), (row.cell_id, row.method, a, b)
    provenance = {
        "study": "Five-input nonlinear probability forecasts; completed expanded study",
        "production_replications": len(seen), "cells": len(cells),
        "all_statistics_valid": True,
        "checks": ["packaged input hashes verified by reproduce.py", "all replication IDs and counts",
                   "stored rejection rules", "all empirical null cutoffs and adjusted powers"],
        "input_sha256": {name: sha(DATA / name) for name in
                         ("configuration.json", "summary.csv", "diagnostic_summary.csv",
                          "draw_statistics.csv.gz", "learning.csv.gz", "power_comparisons.csv")},
        "scope": "Compact replication-level export; original raw checkpoints are not distributed.",
        "script_sha256": sha(Path(__file__)),
        "bootstrap": "999 centered circular-block bootstrap-t draws; learned objects held fixed",
        "design_history": "Pilots excluded; expansion followed review of initial production; all agreed cells retained",
    }
    return cfg, summary, diagnostic, saved, provenance


def power_uncertainty(cfg, draws, repeats=1000):
    """Paired stratified resampling includes empirical-cutoff uncertainty.

    The first 512 null draws share latent paths with the 512 alternative draws.
    Resample those indices jointly and the extra 512 null indices separately.
    Common resampling indices are also used across all four feasible methods.
    """
    methods = METHODS
    cells = {c["id"]: c for c in cfg["cells"]}
    arrays = {(i, m): g.sort_values("replication").score.to_numpy()
              for (i, m), g in draws.groupby(["cell_id", "method"]) if m in methods}
    rng = np.random.default_rng(2026111419)
    rows, differences = [], []
    for i, cell in cells.items():
        if cell["delta"] == 0:
            continue
        null = next(j for j, c in cells.items() if c["delta"] == 0 and
                    (c["n"], c["T"], c["dgp"]) == (cell["n"], cell["T"], cell["dgp"]))
        n, N = len(arrays[i, "raw"]), len(arrays[null, "raw"])
        assert n == 512 and N == 1024
        boot = np.empty((repeats, len(methods)))
        point = np.array([adjusted(arrays[null, m], arrays[i, m])[0] for m in methods])
        for b in range(repeats):
            shared = rng.integers(n, size=n)
            index = np.r_[shared, rng.integers(n, N, size=N - n)]
            boot[b] = [adjusted(arrays[null, m][index], arrays[i, m][shared])[0] for m in methods]
        for j, method in enumerate(methods):
            lower, upper = np.quantile(boot[:, j], [.025, .975])
            rows.append(dict(cell_id=i, delta=cell["delta"], method=method,
                             adjusted_power=point[j], lower=lower, upper=upper, resamples=repeats))
            if method != "orthogonal":
                k = methods.index("orthogonal")
                lower, upper = np.quantile(boot[:, k] - boot[:, j], [.025, .975])
                differences.append(dict(cell_id=i, delta=cell["delta"], comparison="orthogonal minus " + method,
                                        adjusted_power_difference=point[k] - point[j], lower=lower, upper=upper,
                                        resamples=repeats))
    individual, comparisons = pd.DataFrame(rows), pd.DataFrame(differences)
    existing = pd.read_csv(DATA / "power_comparisons.csv")
    existing = existing[existing.comparison.isin(comparisons.comparison)]
    check = existing.merge(comparisons, on=["cell_id", "comparison"], suffixes=("_old", "_new"))
    assert len(check) == len(existing)
    for name in ("adjusted_power_difference", "lower", "upper"):
        assert np.allclose(check[name + "_old"], check[name + "_new"], atol=1e-12)
    individual.to_csv(OUT / "nn_power_uncertainty.csv", index=False)
    comparisons.to_csv(OUT / "nn_power_comparisons.csv", index=False)
    return individual, comparisons


def configure_plots():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 11, "axes.labelsize": 10.5,
                         "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "legend.fontsize": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#777777",
                         "axes.linewidth": .65, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "savefig.facecolor": "white", "axes.axisbelow": True})


def handles(methods=METHODS):
    output = []
    for method in methods:
        color, marker, line, filled = STYLES[method]
        output.append(Line2D([0], [0], color=color, marker=marker, linestyle=line,
                             markerfacecolor=color if filled else "white", markersize=5,
                             linewidth=1.4, label=LABELS[method]))
    return output


def line(ax, x, y, method, **kwargs):
    color, marker, style, filled = STYLES[method]
    ax.plot(x, y, color=color, marker=marker, linestyle=style,
            markerfacecolor=color if filled else "white", markersize=4.5,
            linewidth=1.5, label=LABELS[method], **kwargs)


def save(fig, name, plotted):
    FIGURES.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(FIGURES / (name + ".pdf"), bbox_inches="tight", pad_inches=.04,
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(OUT / "figures" / (name + ".png"), dpi=180, bbox_inches="tight", pad_inches=.04)
    plt.close(fig)
    plotted.to_csv(OUT / (name + "_plot.csv"), index=False)


def wilson(rate, n):
    z = 1.959963984540054
    denominator = 1 + z * z / n
    center = (rate + z * z / (2 * n)) / denominator
    half = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
    return center - half, center + half


def size_figure(summary):
    data = summary[(summary.delta == 0) & summary.method.isin(METHODS)].copy()
    data["lower"], data["upper"] = wilson(data.rejection, data.valid)
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.65), sharey=True)
    fig.subplots_adjust(left=.235, right=.99, bottom=.20, top=.85, wspace=.17)
    for i, ax in enumerate(axes):
        panel = data[data.cell_id == i].set_index("method")
        row = panel.iloc[0]
        ax.axvspan(0, 5, color="#F4F5F6", zorder=0)
        ax.axvline(5, color="#606060", linestyle="--", linewidth=.9)
        ax.axhline(1.5, color="#D5D5D5", linewidth=.6)
        for j, method in enumerate(METHODS):
            r = panel.loc[method]
            color, marker, _, filled = STYLES[method]
            ax.errorbar(100 * r.rejection, j, xerr=100 * np.array([[r.rejection - r.lower], [r.upper - r.rejection]]),
                        color=color, marker=marker, markerfacecolor=color if filled else "white",
                        markersize=5, capsize=2, elinewidth=1.05, linestyle="none")
        ax.set_title(f"$n={int(row.n):,}$\n$T={int(row['T']):,}$", pad=8)
        ax.set_xlim(0, 20)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.grid(axis="x", color="#E8E8E8", linewidth=.5)
        ax.set_yticks(range(len(METHODS)), [LABELS[m] for m in METHODS])
        ax.tick_params(axis="y", length=0, pad=8)
        ax.spines["left"].set_visible(False)
    axes[0].set_ylim(len(METHODS)-.45, -.55)
    for tick, method in zip(axes[0].get_yticklabels(), METHODS):
        tick.set_color(STYLES[method][0])
    fig.supxlabel("Null rejection (%)", x=.61, y=.015, fontsize=10.5)
    save(fig, "nn_size", data)


def power_figure(summary):
    data = summary[(summary.n == 1000) & (summary['T'] == 4000) & summary.method.isin(METHODS)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.55), sharey=True)
    fig.subplots_adjust(left=.085, right=.99, bottom=.31, top=.90, wspace=.18)
    for ax, column, title in zip(axes, ("rejection", "adjusted_power"), ("Nominal rejection", "Size-adjusted power")):
        for method in METHODS:
            panel = data[data.method == method].sort_values("delta")
            line(ax, panel.delta, 100 * panel[column], method)
        ax.axhline(5, color="#777777", linestyle="--", linewidth=.75)
        ax.axvline(0, color="#DDDDDD", linewidth=.7)
        ax.set_title(title)
        ax.set_xlabel(r"Local effect $\delta$")
        ax.set_xticks([-.6, -.4, -.2, 0, .2, .4, .6])
        ax.set_ylim(0, 62)
        ax.set_yticks([0, 20, 40, 60])
        ax.grid(axis="y", color="#EEEEEE", linewidth=.6)
    axes[0].set_ylabel("Rejection (%)")
    legend_handles = handles()
    fig.legend(handles=[legend_handles[j] for j in (0, 2, 1, 3)], loc="lower center", bbox_to_anchor=(.54, -.015),
               ncol=2, frameon=False, columnspacing=1.0, handletextpad=.5)
    save(fig, "nn_power", data)


def diagnostic_figure(summary, diagnostics, draws, learning):
    # Ground-truth population drift is a learning diagnostic. Its scale here is
    # sqrt(n), with no variance from an infeasible test in the denominator.
    plotted = diagnostics[diagnostics.method.isin(("raw", "orthogonal"))][[
        "cell_id", "n", "T", "delta", "method", "bias", "empirical_sd", "rms_se"]].copy()
    drift = draws[draws.method.eq("orthogonal")].copy()
    drift["abs_raw_drift"] = (drift.first_order_drift + drift.quadratic_raw_drift).abs()
    drift["abs_corrected_drift"] = drift.corrected_drift.abs()
    drift = drift.groupby("cell_id")[["abs_raw_drift", "abs_corrected_drift"]].mean()
    plotted = plotted.merge(drift, on="cell_id", validate="many_to_one")
    for method, column in [("raw", "abs_raw_drift"), ("corrected", "abs_corrected_drift")]:
        plotted["mean_abs_scaled_" + method + "_drift"] = np.sqrt(plotted.n) * plotted[column]
    plotted["rms_se_over_empirical_sd"] = plotted.rms_se / plotted.empirical_sd
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.8))
    fig.subplots_adjust(left=.095, right=.99, bottom=.10, top=.94, wspace=.29, hspace=.60)
    x = np.arange(3)
    labels = ["(500, 4000)", "(1000, 4000)", "(1000, 8000)"]
    null = plotted[(plotted.delta == 0) & (plotted.method == "orthogonal")].sort_values("cell_id")
    ax = axes[0, 0]
    for method, column in [("raw", "mean_abs_scaled_raw_drift"), ("orthogonal", "mean_abs_scaled_corrected_drift")]:
        line(ax, x, null[column], method)
    ax.set_title("(a) Population drift", loc="left")
    ax.set_ylabel(r"Mean absolute scaled drift $|D_a|$", fontsize=10)
    ax.set_ylim(0, .20)
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    ax = axes[0, 1]
    for method in ("raw", "orthogonal"):
        panel = plotted[(plotted.delta == 0) & (plotted.method == method)].sort_values("cell_id")
        line(ax, x, panel.rms_se / panel.empirical_sd, method)
    ax.axhline(1, color="#777777", linestyle="--", linewidth=.8)
    ax.set_ylim(.65, 1.07)
    ax.set_title("(b) Studentization", loc="left")
    ax.set_ylabel("RMS standard error / MC SD", fontsize=10)
    ax.legend(frameon=False, fontsize=10, ncol=2, loc="lower center", columnspacing=.7,
              handlelength=1.4, handletextpad=.4, borderpad=.2)
    central = summary[(summary.n == 1000) & (summary['T'] == 4000)]
    ax = axes[1, 0]
    for column, label, color, marker in [
            ("mean_forecast_error", "Forecast", "#44545D", "s"),
            ("mean_representer_error", "Correction", "#55846A", "^")]:
        for rule, style in [(1, "-"), (2, "--")]:
            panel = learning[(learning.cell_id < 3) & (learning.rule == rule)].sort_values("cell_id")
            ax.plot(x, panel[column], color=color, marker=marker, linestyle=style,
                    label=f"{label}, rule {rule}", markersize=4, linewidth=1.3)
    ax.set_title("(c) Forecast and correction accuracy", loc="left")
    ax.set_ylabel(r"Mean $L^2$ error")
    ax.set_ylim(0, .22)
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2,
              columnspacing=.5, handlelength=1.7, handletextpad=.3)
    ax = axes[1, 1]
    for method in ("raw", "orthogonal"):
        panel = central[central.method == method].sort_values("delta")
        line(ax, panel.delta, 1000 * panel.bias, method)
    ax.axhline(0, color="#777777", linestyle="--", linewidth=.75)
    ax.set_title("(d) Signed estimation bias", loc="left")
    ax.set_ylabel(r"Bias $\times 1000$")
    ax.legend(frameon=False, fontsize=10, loc="upper right", handlelength=1.8)
    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_xlabel(r"Evaluation and training sizes $(n,T)$", fontsize=10)
    axes[1, 1].set_xticks([-.6, -.3, 0, .3, .6])
    axes[1, 1].set_xlabel(r"Local effect $\delta$")
    axes[1, 1].axvline(0, color="#DDDDDD", linewidth=.6)
    for ax in axes.ravel():
        ax.grid(axis="y", color="#EEEEEE", linewidth=.5)
    save(fig, "nn_diagnostics", plotted)


def table(lines, name):
    path = OUT / "tables" / name
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def percent(value):
    """One-decimal percentages with the conventional half-up display rule."""
    return str((100 * Decimal(str(value))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def tables(summary, comparisons):
    null = summary[summary.delta == 0]
    size = [r"\begin{tabular}{rr*{4}{r}}", r"\toprule",
            r"$n$ & $T$ & Raw & Boot. Raw & Orth. & Boot. Orth.\\", r"\midrule"]
    for i, group in null.groupby("cell_id"):
        row = group.iloc[0]
        rates = group.set_index("method").rejection
        size.append(f"{int(row.n):,} & {int(row['T']):,} & " + " & ".join(percent(rates[m]) for m in METHODS) + r"\\")
    table(size + [r"\bottomrule", r"\end{tabular}"], "nn_size.tex")
    sensitivity = [r"\begin{tabular}{l*{6}{r}}", r"\toprule",
        r"$(n,T)$ & \multicolumn{2}{c}{$(500,4{,}000)$} & \multicolumn{2}{c}{$(1{,}000,4{,}000)$} & \multicolumn{2}{c}{$(1{,}000,8{,}000)$}\\",
        r"Calibration & Raw & Orth. & Raw & Orth. & Raw & Orth.\\", r"\midrule"]
    for title, suffix in [("HAC half bandwidth", "_hac_half"), ("HAC baseline", ""),
                          ("HAC double bandwidth", "_hac_double"), ("Bootstrap short blocks", "_bootstrap_short"),
                          ("Bootstrap baseline blocks", "_bootstrap"), ("Bootstrap long blocks", "_bootstrap_long")]:
        values = []
        for i in range(3):
            rates = null[null.cell_id == i].set_index("method").rejection
            values.extend([100 * rates[method + suffix] for method in ("raw", "orthogonal")])
        sensitivity.append(title + " & " + " & ".join(f"{x:.1f}" for x in values) + r"\\")
    table(sensitivity + [r"\bottomrule", r"\end{tabular}"], "nn_sensitivity.tex")
    comp = [r"\begin{tabular}{r*{2}{r}}", r"\toprule",
            r"$\delta$ & Orth. $-$ Raw & Orth. $-$ Boot. Raw\\", r"\midrule"]
    for delta, group in comparisons.groupby("delta", sort=True):
        group = group.set_index("comparison")
        values = []
        for method in ("raw", "raw_bootstrap"):
            r = group.loc["orthogonal minus " + method]
            values.append(f"{100*r.adjusted_power_difference:.1f} [{100*r.lower:.1f}, {100*r.upper:.1f}]")
        comp.append(f"{delta:+.1f} & " + " & ".join(values) + r"\\")
    table(comp + [r"\bottomrule", r"\end{tabular}"], "nn_power_comparisons.tex")
    learning = pd.read_csv(DATA / "learning.csv.gz")
    per_draw = learning.groupby(["cell_id", "replication", "rule"]).agg(
        relative_forecast_error=("relative_forecast_error", "mean"),
        forecast_error=("forecast_error", "mean"), representer_error=("representer_error", "mean"),
        validation_mse=("validation_mse", "mean"), validation_constant_mse=("validation_constant_mse", "mean"),
        epochs=("epochs", "mean"))
    rows = []
    for (i, rule), group in per_draw.reset_index().groupby(["cell_id", "rule"]):
        rows.append(dict(cell_id=i, rule=rule + 1,
                         median_relative_forecast_error=group.relative_forecast_error.median(),
                         mean_forecast_error=group.forecast_error.mean(),
                         mean_representer_error=group.representer_error.mean(),
                         mean_validation_gain=(group.validation_constant_mse - group.validation_mse).mean(),
                         mean_selected_epochs=group.epochs.mean()))
    learn = pd.DataFrame(rows)
    learn.to_csv(OUT / "nn_learning.csv", index=False)
    lines = [r"\begin{tabular}{rrr*{4}{r}}", r"\toprule",
             r"$n$ & $T$ & Rule & Rel. RMSE & Forecast & Correction & Val. gain\\", r"\midrule"]
    for r in learn[learn.cell_id < 3].itertuples():
        design = null[null.cell_id == r.cell_id].iloc[0]
        lines.append(f"{int(design.n):,} & {int(design['T']):,} & {r.rule} & {r.median_relative_forecast_error:.3f} & "
                     f"{r.mean_forecast_error:.3f} & {r.mean_representer_error:.3f} & {1000*r.mean_validation_gain:.3f}" + r"\\")
    table(lines + [r"\bottomrule", r"\end{tabular}"], "nn_learning.tex")
    return learn


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    previous = OUT / "nn_six_methods.csv"
    if previous.exists():
        history = OUT / "history"
        history.mkdir(exist_ok=True)
        archived = history / previous.name
        if archived.exists():
            assert archived.read_bytes() == previous.read_bytes()
            previous.unlink()
        else:
            previous.rename(archived)
    cfg, summary, diagnostics, draws, provenance = audit()
    print("NN audit passed: 7,168 production draws, 11 cells, all stored tests and adjusted powers verified.")
    _, comparisons = power_uncertainty(cfg, draws)
    configure_plots()
    size_figure(summary)
    power_figure(summary)
    learning = tables(summary, comparisons)
    diagnostic_figure(summary, diagnostics, draws, learning)
    summary[summary.method.isin(METHODS)].to_csv(OUT / "nn_four_methods.csv", index=False)
    provenance["power_uncertainty"] = "1000 paired stratified percentile resamples; empirical null cutoff re-estimated; pointwise"
    provenance["outputs_sha256"] = {str(p.relative_to(ROOT)): sha(p) for p in
        sorted(FIGURES.glob("nn_*.pdf")) + sorted(OUT.glob("*.csv")) + sorted((OUT / "tables").glob("*.tex"))}
    (OUT / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("Saved nn_size, nn_power, nn_diagnostics, table fragments, plotted data, and audit manifest.")


if __name__ == "__main__":
    main()
