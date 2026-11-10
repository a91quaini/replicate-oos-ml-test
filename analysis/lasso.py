#!/usr/bin/env python3
"""Audited four-test lasso presentation from completed frozen simulations.

No model fitting occurs. Inputs are the packaged replication statistics.
All power calibration uses each method's matched empirical null distribution.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'output/plot-cache'))
import numpy as np
import pandas as pd
from scipy.stats import binomtest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DEFAULT = ROOT / 'data/lasso'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

METHODS = ['raw', 'raw_bootstrap', 'orthogonal', 'orthogonal_bootstrap']
STYLE = {
    'raw': ('Raw', '#B65035', 'o', '-'),
    'raw_bootstrap': ('Bootstrap Raw', '#B65035', '^', '--'),
    'orthogonal': ('Orthogonal', '#0072B2', 'o', '-'),
    'orthogonal_bootstrap': ('Bootstrap Orthogonal', '#0072B2', '^', '--'),
}
KEYS = ['family', 'n', 'T', 'complexity', 'tuning', 'schedule']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 11, 'axes.labelsize': 10, 'xtick.labelsize': 9,
    'ytick.labelsize': 9, 'legend.fontsize': 9, 'pdf.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#AEB7BB', 'axes.linewidth': .6,
    'axes.labelcolor': '#29343A', 'text.color': '#29343A',
    'xtick.color': '#4E5A61', 'ytick.color': '#4E5A61', 'savefig.facecolor': 'white'})


def wilson(p, n):
    p, n = np.asarray(p), np.asarray(n)
    n = np.where(n > 0, n, np.nan)
    z = 1.959963984540054
    den = 1 + z*z/n
    centre = (p + z*z/(2*n))/den
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return centre-half, centre+half


def calibrate(scores, alpha=.05):
    critical = float(np.quantile(scores, 1-alpha, method='inverted_cdf'))
    above, tied = np.mean(scores > critical), np.mean(scores == critical)
    gamma = float(np.clip((alpha-above)/tied, 0, 1))
    return critical, gamma


def rejection_weights(scores, critical, gamma):
    return (scores > critical).astype(float) + gamma * (scores == critical)


def summaries(root, out, draws):
    original = pd.read_csv(root / 'summary.csv', dtype={'complexity': str})
    original = original[original.family.eq('lasso')].copy()
    cfg = json.loads((root / 'configuration.json').read_text())
    assert not draws.duplicated(['cell_id', 'replication', 'method']).any()
    assert set(draws.method) == set(METHODS)
    for (cell_id, method), group in draws.groupby(['cell_id', 'method']):
        count = cfg['counts'][str(cell_id)]
        assert len(group) == count and set(group.replication) == set(range(1, count + 1))
    assert set(draws.cell_id) == {c['id'] for c in cfg['cells']}
    metadata = original.drop_duplicates('cell_id').set_index('cell_id')
    derived = []
    for (i, method), group in draws.groupby(['cell_id', 'method'], sort=True):
        valid = group[group.valid]
        saved = original[(original.cell_id == i) & original.method.eq(method)].iloc[0]
        assert len(group) == saved.completed and len(valid) == saved.valid
        if len(valid):
            assert np.isclose(valid.reject.astype(float).mean(), saved.rejection)
    summary = pd.concat([original, pd.DataFrame(derived)], ignore_index=True)
    assert not summary.duplicated(['cell_id', 'method']).any()
    summary = summary[summary.method.isin(METHODS)].copy()
    lo, hi = wilson(summary.rejection, summary.valid)
    summary['rejection_lower'], summary['rejection_upper'] = lo, hi
    summary.to_csv(out / 'summary.csv', index=False)
    return summary


def calibrate_power(summary, draws, out):
    metadata = summary.drop_duplicates('cell_id')
    nulls = {tuple(row[k] for k in KEYS): int(row['cell_id'])
             for _, row in metadata[metadata.delta.eq(0)].iterrows()}
    grouped = {(int(i), m): g.sort_values('replication')
               for (i, m), g in draws.groupby(['cell_id', 'method'])}
    rows = []
    for original in summary.to_dict('records'):
        i, method = original['cell_id'], original['method']
        null_id = nulls[tuple(original[k] for k in KEYS)]
        base, target = grouped[null_id, method], grouped[i, method]
        field, sign = ('p_value', -1) if method.endswith('_bootstrap') else ('abs_t', 1)
        b = sign * base.loc[base.valid, field].to_numpy(float)
        a = sign * target.loc[target.valid, field].to_numpy(float)
        row = {k: original[k] for k in ['cell_id'] + KEYS + ['ratio', 'refits', 'delta', 'groups', 'method']}
        row.update(null_cell_id=null_id, null_valid=len(b), alternative_valid=len(a),
            nominal_rejection=original['rejection'], adjusted_power=np.nan,
            critical=np.nan, boundary_weight=np.nan, conditional_mcse=np.nan)
        if len(a) and len(b):
            assert np.isfinite(a).all() and np.isfinite(b).all()
            critical, gamma = calibrate(b)
            w = rejection_weights(a, critical, gamma)
            assert np.isclose(rejection_weights(b, critical, gamma).mean(), .05)
            row.update(adjusted_power=w.mean(), critical=critical, boundary_weight=gamma,
                       conditional_mcse=w.std(ddof=1)/np.sqrt(len(w)) if len(w)>1 else np.nan)
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'adjusted_power.csv', index=False)
    return frame, grouped


def paired_uncertainty(summary, adjusted, grouped, out, bootstraps=1000):
    """Paired MC bootstrap; re-estimate null cutoffs, preserving shared seeds.

    Alternatives use draws 1:500 and nulls 1:1000. Resample the shared first
    stratum jointly across methods/null/alternative, and the extra null stratum
    separately. This is an uncertainty calculation for Monte Carlo power, not
    the score bootstrap test compared in the experiment.
    """
    power_rows, intervals, size_rows = [], [], []
    meta = summary.drop_duplicates('cell_id').set_index('cell_id')
    for i, row in meta.iterrows():
        if row.tuning != 'cv':
            continue
        if row.delta == 0:
            for m1, m0 in [('orthogonal', 'raw'), ('orthogonal', 'raw_bootstrap'),
                           ('orthogonal_bootstrap', 'raw_bootstrap')]:
                a, b = grouped[i, m1], grouped[i, m0]
                pair = a[['replication', 'valid', 'reject']].merge(
                    b[['replication', 'valid', 'reject']], on='replication', suffixes=('_a', '_b'))
                pair = pair[pair.valid_a & pair.valid_b]
                d = pair.reject_a.astype(float).to_numpy()-pair.reject_b.astype(float).to_numpy()
                discordant = np.count_nonzero(d)
                mcse = d.std(ddof=1)/np.sqrt(len(d))
                size_rows.append(dict(cell_id=i, n=row.n, T=row['T'], complexity=row.complexity,
                    schedule=row.schedule, method=m1, reference=m0, paired_valid=len(d),
                    difference=d.mean(), mcse=mcse, lower=d.mean()-1.96*mcse,
                    upper=d.mean()+1.96*mcse,
                    exact_mcnemar_p=binomtest(np.sum(d>0), discordant).pvalue if discordant else 1.))
            continue
        rows = adjusted[(adjusted.cell_id == i) & adjusted.method.isin(METHODS)].set_index('method').loc[METHODS]
        null_id = int(rows.null_cell_id.iloc[0])
        assert rows.null_cell_id.nunique() == 1
        a, b = [], []
        for method in METHODS:
            ga, gb = grouped[i, method], grouped[null_id, method]
            assert ga.valid.all() and gb.valid.all()
            assert ga.replication.tolist() == list(range(1, len(ga)+1))
            assert gb.replication.tolist() == list(range(1, len(gb)+1))
            field, sign = ('p_value', -1) if method.endswith('_bootstrap') else ('abs_t', 1)
            a.append(sign*ga[field].to_numpy(float)); b.append(sign*gb[field].to_numpy(float))
        a, b = np.column_stack(a), np.column_stack(b)
        n1, n0 = len(a), len(b)
        assert n1 <= n0
        rng = np.random.default_rng(np.random.SeedSequence([20261110, int(i), 811]))
        sampled = []
        for start in range(0, bootstraps, 100):
            count = min(100, bootstraps-start)
            shared = rng.integers(n1, size=(count, n1))
            extra = rng.integers(n1, n0, size=(count, n0-n1)) if n1 < n0 else np.empty((count, 0), int)
            bb, aa = b[np.concatenate([shared, extra], axis=1)], a[shared]
            critical = np.quantile(bb, .95, axis=1, method='inverted_cdf')
            above = (bb > critical[:, None, :]).mean(axis=1)
            tied = (bb == critical[:, None, :]).mean(axis=1)
            gamma = np.clip((.05-above)/tied, 0, 1)
            weights = (aa > critical[:, None, :]) + gamma[:, None, :]*(aa == critical[:, None, :])
            sampled.append(weights.mean(axis=1))
        sampled = np.concatenate(sampled)
        values = rows.adjusted_power.to_numpy()
        for j, method in enumerate(METHODS):
            lo, hi = np.quantile(sampled[:, j], [.025, .975])
            intervals.append(dict(cell_id=i, method=method, adjusted_lower=lo, adjusted_upper=hi,
                                  adjusted_mcse=sampled[:, j].std(ddof=1)))
        for m1, m0 in [('orthogonal', 'raw'), ('orthogonal', 'raw_bootstrap'),
                       ('orthogonal_bootstrap', 'raw_bootstrap')]:
            j, k = METHODS.index(m1), METHODS.index(m0)
            dd = sampled[:, j]-sampled[:, k]
            lo, hi = np.quantile(dd, [.025, .975])
            power_rows.append(dict(cell_id=i, null_cell_id=null_id, n=row.n, T=row['T'],
                complexity=row.complexity, schedule=row.schedule, delta=row.delta,
                method=m1, reference=m0, difference=values[j]-values[k], lower=lo, upper=hi,
                mcse=dd.std(ddof=1), resamples=bootstraps))
    pd.DataFrame(power_rows).to_csv(out / 'power_comparisons.csv', index=False)
    pd.DataFrame(size_rows).to_csv(out / 'size_comparisons.csv', index=False)
    adjusted = adjusted.merge(pd.DataFrame(intervals), on=['cell_id', 'method'], how='left')
    adjusted.to_csv(out / 'adjusted_power.csv', index=False)
    return adjusted


def setup(ax, ylim, ticks=None, target=True):
    ax.set_ylim(*ylim)
    if ticks is not None:
        ax.set_yticks(ticks)
    ax.grid(axis='y', color='#E5E9EB', linewidth=.6, zorder=0)
    ax.set_axisbelow(True)
    if target:
        ax.axhline(5, color='#89969B', linewidth=.85, dashes=(2, 3), zorder=1)
    ax.tick_params(length=3, width=.6)


def legend(fig, methods=METHODS, y=1., ncol=2):
    if methods == METHODS:
        methods = [METHODS[j] for j in [0, 2, 1, 3]]
    handles = [Line2D([0], [0], label=STYLE[m][0], color=STYLE[m][1],
               marker=STYLE[m][2], linestyle=STYLE[m][3], linewidth=1.4, markersize=4)
               for m in methods]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.52, y),
               ncol=ncol, frameon=False, columnspacing=1.2, handlelength=1.8, handletextpad=.5)


def series(ax, g, method, positions, offset=0., bars=True, column='rejection'):
    _, color, marker, line = STYLE[method]
    assert len(g) == len(positions)
    kw = dict(color=color, marker=marker, linestyle=line, linewidth=1.3,
              markersize=4, markeredgewidth=.7, zorder=3)
    if bars:
        if column == 'adjusted_power':
            lo, hi = g.adjusted_lower.to_numpy(), g.adjusted_upper.to_numpy()
        else:
            lo, hi = wilson(g[column].to_numpy(), g.valid.to_numpy())
        p = g[column].to_numpy()
        errors = 100*np.array([np.maximum(p-lo, 0), np.maximum(hi-p, 0)])
        ax.errorbar(np.asarray(positions)+offset, 100*p, yerr=errors,
                    capsize=1.7, elinewidth=.65, **kw)
    else:
        ax.plot(np.asarray(positions)+offset, 100*g[column].to_numpy(), **kw)


def save(fig, name, rows, figures, out):
    fig.savefig(figures / f'{name}.pdf', bbox_inches='tight', pad_inches=.04,
                metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(out / f'{name}.png', dpi=180, bbox_inches='tight', pad_inches=.04)
    rows.to_csv(out / f'{name}.csv', index=False)
    plt.close(fig)


def size_grid(df, figures, out):
    g = df[(df.tuning=='cv') & (df.n==1000) & (df.schedule=='infrequent') &
           (df.delta==0) & df.ratio.isin([.5, 1, 2])].copy()
    g['p'] = g.complexity.astype(int)
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.1), sharex=True)
    for j, ratio in enumerate([.5, 1, 2]):
        axes[0, j].set_title(r'$T/n='+f'{ratio:g}'+r'$')
        for i in range(2):
            ax = axes[i, j]
            mm = METHODS if i == 0 else ['orthogonal', 'orthogonal_bootstrap']
            setup(ax, (0, 90) if i==0 else (0, 18), [0, 30, 60, 90] if i==0 else [0, 5, 10, 15])
            for k, method in enumerate(mm):
                q = g[(g.ratio==ratio) & (g.method==method)].sort_values('p')
                series(ax, q, method, [0, 1, 2], offset=(k-(len(mm)-1)/2)*.07)
            ax.set_xticks([0, 1, 2], ['32', '128', '256'])
            ax.set_xlim(-.25, 2.25)
            if i == 1:
                ax.set_xlabel(r'Dictionary size $p$')
            if j:
                ax.tick_params(labelleft=False)
    axes[0, 0].set_ylabel('Rejection (%)')
    axes[1, 0].set_ylabel('Enlargement (%)')
    legend(fig, y=1.015)
    fig.subplots_adjust(left=.09, right=.99, bottom=.12, top=.81, wspace=.15, hspace=.2)
    save(fig, 'lasso_size', g[g.method.isin(METHODS)], figures, out)


def refitting(df, figures, out):
    g = df[(df.tuning=='cv') & (df.ratio==1) & (df.complexity=='32') & (df.delta==0)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.85), sharey=True)
    schedules = ['fixed', 'infrequent', 'frequent', 'daily']
    for ax, n in zip(axes, [500, 1000]):
        setup(ax, (0, 48), [0, 5, 15, 25, 35, 45])
        ax.set_title(r'$n=T='+str(n)+'$')
        ss = schedules if n==500 else schedules[:-1]
        for k, method in enumerate(METHODS):
            q = g[(g.n==n) & (g.method==method)].set_index('schedule').loc[ss]
            series(ax, q, method, np.arange(len(ss)), offset=(k-(len(METHODS)-1)/2)*.045)
        ax.set_xticks(np.arange(len(ss)), ['Once', 'Five', 'Every 25', 'Every date*'][:len(ss)])
        ax.set_xlabel('Refitting schedule')
        ax.set_xlim(-.25, len(ss)-.75)
    axes[0].set_ylabel('Null rejection (%)')
    legend(fig, y=1.04)
    fig.subplots_adjust(left=.08, right=.99, bottom=.18, top=.73, wspace=.15)
    save(fig, 'lasso_refitting', g[g.method.isin(METHODS)], figures, out)


def diagnostics(df, figures, out):
    g = df[(df.tuning=='cv') & (df.n==1000) & (df.schedule=='infrequent') &
           (df.delta==0) & df.ratio.isin([.5, 1, 2])].copy()
    g['p'] = g.complexity.astype(int)
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.25), sharex=True)
    for j, ratio in enumerate([.5, 1, 2]):
        ax = axes[0, j]
        setup(ax, (0, 4.1), [0, 1, 2, 3, 4], target=False)
        ax.set_title(r'$T/n='+f'{ratio:g}'+'$')
        q = g[(g.ratio==ratio) & (g.method=='orthogonal')].sort_values('p')
        for method, column in [('raw', 'mean_scaled_raw_drift'), ('orthogonal', 'mean_scaled_corrected_drift')]:
            _, color, marker, line = STYLE[method]
            ax.plot([0, 1, 2], q[column], color=color, marker=marker, linestyle=line, linewidth=1.3, markersize=4)
        ax = axes[1, j]
        setup(ax, (0, .40), [0, .1, .2, .3, .4], target=False)
        for column, label, color, marker in [
                ('mean_max_forecast_error', 'Forecast', '#44545D', 's'),
                ('mean_max_representer_error', 'Correction', '#55846A', '^')]:
            ax.plot([0, 1, 2], q[column], label=label, color=color,
                    marker=marker, linewidth=1.3, markersize=4)
        ax.set_xticks([0, 1, 2], ['32', '128', '256'])
        ax.set_xlabel(r'Dictionary size $p$')
        if j:
            axes[0, j].tick_params(labelleft=False)
            ax.tick_params(labelleft=False)
    axes[0, 0].set_ylabel('Mean scaled drift')
    axes[1, 0].set_ylabel(r'Mean maximum $L^2$ error')
    legend(fig, ['raw', 'orthogonal'], y=1.01, ncol=2)
    axes[1, 1].legend(frameon=False, fontsize=8, loc='upper left')
    fig.subplots_adjust(left=.09, right=.99, bottom=.11, top=.81, wspace=.15, hspace=.22)
    save(fig, 'lasso_diagnostics', g, figures, out)


def power(adjusted, figures, out):
    g = adjusted[(adjusted.tuning=='cv') & (adjusted.n==1000) & (adjusted.ratio==1) &
        (adjusted.schedule=='infrequent') & (adjusted.complexity=='32') & adjusted.method.isin(METHODS)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.0))
    for ax, column, title, limit in zip(axes, ['nominal_rejection', 'adjusted_power'],
            ['Nominal critical values', 'Size-adjusted critical values'], [(0, 60), (0, 22)]):
        setup(ax, limit, [0, 20, 40, 60] if limit[1]==60 else [0, 5, 10, 15, 20])
        ax.set_title(title)
        for j, method in enumerate(METHODS):
            q = g[g.method==method].sort_values('delta').copy()
            assert q.delta.tolist()==[-.7, -.35, 0., .35, .7]
            series(ax, q, method, q.delta.to_numpy(), offset=(j-(len(METHODS)-1)/2)*.008, bars=False, column=column)
        ax.set_xticks([-.7, -.35, 0, .35, .7], ['−0.70', '−0.35', '0', '0.35', '0.70'])
        ax.set_xlabel(r'Local effect $\delta$')
        ax.set_ylabel('Rejection (%)')
        ax.set_xlim(-.76, .76)
    legend(fig, y=1.03)
    fig.subplots_adjust(left=.08, right=.99, bottom=.18, top=.72, wspace=.24)
    save(fig, 'lasso_power', g, figures, out)
    g = adjusted[(adjusted.tuning=='cv') & (adjusted.n==1000) & (adjusted.schedule=='infrequent') &
        (adjusted.complexity=='256') & adjusted.method.isin(METHODS)].copy()
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.3), sharex=True)
    for j, ratio in enumerate([.5, 1, 2]):
        axes[0, j].set_title(r'$T/n='+f'{ratio:g}'+'$')
        for i, column in enumerate(['nominal_rejection', 'adjusted_power']):
            ax = axes[i, j]
            setup(ax, (0, 100) if i==0 else (0, 25), [0, 25, 50, 75, 100] if i==0 else [0, 5, 10, 15, 20, 25])
            for k, method in enumerate(METHODS):
                q = g[(g.ratio==ratio) & (g.method==method)].sort_values('delta')
                assert q.delta.tolist()==[-.7, 0., .7]
                series(ax, q, method, q.delta.to_numpy(), offset=(k-(len(METHODS)-1)/2)*.01, bars=False, column=column)
            ax.set_xticks([-.7, 0, .7], ['−0.7', '0', '0.7'])
            if i==1:
                ax.set_xlabel(r'Local effect $\delta$')
            if j:
                ax.tick_params(labelleft=False)
    axes[0, 0].set_ylabel('Nominal rejection (%)')
    axes[1, 0].set_ylabel('Size-adjusted rejection (%)')
    legend(fig, y=1.01)
    fig.subplots_adjust(left=.09, right=.99, bottom=.11, top=.81, wspace=.15, hspace=.22)
    save(fig, 'lasso_power_complexity', g, figures, out)


def supporting_tables(summary, adjusted, out):
    tables = out / 'tables'
    tables.mkdir(exist_ok=True)
    def tabular(path, alignment, header, rows):
        text = '\\begin{tabular}{'+alignment+'}\n\\toprule\n'
        text += header+' \\\\\n\\midrule\n'
        text += '\n'.join(' & '.join(row)+' \\\\' for row in rows)
        text += '\n\\bottomrule\n\\end{tabular}\n'
        path.write_text(text)

    columns = ['cell_id', 'n', 'T', 'ratio', 'complexity', 'tuning', 'schedule', 'method',
               'valid', 'completed', 'rejection', 'rejection_lower', 'rejection_upper']
    g = summary[(summary.delta==0) & summary.method.isin(METHODS)]
    g[columns].to_csv(out / 'size_all.csv', index=False)
    selected = g[(g.n==1000) & (g.schedule=='infrequent') & (g.ratio.isin([.5, 1, 2])) &
        (g.complexity.isin(['32', '256'])) & (g.tuning.isin(['cv', 'aic']))]
    selected[columns].to_csv(out / 'size_tuning.csv', index=False)
    compact = selected.pivot(index=['ratio', 'complexity', 'tuning'], columns='method', values='rejection')[METHODS]*100
    compact = compact.reindex(pd.MultiIndex.from_product([[.5, 1, 2], ['32', '256'], ['cv', 'aic']],
        names=['ratio', 'complexity', 'tuning']))
    compact.to_csv(out / 'size_tuning_wide.csv')
    tuning_rows = [[f'{ratio:g}', p, tuning.upper(), *[f'{v:.1f}' for v in values]]
                  for (ratio, p, tuning), values in compact.iterrows()]
    tabular(tables/'lasso_size_tuning.tex', 'rrlrrrr',
        r'$T/n$ & $p$ & Tuning & Raw & Boot. Raw & Orth. & Boot. Orth.', tuning_rows)
    size = g[(g.n==1000) & (g.schedule=='infrequent') & (g.tuning=='cv') & g.ratio.isin([.5, 1, 2])]
    size = size.pivot(index=['ratio', 'complexity'], columns='method', values='rejection')[METHODS]*100
    size = size.reset_index().assign(p=lambda q: q.complexity.astype(int)).sort_values(['ratio', 'p'])
    size_rows = [[f'{row.ratio:g}', str(row.p), *[f'{getattr(row, m):.1f}' for m in METHODS]]
                 for row in size.itertuples()]
    tabular(tables/'lasso_size.tex', 'rrrrrr',
        r'$T/n$ & $p$ & Raw & Boot. Raw & Orth. & Boot. Orth.', size_rows)
    learning = summary[(summary.n==1000) & (summary.schedule=='infrequent') &
        (summary.tuning=='cv') & (summary.delta==0) & summary.ratio.isin([.5, 1, 2])]
    learning_rows = []
    for ratio in [.5, 1, 2]:
        for p in ['32', '128', '256']:
            block = learning[(learning.ratio==ratio) & (learning.complexity==p)].set_index('method')
            q = block.loc['orthogonal']
            learning_rows.append([f'{ratio:g}', p, f'{q.mean_max_forecast_error:.3f}',
                f'{q.mean_max_representer_error:.3f}', f'{q.mean_scaled_raw_drift:.3f}',
                f'{q.mean_scaled_corrected_drift:.3f}', f'{100*block.loc["raw", "coverage"]:.1f}',
                f'{100*q.coverage:.1f}'])
    tabular(tables/'lasso_learning.tex', 'rrrrrrrr',
        r'$T/n$ & $p$ & Forecast & Repr. & $D_{\rm raw}$ & $D_{\rm orth}$ & Raw cov. & Orth. cov.', learning_rows)
    # Show unconditional MC uncertainty for adjusted power in a compact table.
    central = adjusted[(adjusted.tuning=='cv') & (adjusted.n==1000) & (adjusted.ratio==1) &
        (adjusted.schedule=='infrequent') & (adjusted.complexity=='32') & adjusted.method.isin(METHODS)]
    central.to_csv(out / 'power_central.csv', index=False)


def main(results=DEFAULT, figures=ROOT / 'output/figures'):
    root, figures = Path(results), Path(figures)
    out = ROOT / 'output/lasso'
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    draws = pd.read_csv(root / 'draw_statistics.csv.gz', dtype={'valid': bool, 'reject': 'boolean', 'covered': 'boolean'})
    summary = summaries(root, out, draws)
    adjusted, grouped = calibrate_power(summary, draws, out)
    adjusted = paired_uncertainty(summary, adjusted, grouped, out)
    size_grid(summary, figures, out)
    refitting(summary, figures, out)
    diagnostics(summary, figures, out)
    power(adjusted, figures, out)
    supporting_tables(summary, adjusted, out)
    manifest = dict(inputs={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__),
        root/'summary.csv', root/'configuration.json', root/'draw_statistics.csv.gz']},
        figures=['lasso_size', 'lasso_refitting', 'lasso_diagnostics', 'lasso_power', 'lasso_power_complexity'],
        size_intervals='Pointwise Wilson 95% Monte Carlo intervals, conditional on validity.',
        power_intervals='CSV pointwise paired 1000-resample percentile intervals; null cutoffs reestimated; shared null/alternative replication indices resampled jointly. Power curves omit bars to keep four methods readable.',
        power_calibration='Matched empirical 95th percentile; fractional boundary rejection integrates ties to exactly 5% empirical null rejection.',
        invalid='Counts retained for every cell. No invalid draw is silently omitted from reported valid fractions.',
        source='Completed lasso draws only; no old neural observations or refits.')
    generated = [*sorted((out/'tables').glob('*.tex')),
                 *[figures/f'{name}.pdf' for name in manifest['figures']],
                 out/'summary.csv', out/'adjusted_power.csv', out/'power_comparisons.csv',
                 out/'size_comparisons.csv']
    manifest['outputs'] = {str(p.relative_to(ROOT)): sha(p) for p in generated}
    (out/'presentation_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Wrote four-method lasso figures and derived tables: {out}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', type=Path, default=DEFAULT)
    ap.add_argument('--figures', type=Path, default=ROOT/'output/figures')
    args = ap.parse_args()
    main(args.results, args.figures)
