# -*- coding: utf-8 -*-
"""
plot_policies.py — Policy function visualizations
=================================================

plot_policy_snapshots : 2×2 grid of savings rates by age and health
plot_policy           : Savings rate, consumption, and value function for a specific state
"""

import numpy as np
import matplotlib.pyplot as plt
from analysis.aggregation import get_savings_rate
from analysis.plot_health import HEALTH_COLORS, HEALTH_LABELS

def plot_policy_snapshots(diag,
                          ages_to_plot=(10, 25, 40, 55),
                          h_indices=None,
                          starting_age=20,
                          xlim=None,
                          figsize=(13, 10)):
    """
    2×2 grid: each panel pins one age, hue = health state.
    p and m averaged via ergodic distribution from diagnostics.
    Y-axis: savings rate s'/x.
    X-axis: cah_grid, optionally truncated by xlim (in model units).

    Parameters
    ----------
    xlim : float or None
        Upper bound on cah_grid to display (model units).
        e.g. xlim=5.0 shows only the bottom of the CAH distribution.
        None shows the full grid.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize,
                             sharex=False, sharey=False)
    axes = axes.flatten()
    cah  = diag.cah_grid

    # Determine x mask once — same for all panels
    if xlim is not None:
        x_mask = cah <= xlim
    else:
        x_mask = np.ones(len(cah), dtype=bool)
    cah_plot = cah[x_mask]

    if h_indices is None:
        n = diag.n_h_alive
        h_indices = [0, n // 2, n - 1] 

    for panel, age_idx in enumerate(ages_to_plot):
        ax       = axes[panel]
        real_age = starting_age + age_idx

        # Savings rate averaged over (p, m) using ergodic weights
        sr_raw  = get_savings_rate(diag.pol.savings[age_idx], cah)
        # aggregate_pm uses invariant distributions as in diagnostics
        from analysis.aggregation import aggregate_pm
        sr_agg  = aggregate_pm(sr_raw, diag.inv_p, diag.inv_m)
        # shape (n_cah, n_h) — apply x mask
        sr_plot = sr_agg[x_mask, :]

        for h in h_indices:
            ax.plot(cah_plot,
                    sr_plot[:, h],
                    color=HEALTH_COLORS[h % len(HEALTH_COLORS)],
                    label=HEALTH_LABELS[h] if panel == 0 else None,
                    linewidth=2.0)

        # Reference line: sr = 0 (neither saving nor dissaving)
        ax.axhline(0, color='black', linestyle='--',
                   linewidth=0.8, alpha=0.5)

        ax.set_title(f'Age {real_age}')
        ax.set_xlabel('Cash-at-hand $x$')
        ax.set_ylabel("Savings rate $s'/x$")
        ax.set_ylim(-0.1, 1.05)
        if xlim is not None:
            ax.set_xlim(0, xlim)
        ax.grid(alpha=0.25)

    axes[0].legend(fontsize=8, framealpha=0.9)
    fig.suptitle(
        r"Savings rate by health state (ergodic avg over $p$, $m$)",
        fontsize=12
    )
    plt.tight_layout()

    return fig


def plot_policy(diag,
                diag2=None,
                label1="Baseline",
                label2="Comparison",
                age_idx=60,
                h_idx=None,
                p_idx=0,
                m_idx=None,
                starting_age=20,
                xlim=None,
                show_value=True,
                figsize=None):
    """
    

    Parameters
    ----------
    diag : TYPE
        diagnostics class with ergodic distribution and policy.
    diag2 : TYPE, optional
        second diagnostics class with ergodic distribution and policy. The default is None.
    label1 : TYPE, optional
        DESCRIPTION for diag. The default is "Baseline".
    label2 : TYPE, optional
        DESCRIPTION for diag2. The default is "Comparison".
    age_idx : TYPE, optional
        Model age idnex. Possible range [0, 89]. The default is 60.
    h_worst : TYPE, optional
        Health states of agents. Range from [0,5] where 5 is death and in NHH only [0,1] The default is None.
    p_worst : TYPE, optional
        Productivity state of agent. The default is 0.
    m_worst : TYPE, optional
        MEdical state. The default is None.
    starting_age : TYPE, optional
        Age to shift the final plot with. The default is 20 to match to demographics.
    xlim : TYPE, optional
        Zoom in on axis to focus on important CAH areas depending on the stae space. The default is None.
    show_value : TYPE, optional
        False turns off value fucniton plot. The default is True.
    figsize : TYPE, optional
        Figure size. The default is None.

    Returns
    -------
    figure
        Savings Rate, Consumption and optionally Value Fucntion of policy given chosen state spaces.

    """
    cfg = diag.cfg
    cah = diag.cah_grid
    #baseline is worst state individual!
    # cannot has to be markov chain dependent thus not possible in the function call
    if h_idx is None: h_idx = diag.n_h_alive - 1
    if p_idx is None: p_idx = cfg.n_p - 1
    if m_idx is None: m_idx = cfg.n_m - 1

    real_age = starting_age + age_idx

    if xlim is not None:
        x_mask = cah <= xlim
    else:
        x_mask = np.ones(len(cah), dtype=bool)
    cah_plot = cah[x_mask]

    def _slice(d, kind):
        arr = getattr(d.pol, kind)[age_idx]
        return arr[:, h_idx, p_idx, m_idx][x_mask]

    def _sr(sav):
        clean_sav = np.maximum(0, sav)
        return np.where(cah_plot > 1e-8, clean_sav / cah_plot, 0.0)

    marker_style = dict(marker='o', markersize=4, markerfacecolor='white',
                        markeredgewidth=1, markevery=1)

    series = [
        (_slice(diag, 'savings'), _slice(diag, 'consumption'), _slice(diag, 'value'),
         dict(color='steelblue', linewidth=2, label=label1, **marker_style))
    ]
    if diag2 is not None:
        series.append(
            (_slice(diag2, 'savings'), _slice(diag2, 'consumption'), _slice(diag2, 'value'),
             dict(color='darkorange', linewidth=1.8, linestyle='--', label=label2, **marker_style))
        )

    # Layout: 1x3 with value function, or 1x2 without
    n_cols   = 3 if show_value else 2
    figsize  = figsize or (13, 5) if show_value else (9, 5)
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)

    for sav, con, val, kw in series:
        axes[0].plot(cah_plot, _sr(sav), **kw)
        axes[1].plot(cah_plot, con,      **kw)
        if show_value:
            axes[2].plot(cah_plot, val,  **kw)

    axes[0].set_ylim(-0.05, 1.05)
    axes[0].axhline(0, color='black', linewidth=0.8, linestyle=':', alpha=0.5)
    axes[0].set(xlabel='Cash-at-hand $x$', ylabel="$s'/x$", title='Savings Rate')

    if cfg.c_floor is not None:
        axes[1].axhline(cfg.c_floor, color='red', linestyle='--',
                        linewidth=0.8, label='Floor')
    axes[1].plot(cah_plot, cah_plot, color='black', linestyle=':',
                 linewidth=0.8, alpha=0.4, label='$c=x$')
    axes[1].set(xlabel='Cash-at-hand $x$', ylabel='$c$', title='Consumption')
    axes[1].set_ylim(0, 0.5) # will be half of savings rate! better visualization

    if show_value:
        axes[2].set(xlabel='Cash-at-hand $x$', ylabel='$V(x)$', title='Value Function')

    for ax in axes:
        ax.grid(alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if xlim is not None:
            ax.set_xlim(0, xlim)

    if diag2 is not None or cfg.c_floor is not None:
        axes[1].legend(fontsize=8, framealpha=0.8)

    if h_idx < len(HEALTH_LABELS):
        h_lbl = HEALTH_LABELS[h_idx]
    else:
        h_lbl = f"Health State {h_idx}"

    plt.tight_layout()
    return fig