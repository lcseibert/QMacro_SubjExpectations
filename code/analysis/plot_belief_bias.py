# -*- coding: utf-8 -*-
"""
plot_belief_bias.py — Savings rate decomposition by belief economy
==================================================================

plot_savings_belief_bias : Bar chart decomposing savings rate differences
                           (SSH−OSH) and (OSH−NHH) by cash-on-hand decile

Internal
--------
_compute_cah_pmf : Build cash-on-hand distribution from previous-period mass
_binned_sr       : Compute weighted average savings rate within CAH quantile bins
"""


import numpy as np
import matplotlib.pyplot as plt
from analysis.aggregation import get_savings_rate

HEALTH_LABELS = ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor']
HEALTH_COLORS = ['#1b5e20', '#4caf50', '#fdd835', '#ff8f00', '#c62828']


def _compute_cah_pmf(diag, age_idx):
    """
    Reconstructs the cross-sectional cash-at-hand distribution at a given age.
    
    The forward-simulated distribution Gamma lives on the savings grid, but
    the savings-rate decomposition conditions on CAH deciles.
    To bridge the gap:
    
    1. Take Gamma_{t-1}(s, h, p, m) and apply Markov transitions to get
       the post-transition mass entering age t.
    2. For each (s, h', p', m') state, compute implied CAH by integrating
       over all transitory shock realizations:
           x = R*s + y(h', p', eps) - med(h', m', nu)
    3. Bin the resulting CAH values onto cah_grid via weighted bincount,
       producing a PMF over (cah_idx, h, p, m).
    
    Returns shape (n_cah, n_h, n_p, n_m)
    """
    if age_idx == 0:
        return None
    cfg   = diag.cfg
    res   = diag.core.resources
    trans = diag.core.trans
    R     = (1 + cfg.r)

    cah_grid = diag.cah_grid
    sav_grid = diag.sav_grid
    n_cah    = len(cah_grid)
    n_h      = diag.n_h_alive
    n_p      = cfg.n_p
    n_m      = cfg.n_m

    eps_p = trans.lab_eps.probs   # (n_eps_p,)
    eps_m = trans.med_eps.probs   # (n_eps_m,)

    # Transition matrices
    Pi_h       = trans.h_Q[age_idx - 1, :n_h, :n_h]
    Pi_p       = trans.lab_p.probs
    Pi_m       = trans.med_p.probs
    dist_prev  = diag.dist[age_idx - 1, :, :n_h, :, :]  # (n_sav, n_h, n_p, n_m)
    mass_trans = np.einsum('shpm,hH,pP,mM->sHPM',
                            dist_prev, Pi_h, Pi_p, Pi_m)

    # get income/expenditure combination for given age
    if age_idx < cfg.ret_age:
        inc = res.income_working[age_idx, :n_h]              # (n_h, n_p, n_eps_p)
    else:
        pen = res.income_retired[age_idx - cfg.ret_age]      # (n_p,)
        inc = pen[np.newaxis, :, np.newaxis] * np.ones((n_h, 1, 1))

    if age_idx >= cfg.medex_age:
        med = res.med_costs[age_idx - cfg.medex_age, :n_h]   # (n_h, n_m, n_eps_m)
    else:
        med = np.zeros((n_h, n_m, 1))

    # CAH for every (s, h, p, eps_p, m, eps_m) via broadcasting
    Rs    = (R * sav_grid)[:, None, None, None, None, None]   # (s,1,1,1,1,1)
    y_arr = inc[None, :, :, :, None, None]                    # (1,h,p,ep,1,1)
    m_arr = med[None, :, None, None, :, :]                    # (1,h,1,1,m,em)

    cah_arr = np.maximum(Rs + y_arr - m_arr, cfg.c_floor)
    # shape: (n_sav, n_h, n_p, n_eps_p, n_m, n_eps_m)

    # cah_grid indices — clip to valid range
    cah_idx = np.searchsorted(cah_grid, cah_arr).clip(0, n_cah - 1).astype(np.intp)

    # Step 4: joint mass weighted by transitory shock probabilities
    mass = (mass_trans[:, :, :, None, :, None]
            * eps_p[None, None, None, :, None, None]
            * eps_m[None, None, None, None, None, :])
    # shape: (n_sav, n_h, n_p, n_eps_p, n_m, n_eps_m)

    # Step 5: linear index encoding (cah, h, p, m) — eps dimensions already
    # contracted into mass weights, h/p/m positions encoded in flat index
    h_off = (np.arange(n_h) * (n_p * n_m))[None, :, None, None, None, None]
    p_off = (np.arange(n_p) * n_m        )[None, None, :, None, None, None]
    m_off =  np.arange(n_m)               [None, None, None, None, :, None]
    
    # Replace the final two lines of _compute_cah_pmf from lin= onward
    
    lin = (cah_idx * (n_h * n_p * n_m) + h_off + p_off + m_off).astype(np.intp)
    
    # Explicitly broadcast to identical shape before raveling —
    # numpy's lazy broadcasting means lin and mass may have different
    # in-memory shapes even if logically equal
    lin_b, mass_b = np.broadcast_arrays(lin, mass)
    
    pmf_flat = np.bincount(
        lin_b.ravel(),
        weights=mass_b.ravel(),
        minlength=n_cah * n_h * n_p * n_m
    )
    
    return pmf_flat.reshape(n_cah, n_h, n_p, n_m)


def _binned_sr(diag, age_idx, pmf_cah, h_idx, n_bins=10):
    """
    Marginalizes over ergodic distribution ot get ergodic CAH given age index.
    Computes percentile buckets. Default is 10 bins -> deciles.
    Returns (n_bins,).
    """
    n_cah = len(diag.cah_grid)

    # Marginal over (p, m) for bins of CAH
    mass_h = pmf_cah[:, h_idx, :, :].sum(axis=(1, 2))  # (n_cah,)
    cum    = np.cumsum(mass_h)
    total  = cum[-1]

    # Quantile edges as cah_grid indices
    edge = np.zeros(n_bins + 1, dtype=int)
    edge[-1] = n_cah
    for k in range(1, n_bins):
        edge[k] = int(np.searchsorted(cum, (k / n_bins) * total))

    # pol.savings[age]: (n_cah, n_h, n_p, n_m)
    sr = get_savings_rate(diag.pol.savings[age_idx], diag.cah_grid)
    # sr shape: (n_cah, n_h, n_p, n_m)

    out = np.zeros(n_bins)
    for k in range(n_bins):
        lo = int(edge[k])
        hi = int(edge[k + 1])
        if hi <= lo:
            continue
        # Both weight and policy slice: (bin_size, n_p, n_m)
        w    = pmf_cah[lo:hi, h_idx, :, :]   # joint pmf preserves (p,m)|cah correlation
        sr_k = sr[lo:hi,      h_idx, :, :]
        denom = w.sum()
        if denom < 1e-12:
            continue
        out[k] = (w * sr_k).sum() / denom

    return out


import matplotlib.patches as mpatches

def plot_savings_belief_bias(diag_ssh, diag_osh, diag_nhh,
                              ages_to_plot=(30, 40, 50, 60),
                              real_age_offset=20,
                              n_bins=10,
                              figsize=(14, 14)):
    """
    Reconstructing the savigns bias plot
    """
    n_panels = len(ages_to_plot)
    ncols    = 2
    nrows    = (n_panels + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
    axes = np.array(axes).flatten()

    n_h_osh = diag_osh.n_h_alive
    mid_h   = n_h_osh // 2

    show_health = [
        (0,           HEALTH_COLORS[0], 'Excellent'),
        (mid_h,       HEALTH_COLORS[2], 'Good'),
        (n_h_osh - 1, HEALTH_COLORS[4], 'Poor'),
    ]
    
    bar_w   = 0.12
    n_bars  = 2 * len(show_health)
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2.0) * bar_w
    x       = np.arange(1, n_bins + 1, dtype=float)

    for panel_idx, age_idx in enumerate(ages_to_plot):
        ax       = axes[panel_idx]
        real_age = real_age_offset + age_idx

        pmf_ssh = _compute_cah_pmf(diag_ssh, age_idx)
        pmf_osh = _compute_cah_pmf(diag_osh, age_idx)
        pmf_nhh = _compute_cah_pmf(diag_nhh, age_idx)

        bar_pos = 0
        for (h, color, h_label) in show_health:
            sr_osh = _binned_sr(diag_osh, age_idx, pmf_osh, h,   n_bins)
            sr_nhh = _binned_sr(diag_nhh, age_idx, pmf_nhh, 0,   n_bins)  
            sr_ssh = _binned_sr(diag_ssh, age_idx, pmf_ssh, h,   n_bins)

            diff_obj = sr_osh - sr_nhh  
            diff_sub = sr_ssh - sr_osh  
            
            # Removed labels from here to prevent cluttering the legend
            kw = dict(width=bar_w, edgecolor='black', linewidth=0.4)
            ax.bar(x + offsets[bar_pos],     diff_obj,
                   color=color, alpha=0.90, hatch='////', **kw)
            ax.bar(x + offsets[bar_pos + 1], diff_sub,
                   color=color, alpha=0.50,                **kw)
            bar_pos += 2

        ax.axhline(0, color='black', linewidth=1.0, linestyle='--')
        ax.set_title(f'Age {real_age}', fontsize=11, fontweight='bold')
        ax.set_xlabel(f'Cash-at-hand decile (age {real_age})', fontsize=9)
        ax.set_xticks(np.arange(1, n_bins + 1))
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(-0.4, 0.2)
        ax.set_yticks([-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2])
        if panel_idx % ncols == 0:
            ax.set_ylabel('Change in savings rate', fontsize=9)

    # --- CUSTOM GREY LEGEND
    # neutral grey patches that only show the hatch/alpha style
    osh_nhh_patch = mpatches.Patch(
        facecolor='grey', alpha=0.9, hatch='////', edgecolor='black', 
        label='OSH − NHH'
    )
    ssh_osh_patch = mpatches.Patch(
        facecolor='grey', alpha=0.5, edgecolor='black', 
        label='SSH − OSH'
    )

    # Apply legend only to the first panel
    axes[0].legend(
        handles=[osh_nhh_patch, ssh_osh_patch], 
        fontsize=9, framealpha=0.95, loc='upper right'
    )

    for k in range(n_panels, len(axes)):
        axes[k].set_visible(False)

    fig.suptitle('Differences in Savings Rates\n', fontsize=13, fontweight='bold')
    plt.tight_layout()

    return fig