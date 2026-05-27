# -*- coding: utf-8 -*-
"""
plot_wealth.py — Wealth and lifecycle profile visualizations
============================================================

plot_median_wealth     : Median assets by age and health state
plot_lifecycle_profiles: Income (working/pension) and medical expenses panels
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from analysis.plot_health import HEALTH_COLORS, HEALTH_LABELS
import matplotlib.ticker as ticker


def plot_median_wealth(diags_dict, usd_scaling=1.0, health_indices=[0, 2, 4], 
                                  starting_age=20, figsize=(14, 6), ymax=None):
    """
    Side-by-side panels (1x2), one per economy.
    Smartly handles labeling and Foltyn-style aesthetics.
    """
    n = len(diags_dict)
    # Smart layout: 1 row, N columns
    nrows = 1
    ncols = n
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
    
    if n == 1:
        axes = [axes]
    axes = np.array(axes).flatten()

    for idx, (title, diag) in enumerate(diags_dict.items()):
        ax = axes[idx]
        
        # Data extraction: (n_ages, n_h_alive)
        med = diag.get_median_assets(by_health=True) * usd_scaling  
        n_ages = med.shape[0]
        ages = np.arange(starting_age, starting_age + n_ages)

        lines_plotted = 0
        for h in health_indices:
            if h < med.shape[1]:
                ax.plot(ages, med[:, h],
                        color=HEALTH_COLORS[h],
                        label=HEALTH_LABELS[h],
                        linewidth=2.0)
                lines_plotted += 1
            
        # Foltyn Styling
        ax.set_title(title, fontsize=13, fontweight='bold', loc='left', pad=12)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # --- BRUTE FORCE THE BOX & TICKS ---
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.0)
            
        ax.tick_params(direction='in', length=5, top=True, right=True, bottom=True, left=True)
        # -----------------------------------

        
        # USD Axis Formatting
        if usd_scaling != 1.0:
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        if ymax:
            ax.set_ylim(-5, ymax)
        ax.set_xlim(20, 100)
        # X-axis label on both plots (since they are side-by-side)
        ax.set_xlabel('Age', fontsize=11)
        
        # Y-axis label ONLY on the leftmost plot
        if idx == 0:
            ylabel = 'Median Assets (USD)' if usd_scaling != 1.0 else 'Assets (Units)'
            ax.set_ylabel(ylabel, fontsize=11)
        
        # Legend logic: trigger only for heterogeneous cases
        if lines_plotted > 1:
            ax.legend(loc='upper left', fontsize=9, framealpha=0.8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig



def plot_lifecycle_profiles(model, usd_scaling = 1):
    """
    Analyzes and plots the deterministic lifecycle profiles in 3 panels:
    1. Working Income by Health State
    2. Retirement Income by Productivity State
    3. Medical Costs by Health State
    """
    # 1. Shock Probabilities
    stat_p = model.invariant_Q.p
    stat_m = model.invariant_Q.m
    prob_p_trans = model.trans.lab_eps.probs
    prob_m_trans = model.trans.med_eps.probs

    # ==========================================
    # 1. PROCESS WORKING INCOME (By Health)
    # ==========================================
    inc_work = model.resources.income_working
    N_work = inc_work.shape[0]
    n_h_alive = inc_work.shape[1]
    
    # Aggregate Transitory -> (N_work, H_alive, P_persist)
    inc_work_no_trans = np.tensordot(inc_work, prob_p_trans, axes=([-1], [0]))
    # Aggregate Persistent -> (N_work, H_alive)
    expected_inc_work = np.tensordot(inc_work_no_trans, stat_p, axes=([-1], [0]))

    # ==========================================
    # 2. PROCESS RETIRED INCOME (By Productivity)
    # ==========================================
    inc_ret = model.resources.income_retired # Shape: (N_ret, P_persist)
    N_ret = inc_ret.shape[0]
    n_p_states = inc_ret.shape[1]
    # No aggregation needed! We want to plot this directly for each productivity state.

    # ==========================================
    # 3. PROCESS MEDICAL EXPENSES (By Health)
    # ==========================================
    med_costs = model.resources.med_costs # Shape: (Total_Ages, H_alive, M_persist, M_trans)
    Total_Ages = med_costs.shape[0]
    
    # Aggregate Transitory -> (Total_Ages, H_alive, M_persist)
    med_costs_no_trans = np.tensordot(med_costs, prob_m_trans, axes=([-1], [0]))
    # Aggregate Persistent -> (Total_Ages, H_alive)
    expected_med_costs = np.tensordot(med_costs_no_trans, stat_m, axes=([-1], [0]))

    # ==========================================
    # AGE GRIDS
    # ==========================================
    # We dynamically create the age axes based on the array lengths
    age_work = np.arange(20, 20 + N_work)
    age_ret = np.arange(20 + N_work, 20 + N_work + N_ret)
    age_all = np.arange(20, 20 + Total_Ages)

    # ==========================================
    # PLOTTING
    # ==========================================
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    colors_h = plt.cm.viridis(np.linspace(0, 0.9, n_h_alive))
    colors_p = plt.cm.plasma(np.linspace(0, 0.9, n_p_states))

    # --- Panel 1: Working Income ---
    for h in range(n_h_alive):
        ax1.plot(age_work, usd_scaling * expected_inc_work[:, h], color=colors_h[h], lw=2, label=f'Health {h}')
    
    ax1.set_title("Working Income by Health")
    ax1.set_xlabel("Age")
    ax1.set_ylabel("Income")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # --- Panel 2: Retirement Income ---
    for p in range(n_p_states):
        ax2.plot(age_ret, usd_scaling * inc_ret[:, p], color=colors_p[p], lw=2, label=f'Productivity {p}')
        
    ax2.set_title("Pension Income by Productivity")
    ax2.set_xlabel("Age")
    ax2.set_ylabel("Income")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # --- Panel 3: Medical Costs ---
    for h in range(n_h_alive):
        ax3.plot(age_all, usd_scaling * expected_med_costs[:, h], color=colors_h[h], lw=2, label=f'Health {h}')
        
    ax3.set_title("Medical Expenses by Health")
    ax3.set_xlabel("Age")
    ax3.set_ylabel("Medical Costs")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    return fig