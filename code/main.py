# -*- coding: utf-8 -*-
"""
main.py — Final Production Run
===============================
Generates all plots and tables for the term paper.

Outputs
-------
plots/lifecycle/
    lifecycle_profiles_py.pdf, ergodic_health_py.pdf
plots/wealth/
    median_wealth_{SSH,OSH,NHH}_py.pdf
plots/savings_bias/
    savings_belief_bias_py.pdf
plots/gss/
    gss_{lo,med,hi}_n{50,250,2000}_{discrete,hybrid}_py.pdf
plots/bequests/
    median_wealth_bequest_{baseline,luxury}_py.pdf
    policy_bequest_age{50,85}_{excellent,poor}_py.pdf
"""

# ===========================================================================
# 0. IMPORTS
# ===========================================================================
import time
from dataclasses import replace

from config import ModelSpecs, BeliefType
from src.primitives   import ModelPrimitives
from src.ge_solver    import GESolver
from src.hh_solver    import solve_lifecycle
from src.simulation   import run_forward_simulation

from analysis.diagnostics      import diagnostics
from analysis.plot_health      import plot_ergodic_health
from analysis.plot_wealth      import plot_median_wealth, plot_lifecycle_profiles
from analysis.plot_belief_bias import plot_savings_belief_bias
from analysis.plot_policies    import plot_policy

from plot_config import (
    apply_style, init_plot_dirs, save_fig, set_save_mode,
    PLOT_DIRS, USD, EQUILIBRIUM_R_RANGE,
)

# ===========================================================================
# 1. INITIALIZATION
# ===========================================================================
apply_style()
init_plot_dirs()

# True  = save plots as PDF to plots/ directories
# False = display plots interactively
set_save_mode(True)

# ===========================================================================
# 2. SOLVER FUNCTIONS
# ===========================================================================

def solve_ge_economy(cfg, r_range=EQUILIBRIUM_R_RANGE):
    """Solve for general equilibrium and return all outputs."""
    r_star, A_star, w_star = GESolver(ModelPrimitives(cfg), r_range).solve()
    cfg_eq = replace(cfg, r=r_star, w=w_star, A_anchor=A_star)
    core, pol, dist, diag = solve_pe_economy(cfg_eq)
    return cfg_eq, core, pol, dist, diag, r_star, A_star, w_star


def solve_pe_economy(cfg):
    """Solve partial equilibrium given fixed prices."""
    core = ModelPrimitives(cfg)
    pol  = solve_lifecycle(core)
    dist = run_forward_simulation(core, pol)
    diag = diagnostics(dist, pol, core)
    return core, pol, dist, diag


def display_ge_table(results: dict):
    """Print formatted GE results table."""
    headers = ['Economy', 'r*', 'w*', 'K/Y', 'Gini']
    widths  = [24, 8, 8, 8, 8]
    
    sep = '+' + '+'.join('-' * (w + 2) for w in widths) + '+'
    row = lambda cells: '|' + '|'.join(f' {c:<{w}} ' for c, w in zip(cells, widths)) + '|'
    
    print(sep)
    print(row(headers))
    print(sep)
    for label, res in results.items():
        r, w, diag = res['r'], res['w'], res['diag']
        print(row([
            label,
            f'{r:.2%}',
            f'{w:.4f}',
            f'{diag.compute_ky_ratio(r):.3f}',
            f'{diag.compute_gini_wealth():.3f}',
        ]))
    print(sep)


# ===========================================================================
# 3. SOLVE ECONOMIES
# ===========================================================================

# ---------------------------------------------------------------------------
# 3A. BASE ECONOMIES (no bequests)
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("  SECTION 1: BASE ECONOMIES")
print("="*60)

# SSH GE — baseline, sets TFP anchor
print("\n>>> SSH GE (baseline)")
cfg_ssh = ModelSpecs(
    belief_mode=BeliefType.subjective,
    is_anchored=False,
    phi_1=0,
    gss_stride=16,
)
cfg_ssh, core_ssh, pol_ssh, dist_ssh, diag_ssh, r_ssh, A_ssh, w_ssh = \
    solve_ge_economy(cfg_ssh)

# OSH GE — anchored to SSH
print("\n>>> OSH GE")
cfg_osh = replace(cfg_ssh,
    belief_mode=BeliefType.objective,
    is_anchored=True,
    A_anchor=A_ssh,
)
cfg_osh, core_osh, pol_osh, dist_osh, diag_osh, r_osh, A_osh, w_osh = \
    solve_ge_economy(cfg_osh)

# NHH GE — anchored to SSH
print("\n>>> NHH GE")
cfg_nhh = replace(cfg_ssh,
    belief_mode=BeliefType.average,
    is_anchored=True,
    A_anchor=A_ssh,
)
cfg_nhh, core_nhh, pol_nhh, dist_nhh, diag_nhh, r_nhh, A_nhh, w_nhh = \
    solve_ge_economy(cfg_nhh)

# ---------------------------------------------------------------------------
# 3B. BEQUEST ECONOMIES
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("  SECTION 2: BEQUEST ECONOMIES")
print("="*60)

BEQUEST_BASE = dict(phi_1=11.127, phi_2=0.001, beta=0.96)
BEQUEST_LUX  = dict(phi_1=11.127, phi_2=0.1,   beta=0.96)

# SSH + Bequest (baseline φ₂)
print("\n>>> SSH + Bequest (baseline)")
cfg_beq_base = replace(cfg_ssh, is_anchored=False, **BEQUEST_BASE)
cfg_beq_base, core_beq_base, pol_beq_base, dist_beq_base, diag_beq_base, \
    r_beq_base, A_beq_base, w_beq_base = solve_ge_economy(cfg_beq_base)

# SSH + Bequest (luxury φ₂) — anchored to baseline bequest
print("\n>>> SSH + Bequest (luxury)")
cfg_beq_lux = replace(cfg_ssh, is_anchored=True, A_anchor=A_beq_base, **BEQUEST_LUX)
cfg_beq_lux, core_beq_lux, pol_beq_lux, dist_beq_lux, diag_beq_lux, \
    r_beq_lux, A_beq_lux, w_beq_lux = solve_ge_economy(cfg_beq_lux)

# ---------------------------------------------------------------------------
# 3C. ROBUSTNESS / GSS VARIANTS
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("  SECTION 3: ROBUSTNESS / GSS")
print("="*60)

N_LO, N_MED, N_HI = 50, 250, 2000

variants = [
    ("Low (50) Discrete",       dict(n_assets=N_LO,  gss_stride=0)),
    ("Medium (250) Discrete",   dict(n_assets=N_MED, gss_stride=0)),
    ("High (2000) Discrete",    dict(n_assets=N_HI,  gss_stride=0)),
    ("Medium (250) Hybrid GSS", dict(n_assets=N_MED, gss_stride=12)),
]

runtimes, rob_diags = {}, {}
for label, overrides in variants:
    print(f"\n>>> {label}")
    t0 = time.time()
    cfg = replace(cfg_osh, **overrides)
    _, _, _, diag = solve_pe_economy(cfg)
    runtimes[label] = time.time() - t0
    rob_diags[label] = diag
    print(f"  Done in {runtimes[label]:.1f}s")

diag_rob_lo  = rob_diags["Low (50) Discrete"]
diag_rob_med = rob_diags["Medium (250) Discrete"]
diag_rob_hi  = rob_diags["High (2000) Discrete"]
diag_rob_gss = rob_diags["Medium (250) Hybrid GSS"]

# ===========================================================================
# 4. GENERATE PLOTS
# ===========================================================================
print("\n" + "="*60)
print("  GENERATING PLOTS")
print("="*60)

# ---------------------------------------------------------------------------
# 4.1 Lifecycle profiles & Ergodic health
# ---------------------------------------------------------------------------
print("\n--- Lifecycle ---")

fig = plot_lifecycle_profiles(core_ssh)
save_fig(fig, 'lifecycle', "lifecycle_profiles_py.pdf")

fig = plot_ergodic_health(diag_osh)
save_fig(fig, 'lifecycle', "ergodic_health_py.pdf")

# ---------------------------------------------------------------------------
# 4.2 Median wealth (3 base economies)
# ---------------------------------------------------------------------------
print("\n--- Median Wealth ---")

for name, diag in [('SSH', diag_ssh), ('OSH', diag_osh), ('NHH', diag_nhh)]:
    fig = plot_median_wealth({name: diag}, usd_scaling=USD, ymax=300, figsize=(8, 5.5))
    save_fig(fig, 'wealth', f"median_wealth_{name}_py.pdf")

# ---------------------------------------------------------------------------
# 4.3 Savings belief bias
# ---------------------------------------------------------------------------
print("\n--- Savings Belief Bias ---")

fig = plot_savings_belief_bias(
    diag_ssh, diag_osh, diag_nhh,
    ages_to_plot=(30, 40, 50, 60),
    real_age_offset=20,
    figsize=(9, 8),
)
save_fig(fig, 'savings_bias', "savings_belief_bias_py.pdf")

# ---------------------------------------------------------------------------
# 4.4 GSS Robustness
# ---------------------------------------------------------------------------
print("\n--- GSS Robustness ---")

gss_plots = [
    (diag_rob_lo,  "gss_lo_n50_discrete_py.pdf"),
    (diag_rob_med, "gss_med_n250_discrete_py.pdf"),
    (diag_rob_hi,  "gss_hi_n2000_discrete_py.pdf"),
    (diag_rob_gss, "gss_med_n250_hybrid_py.pdf"),
]
for diag, fname in gss_plots:
    fig = plot_policy(diag=diag, age_idx=60, xlim=1, show_value=False)
    save_fig(fig, 'gss', fname)

# ---------------------------------------------------------------------------
# 4.5 Bequest Extension
# ---------------------------------------------------------------------------
print("\n--- Bequest Extension ---")

# Median wealth
fig = plot_median_wealth({'SSH + Bequest': diag_beq_base}, usd_scaling=USD, ymax=300, figsize=(8, 5.5))
save_fig(fig, 'bequests', "median_wealth_bequest_baseline_py.pdf")

fig = plot_median_wealth({'SSH + Bequest': diag_beq_lux}, usd_scaling=USD, ymax=300, figsize=(8, 5.5))
save_fig(fig, 'bequests', "median_wealth_bequest_luxury_py.pdf")

# Policy comparisons
fig = plot_policy(
    diag=diag_beq_base, diag2=diag_beq_lux,
    label1=r"Baseline ($\phi_2 = 0.001$)",
    label2=r"Luxury ($\phi_2 = 0.1$)",
    age_idx=65, xlim=1.5, h_idx=4,
)
save_fig(fig, 'bequests', "policy_bequest_age85_poor_py.pdf")

fig = plot_policy(
    diag=diag_beq_base, diag2=diag_beq_lux,
    label1=r"Baseline ($\phi_2 = 0.001$)",
    label2=r"Luxury ($\phi_2 = 0.1$)",
    age_idx=30, xlim=1.5, h_idx=0,
)
save_fig(fig, 'bequests', "policy_bequest_age50_excellent_py.pdf")

# ===========================================================================
# 5. PRINT TABLES
# ===========================================================================
print("\n" + "="*60)
print("  GE SUMMARY — BASE ECONOMIES")
print("="*60)
display_ge_table({
    'SSH (Subjective)': {'r': r_ssh, 'w': w_ssh, 'diag': diag_ssh},
    'OSH (Objective)':  {'r': r_osh, 'w': w_osh, 'diag': diag_osh},
    'NHH (Average)':    {'r': r_nhh, 'w': w_nhh, 'diag': diag_nhh},
})

print("\n" + "="*60)
print("  GE SUMMARY — BEQUEST ECONOMIES")
print("="*60)
display_ge_table({
    'SSH + Bequest (baseline)': {'r': r_beq_base, 'w': w_beq_base, 'diag': diag_beq_base},
    'SSH + Bequest (luxury)':   {'r': r_beq_lux,  'w': w_beq_lux,  'diag': diag_beq_lux},
})

print("\n" + "="*60)
print("  RUNTIME TABLE")
print("="*60)
print(f"{'Configuration':<30} | {'Time (s)':>10}")
print("-" * 45)
for label, t in runtimes.items():
    print(f"{label:<30} | {t:>10.2f}")


# ===========================================================================
# 6. SUMMARY
# ===========================================================================
print("\n" + "="*60)
print("  ALL DONE")
print("="*60)
print("\nGenerated plots:")
for folder, path in PLOT_DIRS.items():
    files = sorted(path.glob("*_py.pdf"))
    if files:
        print(f"\n  {folder}/")
        for f in files:
            print(f"    - {f.name}")