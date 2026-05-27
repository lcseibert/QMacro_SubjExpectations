# -*- coding: utf-8 -*-
"""
diagnostics.py — Post-solution analysis and statistics
=======================================================

Central class for extracting statistics from solved economies.

diagnostics
-----------
Wraps distribution, policy, and model primitives for analysis:

Marginals (ergodics):
    marginal_health_age      : Health distribution by age
    marginal_sav_health_age  : Savings × health distribution by age

Wealth statistics:
    get_median_assets        : Median savings by age (optionally by health)
    compute_gini_wealth      : Gini coefficient on pooled savings

Ergodic flows:
    get_ergodic_wincome      : Mean working income by age
    get_ergodic_pension      : Mean pension by retirement age
    get_ergodic_health_costs : Mean medical costs by age

Policy aggregation:
    get_policy_agg           : Policy averaged over (p,m) via invariant dist
    get_savings_rate_agg_ergodic : Savings rate weighted by simulated distribution

GE statistics:
    compute_ky_ratio         : Capital-output ratio at equilibrium r
"""

import numpy as np
from src.numerics import compute_gini
from analysis.aggregation import aggregate_pm, get_savings_rate, weighted_median


class diagnostics:
    """
    Central data object for post-solution analysis of one economy.

    Shapes
    ------
    dist : (n_ages, n_sav, n_h, n_p, n_m)   forward-simulated distribution
    pol  : PolicyLibrary  .savings / .consumption / .value
           each (n_ages, n_cah, n_h, n_p, n_m)
    core : ModelPrimitives
    """

    def __init__(self, dist, pol, core):
        self.dist     = dist
        self.pol      = pol
        self.core     = core
        self.cfg      = core.cfg

        self.cah_grid  = pol.cah_grid         # (n_cah,)
        self.sav_grid  = core.sav_grid        # (n_sav,)
        self.inv_p     = core.invariant_Q.p   # (n_p,)
        self.inv_m     = core.invariant_Q.m   # (n_m,)
        self.n_h_alive = dist.shape[2] - 1    # exclude dead state

    # ------------------------------------------------------------------
    # Marginals from the distribution (dist lives on sav_grid)
    # ------------------------------------------------------------------

    def marginal_health_age(self):
        """(n_ages, CAH/SAV, n_h, n_p, n_m)->(n_ages, n_h) — sum out sav, p, m."""
        return self.dist.sum(axis=(1, 3, 4))

    def marginal_sav_health_age(self):
        """(n_ages, CAH/SAV, n_h, n_p, n_m)->(n_ages, CAH/SAV, n_h) — sum out p, m."""
        return self.dist.sum(axis=(3, 4))

    # ------------------------------------------------------------------
    # Median assets
    # ------------------------------------------------------------------

    def get_median_assets(self, by_health=True):
        """
        by_health=True  → (n_ages, n_h_alive)
        by_health=False → (n_ages,)
        """
        marg = self.marginal_sav_health_age()   # (n_ages, n_sav, n_h)
        n_ages = marg.shape[0]

        if by_health:
            out = np.zeros((n_ages, self.n_h_alive))
            for t in range(n_ages):
                for h in range(self.n_h_alive):
                    w = marg[t, :, h]
                    w_sum = w.sum()
                    if w_sum > 1e-12:
                        w_norm = w / w_sum  # <-- Transform to conditional PDF
                        out[t, h] = weighted_median(self.sav_grid, w_norm)
        else:
            out = np.zeros(n_ages)
            for t in range(n_ages):
                w = marg[t, :, :self.n_h_alive].sum(axis=1)
                w_sum = w.sum()
                if w_sum > 1e-12:
                    w_norm = w / w_sum  # normalize by population at t
                    out[t] = weighted_median(self.sav_grid, w_norm)
        return out

    # ------------------------------------------------------------------
    # Ergodic income / medical / consumption flows
    # ------------------------------------------------------------------

    def get_ergodic_wincome(self):
        """Ergodic mean working income by age. Returns (n_ret_ages,)."""
        cfg = self.cfg
        res = self.core.resources
        eps_p = self.core.trans.lab_eps.probs   # (n_eps_p,)
        out = np.zeros(cfg.ret_age)
        for t in range(cfg.ret_age):
            # mass over (h_alive, p), summed out sav and m
            mass = self.dist[t, :, :self.n_h_alive, :, :].sum(axis=(0, 3))  # (n_h, n_p)
            total = mass.sum()
            if total < 1e-12:
                continue
            mass /= total
            # smooth out transitory shocks by probability weights
            inc_hp = res.income_working[t, :self.n_h_alive] @ eps_p  # (n_h, n_p)
            out[t] = np.sum(mass * inc_hp)
        return out

    def get_ergodic_pension(self):
        """Ergodic mean pension by retirement age. Returns (n_ret_ages,)."""
        cfg = self.cfg
        res = self.core.resources
        n_ret = cfg.n_ages - cfg.ret_age
        out = np.zeros(n_ret)
        for t in range(n_ret):
            age = cfg.ret_age + t
            mass_p = self.dist[age, :, :self.n_h_alive, :, :].sum(axis=(0, 2, 3))  # (n_p,)
            total = mass_p.sum()
            if total < 1e-12:
                continue
            out[t] = np.dot(mass_p / total, res.income_retired[t])
        return out

    def get_ergodic_health_costs(self):
        """Ergodic mean medical cost by medex age. Returns (n_medex_ages,)."""
        cfg = self.cfg
        res = self.core.resources
        eps_m = self.core.trans.med_eps.probs   # (n_eps_m,)
        n_medex = res.med_costs.shape[0]
        out = np.zeros(n_medex)
        for t in range(n_medex):
            age = cfg.medex_age + t
            if age >= cfg.n_ages:
                break
            # mass over (h_alive, m)
            mass = self.dist[age, :, :self.n_h_alive, :, :].sum(axis=(0, 2))  # (n_h, n_m)
            total = mass.sum()
            if total < 1e-12:
                continue
            mass /= total
            # smooth out transitory shocks by probability weights
            med_hm = res.med_costs[t, :self.n_h_alive] @ eps_m   # (n_h, n_m)
            out[t] = np.sum(mass * med_hm)
        return out

    # ------------------------------------------------------------------
    # Policy aggregation  (pol lives on cah_grid)
    # ------------------------------------------------------------------

    def get_policy_agg(self, age, kind='savings'):
        """
        Policy averaged over p and m via invariant distributions.
        Returns (n_cah, n_h).
        kind: 'savings' | 'consumption' | 'value'
        """
        raw = getattr(self.pol, kind)[age]   # (n_cah, n_h, n_p, n_m)
        return aggregate_pm(raw, self.inv_p, self.inv_m)

    def get_savings_rate_agg_ergodic(self, age):
        """
        Savings rate s'/x averaged over (p, m) using the ergodic distribution
        conditional on (age, health).
        Returns (n_cah, n_h_alive).
        """
        sr_raw = get_savings_rate(
            self.pol.savings[age], self.cah_grid
        )  # (n_cah, n_h, n_p, n_m)
    
        # marginal distribution over (p, m) conditional on (age, health)
        # sum dist over savings dimension → (n_h, n_p, n_m)
        mass_hpm = self.dist[age, :, :self.n_h_alive, :, :].sum(axis=0)
    
        out = np.zeros((len(self.cah_grid), self.n_h_alive))
        for h in range(self.n_h_alive):
            # weights over (p, m) for this health state, marginalised over savings
            w = mass_hpm[h, :, :]          # (n_p, n_m)
            w_total = w.sum()
            if w_total < 1e-12:
                continue
            w /= w_total
            # weighted average of savings rate over (p, m) at each cah point
            # sr_raw[:, h, :, :] shape (n_cah, n_p, n_m)
            out[:, h] = np.einsum('xpm,pm->x', sr_raw[:, h, :, :], w)
    
        return out

    # ------------------------------------------------------------------
    # GE summary statistics
    # ------------------------------------------------------------------

    def compute_gini_wealth(self):
        """Gini on the marginal savings distribution (all ages pooled)."""
        sav_marg = self.dist[:, :, :self.n_h_alive, :, :].sum(axis=(0, 2, 3, 4))
        return compute_gini(self.sav_grid, sav_marg)

    def compute_ky_ratio(self, r):
        """K/Y ratio at equilibrium r assuming Cobb-Douglas, w=1."""
        cfg = self.cfg
        k_d = cfg.alpha / ((1 - cfg.alpha) * (r + cfg.delta))
        A   = 1.0 / ((1 - cfg.alpha) * (k_d ** cfg.alpha))
        Y   = A * (k_d ** cfg.alpha)
        return k_d / Y
