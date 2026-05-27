# -*- coding: utf-8 -*-
"""
aggregation.py — Distribution aggregation utilities
====================================================

aggregate_pm    : Average over (p,m) dimensions using ergodic weights
get_savings_rate: Compute savings rate s'/x from policy array
weighted_median : Weighted median for distribution statistics
"""

# plotting helper functions
import numpy as np


def aggregate_pm(tensor, inv_p, inv_m):
    """
    Average out productivity and medical costs weighted by ergodic distribution
    Works for any leading shape: (age, n_cah, n_h, n_p, n_m) → (age, n_cah, n_h),
    or (n_h, n_p, n_m) → (n_h,), etc.
    """
    return np.einsum('...pm,p,m->...', tensor, inv_p, inv_m)


def get_savings_rate(pol_sav, cah_grid):
    """
    s'/x at a single age. pol_sav: (n_cah, n_h, n_p, n_m), cah_grid: (n_cah,).
    Returns same shape. Contains division safetey check
    """
    cah = cah_grid[:, None, None, None]
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(cah > 1e-8, pol_sav / cah, 0.0)


def weighted_median(values, weights):
    """
    Weighted median. values and weights are 1D, values need not be sorted.
    """
    idx  = np.argsort(values)
    sv   = values[idx]
    sw   = weights[idx]
    cum  = np.cumsum(sw)
    pos  = np.searchsorted(cum, cum[-1] / 2.0)
    return sv[min(pos, len(sv) - 1)]
