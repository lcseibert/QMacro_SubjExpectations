# -*- coding: utf-8 -*-
"""
plot_health.py — Health distribution visualization
===================================================

plot_ergodic_health : Stacked area chart of health state mass over the lifecycle

Constants
---------
HEALTH_LABELS : Labels for health states (Excellent → Poor)
HEALTH_COLORS : Color palette matching paper aesthetics (green → red)
"""

import numpy as np
import matplotlib.pyplot as plt

HEALTH_LABELS = ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor']
HEALTH_COLORS = ['#1b5e20', '#4caf50', '#fdd835', '#ff8f00', '#c62828']


def plot_ergodic_health(diag, starting_age=20, ax=None):
    """
    Stacked area of absolute health state mass over the lifecycle.
    Input: 
        diag: diagnostics class for policy evaluation.
    """
    health_age = diag.marginal_health_age()[:, :diag.n_h_alive]  # (n_ages, n_h_alive)
    n_ages = health_age.shape[0]
    ages   = np.arange(starting_age, starting_age + n_ages)
    n_h    = health_age.shape[1]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 5))

    ax.stackplot(ages, health_age.T,
                 labels=HEALTH_LABELS[:n_h],
                 colors=HEALTH_COLORS[:n_h],
                 alpha=0.85)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper right', fontsize=9, framealpha=0.9)
    ax.set_xlabel('Age');  ax.set_ylabel('Relative cohort size')
    ax.set_xlim(ages[0], 110)
    ax.set_ylim(0, health_age.sum(axis=1).max() * 1.05)
    ax.set_title('Ergodic health distribution');  ax.grid(alpha=0.25)

    if standalone:
        plt.tight_layout()
        return fig