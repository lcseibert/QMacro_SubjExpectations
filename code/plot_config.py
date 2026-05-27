# -*- coding: utf-8 -*-
"""
plot_config.py — Matplotlib configuration for publication-quality figures
=========================================================================
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
SAVE_PLOTS = True  # Toggle: True = save to file, False = display in console

# ---------------------------------------------------------------------------
# Plot directories
# ---------------------------------------------------------------------------
PLOT_DIRS = {
    'lifecycle'    : Path("plots/lifecycle"),
    'wealth'       : Path("plots/wealth"),
    'savings_bias' : Path("plots/savings_bias"),
    'policies'     : Path("plots/policies"),
    'gss'          : Path("plots/gss"),
    'bequests'     : Path("plots/bequests"),
}

def init_plot_dirs():
    """Create plot directories if they don't exist."""
    for d in PLOT_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Matplotlib styling
# ---------------------------------------------------------------------------
PLOT_STYLE = {
    # Font sizes
    'font.size':        16,
    'axes.titlesize':   18,
    'axes.labelsize':   16,
    'xtick.labelsize':  14,
    'ytick.labelsize':  14,
    'legend.fontsize':  13,
    'figure.titlesize': 20,
    
    # Line widths
    'lines.linewidth':  2.5,
    'axes.linewidth':   1.4,
    'grid.linewidth':   0.8,
    
    # Resolution
    'figure.dpi':       150,
    'savefig.dpi':      400,
    'savefig.bbox':     'tight',
    'savefig.pad_inches': 0.05,
    'savefig.format':   'pdf',
    
    # Axes styling
    'axes.grid':        False,
    'axes.spines.top':  True,
    'axes.spines.right': True,
    
    # Font family (serif + Computer Modern math)
    'font.family':      'serif',
    'mathtext.fontset': 'cm',
    
    # Legend
    'legend.framealpha': 0.9,
    'legend.edgecolor':  '0.8',
}

def apply_style():
    """Apply publication-quality matplotlib style."""
    mpl.rcParams.update(PLOT_STYLE)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def save_fig(fig, folder_key: str, filename: str):
    """Save figure to file or display in console based on SAVE_PLOTS setting."""
    if SAVE_PLOTS:
        path = PLOT_DIRS[folder_key] / filename
        fig.savefig(path, dpi=400, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        print(f"  -> saved {path}")
    else:
        plt.show()

def set_save_mode(save: bool):
    """Set whether to save plots or display them."""
    global SAVE_PLOTS
    SAVE_PLOTS = save

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
USD = 46.500                        # Scaling factor: model units -> $1000s
EQUILIBRIUM_R_RANGE = [0.01, 0.04]  # Brent search bounds for r*