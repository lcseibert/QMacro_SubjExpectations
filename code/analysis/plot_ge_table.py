# -*- coding: utf-8 -*-
"""
plot_ge_table.py — GE results table display
============================================
"""

def display_ge_table(results_dict):
    """
    Display general equilibrium results in a formatted table.
    
    Parameters
    ----------
    results_dict : dict[str, dict]
        Each value must have keys 'r', 'w', 'diag' (and optionally 'A').
    
    Example
    -------
    display_ge_table({
        'SSH GE': {'r': r_ssh, 'w': w_ssh, 'A': A_ssh, 'diag': diag_ssh},
        'OSH GE': {'r': r_osh, 'w': w_osh, 'A': A_osh, 'diag': diag_osh},
    })
    """
    headers = ['Economy', 'r*', 'w*', 'K/Y', 'Gini (wealth)']
    col_w   = [22, 8, 8, 8, 14]
    
    sep = '+' + '+'.join('-' * (w + 2) for w in col_w) + '+'
    row = lambda cells: '|' + '|'.join(f' {str(c):<{w}} ' for c, w in zip(cells, col_w)) + '|'
    
    print(sep)
    print(row(headers))
    print(sep)
    
    for label, res in results_dict.items():
        r = res['r']
        w = res['w']
        diag = res['diag']
        
        print(row([
            label,
            f'{r:.2%}',
            f'{w:.4f}',
            f'{diag.compute_ky_ratio(r):.3f}',
            f'{diag.compute_gini_wealth():.4f}',
        ]))
    
    print(sep)


def display_ge_table_latex(results_dict):
    """
    Output GE results as LaTeX tabular rows for direct copy-paste.
    """
    print("% LaTeX table rows:")
    print("% Economy & $r^*$ & $w^*$ & K/Y & Gini \\\\")
    print("\\midrule")
    
    for label, res in results_dict.items():
        r = res['r']
        w = res['w']
        diag = res['diag']
        
        ky = diag.compute_ky_ratio(r)
        gini = diag.compute_gini_wealth()
        
        print(f"{label} & {r:.2%} & {w:.4f} & {ky:.3f} & {gini:.4f} \\\\")