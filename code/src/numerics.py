# -*- coding: utf-8 -*-
"""
numerics.py — Core numerical routines
======================================

Discretization
--------------
rouwenhorst         : Discretize AR(1) process via Rouwenhorst method
tauchen             : Discretize AR(1) process via Tauchen method
stationary_distribution : Compute ergodic distribution of Markov chain

Grid Construction
-----------------
power_grid          : Generate power-spaced asset grid (dense near zero)

Interpolation & Optimization
----------------------------
manual_search       : Binary search for bracketing index (Numba JIT)
interp_1d_numba     : Linear interpolation with manual search (Numba JIT)
evaluate_bellman    : Bellman objective for arbitrary savings choice
golden_section_search : GSS optimizer for 1D maximization problems

Descriptive Statistics
----------------------
compute_gini        : Gini coefficient from values and weights
"""

import numpy as np
from scipy.stats import norm
from numba import njit


# =======================================
# DISCRETIZE 
# =======================================
def rouwenhorst(rho, sigma, mu=0, N=7):
    """Speed-optimized Rouwenhorst method."""
    p = (1 + rho) / 2
    
    def _recursive_matrix(n):
        if n == 2:
            return np.array([[p, 1-p], [1-p, p]])
        else:
            Q_prior = _recursive_matrix(n-1)
            Q = np.zeros((n, n))
            Q[:n-1, :n-1] += p * Q_prior
            Q[:n-1, 1:] += (1-p) * Q_prior
            Q[1:, :n-1] += (1-p) * Q_prior
            Q[1:, 1:] += p * Q_prior
            Q[1:-1, :] /= 2
            return Q

    # Grid construction
    uncond_sd = sigma / np.sqrt(1 - rho**2)
    z_end = np.sqrt(N-1) * uncond_sd
    zgrid = np.linspace(-z_end, z_end, N) + mu
    
    Q = _recursive_matrix(N)
    return zgrid, Q


def tauchen(rho, mu, sigma, N, m=3):
    """
    Discretize the AR(1) process using Tauchen's method.

    Tauchen's method approximates a continuous AR(1) process with a discrete Markov chain.
    It is particularly useful for solving dynamic programming problems numerically.

    Parameters
    ----------
    rho : float
        AR(1) persistence parameter.
    mu : float
        Unconditional mean of the AR(1) process.
    sigma : float
        Standard deviation of AR(1) innovations.
    N : int
        Number of discrete states (grid points) for the AR(1) process.
    m : float, optional
        Scaling parameter that sets the grid width in multiples of the unconditional standard deviation.
        Default is 3.

    Returns
    -------
    zgrid : ndarray of shape (N,)
        Grid of discrete state values.
    Q : ndarray of shape (N, N)
        Transition probability matrix. Each row sums to 1.
    """

    z_max = mu + (m * sigma)/(np.sqrt(1-rho**2))
    z_min = -z_max + 2*mu
    zgrid = np.linspace(z_min, z_max, N)
    dz = (zgrid[1] - zgrid[0])    
    
    # Mean conditional on today's state
    mean = rho*zgrid[:, None] + mu
    upper_std = (zgrid[None, :] + dz/2 - mean) / sigma
    lower_std = (zgrid[None, :] - dz/2 - mean) / sigma

    # Transition matrix
    Q = norm.cdf(upper_std) - norm.cdf(lower_std)     
    # Edge cases
    Q[:, 0] = norm.cdf((zgrid[0] + dz/2 - mean.flatten()) / sigma)  # First grid
    Q[:, -1] = 1 - norm.cdf((zgrid[-1] - dz/2 - mean.flatten()) / sigma)  # Last grid
     
    return zgrid, Q

def stationary_distribution(P):
    """
    Compute the stationary distribution of a Markov chain
    with row-stochastic transition matrix P.
    This function is ought to be used for small nxn.
    For the large space simulation I  refer to hsitogram iteration.
    """
    n = P.shape[0]

    A = P.T - np.eye(n)
    A[-1] = np.ones(n)

    b = np.zeros(n)
    b[-1] = 1

    pi = np.linalg.solve(A, b)
    return pi

# =======================================
# GRID CONSTRUCTION 
# =======================================
    
def power_grid(b, na, a_max = None, zeta=0.15):
    '''
    Return a power grid that approximately matches the target asset possibilites of the agents.
    Upper bound determined by 10 * maximum savings
    
    Parameters
    ----------
    b : TYPE
        DESCRIPTION.
    w_ref : TYPE
        Reference value for the wage. Default is 2
    max_state : float
        Highest value of discretized markov shock.
    na : Int
        Number of grid points.
    zeta : TYPE, optional
        DESCRIPTION. The default is 0.15.
    
    Returns
    -------
    TYPE
        1d-array with asset grid.
    
    '''
    if a_max == None: 
        a_max = 10_000 
        
    s = np.linspace(0, 1, na)
    return -b + (a_max + b) * s**(1/zeta)


# =======================================
# INTERPOLATION
# =======================================
@njit(inline='always')
def manual_search(grid, val):
    """Finds the left index i such that grid[i] <= val < grid[i+1]"""
    nA = len(grid)
    if val <= grid[0]: return 0
    if val >= grid[nA - 1]: return nA - 2
    
    low = 0
    high = nA - 1
    # biscetion type logic -> divided search space by half until we bracket the grid point
    while high - low > 1:
        mid = (low + high) // 2
        if grid[mid] <= val:
            low = mid
        else:
            high = mid
    return low

@njit(inline='always')
def interp_1d_numba(x, x_grid, y_vals):
    """Linear interpolation that uses manual search"""
    idx = manual_search(x_grid, x)
    x0, x1 = x_grid[idx], x_grid[idx+1]
    y0, y1 = y_vals[idx], y_vals[idx+1]
    
    if x1 == x0: #if no mass between the poitns just return the point! otherwise division error
        return y0
    # Linear interpolation formula
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)

@njit(fastmath=True, cache=True)
def evaluate_bellman(s_prime, cah_current, w_curve, sav_grid, rra):
    """
    Bellman objective at an arbitrary (possibly off-grid) s'.
    u(c) is analytic; W(s') is linearly interpolated from the dense w_curve.
    """
    if s_prime >= cah_current:
        return -1e20, 0.0
    cons = cah_current - s_prime
    if cons <= 1e-6:
        return -1e20, cons
    if rra == 1.0:
        util = np.log(cons)
    else:
        util = (cons ** (1.0 - rra)) / (1.0 - rra)
    w_val = interp_1d_numba(s_prime, sav_grid, w_curve)
    return util + w_val, cons


@njit(fastmath=True, cache=True)
def golden_section_search(cah_current, w_curve, sav_grid, rra, a, b, tol=1e-7):
    """
    Pure GSS algorithm over the bracket [a, b].
    Watch out that a < b and b < cah_current.
    Returns (max_value, optimal_s, optimal_consumption).
    """
    phi = (np.sqrt(5.0) - 1.0) / 2.0   # ≈ 0.618 golden ration inverse
    # get evaluation points
    c_pt = b - phi * (b - a)
    d_pt = a + phi * (b - a)
    
    # evaluate points at c and d
    fc, cons_c = evaluate_bellman(c_pt, cah_current, w_curve, sav_grid, rra)
    fd, cons_d = evaluate_bellman(d_pt, cah_current, w_curve, sav_grid, rra)

    for _ in range(60):  # 60 iters → bracket shrink 0.618^60 ≈ 1e-13s by
        if (b - a) < tol:
            break
        # compare cases
        if fc > fd: # left side is higher
            b    = d_pt
            d_pt = c_pt;  cons_d = cons_c;  fd = fc
            c_pt = b - phi * (b - a)
            fc, cons_c = evaluate_bellman(c_pt, cah_current, w_curve, sav_grid, rra)
        else: # right side is higher
            a    = c_pt
            c_pt = d_pt;  cons_c = cons_d;  fc = fd
            d_pt = a + phi * (b - a)
            fd, cons_d = evaluate_bellman(d_pt, cah_current, w_curve, sav_grid, rra)

    # final return statement
    if fc > fd:
        return fc, c_pt, cons_c
    else:
        return fd, d_pt, cons_d


# =======================================
# DESCRIPTIVE STATISTICS
# =======================================

def compute_gini(values, weights=None):
    """
    Computes the Gini Coefficient for a set of values and corresponding weights.
    
    Parameters:
    - values:  1D array of the variable (Assets, Income, etc.)
    - weights: 1D array of population weights/mass. 
               If None, assumes all observations are equally weighted.
    """
    values = np.asarray(values).flatten()
    if weights is None:
        weights = np.ones_like(values)
    else:
        weights = np.asarray(weights).flatten()

    # Sort values and weights by the values
    idx = np.argsort(values)
    values = values[idx]
    weights = weights[idx]

    # Cumulative sums for the Lorenz curve logic
    # Population share
    cum_weights = np.cumsum(weights)
    sum_weights = cum_weights[-1]
    
    # Wealth/Income share
    weighted_values = values * weights
    cum_weighted_values = np.cumsum(weighted_values)
    sum_weighted_values = cum_weighted_values[-1]

    # Standard Gini Formula via Trapezoidal Rule
    # G = 1 - sum( pdf_i * (cdf_wealth_i + cdf_wealth_{i-1}) )
    pdf_pop = weights / sum_weights
    cdf_wealth = cum_weighted_values / sum_weighted_values
    cdf_wealth_prev = np.insert(cdf_wealth[:-1], 0, 0)
    
    gini = 1.0 - np.sum(pdf_pop * (cdf_wealth + cdf_wealth_prev))
    
    return gini