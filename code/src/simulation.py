# -*- coding: utf-8 -*-
"""
simulation.py — Forward simulation and distribution dynamics
============================================================

This module implements the histogram method for the forward simulation and the auxiliary process.

Core Functions
--------------
get_initial_distribution : Initialize mass at age 20 (zero assets, ergodic exogenous states)
run_forward_simulation   : Main forward simulation via histogram method with lottery
lifecycle_simulation     : Auxiliary simulation for calibration (computes median/average income)


The simulation uses Numba-accelerated kernels (forward_step_numba) for performance.

"""

import numpy as np
import time
from scipy.stats import lognorm
from src.worker_funcs import forward_step_numba, forward_life_cycle, compute_income_stats

def get_initial_distribution(primitives):
    """
    Creates a heterogeneous initial distribution at Age 20 using 
    a log-normal distribution for starting assets.
    """
    p = primitives
    
    # collect invariant distributions for the persistent processes
    stat_p = p.invariant_Q.p
    stat_m = p.invariant_Q.m
    # initiate health states at invariant distribution of model age 0
    alive_states = p.invariant_Q.h_0.shape[0]
    init_h = np.zeros(alive_states + 1)
    init_h[:alive_states] = p.invariant_Q.h_0
    init_h[alive_states] = 0.0 # no mortality yet
    
    # combine to exogenous state matrix
    prob_exog = (init_h[:, None, None] * stat_p[None, :, None] * stat_m[None, None, :])
    
    # generate initial savings mass | everyone starts with zero assets eg at savings = 0 with probability 1.0
    asset_probs = np.zeros(len(p.sav_grid))
    asset_probs[0] = 1.0
    
    dist_0 = asset_probs[:, None, None, None] * prob_exog[None, :, :, :]
    
    return dist_0


# =======================================
# ERGODIC FORWARD SIMULATIONS
# =======================================

def run_forward_simulation(primitives, policy_lib, start_asset_idx=0, verbose = False):
    p = primitives
    cfg = p.cfg
    
    print(f"Starting Lifecycle Simulation ({cfg.n_ages} periods)...")
    
    # start up with distribution
    curr_dist = get_initial_distribution(p)
    history = [curr_dist]
    global_cah_max = 0.0
    # iterate over life time
    for age in range(cfg.n_ages - 1):
        # --- DEFAULT WITH ALL SHOCKS ON
        Pi_h = p.trans.h_Q_forward[age]   # health shocks w/ survival probabilities
        Pi_p = p.trans.lab_p.probs        # Persistent Labor
        Pi_m = p.trans.med_p.probs        # Persistent MedEx
        
        prob_trans_p = p.trans.lab_eps.probs # Transitory Labor
        prob_trans_m = p.trans.med_eps.probs # Transitory MedEx
            
        # Policy for this age
        pol_age = policy_lib.savings[age]
        
        if age < cfg.medex_age:
            # no medical shocks, normal income process
            inc_slice = p.resources.income_working[age]
            med_slice = np.zeros((len(Pi_h), cfg.n_m, 1)) # quick fix but probably bad! Actually reasonable
            prob_trans_m = np.array([1.0])
            # Medical State is irrelevant, but we need valid indices. 
            # Identity matrix ensures m stays at 0 (if initialized at 0).
            Pi_m = np.eye(cfg.n_m)
            
        elif age < cfg.ret_age:
            inc_slice = p.resources.income_working[age] # can only index up to age 44 (index 45)
            med_slice = p.resources.med_costs[age - cfg.medex_age]
        else: #retired
            prob_trans_p = np.array([1.0])
            Pi_p = np.eye(cfg.n_p) # Freeze productivity state

            # Slice: (H, P) -> (H, P, 1) to match prob=[1.0]
            pension = p.resources.income_retired[age - cfg.ret_age] # normalize for 45 periods of pension income 
            inc_slice = pension[np.newaxis, :, np.newaxis]
            med_slice = p.resources.med_costs[age - cfg.medex_age] # - 30 to normalize for length of medical expenditures to actual age
        
        time_numba_start = time.time()
        
        # --- EXECUTION
        next_dist, step_max_cah = forward_step_numba(
            dist_curr = curr_dist,                # Current Mass
            policy_sav = pol_age,                 # Policy Rule
            sav_grid = p.sav_grid,                # Asset Grid
            cah_grid = p.cah_grid,                # Cash-on-Hand Grid
            inc_grid = inc_slice,                 # Income (Stage dependent)
            med_grid = med_slice,                 # Medical Costs
            Pi_h = Pi_h,                          # Health Transition (Age dependent) objective!
            Pi_p = Pi_p,                          # Prod Transition (Stage dependent)
            Pi_m = Pi_m,                          # Med Transition (Standard)
            prob_eps_p = prob_trans_p,            # Prod Shock Probs (Stage dependent)
            prob_eps_m = prob_trans_m,            # Med Shock Probs
            r = cfg.r                             # Interest Rate
        )
        # track highest observed cah!
        if step_max_cah > global_cah_max:  global_cah_max = step_max_cah
        time_numba_end = time.time()
        if verbose:
            print(f"Solved age {age} in {time_numba_end - time_numba_start}s")
        # Store and Update
        history.append(next_dist)
        curr_dist = next_dist

        if age % 30 == 0:
            print(f"   -> Completed Age {20 + age}")

    # CHECK ASSET & CAH UPPER BOUND 
    history_arr = np.array(history)
    mass_by_asset = np.sum(history_arr[:, :, :, :, :], axis=(0,2,3,4))
    
    tol_mass_a = 1e-8
    highest_idx = np.flatnonzero(mass_by_asset > tol_mass_a).max()
    highest_asset = p.sav_grid[highest_idx]
    mass_a_top = mass_by_asset[-1]
    if mass_a_top > tol_mass_a:
        print(f"Mass at top asset grid point: {mass_a_top} (grid may be too small)")
        
    print(f"MAX ASSET OBSERVED DURING SIMULATION: {highest_asset}")
    print(f"MAX CAH OBSERVED DURING SIMULATION: {global_cah_max}")
    return np.array(history)


# testing code, will probably go
# bequests are redistributed to households in a PE manner without adjusting policy functions
# the reason is to have a small peek into the effects that potentially hide under the stochastic intergenerational bequest system of the paper
def run_simulation_bequests(primitives, policy_lib, injection_pool=0.0, verbose=False):
    p = primitives
    cfg = p.cfg
    
    print(f"Starting Lifecycle Simulation ({cfg.n_ages} periods)...")
    
    curr_dist = get_initial_distribution(p)
    history = [curr_dist]
    
    # store running count of bequests
    total_bequests = 0.0 
    
    for age in range(cfg.n_ages - 1):
        # --- DEFAULT WITH ALL SHOCKS ON
        Pi_h = p.trans.h_Q_forward[age]           # health shocks w/ survival probabilities
        Pi_p = p.trans.lab_p.probs        # Persistent Labor
        Pi_m = p.trans.med_p.probs        # Persistent MedEx
        
        prob_trans_p = p.trans.lab_eps.probs # Transitory Labor
        prob_trans_m = p.trans.med_eps.probs # Transitory MedEx
            
        # Policy for this age
        pol_age = policy_lib.savings[age]
        
        if age < cfg.medex_age:
            # no medical shocks, normal income process
            inc_slice = p.resources.income_working[age]
            med_slice = np.zeros((len(Pi_h), cfg.n_m, 1)) # quick fix but probably bad!
            prob_trans_m = np.array([1.0])
            # Medical State is irrelevant, but we need valid indices. 
            # Identity matrix ensures m stays at 0 (if initialized at 0).
            #Pi_m = np.array([1.0])# np.eye(cfg.n_m)
            Pi_m = np.eye(cfg.n_m)
            
        elif age < cfg.ret_age:
            inc_slice = p.resources.income_working[age] # can only index up to age 44 (index 45)
            med_slice = p.resources.med_costs[age - cfg.medex_age]
        else: #retired
            prob_trans_p = np.array([1.0])
            Pi_p = np.eye(cfg.n_p) # Freeze productivity state

            # Slice: (H, P) -> (H, P, 1) to match prob=[1.0]
            pension = p.resources.income_retired[age - cfg.ret_age] # normalize for 45 periods of pension income 
            inc_slice = pension[np.newaxis, :, np.newaxis]
            med_slice = p.resources.med_costs[age - cfg.medex_age] # - 30 to normalize for length of medical expenditures to actual age
        
        inc_slice_sim = np.copy(inc_slice)
        
        # assuming wealth distributed at model age 30
        if age == 30 and injection_pool > 0.0: 
            bonus = np.zeros(cfg.n_p)
            
            # 1. Create a skewed distribution (Power Law)
            # theta controls inequality. theta=1 is linear, theta=3 is highly skewed to the top.
            theta = 3.0 
            
            # Base weights: [1^3, 2^3, 3^3, 4^3] -> [1, 8, 27, 64]
            weights = np.array([(i + 1)**theta for i in range(cfg.n_p)])
            
            # Normalize so they sum to exactly 1.0
            weights = weights / np.sum(weights) 
            
            # 2. Distribute the pool
            for ip in range(cfg.n_p):
                # Protect against divide-by-zero if a state has no mass
                mass_ip = max(p.invariant_Q.p[ip], 1e-6) 
                
                # Assign the per-capita bonus to this productivity state
                bonus[ip] = (injection_pool * weights[ip]) / mass_ip
                
            # 3. Add to the simulator's income grid
            inc_slice_sim = inc_slice_sim + bonus[np.newaxis, :, np.newaxis]
            
            if verbose:
                print(f"   -> Injecting bequests at age {age + 20}: {bonus[4]:.2f} to top earners")

        # --- 2. FORWARD STEP ---
        # Note: We pass inc_slice_sim, NOT the standard inc_slice
        next_dist = forward_step_numba(
            dist_curr=curr_dist, policy_sav=pol_age, sav_grid=p.sav_grid,
            cah_grid=p.cah_grid, inc_grid=inc_slice_sim, med_grid=med_slice,
            Pi_h=Pi_h, Pi_p=Pi_p, Pi_m=Pi_m, prob_eps_p=prob_trans_p, 
            prob_eps_m=prob_trans_m, r=cfg.r
        )
        
        # --- 3. HARVEST THE DEAD ---
        # next_dist shape: (nA, nH, nP, nM). Dead state is index -1.
        # Sum out productivity and medical states to get dead mass by asset level
        dead_mass_by_asset = np.sum(next_dist[:, -1, :, :], axis=(1, 2)) 
        
        # Multiply by the asset grid and accrue interest for the year
        ab_this_period = np.sum(dead_mass_by_asset * p.sav_grid) * (1 + cfg.r)
        total_bequests += ab_this_period
        
        # Store and Update
        history.append(next_dist)
        curr_dist = next_dist

    return np.array(history), total_bequests


# =======================================
# LIFECYCLE PREMODEL SETUP
# =======================================

def lifecycle_simulation(primitives, inc_grid):
    """
    Simulates the exogenous life cycle (Health and Productivity) to find
    the cross-sectional median, average and last retirement period average gross income.
    """
    p = primitives
    
    # Invariant Distributions
    # Inititate population with invariant distribution at initial period
    init_h_living = p.invariant_Q.h_0 
    inv_p = p.invariant_Q.p
    
    # combine to 2d probs | grid over productivity and health states
    initial_dist = init_h_living[:, None] * inv_p[None, :]
    
    # iterate the mass forward
    mass_grid = forward_life_cycle(
        initial_dist=initial_dist,
        Pi_h=p.trans.h_Q_forward,  # Shape: (n_ages, n_h_total, n_h_total) obj_beliefs!
        Pi_p=p.trans.lab_p.probs,  # Shape: (n_p, n_p)
        n_working_ages=p.cfg.ret_age  # Working Ages
    )
    
    # Evaluate descriptive statistics of interest
    median_y, average_y, l_supply = compute_income_stats(
        mass_grid=mass_grid,
        inc_grid=inc_grid, 
        prob_eps_p=p.trans.lab_eps.probs
    )
    
    return median_y, average_y, l_supply

