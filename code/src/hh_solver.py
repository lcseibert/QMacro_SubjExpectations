# -*- coding: utf-8 -*-
"""
hh_solver.py — Household problem solver via backward induction
==============================================================

Implements Algorithm 1 from the paper: backward induction over the
lifecycle with E-step (continuation value) and M-step (optimization).

Main Function
-------------
solve_lifecycle : Solve household problem for all ages, returns PolicyLibrary

Data Structures
---------------
AgeInputs       : NamedTuple container for Numba-compatible age-specific inputs
PolicyLibrary   : Storage for value/savings/consumption policies (from data_management)

Internal
--------
build_age_problem : Construct age-specific environment (transitions, income, medical)
debug_age_inputs  : Print diagnostic info for AgeInputs container

"""

import numpy as np
import os
import time
from typing import NamedTuple
from src.worker_funcs import solve_age_layer, m_step_numba
from src.data_management import PolicyLibrary

def solve_lifecycle(model, solution_dir="./sol_data", save_to_disk = False, verbose = False):
    """
    Solves the household problem via Backward Induction as described in the pseudocde
    
    model: Container with 'cfg', 'trans', 'resources', 'grids'
    """
    cfg = model.cfg
    
    # decide on type of storage
    if save_to_disk and not os.path.exists(solution_dir):
        os.makedirs(solution_dir)
        print(f"Created solution directory: {solution_dir}")
        
    n_h = model.trans.h_Q.shape[1] # health states
    death_state = n_h - 1          # death state
    
    # --- ALLOCATE MEMORY ---
    # Dimensions: (n_cah, n_h, n_p, n_m)
    shape_full = (cfg.n_ages, cfg.n_cah, n_h, cfg.n_p, cfg.n_m)
    
    V_master = np.zeros(shape_full)       # Value Function
    S_master = np.zeros(shape_full)       # Savings Policy
    C_master = np.zeros(shape_full)       # Consumption Policy
    
    # --- TERMINAL PERIOD AGE (89) ---
    T = cfg.max_age - 1 # 0-indexed last age
    print("Solving life cycle optimization")
    # depending on whether a bequest motive has been choosen:
        # either consume everything and safe nothing -> V() = log(cash)
        # optimization with bequest utiltity tomorrow: V() = max_s log(cash-s) + beta bequest_util(s)
    
    # store valeus in per age time path
    V_term = np.zeros((cfg.n_cah, n_h, cfg.n_p, cfg.n_m))
    C_term = np.zeros((cfg.n_cah, n_h, cfg.n_p, cfg.n_m))
    S_term = np.zeros((cfg.n_cah, n_h, cfg.n_p, cfg.n_m))
    
    
    w_bequest = np.zeros(len(model.sav_grid))
    if cfg.phi_1 != 0:
        
        # This becomes the 'w_curve' for the last period
        for i, s_prime in enumerate(model.sav_grid):
            sav = s_prime * (1 + cfg.r)
            taxable = max(0.0, sav - cfg.chi_b)
            net_beq = sav - (cfg.tau_b * taxable)
            if cfg.rra == 1.0:
                w_bequest[i] = cfg.phi_1 * np.log(net_beq + cfg.phi_2)
            else:
                w_bequest[i] = cfg.phi_1 * ((net_beq + cfg.phi_2)**(1.0 - cfg.rra)) / (1.0 - cfg.rra)
    w_bequest *= cfg.beta
    # Loop through your grids and call your existing m_step
    for i_c in range(cfg.n_cah):
        cah_val = model.cah_grid[i_c]
        
        # mini maximization
        v, s, c = m_step_numba(cah_val, w_bequest, model.sav_grid, cfg.rra, stride=cfg.gss_stride)
        
        # Store in master (broadcasting across health/prod states)
        V_term[i_c, :, :, :] = v
        S_term[i_c, :, :, :] = s
        C_term[i_c, :, :, :] = c

        
    V_term[:, death_state, :, :] = 0.0
    C_term[:, death_state, :, :] = 0.0
   
        
    # Store in master tensors
    V_master[T] = V_term
    C_master[T] = C_term
    S_master[T] = S_term 
    
    Vfun_next = V_master[T]

    # --- BACKWARD INDUCTION LOOP ---
    start_time = time.time()
    
    for age in range(T - 1, -1, -1): # iterate backwards from index 88 to 0
        if verbose and age % 5 == 0:
            print(f"Solving Age: {age}...")
        # BUILD AGE SPECIFIC INPUTS
        inputs = build_age_problem(age, model, Vfun_next)

        #debug_age_inputs(inputs) # -> comment out to print summary about what I put into solve_age_layer()
        Vfun, Pfun, Pfun_c = solve_age_layer(inputs, Vfun_next, verbose = verbose)
        
        # Store directly in memory
        V_master[age] = Vfun
        S_master[age] = Pfun
        C_master[age] = Pfun_c
        
        # Update V_next for the next iteration (age - 1)
        Vfun_next = Vfun
        
    total_time = time.time() - start_time
    
    # --- OPTIONAL DISK I/O --- for even larger problems, if RAM is too small
    if save_to_disk:
        print(f"Saving master policy tensors to {solution_dir}...")
        np.save(os.path.join(solution_dir, "Vfun.npy"), V_master)
        np.save(os.path.join(solution_dir, "Pfun.npy"), S_master)
        np.save(os.path.join(solution_dir, "Cfun.npy"), C_master)
        print(f"Solved Lifecycle in {total_time:.2f} seconds. Data saved in {solution_dir}")
    else:

        print(f"Solved Lifecycle in {total_time:.2f} seconds")
        return PolicyLibrary(savings=S_master, consumption=C_master, value=V_master, cah_grid=model.cah_grid)
    
class AgeInputs(NamedTuple):
    """
    A lightweight container for Numba. 
    Only Arrays, Ints, Floats, and Bools allowed!
    Will directly link model specs and config to throw data into the numba functions
    """
    # --- SCALARS ---
    age: int
    R: float             
    beta_eff: float      
    c_floor: float
    rra: float

    phi_1: float
    phi_2: float
    chi_b: float
    tau_b: float
    # --- GRIDS ---
    sav_grid: np.ndarray 
    cah_grid: np.ndarray 
    gss_stride: int

    # --- TRANSITIONS ---
    # shape here determines the number of state points in the state loop
    # e.g., len(Pi_h) -> n_h so must match structure of income and medical expenses
    # we have however the following critical points
    # no medical shocks -> transitory part should be 1 and income & medical should always have that
    Pi_h: np.ndarray     
    Pi_p: np.ndarray     
    Pi_m: np.ndarray     
    prob_trans_p: np.ndarray 
    prob_trans_m: np.ndarray 
    
    # --- RESOURCES ---
    income_next: np.ndarray  
    med_cost_next: np.ndarray
    
    # --- VALUE FUNCTION ---
    V_next: np.ndarray # -> shape (ncah, nh, nm, np) expected
    
def build_age_problem(age, model, V_next):
    """
    Transforms the current problem into consistent shapes given age.
    Age acts as a hyper state space as it alters the structure of the problem (e.g., transtion matrices, rectangular VF, etc.)    
    Handles phase transitions (Entry to MedEx, Retirement).
    """
    cfg = model.cfg         # config values
    res = model.resources   # income, pension and medical expenses
    trans = model.trans     # transtition probabilities
    
    # --- DEFAULT WITH ALL SHOCKS ON
    Pi_h = trans.h_Q[age]           # health shocks w/ survival probabilities
    Pi_p = trans.lab_p.probs        # Persistent Labor
    Pi_m = trans.med_p.probs        # Persistent MedEx
    
    prob_trans_p = trans.lab_eps.probs # Transitory Labor
    prob_trans_m = trans.med_eps.probs # Transitory MedEx

    # Resources for Next Period (t+1) | WORKER
    if (age + 1) < cfg.medex_age: # medex_age = 30
        # working w/out med shocks!
        inc_slice = res.income_working[age + 1]
        med_slice = np.zeros((len(Pi_h), cfg.n_m, 1)) # quick fix but probably bad!
        # No Med Shocks
        prob_trans_m = np.array([1.0])
        # Medical State is irrelevant, but we need valid indices. 
        # Identity matrix ensures m stays at 0 (if initialized at 0).
        #Pi_m = np.array([1.0])# np.eye(cfg.n_m)
        Pi_m = np.eye(cfg.n_m)
        
    elif (age + 1) < cfg.ret_age: # = 45
        inc_slice = res.income_working[age + 1] # can only index up to 44 (lenghth 45)
        med_slice = res.med_costs[age + 1 - cfg.medex_age]
    else: # retired!
        # no transitory productivity risk
        prob_trans_p = np.array([1.0])
        Pi_p = np.eye(cfg.n_p) # Freeze productivity state

        # Slice: (H, P) -> (H, P, 1) to match prob=[1.0]
        pension = res.income_retired[age + 1 - cfg.ret_age] # normalize for 45 periods of pension income 
        inc_slice = pension[np.newaxis, :, np.newaxis]
        med_slice = res.med_costs[age + 1 - cfg.medex_age] # - 30 to normalize for length of medical expenditures to actual age
        
    # entry int oemdical shocks | intiated using the invariant distribution
    if age == (cfg.medex_age - 1): # 29
        # Use the Entry Matrix (Rows = Invariant Dist)
        # trans.Q_m_init is the Tiled Matrix (N_m x N_m)
        Pi_m = np.tile(model.invariant_Q.m, (cfg.n_m, 1))  # -> important that it is tiled so that in the loop iteration len() extracts still the relevant numbers to iterate on


    return AgeInputs(
        age = age,
        R = cfg.r + 1,
        beta_eff = cfg.beta ,
        c_floor = cfg.c_floor,
        rra = cfg.rra,

        phi_1 = cfg.phi_1,
        phi_2 = cfg.phi_2,
        chi_b = cfg.chi_b,
        tau_b = cfg.tau_b,
        
        sav_grid = model.sav_grid,
        cah_grid = model.cah_grid,
        gss_stride = cfg.gss_stride,
        Pi_h = Pi_h,   
        Pi_p = Pi_p,
        Pi_m = Pi_m,
        
        prob_trans_p = prob_trans_p,
        prob_trans_m = prob_trans_m,
        
        income_next = inc_slice,
        med_cost_next = med_slice,
        
        V_next = V_next
    )
    

#%%
def debug_age_inputs(age_inputs):
    print("\n===== DEBUG AGE INPUTS =====")
    
    for field_name in age_inputs._fields:
        value = getattr(age_inputs, field_name)
        
        if isinstance(value, np.ndarray):
            print(f"{field_name:15s} | ARRAY | shape={value.shape} | dtype={value.dtype}")
        else:
            print(f"{field_name:15s} | SCALAR | value={value}")
    
    print("===== END DEBUG =====\n")

