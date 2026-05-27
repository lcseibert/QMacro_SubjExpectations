# -*- coding: utf-8 -*-
"""
worker_funcs.py — Numba-accelerated computational kernels
=========================================================

Core routines for the household problem and forward simulation,
compiled with Numba for performance.

Household Solver
----------------
e_step_numba      : Compute continuation value W(s') by integrating over future states
m_step_numba      : Maximize U(c) + W(s') via coarse scan + GSS refinement
solve_age_kernel  : Parallel loop over (h,p,m) states for one age
solve_age_layer   : Python wrapper for solve_age_kernel

Forward Simulation
------------------
forward_step_numba   : One-period distribution update with lottery mechanism
forward_life_cycle   : Propagate (h,p) distribution for calibration
compute_income_stats : Cross-sectional median/mean income from mass grid

"""

import numpy as np
from numba import njit, prange, config
from src.numerics import interp_1d_numba, manual_search, evaluate_bellman, golden_section_search

print(f"Numba Threads: {config.NUMBA_NUM_THREADS}")

# --- CONTROL PANEL ---
# because of the compiled style fo numba debugging is really hard. Toggle between states to go in plain python
# reduces speed but allows for debugging statements during the run
# Create a toggle at the top
DEBUG_MODE = False # False for speed mode

if DEBUG_MODE:
    # Dummy decorators that do nothing
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    # Dummy prange
    prange = range
else:
    from numba import njit, prange



@njit(fastmath=True, cache=True)
def e_step_numba(
    s_grid,         # Savings grid
    cah_grid_next,  # Next period's CAH grid
    V_next,         # Next period's Value Function
    probs_h, probs_p, probs_m, # Transition probability vectors (sliced 1d: current state -> trans probs)
    prob_trans_p, prob_trans_m, # Transitory weights
    income_next,    # Future Income Table | correct slcie expected
    med_cost_next,  # Future MedEx Table | correct slcie expected
    c_floor, beta, R, phi_1, phi_2, tau_b, chi_b, # economic parameters from config
    rra
):
    """
    Computes W(s') = beta * E[ V_{t+1}( s'*(1+r) + y' - m' ) ]
    Returns a 1D array aligned with s_grid.
    """
    # build problem skeleton
    n_sav = len(s_grid)
    n_h_next = len(probs_h)
    n_p_next = len(probs_p)
    n_m_next = len(probs_m)
    n_trans_p = len(prob_trans_p)
    n_trans_m = len(prob_trans_m)
    
    # initiate storage
    w_curve = np.zeros(n_sav)
    dead_idx = n_h_next - 1
    has_bequest = (phi_1 != 0.0)
    
    # pre comute returns on savings
    RS = np.empty(n_sav)
    for i_s in range(n_sav):
        RS[i_s] = R * s_grid[i_s]
        
    # LOOP OVER FUTURE STATES
    # loop over health states
    for ih in range(n_h_next): 
        if ih == dead_idx and not has_bequest: # reason is that index 5 is state 6 (or 1) -> death at the time no bequests so no continuation value
            continue
        ph = probs_h[ih]
        
        if ph == 0:
            continue
        
        is_dead = (ih == dead_idx)
        
        # persistent productivity
        for ip in range(n_p_next):
            pp = probs_p[ip]
            if pp == 0: continue
            
            # persistent medical
            for im in range(n_m_next):
                pm = probs_m[im]
                if pm == 0: continue
                
                # transitory productivity
                for it_p in range(n_trans_p):
                    pt_p = prob_trans_p[it_p]
                    
                    # transitory medical
                    for it_m in range(n_trans_m):
                        pt_m = prob_trans_m[it_m]
                    
                        # Joint Probability | everything is independent
                        prob = ph * pp * pm * pt_p * pt_m
                        if prob < 1e-50: # if overall probability to low skip
                            continue
                        # collect income given joint state space point
                        # note that death state experience medical costs while having zero income thus
                        if not is_dead:
                            y_val = income_next[ih, ip, it_p] # Look up income
                        else:
                            y_val = 0.0
                        
                        m_val = med_cost_next[ih, im, it_m] # Look up medex
                        
                        # loop over savings
                        for i_s in range(n_sav):
                            # compute implied next period CAH
                            cah_prime = RS[i_s] + y_val - m_val
                            
                            if is_dead: # then only utility from bequests
                                # no debt bequests
                                estate = max(RS[i_s] - m_val, 0.0)
                                
                                # apply estate taxes
                                taxable = max(0.0, estate - chi_b)
                                net_beq = estate - (tau_b * taxable)
                                
                                # utility
                                if rra == 1.0:
                                    val = phi_1 * np.log(net_beq + phi_2)
                                else:
                                    val = phi_1 * ((net_beq + phi_2)**(1.0 - rra)) / (1.0 - rra)
                                    
                                w_curve[i_s] += prob * val
                            else:
                                
                                # Consumption Floor
                                cah_prime = max(cah_prime, c_floor)
                                
                                # Interpolate V_{t+1} at implied CAH values
                                # V_next shape: (n_cah, n_h, n_p, n_m)
                                v_val = interp_1d_numba(cah_prime, cah_grid_next, V_next[:, ih, ip, im])
                                w_curve[i_s] += prob * v_val
                                
    for i_s in range(n_sav):
        w_curve[i_s] *= beta # return discounted sum
        
    return w_curve # return discounted continuation value



# note that when strides = 0 its just a discrete grid search algorithm 
@njit(fastmath=True, cache=True)
def m_step_numba(cah_current, w_curve, sav_grid, rra, stride=8, tol=1e-7):
    """
    Coarse grid scan (every `stride` points) to locate the bracket,
    then GSS refinement within that bracket.
    Falls back to the coarse grid point if GSS cannot improve on it.
    If stride 0 or 1 then opting to discrete grid brute force method!
    """
    n_sav = len(sav_grid)
    
    if stride <= 1:
        best_val = -1e20
        best_s   = 0.0
        best_c   = 0.0
        for i in range(n_sav):
            s_prime = sav_grid[i]
            if s_prime >= cah_current:
                break
            cons = cah_current - s_prime
            if cons > 1e-6:
                util = np.log(cons) if rra == 1.0 else (cons ** (1.0 - rra)) / (1.0 - rra)
                val  = util + w_curve[i]
                if val > best_val:
                    best_val = val
                    best_s   = s_prime
                    best_c   = cons
        return best_val, best_s, best_c
    
    # --- coarse scan ---
    best_coarse_val = -1e20
    best_dense_idx  = 0
    i = 0
    
    # pre scan of grid
    while i < n_sav:
        s_prime = sav_grid[i]
        if s_prime >= cah_current:
            break
        cons = cah_current - s_prime
        if cons > 1e-6:
            util = np.log(cons) if rra == 1.0 else (cons ** (1.0 - rra)) / (1.0 - rra)
            val  = util + w_curve[i]
            if val > best_coarse_val:
                best_coarse_val = val
                best_dense_idx  = i
        i += stride # stride determines the amount of "empty holes" I leave in the asset savigns grid
        # note that last i must not be the last savigns grid point
        # the idea is that if the last point is not sav[-1] then + stride will by defintion include that point in the bracket
        
    # bracket the current best savings point given the stride points
    # doing this we cover the entire range valeus we "jumped over"
    lo_idx = max(0,          best_dense_idx - stride)
    hi_idx = min(n_sav - 1,  best_dense_idx + stride)
    a = sav_grid[lo_idx]                            # left bracket
    b = min(sav_grid[hi_idx], cah_current - 1e-8)   # right bracket | - 1e-8 to cap solutions with full savings!

    if a >= b:
        # corner solution — return coarse grid best directly -> happens especially for agents at c_floor
        return best_coarse_val, sav_grid[best_dense_idx], cah_current - sav_grid[best_dense_idx]

    # run GSS on brackets
    val, s_star, cons = golden_section_search(
        cah_current, w_curve, sav_grid, rra, a, b, tol
    )

    # robustness check
    if best_coarse_val > val:
        return best_coarse_val, sav_grid[best_dense_idx], cah_current - sav_grid[best_dense_idx]
    return val, s_star, cons

# outdated fucntion I embed it in the gss version to carry both ideas in one fucntion
# @njit(fastmath=True, cache=True)
# def m_step_numba(cah_current, w_curve, s_grid, rra):
#     best_val = -1e20
#     best_s   = 0.0
#     best_c   = 0.0
#     n_sav    = len(s_grid)

#     # loop over savings
#     for i in range(n_sav):
#         s_prime = s_grid[i]
#         if s_prime >= cah_current: # cannot safe more than you have
#             break
#         # evaluate bellman at the point
#         current_VF, curr_c = evaluate_bellman(s_prime, cah_current, w_curve, s_grid, rra)
        
#         #collect maximizing sav
#         if current_VF > best_val:
#             best_val = current_VF
#             best_s   = s_prime
#             best_c   = curr_c

#     return best_val, best_s, best_c


@njit(parallel=True, fastmath=True, cache=True)
def solve_age_kernel(
    cah_grid, sav_grid, gss_stride,       # Grids
    V_next,                     # Future Value (4D) no age just (x, h, p, m)
    Pi_h, Pi_p, Pi_m,           # Transition Matrices
    prob_trans_p, prob_trans_m, # Transitory Weights
    income_next, med_cost_next, # pre-computed income/expenses
    c_floor, beta_eff, R, rra, phi_1, phi_2, tau_b, chi_b # economic parameters
):
    """
    This function takes the current problem defined by age.
    It iterates in parallel over the exogenous state space as described in the pseudocode and calls E and M Step
    """
    n_cah = len(cah_grid)
    n_h = len(Pi_h)
    n_p = len(Pi_p)
    n_m = len(Pi_m)

    # Pre-allocate output arrays
    V_curr = np.zeros((n_cah, n_h, n_p, n_m))
    Pol_sav = np.zeros((n_cah, n_h, n_p, n_m))
    Pol_cons = np.zeros((n_cah, n_h, n_p, n_m))

    # --- PARALLEL LOOP OVER STATES
    # embarrassingly parallel since states are independent of today's income (we track via CAH) and of the transition
    for h in prange(n_h - 1): # since dead state which means that savings, consumption and value function is zero
        for p in prange(n_p):
            for m in prange(n_m):

                # E-STEP -> EXPECTED CONTINUATION VALUE GIVEN STATE SPACE TODAY
                w_curve = e_step_numba(
                    sav_grid, cah_grid, V_next,
                    Pi_h[h], Pi_p[p], Pi_m[m],
                    prob_trans_p, prob_trans_m,
                    income_next, med_cost_next,
                    c_floor, beta_eff, R, phi_1, phi_2, tau_b, chi_b,
                    rra
                )
                
                # M-STEP -> MAXIMIZE 1D PROBLEM: max U(X-S) + BETA*PI*V(S)
                for i_c in range(n_cah):
                    cah_val = cah_grid[i_c]
                    if cah_val <= c_floor:
                        best_s = 0.0
                        best_c = c_floor
                        # Utility at the floor + continuation value of 0 savings -> e.g., government stepping in to guarantee c value
                        if rra == 1.0:
                            util = np.log(c_floor)
                        else:
                            util = (c_floor**(1.0 - rra)) / (1.0 - rra)
                        best_val = util + w_curve[0] # util + expected continuation value
                    else:
                            best_val, best_s, best_c = m_step_numba(
                                cah_val, w_curve, sav_grid, rra, gss_stride
                            )

                    # Store results in the 4D arrays
                    V_curr[i_c, h, p, m]   = best_val
                    Pol_sav[i_c, h, p, m]  = best_s
                    Pol_cons[i_c, h, p, m] = best_c
                    
    return V_curr, Pol_sav, Pol_cons


# Python Wrapper
def solve_age_layer(inputs, V_next, verbose = True):
    """
    Python wrapper that unpacks the 'inputs' class and calls the Numba kernel.
    """
    # optional debugging point
    if verbose:
        debug_age_inputs(inputs)
        
             
    return solve_age_kernel(
        inputs.cah_grid, 
        inputs.sav_grid,
        inputs.gss_stride,
        V_next,
        inputs.Pi_h, 
        inputs.Pi_p, 
        inputs.Pi_m,
        inputs.prob_trans_p, 
        inputs.prob_trans_m,
        inputs.income_next, 
        inputs.med_cost_next,
        inputs.c_floor, 
        inputs.beta_eff, 
        inputs.R, 
        inputs.rra,
        inputs.phi_1,
        inputs.phi_2,
        inputs.tau_b,
        inputs.chi_b
    )

#optional debugging step that tracks the entire inputs passed towards the numba solver
def debug_age_inputs(age_inputs):
    print("\n===== DEBUG AGE INPUTS =====")
    
    for field_name in age_inputs._fields:
        value = getattr(age_inputs, field_name)
        
        if isinstance(value, np.ndarray):
            print(f"{field_name:15s} | ARRAY | shape={value.shape} | dtype={value.dtype}")
        else:
            print(f"{field_name:15s} | SCALAR | value={value}")
    
    print("===== END DEBUG =====\n")
    
    
# ---------------------------------------
# FORWARD SIMULATION
# ---------------------------------------

@njit(fastmath=True, cache=True)
def forward_step_numba(
    dist_curr, policy_sav, sav_grid, cah_grid, inc_grid, med_grid, 
    Pi_h, Pi_p, Pi_m, prob_eps_p, prob_eps_m, r
):
    """
    This function defines the mapping Gamma_t -> Gamma_{t+1} described in the pseudocode. 
    Computes cah and moves mass forward give nthe discussed lottery system to bring back savings back on grid.
    """
    nA, nH, nP, nM = dist_curr.shape
    n_eps_p = len(prob_eps_p)
    n_eps_m = len(prob_eps_m)
    
    dist_next = np.zeros_like(dist_curr)
    step_max_cah = 0.0
    # all variables of interest follow a clear markov transition path except for 
    # savings, iid transitory shocks
    # because savings is a state variable in the mass -> lottery to bring pack to grid
    # iid shocks can be integrated out
    # thus create points to integrate over
    max_dest = n_eps_p * n_eps_m * 2  # times 2 because of the lottery -> high / low grid points 
    
    # initialize memory storage
    dest_indices = np.zeros(max_dest, dtype=np.int32)
    dest_weights = np.zeros(max_dest, dtype=np.float64)

    # iterate over savings grid
    for ia in range(nA):
        a_val = sav_grid[ia]
        
        #iterate over states space given age -> (h, p, m): income and med shocks independent of x -> get x with resosurces!
        for ih in range(nH - 1):
            for ip in range(nP):
                for im in range(nM):
                    
                    mass = dist_curr[ia, ih, ip, im] # how probable to be in that state
                    if mass < 1e-24: continue
                    
                    # given state mass compute x given transitory shocks
                    counter = 0 # counting dummy for asset grids
                    # iterate over transitory iid shocks
                    for ie_p in range(n_eps_p):
                        p_ep = prob_eps_p[ie_p]
                        for ie_m in range(n_eps_m):
                            p_em = prob_eps_m[ie_m]
                            
                            # probability mass spreaded by iid shocks
                            sub_mass = mass * p_ep * p_em # -> mass at each iid instance given today
                            
                            # compute x
                            y = inc_grid[min(ih, inc_grid.shape[0]-1), ip, ie_p]
                            m_cost = med_grid[ih, im, ie_m]
                            cah = (1 + r) * a_val + y - m_cost
                        
                            # track maximum cah observed to verify grid bounds
                            if cah > step_max_cah: step_max_cah = cah 
                            
                            # INTERPOLATION KEY STEP | Given policy what is the savings part to x?
                            pol_vector = policy_sav[:, ih, ip, im]
                            s_prime = interp_1d_numba(cah, cah_grid, pol_vector)
                            
                            # Bound check
                            if s_prime < sav_grid[0]: s_prime = sav_grid[0]
                            if s_prime > sav_grid[-1]: s_prime = sav_grid[-1]
                            
                            # apply lottery idea here: split probabilities to adjacent grid points weighted by linear distance
                            # find grid points
                            idx = manual_search(sav_grid, s_prime)
                            
                            s_low  = sav_grid[idx] # lower grid point
                            s_high = sav_grid[idx + 1] # higher grid point
                            
                            # Weights for linear interpolation
                            w_high = (s_prime - s_low) / (s_high - s_low)
                            w_low  = 1.0 - w_high
                            
                            # store weights and indices
                            dest_indices[counter] = idx
                            dest_weights[counter] = sub_mass * w_low
                            counter += 1
                            
                            dest_indices[counter] = idx + 1
                            dest_weights[counter] = sub_mass * w_high
                            counter += 1
                            # counter: uneven -> iterating over low shocks with (ia * 2) + 1
                            # counter: even -> iterating over higher indices with ((ia + 1) * 2)
                            # all in all counter just stores the asset grid points that we assign probabilities to
                            # overall we get the next states asset savings given the probabbility mass today + iid shocks
                    
                    # MAP MASS INTO NEXT PERIOD GIVEN MARKOV TRANSITION
                    for ih_next in range(nH):
                        prob_h = Pi_h[ih, ih_next]
                        if prob_h == 0: continue
                        
                        for ip_next in range(nP):
                            prob_p = Pi_p[ip, ip_next]
                            if prob_p == 0: continue
                            
                            for im_next in range(nM):
                                prob_m = Pi_m[im, im_next]
                                if prob_m == 0: continue
                                # get joint probability
                                prob_total = prob_h * prob_p * prob_m
                                
                                # adjust the joint with the pre-computed weights from the iid shocks and asset lottery
                                for k in range(counter):
                                    idx_dest = dest_indices[k]
                                    mass_to_move = dest_weights[k]
                                    dist_next[idx_dest, ih_next, ip_next, im_next] += mass_to_move * prob_total

    return dist_next, step_max_cah


@njit(fastmath=True, cache=True)
def forward_life_cycle(initial_dist, Pi_h, Pi_p, n_working_ages):
    """
    initial_dist: (nH-1, nP) Newborns only exist in living states
    Pi_h: (n_working_ages, nH, nH) Health transitions, last index is 'Dead'
    Pi_p: (nP, nP) Persistent productivity transitions
    """
    nH = Pi_h.shape[1]
    nP = Pi_p.shape[0]
    
    mass_grid = np.zeros((n_working_ages, nH, nP))
    
    # Initialize Age 0
    for ih in range(nH - 1):
        for ip in range(nP):
            mass_grid[0, ih, ip] = initial_dist[ih, ip]
            
    # Iterate forward
    for age in range(n_working_ages - 1):
        # push mass only from living states. Accidental bequests fully collected by the Government
        for ih in range(nH - 1): 
            for ip in range(nP):
                mass = mass_grid[age, ih, ip]
                if mass < 1e-12: continue
                
                # We push mass TO all states, including the dead state (ih_next = nH - 1)
                for ih_next in range(nH):
                    prob_h = Pi_h[age, ih, ih_next]
                    if prob_h == 0.0: continue
                    
                    for ip_next in range(nP):
                        prob_p = Pi_p[ip, ip_next]
                        if prob_p == 0.0: continue
                        
                        mass_grid[age + 1, ih_next, ip_next] += mass * prob_h * prob_p
                        
    return mass_grid

@njit(fastmath=True, cache=True)
def compute_income_stats(mass_grid, inc_grid, prob_eps_p):
    """
    Computes cross-sectional median income, mean income, and total labor supply.
    """
    n_ages, nH, nP = mass_grid.shape
    n_eps_p = len(prob_eps_p)
    
    # Pre-allocate arrays for maximum possible living states
    total_living_states = n_ages * (nH - 1) * nP * n_eps_p
    y_flat = np.zeros(total_living_states)
    weight_flat = np.zeros(total_living_states)
    
    counter = 0
    total_living_mass = 0.0
    l_supply = 0.0
    
    # iterate over the life cycle ages
    for age in range(n_ages):
        for ih in range(nH - 1): # Exclude dead state
            for ip in range(nP):
                base_mass = mass_grid[age, ih, ip]
                
                # Skip zero-mass states
                if base_mass < 1e-16: continue
                
                for ie_p in range(n_eps_p):
                    trans_prob = prob_eps_p[ie_p]
                    final_mass = base_mass * trans_prob
                    
                    val_y = inc_grid[age, ih, ip, ie_p]
                    
                    # Store values in flat arrays
                    y_flat[counter] = val_y
                    weight_flat[counter] = final_mass
                    
                    # Accumulate totals
                    total_living_mass += final_mass
                    l_supply += final_mass * val_y
                    counter += 1
                        
    # compute average
    average_y = 0.0
    for i in range(counter):
        weight_flat[i] /= total_living_mass
        average_y += y_flat[i] * weight_flat[i]
        
    # compute weighted median
    y_pop = y_flat[:counter]
    w_pop = weight_flat[:counter]
    
    # Sort for CDF calculation
    sort_idx = np.argsort(y_pop)
    y_sorted = y_pop[sort_idx]
    w_sorted = w_pop[sort_idx]
    
    cdf = 0.0
    median_y = 0.0
    # iterate over grid until mass is greater than 0.5: Might want to consider interpolation betwen grid and grid -1 here
    for i in range(counter):
        cdf += w_sorted[i]
        if cdf >= 0.5:
            median_y = y_sorted[i]
            break
            
    return median_y, average_y, l_supply