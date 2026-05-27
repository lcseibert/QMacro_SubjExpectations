# -*- coding: utf-8 -*-
"""
primitives.py — Model construction and precomputation
=====================================================

This module builds the complete model environment from configuration,
including shock discretization, lifecycle income/expenditure grids,
and state space construction. 

Data Structures
---------------
ShockStruct           : Container for shock grid and transition probabilities
TransitionData        : Warehouse for all stochastic elements (labor, medical, health)
InvariantDistributions: Ergodic distributions for exogenous Markov processes
LifecycleResources    : Net income and medical cost grids by age for hh_solver

Main Class
----------
ModelPrimitives       : Constructs full model environment from ModelSpecs config
    - Discretizes persistent/transitory shocks (Rouwenhorst, Tauchen)
    - Loads health transition matrices by belief mode (subjective/objective/average)
    - Builds lifecycle income tables (working + retirement)
    - Builds medical expenditure tables
    - Constructs savings and cash-on-hand grids
    - Handles calibration anchoring across economies (SSH → OSH/NHH)

Helper Functions
----------------
prepare_med_table       : Build medical cost grid from econometric estimates
prepare_incomew_table   : Build working-age productivity grid
prepare_incomer_table   : Build retirement pension grid with SS bend points
regressive_rr           : US Social Security replacement rate formula
compute_ret_health_dist : Compute health distribution at retirement entry
get_inv_health          : Quasi-stationary health distribution (excluding death)
"""

import src.numerics as numerics
from dataclasses import dataclass
import numpy as np
from src.simulation import lifecycle_simulation
from config import BeliefType


# ==============================================================================
# MODEL STRUCTURE
# ==============================================================================
@dataclass
class ShockStruct:
    """Generic container for a shock process"""
    grid: np.ndarray          # The values (e.g. -0.2, 0.0, 0.2)
    probs: np.ndarray         # The probabilities (Matrix if persistent, Weights if i.i.d.)

@dataclass
class TransitionData:
    """
    The Central Warehouse for all stochastic elements.
    """
    # Labor Shocks
    lab_p: ShockStruct        # Persistent (Grid + Q Matrix)
    lab_eps: ShockStruct      # Transitory (Grid + Weights)
    
    # Medical Shocks
    med_p: ShockStruct        # Persistent (Grid + Q Matrix)
    med_eps: ShockStruct      # Transitory (Grid + Weights)
    
    # Health State
    # Only needs Matrix (Age-dependent), grid is just integer 0,1
    h_Q: np.ndarray      # Shape (Age, H, H')
    h_Q_forward: np.ndarray # for the forward simulation only use objective or no health beliefs
    
@dataclass
class InvariantDistributions:
    """
    Computes and stores the ergodic (stationary) distributions 
    for the exogenous Markov processes.
    """
    p: np.array  # Productivity Invariant Distribution
    m: np.array  # Medical Shocks Invariant Distribution
    h_0: np.array # Invariant of first transition matrix for forward simulation
    
@dataclass
class LifecycleResources:
    """
    Holds the pre-computed income and medical grids for the entire lifecycle.
    We split them by logic, but they reside here.
    """
    # Labor Income (Age 20-64): (n_work_years, n_h, n_p, n_eps)
    income_working: np.ndarray 
    
    # Pension Income (Age 65+): (n_ret_years, n_h, n_p)
    # Note: Pension usually depends on P (proxy for lifetime earnings) but not eps.
    income_retired: np.ndarray
    
    # Medical Costs (Age 20-90): (n_total_years, n_h, n_m, n_m_eps)
    med_costs: np.ndarray

# ==============================================================================
# PRECOMPUTATION | SETTING UP THE MODEL
# ==============================================================================
class ModelPrimitives:
    def __init__(self, config):
        # get config data
        self.cfg = config
        # discretize shocks and get transsition matrices
        self.trans = self._build_transitions()
        # construct pre-computed income & expenditures
        self.resources = self._build_life_finances()
        # create savings and cah grids
        self._build_grids()
        
        # set up inital net earnings!
        # if PE we can alter the wage to shift income | baseline is w = 1.0
        w = getattr(self.cfg, 'w', 1.0)
        self.get_net_income(self.gross_productivity, self.pension_gross, w)
        
        
        
        # summary print
        print("\n" + "="*50)
        print("  Discretization Summary")
        print("="*50)
        print(f"Total Life Cycle Length: {self.cfg.n_ages}")
        print(f"Working income shape:      {self.resources.income_working.shape} (w_age, health, p, transitory p)")
        print(f"Retirement income shape:   {self.resources.income_retired.shape}       (ret_age, p)")
        print(f"Medical costs shape:       {self.resources.med_costs.shape} (medex_age, health, m, transitory m)")
        print("\nShock Structure:")
        print(f"  Persistent medical states:      {len(self.trans.med_p.grid)}")
        print(f"  Transitory medical states:      {len(self.trans.med_eps.grid)}")
        print(f"  Persistent productivity states: {len(self.trans.lab_p.grid)}")
        print(f"  Transitory productivity states: {len(self.trans.lab_eps.grid)}")
        if self.cfg.phi_1 == 0:
            bequests_motive_statement = "No Bequest Motive"
        else:
            bequests_motive_statement = "With Bequest Motive"
        if self.cfg.belief_mode == BeliefType.subjective:
            print(f"\n  Beliefs: SUBJECTIVE | {bequests_motive_statement}")
        elif self.cfg.belief_mode == BeliefType.objective:
            print(f"\n  Beliefs: OBJECTIVE | {bequests_motive_statement}")
        else:
            print(f"\n  Beliefs: No Health States | {bequests_motive_statement}")

        print("="*50 + "\n")
        
    def _build_grids(self):
        """Construct the state space grids."""
        self.sav_grid = numerics.power_grid(
            b = self.cfg.sav_min, 
            a_max = self.cfg.sav_max, 
            na = self.cfg.n_assets, 
            zeta = 0.4
        )
        
        # Cash-at-Hand Grid (State variable x)
        # Usually wider than a_grid to catch high income realizations
        # note that if started at zero and zeta low -> massively many points in the c_floor!
        # thus restrict x [c_floor, x_max] and append zero at the beginning
        base_grid = numerics.power_grid(
            b = -self.cfg.c_floor, 
            a_max = self.cfg.cah_max,
            na = self.cfg.n_cah - 1,
            zeta = 0.6
        )
        self.cah_grid = np.append([0.0], base_grid)
        print(f"Max CAH and Sav: {self.cah_grid.max()}, {self.sav_grid.max()}")
        
    # ==============================================================================
    # DISCRETIZING SHOCKS
    # ==============================================================================
    def _build_transitions(self):
        c = self.cfg
        # --- LABOR (Persistent)
        uncond_var_p = c.sigma_kappa**2 / (1 - (c.rho_p**2))
        mu_p = - uncond_var_p / 2
        p_grid, Q_p = numerics.rouwenhorst(c.rho_p, c.sigma_kappa, mu=mu_p, N=c.n_p)
        # p_grid_log  = np.exp(p_grid)
        lab_p = ShockStruct(grid=p_grid, probs=Q_p)
        
        # --- LABOR (Transitory)
        mu_eps = - c.sigma_eps**2 / 2
        eps_grid, P_p = numerics.tauchen(0, mu_eps, c.sigma_eps, c.n_eta_p)
        # eps_grid_log = np.exp(eps_grid)
        eps_p = ShockStruct(grid=eps_grid, probs=P_p)
        
        # --- MEDICAL (Persistent)
        m_grid, Q_m = numerics.rouwenhorst(c.rho_m, c.sigma_zeta,mu=0, N=c.n_m)
        med_p = ShockStruct(grid=m_grid, probs=Q_m)
        
        # --- MEDICAL (Transitory)
        eps_grid_m, P_m = numerics.tauchen(0, 0, c.sigma_v, c.n_eta_m)
        eps_m = ShockStruct(grid=eps_grid_m, probs=P_m)
        
        # --- HEALTH (age, health, health)
        Q_h, Q_h_forw = self._build_health_trans()
        
        # --- initial medical shock period
        Q_m_inv = numerics.stationary_distribution(Q_m)
        # Pi_m = np.tile(Q_m_inv, (c.n_m, 1))
        Q_p_inv = numerics.stationary_distribution(Q_p)
        
        h_0_inv = get_inv_health(Q_h_forw)

        
        self.invariant_Q = InvariantDistributions(
                                                 p = Q_p_inv,
                                                 m = Q_m_inv,
                                                 h_0 = h_0_inv
                                                     )
        
        # since tauchen generates a symmetric Q matrix -> collapse to 1d for iid weights
        if eps_p.probs.ndim == 2:
            eps_p.probs = eps_p.probs[0, :]  # Turn (3,3) into (3,)
            
        if eps_m.probs.ndim == 2:
            eps_m.probs = eps_m.probs[0, :]  # Turn (1,1) into (1,)

        return TransitionData(
            lab_p=lab_p,
            lab_eps=eps_p,
            med_p=med_p,
            med_eps=eps_m,
            h_Q=Q_h,
            h_Q_forward=Q_h_forw
        )
    
    # load the used health beliefs into the model
    def _build_health_trans(self):
        obj_raw = np.load("data/surv_probs.npz")
        
        # note that forward simulation is always based on the objective beliefs!
        if self.cfg.belief_mode == 0: # "subjective"
            return obj_raw["trans_subj"], obj_raw["trans_obj"]
            
        elif self.cfg.belief_mode == 1: # "objective"
            return obj_raw["trans_obj"], obj_raw["trans_obj"]
            
        elif self.cfg.belief_mode == 2: # "average"
            return obj_raw["trans_avg"], obj_raw["trans_avg"]
    
    def _build_life_finances(self):
        """
        Calculates the actual dollar values for Income and MedEx 
        for every possible state in the lifecycle.
        """
        is_average = (self.cfg.belief_mode == 2) 
        h_dist_last_year = compute_ret_health_dist(
                    self.invariant_Q.h_0, 
                    self.trans.h_Q_forward,
                    self.cfg.ret_age
                )
        
        med_costs = prepare_med_table(
            eta_grid = self.trans.med_p.grid,   # Persistent grid
            trans_grid = self.trans.med_eps.grid,  # Transitory grid
            is_average = is_average
        )
        
        # 2. Labor Income (Working Years)
        # Shape: (N_work, H, P_persist, P_trans)
        gross_productivity, omega_ret = prepare_incomew_table(
            p_grid = self.trans.lab_p.grid,   # Persistent grid (log)
            eps_grid = self.trans.lab_eps.grid,  # Transitory grid (log)
            h_weights = h_dist_last_year,
            is_average = is_average
        )

        # Calibration vs. Anchoring
        if not self.cfg.is_anchored:
            # --- OSH BASELINE (Endogenous Calibration) ---
            y_med, y_avg, l_supply = lifecycle_simulation(self, gross_productivity[:self.cfg.ret_age])
            self.cfg.c_floor = 0.05 # fixed to 0.05 in average worker income scale
            self.cfg.p_1_star, self.cfg.p_2_star, self.cfg.p_max_star = self.get_bendpoints(y_med, omega_ret)
            # Store these in the config so they can be injected into SSH/NHH later
            self.cfg.anchor_y_med = y_med
            self.cfg.anchor_y_avg = y_avg
            self.cfg.anchor_omega_ret = omega_ret
            self.cfg.anchor_l_supply = l_supply
            print(f"Baseline Consumption floor set at {self.cfg.c_floor}")
            
        else:
            # --- SSH / NHH COUNTERFACTUALS (Use Frozen Baseline) ---
            print("Using injected baseline parameters. Skipping endogenous simulation.")
            y_med = self.cfg.anchor_y_med
            y_avg = self.cfg.anchor_y_avg
            omega_ret = self.cfg.anchor_omega_ret

        
        # Pension Income
        pension_gross = prepare_incomer_table(
            omega_ret=omega_ret, 
            p_grid_vals=self.trans.lab_p.grid,   
            cfg=self.cfg
        )
        
        # parameter for tax system 
        self.y_max = (self.cfg.e_max / self.cfg.e_med) * y_avg
        # scaling into model units
        self.gross_productivity =  gross_productivity / y_avg
        self.pension_gross = pension_gross / y_avg
        self.med_costs = med_costs / y_avg

        print("Model median simulated income:", y_med)
        print("Model mean simulated income:", y_avg)
    
        return LifecycleResources(gross_productivity, pension_gross, med_costs)
    
    def get_net_income(self,income_work, income_ret, w):
        """
        Updates net income during GE loop to adjust for the wage value. Inititated in primitves with w=1 for PE benchmark.
        Stores values in LifecycleResources container that is used in the backward and forward code.
        """
        income_work_ss = income_work - self.Tax_ss(income_work, self.y_max)
        income_work_net = w*income_work_ss - self.Tax_y(income_work_ss*w)

        income_ret_net = w*income_ret - self.Tax_y(w*income_ret)
        self.resources = LifecycleResources(income_work_net, income_ret_net, self.med_costs)
        return None
        
    def get_bendpoints(self, y_med, omega_ret):
        """
        Mimics pension system of the paper
        """
        scaling = y_med / self.cfg.e_med
        p1_star = (self.cfg.b_1 * scaling) / omega_ret
        p2_star = (self.cfg.b_2 * scaling) / omega_ret
        p_max_star = (self.cfg.e_max * scaling) / omega_ret
        return p1_star, p2_star, p_max_star
    
    def Tax_y(self, income):
        return income - self.cfg.lam * income**(1-self.cfg.tau)
    
    def Tax_ss(self, income, y_cap):
        
        return self.cfg.tau_ss * np.minimum(income, y_cap)
    
# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def prepare_med_table(eta_grid, trans_grid, is_average = False):
        """
        Pre-computes the grid of MEDICAL COST LEVELS (in $).
        Returns shape: (n_ages, n_health, n_persistent, n_transitory)
        """
        med_raw = np.load("data/medex_params.npz")
        
        if is_average: # average mu and sigma
            mu = med_raw['mu_avg'][:, :, np.newaxis, np.newaxis]
            sigma = med_raw['sigma_avg'][:, :, np.newaxis, np.newaxis]
        else: # for both subjective and objective beliefs
            mu = med_raw['mu'][:, :, np.newaxis, np.newaxis]
            sigma = med_raw['sigma'][:, :, np.newaxis, np.newaxis]
            
        # (1, 1, n_eta_grid (persistent), 1) 
        eta_4d = eta_grid[np.newaxis, np.newaxis, :, np.newaxis]
        # (1, 1, 1, n_eta (transitory))
        trans_4d = trans_grid[np.newaxis, np.newaxis, np.newaxis, :]

        # Calculate levels: med = exp(mu + sigma * (persistent + transitory))
        log_med = mu + sigma * (trans_4d + eta_4d)
        
        med_levels = np.exp(log_med) # scale back into thousands of dollars

        return med_levels / 1_000 # back into dollar units 4-D !


def prepare_incomer_table(omega_ret, p_grid_vals, cfg):
    """
    Computes pension income. 
    omega_ret: Scalar average income at the last working age.
    p_grid_vals: 1D array of the persistent productivity states (e.g., np.exp(p_grid)).
    """
    
    # Apply the Regressive Pension Formula
    pension_table = np.array([
        regressive_rr(
            p=p_state,  
            rho_1=cfg.rho_1,
            rho_2=cfg.rho_2,
            rho_3=cfg.rho_3,
            p_1_star=cfg.p_1_star,       
            p_2_star=cfg.p_2_star,       
            p_max_star=cfg.p_max_star    
        )
        for p_state in np.exp(p_grid_vals)
    ])  # shape (n_p,)
    pension_table = pension_table * omega_ret * 1.0
    # Broadcast to the number of retirement years
    n_ret_years = cfg.max_age - cfg.ret_age + 1
    pension_table_full = np.tile(pension_table, (n_ret_years, 1))  # (n_ret_years, n_p)

    return pension_table_full
    

def prepare_incomew_table(p_grid, eps_grid, h_weights, is_average = False):
    """
    Returns shape: (n_ages, n_health, n_persistent_labor, n_transitory_labor)
    """
    omega_ht = np.load('data/earn_profile.npz')
    
    if is_average:
        # omega is (45, 5) -> (45, 1, 1, 1)
        omega_ret_raw = omega_ht["omega_avg"][-1,:]
        omega_4d = omega_ht["omega_avg"][:, :, np.newaxis, np.newaxis]
    else:
        omega_ret_raw = omega_ht["omega"][-1,:]
        omega_4d = omega_ht["omega"][:, :, np.newaxis, np.newaxis]
        
    omega_ret = np.dot(h_weights, omega_ret_raw)
    # Persistent p_grid: (1, 1, n_p, 1)
    p_4d = p_grid[np.newaxis, np.newaxis, :, np.newaxis]
    
    # Transitory eps_grid: (1, 1, 1, n_eps)
    eps_4d = eps_grid[np.newaxis, np.newaxis, np.newaxis, :]
    
    # working life productivity
    y_wor_prod =  omega_4d * np.exp(p_4d + eps_4d) # scale dollar terms

    return y_wor_prod, omega_ret# 4d


def regressive_rr(p, rho_1, rho_2, rho_3, p_1_star, p_2_star, p_max_star):
    if p <= p_1_star:
        return rho_1 * p
    elif (p_1_star <= p ) and (p <= p_2_star):
        return rho_1 * p_1_star + rho_2 * (p-p_1_star)
    else:
        min_star = min(p_max_star, p)
        return rho_1 * p_1_star + rho_2 * (p_2_star - p_1_star) + rho_3 * (min_star - p_2_star)
    
def compute_ret_health_dist(h_0, h_Q_forward, n_work_years):
    """
    Compute the invariant distribution of health at pre-retirement by iterating forward and then constructing the invaraitn distribution.
    h_0: Initial health distribution at age 20 (1D array)
    h_Q_forward: Objective health transition matrices, shape (Total_Ages, H_all, H_all)
    n_work_years: Number of working years (e.g., 45 for ages 20-64)
    """
    n_h_alive = len(h_0)
    current_dist = np.copy(h_0)
    
    # Evolve the distribution forward from Model Age 0 to Model Age (n_work_years - 1)
    # We apply the transition matrices for t = 0 up to t = n_work_years - 2.
    for t in range(n_work_years - 1):
        Q_alive = h_Q_forward[t][:n_h_alive, :n_h_alive]
        current_dist = current_dist @ Q_alive
        
        # Normalize to condition on survival
        survival_mass = np.sum(current_dist)
        if survival_mass > 0:
            current_dist = current_dist / survival_mass
            
    return current_dist # This is pi_{h, 64}



def get_inv_health(Q_h):
    """
    Compute ergodic health distribution by iterating forward on health transitions
    """
    n_alive = Q_h.shape[1] - 1
    # quasi stationary without dead state
    Q_alive = Q_h[0][:n_alive,:n_alive].copy()
    # rescale probs so that rows sum up to 1 (eigenvalue 1 exists)
    row_sums = Q_alive.sum(axis=1)
    Q_alive = Q_alive / row_sums[:, np.newaxis]

    return numerics.stationary_distribution(Q_alive)