# -*- coding: utf-8 -*-
"""
ge_solver.py — General equilibrium solver via Brent's method
============================================================

Implements Algorithm 4: finds equilibrium interest rate
r* that clears the capital market given household behavior.

GESolver
--------
Root finding class:

    solve             : Main entry point, returns (r*, A, w)
    objective_function: K_supply/L - K_demand(r)
"""

# iterate over r and K/L normalize A such that w fixed and also L probably normlaized to 1
# idea is to guess K -> determine A such that w = 1 -> determine r -> check aggregate K supply demand match

import numpy as np
import time
from scipy.optimize import brentq
from src.hh_solver import solve_lifecycle
from src.simulation import run_forward_simulation

class GESolver:
    def __init__(self, model, r_init=[0, 0.2]):
        self.model = model
        self.cfg = model.cfg
        self.r_bracket = r_init
        
    def objective_function(self, r_guess):
        """ The gap function: Aggregate Capital Supply - Aggregate Capital Demand """
        self.cfg.r = r_guess
        n_alive = self.model.trans.h_Q.shape[1] - 1
        # ==========================================
        # SOLVING THE FIRM SIDE
        # ==========================================
        alpha, delta = self.cfg.alpha, self.cfg.delta
        
        if self.cfg.belief_mode.value == 0:  # Subjective (Baseline)
            # Set wage to 1, impute A and K_demand
            self.cfg.w = 1.0
            k_demand = alpha / ((1 - alpha) * (r_guess + delta))
            A = 1.0 / ((1 - alpha) * (k_demand**alpha))
        else:  # Objective / NHH (Counterfactuals)
            # Anchor A, compute standard K_demand and endogenous wage
            A = self.cfg.A_anchor
            k_demand = ((alpha * A) / (r_guess + delta)) ** (1.0 / (1.0 - alpha))
            self.cfg.w = (1 - alpha) * A * (k_demand**alpha)
            
        # Update net income resources using the single source of truth for wage
        self.model.get_net_income(self.model.gross_productivity, self.model.pension_gross, self.cfg.w)

        print(f"\nEvaluating r = {r_guess:.4%} | Target K_demand = {k_demand:.4f} | Implied A = {A:.4f}")
        
        # ==========================================
        # SOLVING HOUSEHOLD PROBLEM & SIMULATING FORWARD
        # ==========================================
        policies = solve_lifecycle(self.model, save_to_disk=False) 
        dist_history = run_forward_simulation(self.model, policies)
        
        # ==========================================
        # AGGREGATION & MARKET CLEARING
        # ==========================================
        k_supply = 0.0
        l_supply = 0.0
        total_mass = 0.0
        
        for age in range(self.cfg.n_ages):
            # collect savings supply
            mass_all = dist_history[age, :, :, :, :]
            k_supply += np.dot(np.sum(mass_all, axis=(1, 2, 3)), self.model.sav_grid)
        
            if age < self.cfg.ret_age:
                # if in wroking life collect productivity for normalization
                mass_alive = dist_history[age, :, :n_alive, :, :]
                mass_by_state = np.sum(mass_alive, axis=(0, 3)) # (Health, P)
                gross_productivity = self.model.gross_productivity[age] # (Health, P, Eps_p)
                expected_prod_at_age = np.tensordot(gross_productivity, self.model.trans.lab_eps.probs, axes=([-1], [0]))
                
                # sum up efficiency units of labor supplys
                l_supply += np.sum(mass_by_state * expected_prod_at_age)
                total_mass += np.sum(mass_alive)
        
        # compute savings gap in normalized terms
        normalized_k_supply = k_supply / l_supply
        gap = normalized_k_supply - k_demand
        
        print(f"  -> K_supply: {normalized_k_supply:.4f} | K_demand: {k_demand:.4f} | Gap: {gap:.6f}")
        
        self.last_results = {
            'NormalizedK': normalized_k_supply, 'K': k_demand, 'L': 1.0, 'r': r_guess, 'A': A, 'w': self.cfg.w
        }
        
        return gap

    def solve(self):
        print("="*50)
        start_GE = time.time()
        print("STARTING GENERAL EQUILIBRIUM ROOT-FINDING USING BRENTQ")
        print("="*50)
        
        r_star = brentq(self.objective_function, self.r_bracket[0], self.r_bracket[1], xtol=1e-4)
        
        # SUMMARY
        A_final = self.last_results['A']
        w_final = self.last_results['w']
        Y = A_final * self.last_results['K']**self.cfg.alpha
        
        print("\n" + "!"*50)
        print("GENERAL EQUILIBRIUM SOLUTION FOUND:")
        print(f"Equilibrium Dynamics FOR BELIEF TYPE: {self.cfg.belief_mode.name}:")
        print(f"  Capital (K):   {self.last_results['NormalizedK']:.4f}")
        print(f"  Labor (L):     {self.last_results['L']:.4f}")
        print(f"  TFP (A):       {A_final:.4f}")
        print(f"  Wage (w):      {w_final:.6f}")
        print(f"  Interest (r):  {r_star:.4%}")
        print(f"  Capital/Output (K/Y): {self.last_results['K']/Y:.4f}")
        print("!"*50 + "\n")
        
        ge_time = time.time() - start_GE
        print(f"GE SOLVER TOOK {ge_time:.4f} seconds")
        print("\n" + "!"*50)        
        
        return r_star, A_final, w_final