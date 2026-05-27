# -*- coding: utf-8 -*-
"""
build_data.py — Raw data processing and model input construction
================================================================

Converts raw econometric estimates from Foltyn & Olsson (2024) into
compressed .npz files used by ModelPrimitives.

Outputs (saved to data/)
------------------------
surv_probs.npz   : Health transition matrices (objective/subjective/average)
medex_params.npz : Medical expenditure moments (μ, σ) by age and health
earn_profile.npz : Age-health earnings profiles (ω)

Functions
---------
build_model_data   : Process survival/health transitions
build_medex_npz    : Process medical expenditure parameters
build_health_income: Process earnings profiles
evolve_demographics: Forward-simulate health distribution for weighting

Usage
-----
Run as script to rebuild all data files:
    python build_data.py
"""

import numpy as np
from pathlib import Path
from src.numerics import stationary_distribution
from config import ModelSpecs

# =================================================================
# Transport Data from Econometric Estimation into model files
# =================================================================

# --- Path Setup ---
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data"

def build_model_data(mu_h_90):
    # Load the Transition Data
    obj_path = RAW_DIR / "health_surv_prob_obj.txt"
    subj_path = RAW_DIR / "health_surv_prob_subj.txt"
    
    # Load and reshape to (Ages, Health_today, Health_tomorrow)
    # 3240 elements -> 540 rows of 6 -> (90, 6, 6)
    trans_obj = np.loadtxt(obj_path).reshape((90, 6, 6))
    trans_subj = np.loadtxt(subj_path).reshape((90, 6, 6))
    trans_avg = np.zeros((90, 2, 2))
    for t in range(90):
        # Column index 5 is the probability of death for the 5 alive states
        mortality_probs = trans_obj[t, :5, 5]
        
        # Weighted mean using the demographics for age t
        avg_mortality = np.sum(mortality_probs * mu_h_90[t])
        
        trans_avg[t, 0, 0] = 1.0 - avg_mortality  # Average Alive -> Average Alive
        trans_avg[t, 0, 1] = avg_mortality        # Average Alive -> Dead
        trans_avg[t, 1, 0] = 0.0                  # Dead -> Average Alive
        trans_avg[t, 1, 1] = 1.0                  # Dead -> Dead (Absorbing)
        
    # Save as a single archive
    np.savez_compressed(
        OUT_DIR / "surv_probs.npz",
        trans_obj=trans_obj,
        trans_subj=trans_subj,
        trans_avg=trans_avg,
        age_start=20,
        n_ages=90
    )
    
    print(f"Build Complete! Saved to {OUT_DIR / 'surv_probs.npz'}")
    print(f"Dimensions: {trans_obj.shape} (Ages, Health_Today, Health_Tomorrow)")
    return trans_avg

def load_and_clean(filename):
    """Reads file, removes Fortran & and commas, returns flat float list"""
    with open(filename, 'r') as f:
        content = f.read()
    clean = content.replace('&', ' ').replace(',', ' ')
    return [float(x) for x in clean.split()]

def build_medex_npz(mu_h_60):
    cfg = ModelSpecs()
    # 1. Load raw data
    mu_surv_raw = load_and_clean("data/raw/medex_mu.txt")
    sig_surv_raw = load_and_clean("data/raw/medex_sigma.txt")
    mu_dead_raw = load_and_clean("data/raw/medex_mu_death.txt")
    sig_dead_raw = load_and_clean("data/raw/medex_sigma_death.txt")
    
    
    # 2. Reshape Survivors (5 states x 60 ages)
    # Using 'F' order is crucial here to map (Health, Age) correctly
    mu_s = np.array(mu_surv_raw).reshape((5, 60), order='F').T
    sig_s = np.array(sig_surv_raw).reshape((5, 60), order='F').T
    
    # 3. Reshape Dead (60 ages x 1 state)
    mu_d = np.array(mu_dead_raw).reshape(60, 1)
    sig_d = np.array(sig_dead_raw).reshape(60, 1)
    
    # 4. Combine into (60, 6) matrix
    # Columns 0-4: Living States | Column 5: Death State
    mu_60 = np.hstack([mu_s, mu_d])
    sig_60 = np.hstack([sig_s, sig_d])
    
    
    mu_alive_avg = np.sum(mu_60[:, :5] * mu_h_60, axis=1)

    # Variance correction for NHH
    var_between_mu = np.sum(mu_60[:, :5]**2 * mu_h_60, axis=1) - mu_alive_avg**2
    e_sigma_sq     = np.sum(sig_60[:, :5]**2 * mu_h_60, axis=1)
    shock_var = cfg.sigma_zeta**2 / (1 - cfg.rho_m**2) + cfg.sigma_v**2

    sigma_eff_sq   = e_sigma_sq + var_between_mu / shock_var
    sigma_eff_avg  = np.sqrt(sigma_eff_sq)

    mu_avg  = np.column_stack((mu_alive_avg, mu_60[:, 5]))
    sig_avg = np.column_stack((sigma_eff_avg, sig_60[:, 5]))  # <-- use eff here
    print(f"Old sigma_avg (simple mean):  {np.mean(sig_s, axis=1)[:3]}")
    print(f"New sigma_eff_avg (corrected): {sigma_eff_avg[:3]}")
    print(f"var_between_mu (first 3 ages): {var_between_mu[:3]}")
    print(f"shock_var: {shock_var:.4f}")
    np.savez_compressed(
        OUT_DIR / "medex_params.npz",
        mu=mu_60,
        sigma=sig_60,
        mu_avg=mu_avg,
        sigma_avg=sig_avg   # now stores the corrected sigma
    )
    
def build_health_income(mu_h_45):
   
    health_inc = load_and_clean("data/raw/earn_profile.txt")
    omega_inc = np.array(health_inc).reshape((5, 45), order='F').T 
    # Collapse the 5 health states into an average efficiency profile (shape: 45,)
    omega_avg = np.sum(omega_inc * mu_h_45, axis=1)
    
    # Save as NPZ so we can store both heterogeneous and average profiles
    np.savez_compressed(
        OUT_DIR / "earn_profile.npz",
        omega=omega_inc,           # Shape (45, 5)
        omega_avg=omega_avg[:, None] # Shape (45, 1)
    )
    print("Succesfully stored health earnings profile.")
    
def evolve_demographics(trans_obj, h_0, n_periods):
    """
    Evolves the population to find the cross-sectional share of alive people
    in each health state at each age.
    """
    mu_h = np.zeros((n_periods, 5))
    mu_h[0] = h_0 
    
    for t in range(1, n_periods):
        Q_alive = trans_obj[t-1, :5, :5]
        next_h = mu_h[t-1] @ Q_alive
        
        survival_mass = np.sum(next_h)
        if survival_mass > 0:
            mu_h[t] = next_h / survival_mass
            
    return mu_h

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    RAW_DIR = ROOT / "data" / "raw"
    OUT_DIR = ROOT / "data"
    obj_path = RAW_DIR / "health_surv_prob_obj.txt"
    obj_beliefs = np.loadtxt(obj_path).reshape((90, 6, 6))
    
    Q_init = obj_beliefs[0, :5, :5]
    Q_init_scaled = Q_init / Q_init.sum(axis=1, keepdims=True)
    h_0 = stationary_distribution(Q_init_scaled)
    
    # Generate demographic weights
    mu_h_90 = evolve_demographics(obj_beliefs, h_0, n_periods=90)
    
    # Now actually call the functions
    build_medex_npz(mu_h_90[30:])       # ages 50-110 -> 60 periods of medex
    build_health_income(mu_h_90[:45])   # ages 20-64 -> 45 working periods
    build_model_data(mu_h_90)           # survival transition matrices
