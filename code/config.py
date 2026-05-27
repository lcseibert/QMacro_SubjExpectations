# -*- coding: utf-8 -*-
"""
config.py — Model configuration and parameter specification
===========================================================

BeliefType : Enum for belief scenarios (subjective/objective/average)

ModelSpecs : Main configuration dataclass
    Demographics   : Lifecycle timing (retirement, medical shock onset)
    Dimensions     : Grid sizes for state variables and shocks
    Grids          : Asset and cash-on-hand bounds
    Shocks         : AR(1) parameters for productivity and medical processes
    Economics      : Preferences, prices, production
    Social Security: Bend points and replacement rates
    Taxes          : Income, payroll, and estate tax parameters
    Bequests       : Warm-glow bequest motive parameters
    Anchoring      : Cross-economy normalization (SSH → OSH/NHH)

"""

from dataclasses import dataclass
import numpy as np
from enum import IntEnum
from typing import Optional

class BeliefType(IntEnum):
    subjective = 0
    objective = 1
    average = 2

@dataclass
class ModelSpecs:
    # --- Beliefs ---
    belief_mode: int = BeliefType.subjective
    subjective: bool = False  # core of the paper, True -> subj, False -> obj

    # Note that indexing in python works differently and 
    # thus my model periods go form 0 to 109 unlike explaiend in the paper!!
    # --- Demographics ---
    max_age: int = 90         # Total model periods
    n_ages: int = 90
    ret_idx: int = 45         # Retirement happens at t=45 (Age 65)
    ret_age: int = 45
    medex_idx: int = 30       # Medex shocks start at t=30 (Age 50)
    medex_age: int = 30
    
    # --- Dimensions ---
    n_assets: int = 500       # Savings grid points
    n_cah: int = 200          # CAH grid points
    n_h: int = 6              # health states
    n_p: int = 5              # persistent productivity states
    n_eta_p: int = 3          # iid productivity states
    n_m: int = 7              # persistent medical expenditure states
    n_eta_m: int = 5          # iid medical expenditure states
    
    # --- Grids --- in model units!!
    sav_max: float = 65 # $950,000
    sav_min: float = 0
    cah_max: float = 80 # $1,000,000
    cah_min: float = 0
    gss_stride: int = 0

    # --- Shocks ---
    rho_p: float = 0.9695       
    sigma_kappa: float = np.sqrt(0.0384)
    sigma_eps: float = np.sqrt(0.0522)
    
    rho_m: float = 0.920
    sigma_zeta: float = np.sqrt(0.084)
    sigma_v: float = np.sqrt(0.457)
    
    
    # --- Economics ---
    w: float = 1.0
    r: float = 0.04
    alpha: float = 0.36     
    beta: float = 0.986
    gamma: float = 1.0    
    rra: float = 1.0
    delta: float = 0.096
    
    
    # --- Anchor the models to follow one normalization ---
    is_anchored: bool = False
    c_floor: Optional[float] = None
    p_1_star: Optional[float] = None
    p_2_star: Optional[float] = None
    p_max_star: Optional[float] = None
    anchor_y_med: Optional[float] = None
    anchor_y_avg: Optional[float] = None
    anchor_omega_ret: Optional[float] = None
    A_anchor: Optional[float] = None


    # --- Social Security Bend Points ---
    rho_1: float = 0.90
    rho_2: float = 0.32
    rho_3: float = 0.15

    b_1: float = 6.384
    b_2: float = 38.424
    e_med: float = 31.5268
    e_max: float = 76.20
    
    # --- Tax Parameter of FolytinOlsson(2024)---
    tau: float = 0.137
    tau_ss: float = 0.124
    lam: float = 0.92222
    tau_b: float = 0.3
    
    # --- Bequests Parameter---
    phi_1: float = 11.127
    phi_2: float = 0.001
    chi_b: float = 18.333
    