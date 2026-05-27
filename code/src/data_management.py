# -*- coding: utf-8 -*-
"""
data_management.py — Policy storage
======================================================

PolicyLibrary
-------------
Storage for solved policy and value functions.

Attributes:
    savings, consumption, value : (age, cah, health, productivity, medical) arrays
    cah_grid                    : Cash-on-hand grid for interpolation
    savings_rate                : Optional precomputed s'/x

Methods:
    load_from_disk    : Load master tensors from saved .npy files
    get_savings       : Extract 4D policy slice for specific age
    get_consumption   : Extract 4D consumption slice for specific age
    get_specific_policy: Extract 1D policy for specific (age, h, p, m) state
    compute_SR        : Compute savings rates
"""
# loading and accessing the policy functions
import numpy as np
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class PolicyLibrary:
    """
    Central storage for Policy and Value functions.
    Loads individual age-files and stacks them into a master 5D tensor.
    
    Master Shape: (n_ages, n_cah, n_h, n_p, n_m)
    """
    savings: np.ndarray
    consumption: np.ndarray
    cah_grid: np.ndarray
    value: Optional[np.ndarray] = None
    savings_rate: Optional[np.ndarray] = None

    @classmethod
    def load_from_disk(cls, solution_dir, cah_grid):
        """
        Factory method to load the master 5D policy tensors.
        """
        print(f"Loading solution data from {solution_dir}...")
        
        # Construct paths (Ensure these match the filenames you used in solve_lifecycle)
        path_s = os.path.join(solution_dir, "Pfun.npy")
        path_c = os.path.join(solution_dir, "Cfun.npy")
        
        # Load directly. np.load automatically reconstructs the 5D shape!
        if os.path.exists(path_s):
            S_master = np.load(path_s)
        else:
            raise FileNotFoundError(f"❌ Master savings policy not found at {path_s}")
            
        # Handle consumption safely
        if os.path.exists(path_c): 
            C_master = np.load(path_c)
        else:
            print("⚠️ Warning: Consumption policy not found. Filling with zeros.")
            C_master = np.zeros_like(S_master)
            
        print(">>> Successfully loaded master policy functions.")
        return cls(
            savings=S_master,
            consumption=C_master,
            cah_grid=cah_grid
        )

    def get_savings(self, age):
        """Returns the full 4D savings policy tensor for a specific age."""
        return self.savings[age]

    def get_consumption(self, age):
        """Returns the full 4D consumption policy tensor for a specific age."""
        return self.consumption[age]
        
    def get_specific_policy(self, age, h_idx, p_idx, m_idx, policy_type='savings'):
        """Returns a 1D array of policy choices across the CAH grid for a specific state."""
        tensor = self.savings if policy_type == 'savings' else self.consumption
        return tensor[age, :, h_idx, p_idx, m_idx]
    
    def compute_SR(self):
        self.savings_rate = np.zeros_like(self.savings)
        cah = self.cah_grid[None, :, None, None, None]
        np.divide(
            self.savings,
            cah,
            out=self.savings_rate,
            where=cah > 1e-10
        )

        print("Computed savings rates as SAV / CAH")