# Life-Cycle Model with Health-Dependent Survival Beliefs

Independent Python replication of the quantitative model without intergenerational bequests in:

Foltyn & Olsson (2024),
"Subjective Life Expectancies, Time Preference Heterogeneity, and Wealth Inequality"
https://doi.org/10.3982/QE2016


This repository contains an independent Python implementation based on the econometric estimates and quantitative framework presented in the original paper and their replication package.
The implementation does not reproduce the authors' original source code directly, but instead reconstructs the model environment and life savings distortion structure in Python.


## Methodology

backward induction
stationary equilibrium
histogram method
heterogeneous agents
subjective survival beliefs


## Features

- Dynamic programming life-cycle model
- Heterogeneous subjective survival beliefs
- Partial equilibrium savings distortion analysis
- General equilibrium solver
- Histogram-based forward simulation
- Numba-accelerated numerical routines
- Wealth distribution diagnostics and plotting


## Requirements

- Python 3.10+
- NumPy, SciPy, Numba, Matplotlib
- 32 GB RAM used

Install dependencies: # should be already available in a standard python setting
```bash
pip install numpy scipy numba matplotlib
```

## Installation

1) Clone the repository and navigate to the code/ directory
2) Install the package in editable mode (once, sets up all import paths):
	pip isntall -e .
3) Install dependencies:
	pip install numpy scipy numba matplotlib


## Project Structure of code directory:

```
code/
├── main.py                 # Main execution script
├── config.py               # Model configuration and parameters
├── plot_config.py          # Matplotlib styling
├── pyproject.toml          # Package setup (enables imports)
├── src/
│   ├── primitives.py       # Model primitives and grids
│   ├── hh_solver.py        # Household backward induction
│   ├── simulation.py       # Forward simulation (histogram method)
│   ├── ge_solver.py        # General equilibrium solver
│   ├── worker_funcs.py     # Numba-optimized routines
│   ├── data_management.py  # Policy Function Storer
│   └── numerics.py         # Interpolation, GSS, utilities, Discetizing Shocks, ..
├── analysis/
│   ├── diagnostics.py      # Stores statistics about economy | e.g., policy & ergodic distribution
│   ├── aggregation.py      # Aggregation and ergodic marignalization to handle
│   └── plot_*.py           # Plotting functions
└── plots/               	 # Output directory
├── data/
│   ├── build_data.py       # Build data inputs from econometric calibration
│   └── data_*              # Data files
```

## Running the Code

Full replication (all economies + plots):
From the code/ directory:
```bash
python main.py
```

### Expected Runtimes

Tested on: **Intel Core Ultra 7 155H** (16 cores, 22 threads), 32 GB RAM

| Task | Time |
|------|------|
| SSH GE (baseline) | ~8 min |
| OSH GE | ~8 min |
| NHH GE | ~8 min |
| SSH + Bequest GE | ~10 min |
| Full replication | ~40 min |
