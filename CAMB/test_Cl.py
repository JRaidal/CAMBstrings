#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import camb
from camb.custom_module import customclass # Ensure this matches your project structure
import os
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline # For Python-side 2D interpolation


# --- Configuration ---
NPZ_FILENAME = "uetc_eigen_data_corrected_with_derivs.npz"
#NPZ_FILENAME = "uetc_eigen_data_medianized.npz"

N_MODES_TO_SUM = 10  # Number of UETC eigenmodes to sum for the string signal
LMAX_PLOT = 4000    # Max multipole for plotting
CMB_UNIT_OUTPUT = 'muK' # 'muK' for muK^2 units, 'K' for K^2 units
pol_mode_idx=0;# !indices: TT, EE, BB, TE

# --- Plotting Style (Optional) ---
plt.style.use('seaborn-v0_8-colorblind') # Or any other style you prefer

print("--- UETC C_l^{EE} Calculation and Plotting Script ---") # MODIFIED SCRIPT TITLE

# ------------------------------------------------------------------------------
# 1. CAMB Parameter Setup
# ------------------------------------------------------------------------------
print("Setting up CAMB parameters...")
pars = camb.CAMBparams()
# Have turned off massive neutrinos to avoid issues with the vector modes
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.0, omk=0, tau=0.06)
pars.max_l_tensor = 1500

# Settings for accurate spectra
#TARGET_LMAX = 4000 # Still use a high LMAX for calculation
#LENS_POTENTIAL_ACCURACY = 2 # Keep if you want lensed EE, less critical than for BB but good for consistency
#pars.set_for_lmax(TARGET_LMAX, lens_potential_accuracy=LENS_POTENTIAL_ACCURACY)

# pars.WantScalars = True
# pars.WantVectors = False
# pars.WantTensors = True
pars.DoLensing = False

if pars.WantTensors:
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=0.1) 
else:
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=0.0) 

# ------------------------------------------------------------------------------
# 2. Load UETC Data and Initialize Custom Object
# ------------------------------------------------------------------------------
print(f"Loading UETC data from: {NPZ_FILENAME}...")
if not os.path.exists(NPZ_FILENAME):
    print(f"ERROR: UETC data file '{NPZ_FILENAME}' not found. Please generate it first.")
    exit()

uetc_data = np.load(NPZ_FILENAME)
k_grid = uetc_data['k_grid']
tau_grid = uetc_data['ktau_grid']
all_eigenfunctions = uetc_data['eigenfunctions']
all_eigenfunctions_d_dlogkt = uetc_data['eigenfunctions_d_dlogkt']
all_eigenvalues_S = uetc_data['eigenvalues_S']
all_eigenvalues_00 = uetc_data['eigenvalues_00']
all_eigenvalues_V = uetc_data['eigenvalues_V']
all_eigenvalues_T = uetc_data['eigenvalues_T']
string_p_mu = uetc_data['string_params_mu'].item()
nmodes_from_file = uetc_data['nmodes'].item()
weighting_from_file = uetc_data['weighting_gamma'].item()
print("UETC data loaded.")

print("Initializing custom Fortran object...")
my_custom_obj = customclass()
my_custom_obj.set_uetc_table_data(
    k_grid=k_grid,
    tau_grid=tau_grid,
    eigenfunctions=all_eigenfunctions,
    eigenfunctions_d_dlogkt=all_eigenfunctions_d_dlogkt,
    eigenvalues_S=all_eigenvalues_S,
    eigenvalues_00=all_eigenvalues_00,
    eigenvalues_V=all_eigenvalues_V,
    eigenvalues_T=all_eigenvalues_T,
    string_params_mu=string_p_mu,
    nmodes_param=nmodes_from_file,
    weighting_param=weighting_from_file
)
pars.Custom = my_custom_obj 
print("Custom object initialized and assigned to CAMB parameters.")

#print(tau_grid)

#print(all_eigenfunctions[0, 3, 0, :])

#my_custom_obj.verify_interpolation_at_point(1e-1, 1e-1, 1)
#stop

# --- Test Points and Indices ---
k_test_values = [k_grid[len(k_grid)//4], k_grid[len(k_grid)//2], k_grid[3*len(k_grid)//4]]
tau_test_values = [tau_grid[len(tau_grid)//4], tau_grid[len(tau_grid)//2], tau_grid[3*len(tau_grid)//4]]
mode_idx_test_py = 0 # Test for the first mode (index 0 in Python)
                     # This corresponds to mode_idx = 1 in Fortran for array access

print(f"\n--- Python-Side Interpolation Verification for Mode {mode_idx_test_py+1} ---")

# Eigenfunction u_00 (type 0 in Python)
eigenfunc_type0_py_table = all_eigenfunctions[:, 0, mode_idx_test_py, :] # Shape (nk, ntau)
# Ensure k_grid and tau_grid are monotonically increasing for RectBivariateSpline
if not (np.all(np.diff(k_grid) > 0) and np.all(np.diff(tau_grid) > 0)):
    print("Warning: k_grid or tau_grid may not be strictly monotonically increasing for RectBivariateSpline.")


# ------------------------------------------------------------------------------
# 3. Calculate Baseline C_l^{EE} (No UETC Sources)
# ------------------------------------------------------------------------------
print("Calculating baseline C_l^{EE} (UETC sources OFF)...")
my_custom_obj.set_active_eigenmode(0) 
results_baseline = camb.get_results(pars)

power_spectra_baseline = results_baseline.get_cmb_power_spectra(pars, CMB_unit=CMB_UNIT_OUTPUT, raw_cl=False)
# For get_cmb_power_spectra, output columns for 'total' are typically: TT, EE, BB, TE
# Index 1 is EE.
cl_ee_baseline_dl = power_spectra_baseline['total'][:,pol_mode_idx]
lmax_calc = cl_ee_baseline_dl.shape[0] - 1
ls_calc = np.arange(lmax_calc + 1)

print(cl_ee_baseline_dl)


print(f"Baseline C_l^{{EE}} calculated up to LMAX={lmax_calc}.")

# ------------------------------------------------------------------------------
# 4. Calculate UETC C_l^{EE} by Summing Modes
# ------------------------------------------------------------------------------

if pars.WantTensors and not pars.WantVectors and not pars.WantScalars:
    pars.InitPower.set_params(As=1, ns=4, r=1, nt=3, pivot_scalar=1.0, pivot_tensor=1.0) 
    scale_factor = 16/(2*np.pi**2)
elif pars.WantVectors and not pars.WantTensors and not pars.WantScalars:
    pars.InitPower.set_params(As=1, ns=4, r=0, nt=3, pivot_scalar=1.0, pivot_tensor=1.0) 
    scale_factor = 8/(2*np.pi**2)
elif pars.WantScalars and not pars.WantTensors and not pars.WantVectors:
    pars.InitPower.set_params(As=1, ns=4, r=0, nt=3, pivot_scalar=1.0, pivot_tensor=1.0) 
    scale_factor = 1/(2*np.pi**2)
else:
    print("Error: Invalid combination of modes.")
    exit()

actual_n_modes_to_sum = min(N_MODES_TO_SUM, nmodes_from_file)
print(f"Calculating UETC C_l^{{EE}} by summing {actual_n_modes_to_sum} eigenmodes...")

cl_ee_strings_sum_dl = np.zeros_like(cl_ee_baseline_dl)

# pars.InitPower.set_params(As=1, ns=1, r=1) 
for i_mode in range(1, actual_n_modes_to_sum + 1):
    print(f"  Processing eigenmode {i_mode}/{actual_n_modes_to_sum}...")
    my_custom_obj.set_active_eigenmode(i_mode)
    
    pars.Custom = my_custom_obj # Re-assign to ensure CAMB sees the updated active_mode

    results_mode_i = camb.get_results(pars)
    power_spectra_mode_i = results_mode_i.get_cmb_power_spectra(pars, CMB_unit=CMB_UNIT_OUTPUT, raw_cl=False)
    cl_ee_mode_i_dl = power_spectra_mode_i['total'][:,pol_mode_idx] # !indices: TT, EE, BB, TE
        
    min_len = min(len(cl_ee_strings_sum_dl), len(cl_ee_mode_i_dl))
    
    cl_ee_strings_sum_dl[:min_len] += cl_ee_mode_i_dl[:min_len]

my_custom_obj.set_active_eigenmode(0) 
print("UETC C_l^{EE} calculation finished.")

# Assumes linear addition of power spectra.

#Cl_strings = abs(cl_ee_strings_sum_dl-actual_n_modes_to_sum*cl_ee_baseline_dl)
#cl_ee_total_with_strings_dl = cl_ee_baseline_dl
Cl_strings = scale_factor*cl_ee_strings_sum_dl

print(Cl_strings)

# ------------------------------------------------------------------------------
# 5. Plotting
# ------------------------------------------------------------------------------
print("Plotting results...")
fig, axs = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns

plot_mask = (ls_calc >= 1) & (ls_calc <= LMAX_PLOT)
ls_plot = ls_calc[plot_mask]

# First plot: Strings EE
axs[0].plot(ls_plot, Cl_strings[plot_mask], label='Strings', color='C1', linestyle='-')
axs[0].set_xlabel(r'$\ell$')
axs[0].set_ylabel(r'$\ell(\ell+1)C_\ell/2\pi \, [\mu K^2]$')
axs[0].set_title('Strings Power Spectrum')
axs[0].set_xlim([2, LMAX_PLOT])
axs[0].set_ylim(ymin=0)
axs[0].set_xscale('log')
axs[0].legend()
axs[0].grid(True, which="both", ls="-", alpha=0.5)

# Second plot: Baseline EE
axs[1].plot(ls_plot, cl_ee_baseline_dl[plot_mask], label='Baseline', color='C0', linestyle='-')
axs[1].set_xlabel(r'$\ell$')
axs[1].set_ylabel(r'$\ell(\ell+1)C_\ell/2\pi \, [\mu K^2]$')
axs[1].set_title('Baseline Power Spectrum')
axs[1].set_xlim([2, LMAX_PLOT])
axs[1].set_ylim(ymin=0)
axs[1].set_xscale('log')
axs[1].legend()
axs[1].grid(True, which="both", ls="-", alpha=0.5)

plt.tight_layout()
plt.savefig("cl_ee_comparison_side_by_side.png")
print("Plot saved to cl_ee_comparison_side_by_side.png")
plt.show()

print("--- Script Finished ---")