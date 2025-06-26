import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

# --- Cosmological Parameters and Constants ---
Omega_Lambda = 0.685
Omega_R = 9.2434441243835e-05
Omega_M = 1.0 - Omega_Lambda - Omega_R
Omega_K = 0.0

OMEGAS = {'R': Omega_R, 'M': Omega_M, 'L': Omega_Lambda, 'K': Omega_K}
Mpc = 3.085677577849e22
hH = 0.673
H0 = (100 * 1000 * hH) / Mpc

def solve_background_cosmology(tau_eval):
    """
    Solves the Friedmann equation for a full ΛCDM model.
    Returns interpolation functions for a(τ) and ℋ(τ), and raw arrays for plotting.
    """
    print("Solving background cosmology...")

    def friedmann_rhs(tau, a, H0, omegas):
        term_R = omegas['R'] / a**2
        term_M = omegas['M'] / a
        term_K = omegas['K']
        term_L = omegas['L'] * a**2
        radicand = term_R + term_M + term_K + term_L
        H_conformal = H0 * np.sqrt(np.maximum(0, radicand))
        return a * H_conformal

    tau_min = tau_eval.min()
    a_initial = np.sqrt(OMEGAS['R']) * H0 * tau_min
    
    sol_cosmo = solve_ivp(
        friedmann_rhs, [tau_min, tau_eval.max()], [a_initial],
        args=(H0, OMEGAS), method='DOP853', t_eval=tau_eval
    )
    
    if not sol_cosmo.success:
        raise RuntimeError(f"Cosmology ODE solver failed: {sol_cosmo.message}")

    a_solution = sol_cosmo.y[0]
    
    term_R = OMEGAS['R'] / a_solution**2
    term_M = OMEGAS['M'] / a_solution
    term_K = OMEGAS['K']
    term_L = OMEGAS['L'] * a_solution**2
    H_conformal_solution = H0 * np.sqrt(term_R + term_M + term_K + term_L)

    a_interp = interp1d(tau_eval, a_solution, kind='cubic', fill_value="extrapolate")
    H_c_interp = interp1d(tau_eval, H_conformal_solution, kind='cubic', fill_value="extrapolate")
    
    print("Cosmology solved. Interpolators created.")
    return a_interp, H_c_interp, tau_eval, a_solution, H_conformal_solution

def k_tilde_func(v):
    """ Implements the velocity-dependent k̃ from Eq. (2.14). """
    v = np.minimum(v, 1.0 - 1e-9)
    v6 = v**6
    return (2.0 * np.sqrt(2.0) / np.pi) * (1.0 - 8.0 * v6) / (1.0 + 8.0 * v6)

def get_hybrid_vos_odes(tau, y, cr, H_c_interp):
    """
    Defines the hybrid VOS ODE system for ξ and v, using the full ℋ(τ).
    The state vector y is [ξ, v].
    """
    xi, v = y
    
    if xi <= 0: return [0, 0]

    # Calculate k_tilde using the current velocity v
    k_tilde = k_tilde_func(v)
    
    # Get the realistic conformal Hubble parameter at the current time
    H_c = H_c_interp(tau)
    
    # Eq (2.15) generalized: replace 1/τ with ℋ(τ)
    dxi_dtau = 1/tau*(v**2 * xi*tau*H_c - xi + cr * v / 2.0)
    
    # Eq (2.16) generalized: replace the prefactor 1/τ with ℋ(τ)
    dv_dtau =  (1.0 - v**2) * (k_tilde / (xi*tau) - 2.0 * v*H_c)
    
    return [dxi_dtau, dv_dtau]

def solve_and_plot():
    """ Solves the ODEs and generates both the cosmology and string plots. """
    # Physical time range in seconds
    tau_min, tau_max = 1, 5e17
    tau_span = [tau_min, tau_max]
    tau_eval = np.logspace(np.log10(tau_min), np.log10(tau_max), 500)
    
    # --- Solve and Plot Background Cosmology ---
    a_interp, H_c_interp, tau_cosmo, a_cosmo, Hc_cosmo = solve_background_cosmology(tau_eval)

    fig_cosmo, ax_a = plt.subplots(figsize=(10, 6))
    color1 = 'C0'
    ax_a.set_xlabel(r'Conformal Time $\tau$ [s]', fontsize=14)
    ax_a.set_ylabel(r'Scale Factor $a(\tau)$', color=color1, fontsize=14)
    ax_a.plot(tau_cosmo, a_cosmo, color=color1, lw=2)
    ax_a.tick_params(axis='y', labelcolor=color1)
    ax_a.set_xscale('log'); ax_a.set_yscale('log')
    ax_a.grid(True, which="both", ls="--", alpha=0.5)
    ax_Hc = ax_a.twinx()
    color2 = 'C3'
    ax_Hc.set_ylabel(r'Conformal Hubble $\mathcal{H}(\tau)$ [s$^{-1}$]', color=color2, fontsize=14)
    ax_Hc.plot(tau_cosmo, Hc_cosmo, color=color2, lw=2, ls='--')
    ax_Hc.tick_params(axis='y', labelcolor=color2)
    ax_Hc.set_yscale('log')
    fig_cosmo.suptitle('Background ΛCDM Cosmology Evolution', fontsize=16)
    fig_cosmo.tight_layout(rect=[0, 0, 1, 0.96])

    # --- Solve and Plot String Evolution using the Hybrid Model ---
    # Initial conditions for ξ and v
    xi0 = 0.15
    v0 = 0.65
    y0 = [xi0, v0]

    cr_values = np.logspace(-2, 0, 100)
    cr_special = 0.23

    fig_string, (ax_xi, ax_v) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 7),
        gridspec_kw={'hspace': 0.05}, constrained_layout=True
    )
    fig_string.text(0.9, 0.95, '3', fontsize=14, ha='right')
    cmap = plt.get_cmap('PRGn')
    norm = colors.LogNorm(vmin=cr_values.min(), vmax=cr_values.max())
    
    print("Solving hybrid VOS equations for a range of c_r...")
    for cr in cr_values:
        sol = solve_ivp(
            get_hybrid_vos_odes, tau_span, y0,
            args=(cr, H_c_interp),
            method='LSODA', t_eval=tau_eval
        )
        # The solution is directly [ξ, v]
        xi_sol, v_sol = sol.y
        
        color = cmap(norm(cr))
        ax_xi.plot(tau_eval, xi_sol, color=color, lw=2)
        ax_v.plot(tau_eval, v_sol, color=color, lw=2)

    sol_special = solve_ivp(
        get_hybrid_vos_odes, tau_span, y0,
        args=(cr_special, H_c_interp),
        method='LSODA', t_eval=tau_eval
    )
    xi_special, v_special = sol_special.y
    
    ax_xi.plot(tau_eval, xi_special, 'k-.', lw=2)
    ax_v.plot(tau_eval, v_special, 'k-.', lw=2)
    
    # Formatting for the string plot
    ax_xi.set_ylabel(r'$\xi$', fontsize=14, rotation=0, labelpad=15)
    ax_xi.set_ylim(0, 0.42); ax_xi.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
    ax_v.set_ylabel(r'$v$', fontsize=14, rotation=0, labelpad=15)
    ax_v.set_ylim(0.48, 0.72); ax_v.set_yticks([0.5, 0.6, 0.7])
    ax_v.set_xlabel(r'Conformal Time $\tau$ [s]', fontsize=14)
    
    for ax in [ax_xi, ax_v]:
        ax.set_xscale('log')
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)

    ax_xi2 = ax_xi.twinx(); ax_xi2.set_yscale('log'); ax_xi2.set_ylim(1e-1, 1)
    ax_xi2.set_yticks([1e-1, 1e0]); ax_xi2.tick_params(axis='y', which='major', labelsize=12, direction='in')
    ax_v2 = ax_v.twinx(); ax_v2.set_yscale('log'); ax_v2.set_ylim(1e-2, 1e-1)
    ax_v2.set_yticks([1e-2, 1e-1]); ax_v2.tick_params(axis='y', which='major', labelsize=12, direction='in')
    
    print("Done. Displaying plots.")
    plt.show()

if __name__ == '__main__':
    solve_and_plot()