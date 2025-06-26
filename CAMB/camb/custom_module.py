# In custom_module.py
from .baseconfig import F2003Class, fortran_class, numpy_1d, numpy_1d_int, CAMBError, np, \
    AllocatableArrayDouble, f_pointer
from ctypes import c_int, c_double, byref, POINTER, c_bool, c_size_t

class OverlordClass(F2003Class):
    pass

@fortran_class
class customclass(OverlordClass):
    _fortran_class_module_ = 'CustomModule'
    _fortran_class_name_ = 'Tcustom'
    _fields_ = []

    _methods_ = [
        ('SetMyDataArray', [numpy_1d, POINTER(c_int)]), 

        ('SetUETCTable', [ 
            numpy_1d, POINTER(c_int),          # k_grid_in, nk_in
            numpy_1d, POINTER(c_int),          # tau_grid_in, ntau_in
            numpy_1d, POINTER(c_int),          # eigenfuncs_flat_in, num_eigen_types_in
            POINTER(c_int),                    # nmodes_in
            numpy_1d,                          # evals_S_flat_in
            numpy_1d,                          # evals_00_flat_in

            numpy_1d,                          # evals_V_flat_in
            numpy_1d,                          # evals_T_flat_in
            numpy_1d,                          # eigenfunc_derivs_logkt_flat_in 
            POINTER(c_double),                 # mu_in
            POINTER(c_double)                  # weighting_param_in
        ]),
        ('SetActiveEigenmode', [POINTER(c_int)]),
        ('VerifyInterpolation', [POINTER(c_double), POINTER(c_double), POINTER(c_int)])
    ]

    def set_my_data_array(self, data_array):
        if not isinstance(data_array, np.ndarray): raise TypeError("data_array must be a NumPy array.")
        if data_array.ndim != 2: raise ValueError("data_array must be 2-dimensional.")
        if data_array.shape[1] != 2: raise ValueError(f"data_array must have shape (N, 2), got {data_array.shape}")
        data_array_c = np.ascontiguousarray(data_array, dtype=np.float64)
        num_rows = data_array_c.shape[0]
        self.f_SetMyDataArray(data_array_c, byref(c_int(num_rows)))
        print(f"Python: customclass.set_my_data_array called with C-array shape {data_array_c.shape}")
        return self

    # --- MODIFIED method for UETC Table ---
    def set_uetc_table_data(self, k_grid, tau_grid, eigenfunctions,
                              eigenfunctions_d_dlogkt, 
                              eigenvalues_S,eigenvalues_00, eigenvalues_V, eigenvalues_T,
                              string_params_mu, nmodes_param, weighting_param):
        print("Python: Preparing UETC table (with derivatives) for Fortran...") # Modified print
        k_grid_c = np.ascontiguousarray(k_grid, dtype=np.float64)
        tau_grid_c = np.ascontiguousarray(tau_grid, dtype=np.float64)
        eigenfunctions_flat_c = np.ascontiguousarray(eigenfunctions, dtype=np.float64).ravel(order='C')
        
        # !JR NEW: Prepare derivative table
        eigenfunctions_derivs_flat_c = np.ascontiguousarray(eigenfunctions_d_dlogkt, dtype=np.float64).ravel(order='C')
        
        eigenvalues_S_flat_c = np.ascontiguousarray(eigenvalues_S, dtype=np.float64).ravel(order='C')
        eigenvalues_00_flat_c = np.ascontiguousarray(eigenvalues_00, dtype=np.float64).ravel(order='C')

        eigenvalues_V_flat_c = np.ascontiguousarray(eigenvalues_V, dtype=np.float64).ravel(order='C')
        eigenvalues_T_flat_c = np.ascontiguousarray(eigenvalues_T, dtype=np.float64).ravel(order='C')

        nk = k_grid_c.shape[0]
        ntau = tau_grid_c.shape[0]
        num_eigen_types = eigenfunctions.shape[1] 
        
        if not (eigenfunctions.shape == eigenfunctions_d_dlogkt.shape): 
            raise ValueError("Shape mismatch between eigenfunctions and their derivatives.")
        if not (eigenfunctions.shape[0] == nk and eigenfunctions.shape[3] == ntau and \
                eigenfunctions.shape[1] == num_eigen_types and eigenfunctions.shape[2] == nmodes_param):
            raise ValueError("Dimension mismatch for eigenfunctions passed to set_uetc_table_data")

        print(f"  Python Sending: nk={nk}, ntau={ntau}, Ntypes={num_eigen_types}, Nmodes={nmodes_param}")

        self.f_SetUETCTable( 
            k_grid_c, byref(c_int(nk)),
            tau_grid_c, byref(c_int(ntau)),
            eigenfunctions_flat_c, byref(c_int(num_eigen_types)), # num_eigen_types is for both func and deriv
            byref(c_int(nmodes_param)),                     # nmodes_param is for both
            eigenvalues_S_flat_c,
            eigenvalues_00_flat_c,

            eigenvalues_V_flat_c,
            eigenvalues_T_flat_c,
            eigenfunctions_derivs_flat_c,
            byref(c_double(string_params_mu)),
            byref(c_double(weighting_param))
        )
        print("Python: Fortran call to SetUETCTable completed.")
        return self
    
    def set_active_eigenmode(self, mode_idx):
            # ... (existing code, unchanged) ...
            if not isinstance(mode_idx, int):
                raise TypeError("mode_idx must be an integer.")
            
            print(f"Python: Setting active UETC eigenmode to: {mode_idx}")
            self.f_SetActiveEigenmode(byref(c_int(mode_idx)))
            return self
        
    # !JR NEW Python wrapper for VerifyInterpolation
    def verify_interpolation_at_point(self, k_val, tau_val, mode_idx_fortran):
        """
        Calls Fortran to interpolate and print values for a given k, tau, and 1-based mode_idx.
        """

        print(f"Python: Requesting Fortran verification for k={k_val}, tau={tau_val}, mode_idx_f={mode_idx_fortran}")
        self.f_VerifyInterpolation(byref(c_double(k_val)), 
                                   byref(c_double(tau_val)), 
                                   byref(c_int(mode_idx_fortran)))
        return self
