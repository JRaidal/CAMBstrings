module CustomModule
    use precision
    use interpolation, only: TInterpGrid2D
    use classes
    implicit none

    private
    public :: Tcustom, Tcustom_SetMyDataArray, Tcustom_SetUETCTable, Tcustom_SetActiveEigenmode, Tcustom_VerifyInterpolation

    type, extends(TPythonInterfacedClass) :: Tcustom
        ! For 2D points
        real(dl), allocatable    :: my_data_array_fortran(:,:) 

        ! For UETC Table
        real(dl), allocatable    :: k_grid_uetc(:)    
        real(dl), allocatable    :: tau_grid_uetc(:)  
        integer                  :: nk_uetc_stored = 0
        integer                  :: ntau_uetc_stored = 0
        
        ! Raw Eigenfunction table (kept for initializing TInterpGrid2D objects)
        real(dl), allocatable    :: eigenfunctions_uetc_raw_table(:,:,:,:) 

        ! Arrays of TInterpGrid2D objects for each eigenfunction type
        type(TInterpGrid2D), allocatable :: ef_interp_00(:) ! Shape (nmodes_uetc_stored)
        type(TInterpGrid2D), allocatable :: ef_interp_S(:)  ! Shape (nmodes_uetc_stored)
        type(TInterpGrid2D), allocatable :: ef_interp_V(:)  ! Shape (nmodes_uetc_stored)
        type(TInterpGrid2D), allocatable :: ef_interp_T(:)  ! Shape (nmodes_uetc_stored)
        
        ! Eigenvalues (still stored as raw tables)
        real(dl), allocatable    :: eigenvalues_S_uetc_raw_table(:,:) 
        real(dl), allocatable    :: eigenvalues_00_uetc_raw_table(:,:) 

        real(dl), allocatable    :: eigenvalues_V_uetc_raw_table(:,:) 
        real(dl), allocatable    :: eigenvalues_T_uetc_raw_table(:,:) 

        !JR For eigenvalue interpolation
        type(TCubicSpline), allocatable :: lambda_interp_S(:) ! Array for S-mode eigenvalues, size (nmodes_uetc_stored)
        type(TCubicSpline), allocatable :: lambda_interp_00(:) ! Array for S-mode eigenvalues, size (nmodes_uetc_stored)

        type(TCubicSpline), allocatable :: lambda_interp_V(:) ! Array for V-mode eigenvalues
        type(TCubicSpline), allocatable :: lambda_interp_T(:) ! Array for T-mode eigenvalues
        logical                         :: eigenvalue_interpolators_set = .false.
        
        integer                  :: nmodes_uetc_stored = 0
        integer                  :: ntypes_uetc_stored = 0 
        real(dl)                 :: string_mu_uetc_stored = 0.0_dl
        real(dl)                 :: weighting_uetc_stored = 0.0_dl
        
        logical                  :: uetc_interp_objects_are_set = .false. ! Flag for interpolators

        integer                  :: active_mode_idx_uetc = 0 ! 0 means no UETC source, 1 to N for modes

        !Storage for du/d(log(kτ))
        real(dl), allocatable :: eigenfunc_derivs_logkt_raw_table(:,:,:,:) ! Raw table (nk, ntypes, nmodes, ntau)
        type(TInterpGrid2D), allocatable :: ef_deriv_logkt_interp_00(:)     ! Interpolators for 00 component derivatives
        type(TInterpGrid2D), allocatable :: ef_deriv_logkt_interp_S(:)      ! Interpolators for S component derivatives
        ! We might add V and T derivatives later if needed
        logical :: uetc_deriv_interp_objects_are_set = .false. ! Flag for derivative interpolators

    contains
        procedure, nopass :: PythonClass => Tcustom_PythonClass
        procedure, nopass :: SelfPointer => Tcustom_SelfPointer
        procedure :: SetMyDataArray => Tcustom_SetMyDataArray 
        procedure :: SetUETCTable => Tcustom_SetUETCTable
        procedure :: SetActiveEigenmode => Tcustom_SetActiveEigenmode
        procedure :: VerifyInterpolation => Tcustom_VerifyInterpolation
    end type Tcustom

contains 

    subroutine Tcustom_VerifyInterpolation(this, k_val, tau_val, mode_idx_f)
    class(Tcustom), intent(inout) :: this ! inout to ensure it's initialized if needed
    real(dl), intent(in)       :: k_val, tau_val
    integer, intent(in)        :: mode_idx_f
    
    real(dl) :: f_u00, f_du00, f_uS, f_duS
    integer :: error_flag_ignored 

    print *, ""
    print '("Fortran VerifyInterpolation: k=",ES12.4,", tau=",ES12.4,", mode_idx_f=",I0)', k_val, tau_val, mode_idx_f

    if (.not. this%uetc_interp_objects_are_set) then
        print *, "  Fortran WARNING: Eigenfunction interpolators not set."
        return
    endif
    if (.not. this%uetc_deriv_interp_objects_are_set) then
        print *, "  Fortran WARNING: Derivative interpolators not set."
        return
    endif
    
    if (mode_idx_f < 1 .or. mode_idx_f > this%nmodes_uetc_stored) then
        print *, "  Fortran ERROR: mode_idx_f out of bounds: ", mode_idx_f
        return
    endif

    ! Eigenfunction u_00
    f_u00 = this%ef_interp_00(mode_idx_f)%Value(k_val, tau_val)
    print*, "  Fortran interp u_00      : ", f_u00

    ! Derivative du_00/d(log(kτ))
    f_du00 = this%ef_deriv_logkt_interp_00(mode_idx_f)%Value(k_val, tau_val)
    print '("  Fortran interp du00/dlogkτ: ",ES12.6)', f_du00

    ! Eigenfunction u_S
    f_uS = this%ef_interp_S(mode_idx_f)%Value(k_val, tau_val)
    print '("  Fortran interp u_S       : ",ES12.6)', f_uS
    
    ! Derivative du_S/d(log(kτ))
    f_duS = this%ef_deriv_logkt_interp_S(mode_idx_f)%Value(k_val, tau_val)
    print '("  Fortran interp duS/dlogkτ : ",ES12.6)', f_duS

    end subroutine Tcustom_VerifyInterpolation  


    subroutine Tcustom_SetActiveEigenmode(this, mode_to_set)
        class(Tcustom), intent(inout) :: this
        integer, intent(in)           :: mode_to_set

        if (mode_to_set >= 0 .and. (mode_to_set == 0 .or. mode_to_set <= this%nmodes_uetc_stored) ) then
            this%active_mode_idx_uetc = mode_to_set
            if (mode_to_set > 0) then
                print *, "Fortran Tcustom: Active UETC eigenmode set to: ", this%active_mode_idx_uetc
            else
                print *, "Fortran Tcustom: UETC sources turned OFF (active_mode_idx_uetc = 0)."
            endif
        else
            print *, "Fortran Tcustom Warning: Invalid active_mode_idx requested: ", mode_to_set, &
                     " Max modes available: ", this%nmodes_uetc_stored, ". Setting to 0 (OFF)."
            this%active_mode_idx_uetc = 0
        endif
    end subroutine Tcustom_SetActiveEigenmode

    subroutine Tcustom_SetMyDataArray(this, data_flat, num_rows_in)
        class(Tcustom), intent(inout) :: this; real(dl), intent(in) :: data_flat(*); integer, intent(in) :: num_rows_in
        integer :: num_rows; integer, parameter :: num_cols = 2; integer :: expected_size
        num_rows = num_rows_in; expected_size = num_rows * num_cols
        if (allocated(this%my_data_array_fortran)) deallocate(this%my_data_array_fortran)
        if (num_rows > 0) then
            allocate(this%my_data_array_fortran(num_rows, num_cols))
            if (expected_size > 0) then 
                 this%my_data_array_fortran = TRANSPOSE(RESHAPE(data_flat(1:expected_size), [num_cols, num_rows]))
                 if (num_rows > 0 .and. num_cols > 0 .and. size(this%my_data_array_fortran,dim=1) > 0) then
                    !print *, "Fortran Tcustom_SetMyDataArray: Set 2D points array, shape (", num_rows, ",", num_cols, ")"
                    !print *, "  First point: ", this%my_data_array_fortran(1,:)
                 end if
            else
                 if(allocated(this%my_data_array_fortran)) deallocate(this%my_data_array_fortran)
                 allocate(this%my_data_array_fortran(0,num_cols))
            endif
        else
             if (allocated(this%my_data_array_fortran)) deallocate(this%my_data_array_fortran)
             allocate(this%my_data_array_fortran(0, num_cols))
        endif
    end subroutine Tcustom_SetMyDataArray

    subroutine Tcustom_SetUETCTable(this, &
        k_grid_in, nk_in, &
        tau_grid_in, ntau_in, &
        eigenfuncs_flat_in, num_eigen_types_in, nmodes_in, &
        evals_S_flat_in, evals_00_flat_in, evals_V_flat_in, evals_T_flat_in, &
        eigenfunc_derivs_logkt_flat_in, & 
        mu_in, weighting_param_in) 

        class(Tcustom), intent(inout) :: this
        integer, intent(in)                :: nk_in, ntau_in, num_eigen_types_in, nmodes_in
        real(dl), intent(in)               :: k_grid_in(nk_in), tau_grid_in(ntau_in)
        real(dl), intent(in)               :: eigenfuncs_flat_in(*)
        real(dl), intent(in)               :: eigenfunc_derivs_logkt_flat_in(*) ! !JR NEW
        real(dl), intent(in)               :: evals_S_flat_in(*),evals_00_flat_in(*), evals_V_flat_in(*), evals_T_flat_in(*)
        real(dl), intent(in)               :: mu_in, weighting_param_in

        integer :: nk_local, ntau_local, nmodes_local, num_eigen_types_local
        integer :: i, j, l, m, mode_idx
        real(dl), allocatable :: slice_2d_for_interp(:,:) 
        real(dl), allocatable :: slice_2d_deriv_for_interp(:,:) ! !JR NEW for derivative slices

        call DeallocateUETCTableData(this) 
        this%uetc_interp_objects_are_set = .false.
        this%eigenvalue_interpolators_set = .false. ! Also reset this flag
        this%uetc_deriv_interp_objects_are_set = .false. ! !JR NEW: Reset derivative flag

        nk_local = nk_in; ntau_local = ntau_in
        nmodes_local = nmodes_in; num_eigen_types_local = num_eigen_types_in
        
        this%nk_uetc_stored = nk_local; this%ntau_uetc_stored = ntau_local
        this%ntypes_uetc_stored = num_eigen_types_local; this%nmodes_uetc_stored = nmodes_local
        this%string_mu_uetc_stored = mu_in; this%weighting_uetc_stored = weighting_param_in
        
        print *, "Fortran Tcustom_SetUETCTable: Received nk=", nk_local, ", ntau=", ntau_local, &
                 ", ntypes=", num_eigen_types_local, ", nmodes=", nmodes_local

        if (nk_local < 2 .or. ntau_local < 2 .or. nmodes_local <= 0 .or. num_eigen_types_local /= 4) then
            print *, "Fortran SetUETCTable: Insufficient dimensions. Aborting setup."
            call EnsureZeroSizeAllocations(this)
            return
        endif

        allocate(this%k_grid_uetc(nk_local)); this%k_grid_uetc = k_grid_in(1:nk_local)
        allocate(this%tau_grid_uetc(ntau_local)); this%tau_grid_uetc = tau_grid_in(1:ntau_local)
        
        ! Store raw eigenfunctions
        allocate(this%eigenfunctions_uetc_raw_table(nk_local, num_eigen_types_local, nmodes_local, ntau_local))
        do i = 1, nk_local; do j = 1, num_eigen_types_local; do l = 1, nmodes_local; do m = 1, ntau_local
            this%eigenfunctions_uetc_raw_table(i,j,l,m) = eigenfuncs_flat_in( &
                ((( (i-1)*num_eigen_types_local + (j-1) )*nmodes_local + (l-1) )*ntau_local + (m-1) ) + 1 )
        end do; end do; end do; end do

        ! Store raw eigenfunction derivatives
        allocate(this%eigenfunc_derivs_logkt_raw_table(nk_local, num_eigen_types_local, nmodes_local, ntau_local))
        do i = 1, nk_local; do j = 1, num_eigen_types_local; do l = 1, nmodes_local; do m = 1, ntau_local
            this%eigenfunc_derivs_logkt_raw_table(i,j,l,m) = eigenfunc_derivs_logkt_flat_in( &
                ((( (i-1)*num_eigen_types_local + (j-1) )*nmodes_local + (l-1) )*ntau_local + (m-1) ) + 1 )
        end do; end do; end do; end do

        ! Store raw eigenvalues
        allocate(this%eigenvalues_S_uetc_raw_table(nk_local, nmodes_local))
        allocate(this%eigenvalues_00_uetc_raw_table(nk_local, nmodes_local))

        allocate(this%eigenvalues_V_uetc_raw_table(nk_local, nmodes_local))
        allocate(this%eigenvalues_T_uetc_raw_table(nk_local, nmodes_local))
        this%eigenvalues_S_uetc_raw_table = TRANSPOSE(RESHAPE(evals_S_flat_in(1:nk_local*nmodes_local), [nmodes_local, nk_local]))
        this%eigenvalues_00_uetc_raw_table = TRANSPOSE(RESHAPE(evals_00_flat_in(1:nk_local*nmodes_local), [nmodes_local, nk_local]))
        this%eigenvalues_V_uetc_raw_table = TRANSPOSE(RESHAPE(evals_V_flat_in(1:nk_local*nmodes_local), [nmodes_local, nk_local]))
        this%eigenvalues_T_uetc_raw_table = TRANSPOSE(RESHAPE(evals_T_flat_in(1:nk_local*nmodes_local), [nmodes_local, nk_local]))
        print *, "Fortran SetUETCTable: Raw data tables (functions, derivatives, eigenvalues) stored."

        ! Initialize eigenfunction interpolators
        allocate(this%ef_interp_00(nmodes_local)); allocate(this%ef_interp_S(nmodes_local))
        allocate(this%ef_interp_V(nmodes_local)); allocate(this%ef_interp_T(nmodes_local))
        allocate(slice_2d_for_interp(nk_local, ntau_local)) 

        do mode_idx = 1, nmodes_local
            do i = 1, nk_local; slice_2d_for_interp(i, :) = this%eigenfunctions_uetc_raw_table(i, 1, mode_idx, :); end do
            call this%ef_interp_00(mode_idx)%Init(this%k_grid_uetc, this%tau_grid_uetc, slice_2d_for_interp)
            
            do i = 1, nk_local; slice_2d_for_interp(i, :) = this%eigenfunctions_uetc_raw_table(i, 2, mode_idx, :); end do
            call this%ef_interp_S(mode_idx)%Init(this%k_grid_uetc, this%tau_grid_uetc, slice_2d_for_interp)
            
            do i = 1, nk_local; slice_2d_for_interp(i, :) = this%eigenfunctions_uetc_raw_table(i, 3, mode_idx, :); end do
            call this%ef_interp_V(mode_idx)%Init(this%k_grid_uetc, this%tau_grid_uetc, slice_2d_for_interp)
            
            do i = 1, nk_local; slice_2d_for_interp(i, :) = this%eigenfunctions_uetc_raw_table(i, 4, mode_idx, :); end do
            call this%ef_interp_T(mode_idx)%Init(this%k_grid_uetc, this%tau_grid_uetc, slice_2d_for_interp)
        end do
        deallocate(slice_2d_for_interp)
        this%uetc_interp_objects_are_set = .true.
        print *, "Fortran SetUETCTable: Initialized TInterpGrid2D for eigenfunctions."

        ! Initialize derivative interpolators (only for 00 and S types for now)
        allocate(this%ef_deriv_logkt_interp_00(nmodes_local))
        allocate(this%ef_deriv_logkt_interp_S(nmodes_local))
        allocate(slice_2d_deriv_for_interp(nk_local, ntau_local))

        do mode_idx = 1, nmodes_local
            ! Type 1 (index 1 in Fortran for raw table) is '00' component
            do i = 1, nk_local; slice_2d_deriv_for_interp(i, :) = this%eigenfunc_derivs_logkt_raw_table(i, 1, mode_idx, :); end do
            call this%ef_deriv_logkt_interp_00(mode_idx)%Init(this%k_grid_uetc, this%tau_grid_uetc, slice_2d_deriv_for_interp)
            
            ! Type 2 (index 2 in Fortran for raw table) is 'S' component
            do i = 1, nk_local; slice_2d_deriv_for_interp(i, :) = this%eigenfunc_derivs_logkt_raw_table(i, 2, mode_idx, :); end do
            call this%ef_deriv_logkt_interp_S(mode_idx)%Init(this%k_grid_uetc, this%tau_grid_uetc, slice_2d_deriv_for_interp)
        end do
        deallocate(slice_2d_deriv_for_interp)
        this%uetc_deriv_interp_objects_are_set = .true.
        print *, "Fortran SetUETCTable: Initialized TInterpGrid2D for eigenfunction derivatives (00, S)."
        
        ! Initialize eigenvalue interpolators (TCubicSpline)
        allocate(this%lambda_interp_S(nmodes_local))
        allocate(this%lambda_interp_00(nmodes_local))

        allocate(this%lambda_interp_V(nmodes_local))
        allocate(this%lambda_interp_T(nmodes_local))
        do mode_idx = 1, nmodes_local
            call this%lambda_interp_S(mode_idx)%Init(Xarr=this%k_grid_uetc, values=this%eigenvalues_S_uetc_raw_table(:, mode_idx), n=nk_local)
            call this%lambda_interp_00(mode_idx)%Init(Xarr=this%k_grid_uetc, values=this%eigenvalues_00_uetc_raw_table(:, mode_idx), n=nk_local)
            call this%lambda_interp_V(mode_idx)%Init(Xarr=this%k_grid_uetc, values=this%eigenvalues_V_uetc_raw_table(:, mode_idx), n=nk_local)
            call this%lambda_interp_T(mode_idx)%Init(Xarr=this%k_grid_uetc, values=this%eigenvalues_T_uetc_raw_table(:, mode_idx), n=nk_local)
        end do
        this%eigenvalue_interpolators_set = .true.
        print *, "Fortran SetUETCTable: Initialized TCubicSpline for eigenvalues."

    contains 
        subroutine DeallocateUETCTableData(obj_internal) 
            class(Tcustom), intent(inout) :: obj_internal 
            integer :: mode_idx_loop

            if (allocated(obj_internal%k_grid_uetc)) deallocate(obj_internal%k_grid_uetc)
            if (allocated(obj_internal%tau_grid_uetc)) deallocate(obj_internal%tau_grid_uetc)
            
            if (allocated(obj_internal%eigenfunctions_uetc_raw_table)) deallocate(obj_internal%eigenfunctions_uetc_raw_table)
            if (allocated(obj_internal%eigenfunc_derivs_logkt_raw_table)) deallocate(obj_internal%eigenfunc_derivs_logkt_raw_table)

            if (allocated(obj_internal%eigenvalues_S_uetc_raw_table)) deallocate(obj_internal%eigenvalues_S_uetc_raw_table)
            if (allocated(obj_internal%eigenvalues_00_uetc_raw_table)) deallocate(obj_internal%eigenvalues_00_uetc_raw_table)
            if (allocated(obj_internal%eigenvalues_V_uetc_raw_table)) deallocate(obj_internal%eigenvalues_V_uetc_raw_table)
            if (allocated(obj_internal%eigenvalues_T_uetc_raw_table)) deallocate(obj_internal%eigenvalues_T_uetc_raw_table)

            obj_internal%nk_uetc_stored = 0; obj_internal%ntau_uetc_stored = 0; 
            obj_internal%ntypes_uetc_stored = 0; obj_internal%nmodes_uetc_stored = 0

            ! Deallocation for eigenfunction TInterpGrid2D arrays
            if (allocated(obj_internal%ef_interp_00)) then
                do mode_idx_loop=1,size(obj_internal%ef_interp_00); call obj_internal%ef_interp_00(mode_idx_loop)%Clear(); enddo
                deallocate(obj_internal%ef_interp_00)
            endif
            if (allocated(obj_internal%ef_interp_S))  then
                do mode_idx_loop=1,size(obj_internal%ef_interp_S); call obj_internal%ef_interp_S(mode_idx_loop)%Clear(); enddo
                deallocate(obj_internal%ef_interp_S)
            endif
            if (allocated(obj_internal%ef_interp_V))  then
                do mode_idx_loop=1,size(obj_internal%ef_interp_V); call obj_internal%ef_interp_V(mode_idx_loop)%Clear(); enddo
                deallocate(obj_internal%ef_interp_V)
            endif
            if (allocated(obj_internal%ef_interp_T))  then
                do mode_idx_loop = 1, size(obj_internal%ef_interp_T); call obj_internal%ef_interp_T(mode_idx_loop)%Clear(); enddo
                deallocate(obj_internal%ef_interp_T)
            endif
            obj_internal%uetc_interp_objects_are_set = .false.

            ! Deallocate eigenfunction derivative TInterpGrid2D arrays
            if (allocated(obj_internal%ef_deriv_logkt_interp_00)) then
                do mode_idx_loop=1,size(obj_internal%ef_deriv_logkt_interp_00); call obj_internal%ef_deriv_logkt_interp_00(mode_idx_loop)%Clear(); enddo
                deallocate(obj_internal%ef_deriv_logkt_interp_00)
            endif
            if (allocated(obj_internal%ef_deriv_logkt_interp_S)) then
                do mode_idx_loop=1,size(obj_internal%ef_deriv_logkt_interp_S); call obj_internal%ef_deriv_logkt_interp_S(mode_idx_loop)%Clear(); enddo
                deallocate(obj_internal%ef_deriv_logkt_interp_S)
            endif
            obj_internal%uetc_deriv_interp_objects_are_set = .false.

            ! Deallocate eigenvalue TCubicSpline interpolators
            if (allocated(obj_internal%lambda_interp_S)) then 
                do mode_idx_loop=1,size(obj_internal%lambda_interp_S); call obj_internal%lambda_interp_S(mode_idx_loop)%Clear(); enddo
                deallocate(obj_internal%lambda_interp_S) 
            endif
            if (allocated(obj_internal%lambda_interp_V)) then 
                do mode_idx_loop=1,size(obj_internal%lambda_interp_V); call obj_internal%lambda_interp_V(mode_idx_loop)%Clear(); enddo
                deallocate(obj_internal%lambda_interp_V) 
            endif
            if (allocated(obj_internal%lambda_interp_T)) then 
                do mode_idx_loop=1,size(obj_internal%lambda_interp_T); call obj_internal%lambda_interp_T(mode_idx_loop)%Clear(); enddo
                deallocate(obj_internal%lambda_interp_T) 
            endif
            obj_internal%eigenvalue_interpolators_set = .false.
            
        end subroutine DeallocateUETCTableData

        subroutine EnsureZeroSizeAllocations(obj_internal)
             class(Tcustom), intent(inout) :: obj_internal
             if (.not. allocated(obj_internal%k_grid_uetc)) allocate(obj_internal%k_grid_uetc(0))
             if (.not. allocated(obj_internal%tau_grid_uetc)) allocate(obj_internal%tau_grid_uetc(0))
             
             if (.not. allocated(obj_internal%eigenfunctions_uetc_raw_table)) allocate(obj_internal%eigenfunctions_uetc_raw_table(0,0,0,0))
             ! For derivatives
             if (.not. allocated(obj_internal%eigenfunc_derivs_logkt_raw_table)) allocate(obj_internal%eigenfunc_derivs_logkt_raw_table(0,0,0,0))

             if (.not. allocated(obj_internal%eigenvalues_S_uetc_raw_table)) allocate(obj_internal%eigenvalues_S_uetc_raw_table(0,0))
             if (.not. allocated(obj_internal%eigenvalues_V_uetc_raw_table)) allocate(obj_internal%eigenvalues_V_uetc_raw_table(0,0))
             if (.not. allocated(obj_internal%eigenvalues_T_uetc_raw_table)) allocate(obj_internal%eigenvalues_T_uetc_raw_table(0,0))
             
             if (.not. allocated(obj_internal%ef_interp_00)) allocate(obj_internal%ef_interp_00(0))
             if (.not. allocated(obj_internal%ef_interp_S)) allocate(obj_internal%ef_interp_S(0))
             if (.not. allocated(obj_internal%ef_interp_V)) allocate(obj_internal%ef_interp_V(0))
             if (.not. allocated(obj_internal%ef_interp_T)) allocate(obj_internal%ef_interp_T(0))
             
             ! !JR NEW: For derivative interpolators
             if (.not. allocated(obj_internal%ef_deriv_logkt_interp_00)) allocate(obj_internal%ef_deriv_logkt_interp_00(0))
             if (.not. allocated(obj_internal%ef_deriv_logkt_interp_S)) allocate(obj_internal%ef_deriv_logkt_interp_S(0))

             ! For eigenvalue interpolators
             if (.not. allocated(obj_internal%lambda_interp_S)) allocate(obj_internal%lambda_interp_S(0))
             if (.not. allocated(obj_internal%lambda_interp_V)) allocate(obj_internal%lambda_interp_V(0))
             if (.not. allocated(obj_internal%lambda_interp_T)) allocate(obj_internal%lambda_interp_T(0))
        end subroutine EnsureZeroSizeAllocations
    end subroutine Tcustom_SetUETCTable

    function Tcustom_PythonClass() result(pyClassName)
        character(LEN=:), allocatable :: pyClassName; pyClassName = 'customclass'
    end function Tcustom_PythonClass
    subroutine Tcustom_SelfPointer(cptr, P)
        use iso_c_binding; Type(C_PTR) :: cptr
        class(TPythonInterfacedClass), pointer :: P; Type(Tcustom), pointer :: P_Specific_Tcustom
        call c_f_pointer(cptr, P_Specific_Tcustom); P => P_Specific_Tcustom
    end subroutine Tcustom_SelfPointer
end module CustomModule