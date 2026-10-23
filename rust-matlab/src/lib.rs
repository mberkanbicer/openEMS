//! C ABI wrapper around openems-core for MATLAB/Octave MEX interop.
//!
//! All functions are `extern "C"` and `#[no_mangle]`. Opaque solver pointers
//! are passed as `*mut c_void` and cast to/from `*mut openems_core::OpenEMS`.
//! Strings use `std::ffi::CStr`/`CString`. Panics are caught at the FFI
//! boundary with `std::panic::catch_unwind` so they never unwind into C or
//! MATLAB. Failures return 0 / false / null as appropriate.

use std::ffi::CStr;
use std::os::raw::{c_char, c_double, c_int, c_uint, c_void};
use std::panic;

#[allow(unused_imports)]
use openems_core::{OpenEMS, BC_PEC, BC_PMC, BC_MUR, BC_PML};

/// Convert a raw void pointer into a mutable reference to an OpenEMS solver.
///
/// Returns `None` if the pointer is null. The `'static` lifetime is an FFI
/// convention: the caller owns the pointer's lifetime.
fn solver_mut(ptr: *mut c_void) -> Option<&'static mut OpenEMS> {
    if ptr.is_null() {
        return None;
    }
    // Safety: the caller guarantees `ptr` is a valid, non-null pointer to an
    // OpenEMS allocated by `openems_new`.
    Some(unsafe { &mut *(ptr as *mut OpenEMS) })
}

/// Run a closure that may panic, returning `None` if it panics.
fn catch_unwind<F: FnOnce() -> R + std::panic::UnwindSafe, R>(f: F) -> Option<R> {
    panic::catch_unwind(f).ok()
}

// ---------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------

/// Create a new OpenEMS solver instance. Returns an opaque pointer to the
/// caller, or null on failure.
#[no_mangle]
pub extern "C" fn openems_new() -> *mut c_void {
    let result = catch_unwind(|| {
        let solver = Box::new(OpenEMS::new());
        Some(Box::into_raw(solver) as *mut c_void)
    });
    match result {
        Some(Some(ptr)) => ptr,
        _ => std::ptr::null_mut(),
    }
}

/// Free an OpenEMS solver previously created by `openems_new`.
#[no_mangle]
pub extern "C" fn openems_free(ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }
    let _ = catch_unwind(|| {
        // Safety: the pointer was allocated by `openems_new` as a Box<OpenEMS>.
        unsafe { drop(Box::from_raw(ptr as *mut OpenEMS)) };
    });
}

/// Set the number of time steps.
#[no_mangle]
pub extern "C" fn openems_set_number_of_time_steps(ptr: *mut c_void, val: c_uint) {
    let _ = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver.set_number_of_time_steps(val);
        Some(())
    });
}

/// Set the end criteria.
#[no_mangle]
pub extern "C" fn openems_set_end_criteria(ptr: *mut c_void, val: c_double) {
    let _ = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver.set_end_criteria(val);
        Some(())
    });
}

/// Enable or disable dumps.
#[no_mangle]
pub extern "C" fn openems_set_enable_dumps(ptr: *mut c_void, val: u8) {
    let _ = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver.set_enable_dumps(val != 0);
        Some(())
    });
}

/// Set cylinder coordinates mode.
#[no_mangle]
pub extern "C" fn openems_set_cylinder_coords(ptr: *mut c_void, val: u8) {
    let _ = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver.set_cylinder_coords(val != 0);
        Some(())
    });
}

/// Set boundary condition type for a given side (0..6).
/// BC_PEC=0, BC_PMC=1, BC_MUR=2, BC_PML=3.
#[no_mangle]
pub extern "C" fn openems_set_bc_type(ptr: *mut c_void, idx: c_uint, typ: c_int) {
    let _ = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver.set_bc_type(idx as usize, typ);
        Some(())
    });
}

/// Set PML size for a given side.
#[no_mangle]
pub extern "C" fn openems_set_pml_size(ptr: *mut c_void, idx: c_uint, size: c_uint) {
    let _ = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver.set_pml_size(idx as usize, size);
        Some(())
    });
}

/// Set MUR phase velocity for a given side.
#[no_mangle]
pub extern "C" fn openems_set_mur_phase_vel(ptr: *mut c_void, idx: c_uint, val: c_double) {
    let _ = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver.set_mur_phase_vel(idx as usize, val);
        Some(())
    });
}

/// Set maximum simulation time (overrides nr_ts if reached first).
#[no_mangle]
pub extern "C" fn openems_set_max_time(ptr: *mut c_void, val: c_double) {
    let _ = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver.set_max_time(val);
        Some(())
    });
}

/// Set the timestep method.
#[no_mangle]
pub extern "C" fn openems_set_time_step_method(ptr: *mut c_void, val: c_int) {
    let _ = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver.set_time_step_method(val);
        Some(())
    });
}

/// Set a fixed timestep (overrides CFL-computed value if > 0).
#[no_mangle]
pub extern "C" fn openems_set_time_step(ptr: *mut c_void, val: c_double) {
    let _ = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver.set_time_step(val);
        Some(())
    });
}

/// Set the timestep factor (multiplier on CFL timestep).
#[no_mangle]
pub extern "C" fn openems_set_time_step_factor(ptr: *mut c_void, val: c_double) {
    let _ = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver.set_time_step_factor(val);
        Some(())
    });
}

/// Set the oversampling factor for time-domain dumps/probes.
#[no_mangle]
pub extern "C" fn openems_set_over_sampling(ptr: *mut c_void, val: c_uint) {
    let _ = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver.set_over_sampling(val);
        Some(())
    });
}

/// Set cell constant material mode.
#[no_mangle]
pub extern "C" fn openems_set_cell_constant_material(ptr: *mut c_void, val: u8) {
    let _ = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver.set_cell_constant_material(val != 0);
        Some(())
    });
}

/// Set the coordinate system (0=Cartesian, 1=Cylindrical).
/// Returns true (1) on success, false (0) on failure (null pointer or invalid code).
#[no_mangle]
pub extern "C" fn openems_set_coord_system(ptr: *mut c_void, val: c_int) -> u8 {
    let result = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver.set_coord_system(val).ok()?;
        Some(1u8)
    });
    match result {
        Some(Some(ok)) => ok,
        _ => 0,
    }
}

/// Set a Gaussian pulse excitation.
#[no_mangle]
pub extern "C" fn openems_set_gauss_excite(ptr: *mut c_void, f0: c_double, fc: c_double) {
    let _ = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver.set_gauss_excite(f0, fc);
        Some(())
    });
}

/// Set a sinusoidal excitation.
#[no_mangle]
pub extern "C" fn openems_set_sinus_excite(ptr: *mut c_void, f0: c_double) {
    let _ = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver.set_sinus_excite(f0);
        Some(())
    });
}

/// Set a Dirac pulse excitation.
#[no_mangle]
pub extern "C" fn openems_set_dirac_excite(ptr: *mut c_void, f_max: c_double) {
    let _ = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver.set_dirac_excite(f_max);
        Some(())
    });
}

/// Set a step function excitation.
#[no_mangle]
pub extern "C" fn openems_set_step_excite(ptr: *mut c_void, f_max: c_double) {
    let _ = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver.set_step_excite(f_max);
        Some(())
    });
}

/// Set the CSXCAD XML geometry string.
#[no_mangle]
pub extern "C" fn openems_set_csx_xml(ptr: *mut c_void, xml: *const c_char) {
    let result = catch_unwind(|| {
        let xml_str = unsafe { CStr::from_ptr(xml) }.to_string_lossy().into_owned();
        let solver = solver_mut(ptr)?;
        solver.set_csx_xml(xml_str);
        Some(())
    });
    let _ = result;
}

/// Initialize the FDTD operator from the CSX geometry.
/// Returns true (1) on success, false (0) on failure.
#[no_mangle]
pub extern "C" fn openems_setup_fdtd(ptr: *mut c_void) -> u8 {
    let result = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver.setup_fdtd().then_some(1u8)
    });
    match result {
        Some(Some(ok)) => ok,
        _ => 0,
    }
}

/// Run the FDTD simulation.
/// Returns true (1) on success, false (0) on failure.
#[no_mangle]
pub extern "C" fn openems_run_fdtd(ptr: *mut c_void) -> u8 {
    let result = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver.run_fdtd().then_some(1u8)
    });
    match result {
        Some(Some(ok)) => ok,
        _ => 0,
    }
}

/// Get the number of time steps executed so far.
#[no_mangle]
pub extern "C" fn openems_get_number_of_timesteps(ptr: *mut c_void) -> c_uint {
    let result = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver
            .fdtd_eng
            .as_ref()
            .map(|eng| eng.get_number_of_timesteps() as c_uint)
            .or(Some(0))
    });
    match result {
        Some(Some(n)) => n,
        _ => 0,
    }
}

/// Get boundary condition type for a given side (0..6).
#[no_mangle]
pub extern "C" fn openems_get_bc_type(ptr: *mut c_void, idx: c_uint) -> c_int {
    let result = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        Some(solver.get_bc_type(idx as usize) as c_int)
    });
    match result {
        Some(Some(typ)) => typ,
        _ => BC_PEC as c_int,
    }
}

/// Get PML size for a given side (0..6).
#[no_mangle]
pub extern "C" fn openems_get_pml_size(ptr: *mut c_void, idx: c_uint) -> c_uint {
    let result = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        Some(solver.get_pml_size(idx as usize) as c_uint)
    });
    match result {
        Some(Some(size)) => size,
        _ => 0,
    }
}

