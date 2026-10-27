//! C ABI wrapper around openems-core for MATLAB/Octave MEX interop.
//!
//! The ABI is the C side of the MATLAB/Octave delivery path. The native
//! contract is the unchanged M-file API plus the Python controller it mirrors:
//! `openEMS/matlab/*.m` and `openEMS/python/openEMS/openEMS.pyx`. Every entry
//! point below is mapped to that contract in `abi_inventory.rs`, and
//! `contract_tests.rs` asserts the observable outcome (return code plus error
//! identifier/message) of the behaviours the native sources define.
//!
//! Conventions:
//!
//! * Handles are opaque `*mut c_void` values produced by [`openems_new`]. They
//!   are tracked in a live-handle registry, so a null, freed or double-freed
//!   handle is reported as the native "deleted object" error instead of being
//!   dereferenced. Native `delete` on a stale pointer is undefined behaviour
//!   (`openEMS/openems.cpp:96-99`, `openEMS/python/openEMS/openEMS.pyx:115-117`).
//! * Every failing entry point records a native error identifier, message and
//!   numeric code. Read them with [`openems_last_error_id`],
//!   [`openems_last_error_message`] and [`openems_last_error_code`]; every
//!   exported call that reports a result clears the previous error first. The
//!   three accessors only read the state, and [`openems_free_string`] is pure
//!   ownership.
//! * Returned strings are `CString`s and must be released with
//!   [`openems_free_string`].
//! * Panics are caught at the boundary with `std::panic::catch_unwind` so they
//!   never unwind into C or MATLAB.
//!
//! This crate *is* the C ABI boundary: every entry point takes raw pointers
//! owned by the C caller, so the pointer contract is part of the C API rather
//! than something a Rust wrapper can enforce. The clippy lint that wants those
//! functions marked `unsafe` is therefore allowed crate-wide; the contract is
//! documented per function and tested by `contract_tests.rs`.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::cell::RefCell;
use std::collections::HashSet;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_double, c_int, c_uint, c_void};
use std::panic;
use std::sync::{Mutex, OnceLock};

use csxcad_core::model::Structure;
use openems_core::{OpenEMS, BC_PML};

/// Native `openEMS::Reset()` initialises every PML size to 8 while the face
/// type stays PEC (`openEMS/openems.cpp:127-132`).
const NATIVE_PML_SIZE_DEFAULT: u32 = 8;

/// Native sentinel for an out-of-range face index and for a face that is not a
/// PML (`openEMS/openems.cpp:817-822` and `:832-839`).
const NATIVE_INVALID_INT: c_int = -1;

/// Native message for a wrapper whose C++ object is gone:
/// `CSXCAD/python/CSXCAD/CSXCAD.pyx:118`
/// `raise RuntimeError('wrapped C++ object of type {} has been deleted')`,
/// with the `openEMS` class name. `openEMS/python/openEMS/openEMS.pyx:115-117`
/// states the same rule for the controller (`__dealloc__` "deletes the C++
/// structure too, which invalidates its wrapper for us").
const ERR_DELETED_MESSAGE: &str = "wrapped C++ object of type openEMS has been deleted";

// ---------------------------------------------------------------------------
// Error channel
// ---------------------------------------------------------------------------

/// The last error recorded by the ABI, in the shape a MEX layer needs to build
/// a MATLAB `MException` (`error(id, msg)`) or to print the native code.
#[derive(Clone, Default)]
struct LastError {
    /// Native identifier, e.g. `openEMS:SetBoundaryCond`. Empty when the native
    /// path raises a plain message without an identifier.
    id: String,
    /// Native message text, quoted from the defining source line.
    message: String,
    /// Native numeric code where one exists (`SetupFDTD` returns 2 or 3,
    /// `openEMS/openems.cpp:1188`, `:1216`, `:1233`). 0 otherwise.
    code: c_int,
}

thread_local! {
    static LAST_ERROR: RefCell<LastError> = RefCell::new(LastError::default());
}

fn clear_error() {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = LastError::default());
}

fn set_error(id: &str, message: &str) {
    set_error_code(id, message, 0);
}

fn set_error_code(id: &str, message: &str, code: c_int) {
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = LastError {
            id: id.to_string(),
            message: message.to_string(),
            code,
        };
    });
}

fn last_error() -> LastError {
    LAST_ERROR.with(|slot| slot.borrow().clone())
}

/// Move an owned string across the boundary, or return null for "no error".
fn error_string(value: String) -> *mut c_char {
    if value.is_empty() {
        return std::ptr::null_mut();
    }
    match CString::new(value) {
        Ok(text) => text.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

// ---------------------------------------------------------------------------
// Handles
// ---------------------------------------------------------------------------

/// One solver handle: the Rust controller plus the geometry it owns.
///
/// `csx` mirrors the native `openEMS::m_CSX` member (`openEMS/openems.h:135`),
/// which `SetCSX` takes ownership of and `Reset`/the destructor delete
/// (`openEMS/openems.cpp:110-111`, `:1163-1170`). The core controller stores
/// geometry as an XML document, so the owned structure is serialised into that
/// slot at setup time; edits made through [`openems_get_csx`] are therefore
/// visible to the next setup, exactly like native `GetCSX()` edits.
struct AbiSolver {
    core: OpenEMS,
    csx: Option<Box<Structure>>,
}

impl AbiSolver {
    fn new() -> Self {
        let mut core = OpenEMS::new();
        // Native `Reset()` default (openEMS/openems.cpp:127-132).
        for face in 0..6 {
            core.set_pml_size(face, NATIVE_PML_SIZE_DEFAULT);
        }
        Self { core, csx: None }
    }
}

fn live_handles() -> &'static Mutex<HashSet<usize>> {
    static LIVE: OnceLock<Mutex<HashSet<usize>>> = OnceLock::new();
    LIVE.get_or_init(|| Mutex::new(HashSet::new()))
}

fn register_handle(ptr: *mut c_void) {
    let mut live = live_handles()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    live.insert(ptr as usize);
}

/// Remove a handle from the registry. Returns false when it was not live, which
/// makes `openems_free` idempotent instead of a double free.
fn unregister_handle(ptr: *mut c_void) -> bool {
    let mut live = live_handles()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    live.remove(&(ptr as usize))
}

fn is_live(ptr: *mut c_void) -> bool {
    let live = live_handles()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    live.contains(&(ptr as usize))
}

/// Convert a raw void pointer into a mutable reference to an [`AbiSolver`].
///
/// Returns `None` and records the native "deleted object" error when the
/// pointer is null or no longer live. A freed handle is never dereferenced.
///
/// Known limitation (shared with every pointer-keyed handle table): once the
/// allocator reuses a freed address for a new solver, the stale pointer becomes
/// indistinguishable from the live one. Callers must not use a handle after
/// `openems_free`, which the native contract states as well
/// (`openEMS/python/openEMS/openEMS.pyx:115-117`).
fn solver_mut(ptr: *mut c_void) -> Option<&'static mut AbiSolver> {
    if !is_live(ptr) {
        set_error("", ERR_DELETED_MESSAGE);
        return None;
    }
    // Safety: the caller guarantees `ptr` came from `openems_new`, and the
    // registry proves the handle has not been freed. The `'static` lifetime is
    // an FFI convention: the caller owns the pointer's lifetime.
    Some(unsafe { &mut *(ptr as *mut AbiSolver) })
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
    clear_error();
    let result = catch_unwind(|| Some(Box::into_raw(Box::new(AbiSolver::new())) as *mut c_void));
    match result {
        Some(Some(ptr)) => {
            register_handle(ptr);
            ptr
        }
        _ => std::ptr::null_mut(),
    }
}

/// Free an OpenEMS solver previously created by `openems_new`.
///
/// Frees the geometry it owns as well (native `~openEMS` -> `Reset`,
/// `openEMS/openems.cpp:96-111`). A null or already-freed handle is a no-op:
/// native `delete` on a stale pointer is undefined behaviour, so the ABI makes
/// the second call well defined.
#[no_mangle]
pub extern "C" fn openems_free(ptr: *mut c_void) {
    clear_error();
    if ptr.is_null() {
        return;
    }
    if !unregister_handle(ptr) {
        return;
    }
    let _ = catch_unwind(|| {
        // Safety: the pointer was allocated by `openems_new` as a
        // Box<AbiSolver> and the registry proved it was still live.
        unsafe { drop(Box::from_raw(ptr as *mut AbiSolver)) };
    });
}

/// Set the number of time steps (native `SetNumberOfTimeSteps`,
/// `openEMS/openems.h:64`).
#[no_mangle]
pub extern "C" fn openems_set_number_of_time_steps(ptr: *mut c_void, val: c_uint) {
    clear_error();
    let _ = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver.core.set_number_of_time_steps(val);
        Some(())
    });
}

/// Set the end criteria (native `SetEndCriteria`, `openEMS/openems.h:66`).
#[no_mangle]
pub extern "C" fn openems_set_end_criteria(ptr: *mut c_void, val: c_double) {
    clear_error();
    let _ = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver.core.set_end_criteria(val);
        Some(())
    });
}

/// Enable or disable dumps (native `SetEnableDumps`, `openEMS/openems.h:65`).
#[no_mangle]
pub extern "C" fn openems_set_enable_dumps(ptr: *mut c_void, val: u8) {
    clear_error();
    let _ = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver.core.set_enable_dumps(val != 0);
        Some(())
    });
}

/// Set cylinder coordinates mode (native `SetCylinderCoords`,
/// `openEMS/openems.h:70`).
#[no_mangle]
pub extern "C" fn openems_set_cylinder_coords(ptr: *mut c_void, val: u8) {
    clear_error();
    let _ = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver.core.set_cylinder_coords(val != 0);
        Some(())
    });
}

/// Set boundary condition type for one face (0..5 = xmin, xmax, ymin, ymax,
/// zmin, zmax).
///
/// Mirrors native `Set_BC_Type`: an index outside 0..5 is a silent no-op and
/// the type value is stored verbatim without enum validation
/// (`openEMS/openems.cpp:810-815`, forwarded unchecked by
/// `openEMS/python/openEMS/openEMS.pyx:322-325`). BC_PEC=0, BC_PMC=1, BC_MUR=2,
/// BC_PML=3.
#[no_mangle]
pub extern "C" fn openems_set_bc_type(ptr: *mut c_void, idx: c_uint, typ: c_int) {
    clear_error();
    let _ = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver.core.set_bc_type(idx as usize, typ);
        Some(())
    });
}

/// Set the PML size for one face.
///
/// Native `Set_BC_PML` also forces that face's type to PML
/// (`openEMS/openems.cpp:824-830`); an index outside 0..5 is a no-op.
#[no_mangle]
pub extern "C" fn openems_set_pml_size(ptr: *mut c_void, idx: c_uint, size: c_uint) {
    clear_error();
    let _ = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        if (idx as usize) < 6 {
            solver.core.set_bc_type(idx as usize, BC_PML);
            solver.core.set_pml_size(idx as usize, size);
        }
        Some(())
    });
}

/// Set the MUR phase velocity for one face (native `Set_Mur_PhaseVel`,
/// `openEMS/openems.cpp:841-846`).
#[no_mangle]
pub extern "C" fn openems_set_mur_phase_vel(ptr: *mut c_void, idx: c_uint, val: c_double) {
    clear_error();
    let _ = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver.core.set_mur_phase_vel(idx as usize, val);
        Some(())
    });
}

/// Set maximum simulation time (native `SetMaxTime`, `openEMS/openems.h:77`).
#[no_mangle]
pub extern "C" fn openems_set_max_time(ptr: *mut c_void, val: c_double) {
    clear_error();
    let _ = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver.core.set_max_time(val);
        Some(())
    });
}

/// Set the timestep method (native `SetTimeStepMethod`, `openEMS/openems.h:74`).
#[no_mangle]
pub extern "C" fn openems_set_time_step_method(ptr: *mut c_void, val: c_int) {
    clear_error();
    let _ = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver.core.set_time_step_method(val);
        Some(())
    });
}

/// Set a fixed timestep (native `SetTimeStep`, `openEMS/openems.h:75`).
#[no_mangle]
pub extern "C" fn openems_set_time_step(ptr: *mut c_void, val: c_double) {
    clear_error();
    let _ = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver.core.set_time_step(val);
        Some(())
    });
}

/// Set the timestep factor (native `SetTimeStepFactor`, `openEMS/openems.h:76`).
#[no_mangle]
pub extern "C" fn openems_set_time_step_factor(ptr: *mut c_void, val: c_double) {
    clear_error();
    let _ = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver.core.set_time_step_factor(val);
        Some(())
    });
}

/// Set the oversampling factor (native `SetOverSampling`,
/// `openEMS/openems.h:67`).
#[no_mangle]
pub extern "C" fn openems_set_over_sampling(ptr: *mut c_void, val: c_uint) {
    clear_error();
    let _ = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver.core.set_over_sampling(val);
        Some(())
    });
}

/// Set cell constant material mode (native `SetCellConstantMaterial`,
/// `openEMS/openems.h:68`).
#[no_mangle]
pub extern "C" fn openems_set_cell_constant_material(ptr: *mut c_void, val: u8) {
    clear_error();
    let _ = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver.core.set_cell_constant_material(val != 0);
        Some(())
    });
}

/// Set the coordinate system (0 = Cartesian, 1 = Cylindrical).
///
/// Native `SetCoordSystem` accepts values that compare equal to 0 or 1 and
/// otherwise raises (`openEMS/python/openEMS/openEMS.pyx:174-179`). The
/// `SetCoordSystem: ` prefix belongs to that wrapper; the ABI owns the message
/// `Invalid coordinate system`.
///
/// Returns true (1) on success, false (0) on failure.
#[no_mangle]
pub extern "C" fn openems_set_coord_system(ptr: *mut c_void, val: c_int) -> u8 {
    clear_error();
    let result = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        match solver.core.set_coord_system(val) {
            Ok(()) => Some(1u8),
            Err(message) => {
                set_error("", &message);
                Some(0u8)
            }
        }
    });
    match result {
        Some(Some(ok)) => ok,
        _ => 0,
    }
}

/// Set a Gaussian pulse excitation (native `SetGaussExcite`,
/// `openEMS/openems.h:107`).
#[no_mangle]
pub extern "C" fn openems_set_gauss_excite(ptr: *mut c_void, f0: c_double, fc: c_double) {
    clear_error();
    let _ = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver.core.set_gauss_excite(f0, fc);
        Some(())
    });
}

/// Set a sinusoidal excitation (native `SetSinusExcite`,
/// `openEMS/openems.h:108`).
#[no_mangle]
pub extern "C" fn openems_set_sinus_excite(ptr: *mut c_void, f0: c_double) {
    clear_error();
    let _ = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver.core.set_sinus_excite(f0);
        Some(())
    });
}

/// Set a Dirac pulse excitation (native `SetDiracExcite`,
/// `openEMS/openems.h:109`).
#[no_mangle]
pub extern "C" fn openems_set_dirac_excite(ptr: *mut c_void, f_max: c_double) {
    clear_error();
    let _ = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver.core.set_dirac_excite(f_max);
        Some(())
    });
}

/// Set a step function excitation (native `SetStepExcite`,
/// `openEMS/openems.h:110`).
#[no_mangle]
pub extern "C" fn openems_set_step_excite(ptr: *mut c_void, f_max: c_double) {
    clear_error();
    let _ = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver.core.set_step_excite(f_max);
        Some(())
    });
}

/// Set the CSXCAD geometry from an XML document.
///
/// A null pointer clears the geometry, which is what native `SetCSX(NULL)` does
/// (`openEMS/openems.cpp:1163-1170`: `delete m_CSX; m_CSX = csx;` is defined for
/// `csx == NULL`). Setting geometry replaces any structure previously handed to
/// [`openems_set_csx`], because native has a single geometry slot.
#[no_mangle]
pub extern "C" fn openems_set_csx_xml(ptr: *mut c_void, xml: *const c_char) {
    clear_error();
    let _ = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver.csx = None;
        if xml.is_null() {
            solver.core.clear_csx_xml();
        } else {
            // Safety: the caller passes a NUL-terminated string for the duration
            // of the call.
            let text = unsafe { CStr::from_ptr(xml) }
                .to_string_lossy()
                .into_owned();
            solver.core.set_csx_xml(text);
        }
        Some(())
    });
}

/// Hand a CSXCAD structure to the solver, transferring ownership.
///
/// Native contract (`openEMS/openems.h:115-119`,
/// `openEMS/openems.cpp:1163-1170`, `openEMS/python/openEMS/openEMS.pyx:519-536`):
/// the solver destroys the structure it is given as well as any structure set
/// before, the caller must not free it afterwards, and `GetCSX` keeps returning
/// the same pointer.
///
/// `csx` must be a pointer produced by `csxcad_new` (`Box<Structure>`); a null
/// pointer clears the geometry. Passing the pointer that is already owned is a
/// no-op: native deletes the stored pointer before storing the new one, so
/// re-setting the same pointer would be a double delete. When the call fails
/// (invalid solver handle) ownership stays with the caller.
///
/// Returns true (1) on success, false (0) on failure.
#[no_mangle]
pub extern "C" fn openems_set_csx(ptr: *mut c_void, csx: *mut c_void) -> u8 {
    clear_error();
    let result = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        if csx.is_null() {
            solver.csx = None;
            solver.core.clear_csx_xml();
            return Some(1u8);
        }
        if let Some(existing) = solver.csx.as_ref() {
            if std::ptr::eq(existing.as_ref(), csx as *const Structure) {
                return Some(1u8);
            }
        }
        // Safety: the caller transfers a Box<Structure> allocated by
        // `csxcad_new`, which uses the same `csxcad_core::model::Structure`.
        solver.csx = Some(unsafe { Box::from_raw(csx as *mut Structure) });
        solver.core.clear_csx_xml();
        Some(1u8)
    });
    match result {
        Some(Some(ok)) => ok,
        _ => 0,
    }
}

/// Get the geometry the solver owns (native `GetCSX`, `openEMS/openems.h:118-119`).
///
/// Returns the owned pointer, or null when no geometry is set (native returns
/// `NULL` and the Python wrapper turns that into `None`,
/// `openEMS/python/openEMS/openEMS.pyx:539-543`). The returned pointer stays
/// valid until the geometry is replaced or the solver is freed, like native.
#[no_mangle]
pub extern "C" fn openems_get_csx(ptr: *mut c_void) -> *mut c_void {
    clear_error();
    let result = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        Some(
            solver
                .csx
                .as_ref()
                .map_or(std::ptr::null_mut(), |structure| {
                    structure.as_ref() as *const Structure as *mut c_void
                }),
        )
    });
    match result {
        Some(Some(value)) => value,
        _ => std::ptr::null_mut(),
    }
}

/// Set all six boundary conditions from an array.
///
/// Native contract: exactly six entries
/// (`openEMS/matlab/SetBoundaryCond.m:51-52`,
/// `openEMS/python/openEMS/openEMS.pyx:320-321`) ordered xmin, xmax, ymin,
/// ymax, zmin, zmax (`openEMS/matlab/SetBoundaryCond.m:55-61`). Entry `i` is
/// face `i`, so the array is zero-based at the C boundary; MATLAB's one-based
/// `BC(1)..BC(6)` map onto the same six faces. Values are forwarded without enum
/// validation, exactly like native.
///
/// Returns true (1) on success, false (0) on failure.
#[no_mangle]
pub extern "C" fn openems_set_boundary_cond(ptr: *mut c_void, bc: *const c_int, len: c_uint) -> u8 {
    clear_error();
    let result = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        if bc.is_null() {
            // Native raises this identifier when the argument is neither a
            // numeric nor a cell array (openEMS/matlab/SetBoundaryCond.m:69-70);
            // at the C boundary that is an absent buffer.
            set_error("openEMS:SetBoundaryCond", "unknown boundary condition type");
            return Some(0u8);
        }
        if len as usize != 6 {
            set_error(
                "openEMS:SetBoundaryCond",
                "wrong number of boundary conditions",
            );
            return Some(0u8);
        }
        // Safety: the caller passes six readable c_int values.
        let values = unsafe { std::slice::from_raw_parts(bc, 6) };
        for (face, value) in values.iter().enumerate() {
            solver.core.set_bc_type(face, *value);
        }
        Some(1u8)
    });
    match result {
        Some(Some(ok)) => ok,
        _ => 0,
    }
}

/// Read all six boundary conditions into a caller-provided array of six ints.
///
/// The native caller this mirrors reads the six faces in the same order
/// (`openEMS/python/openEMS/openEMS.pyx:488-489`, `CreateNF2FFBox`). Exactly six
/// values are written; nothing past index 5 is touched.
///
/// Returns true (1) on success, false (0) on failure.
#[no_mangle]
pub extern "C" fn openems_get_boundary_cond(ptr: *mut c_void, out: *mut c_int, len: c_uint) -> u8 {
    clear_error();
    let result = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        if out.is_null() || len as usize != 6 {
            set_error(
                "openEMS:SetBoundaryCond",
                "wrong number of boundary conditions",
            );
            return Some(0u8);
        }
        // Safety: the caller passes room for six c_int values.
        let values = unsafe { std::slice::from_raw_parts_mut(out, 6) };
        for (face, value) in values.iter_mut().enumerate() {
            *value = solver.core.get_bc_type(face);
        }
        Some(1u8)
    });
    match result {
        Some(Some(ok)) => ok,
        _ => 0,
    }
}

/// Initialize the FDTD operator from the CSX geometry.
///
/// Returns true (1) on success. On failure it returns 0 and records the native
/// code and message: 3 with `openEMS::SetupFDTD: Error: CSXCAD is not set!` when
/// no geometry was given (`openEMS/openems.cpp:1184-1188`), and 2 when the
/// operator setup fails (`openEMS/openems.cpp:1213-1217`, which native reports
/// without a message).
#[no_mangle]
pub extern "C" fn openems_setup_fdtd(ptr: *mut c_void) -> u8 {
    clear_error();
    let result = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        if solver.csx.is_none() && solver.core.csx_xml.is_none() {
            set_error_code("", "openEMS::SetupFDTD: Error: CSXCAD is not set!", 3);
            return Some(0u8);
        }
        if let Some(structure) = solver.csx.as_ref() {
            let xml = match structure.to_xml_string() {
                Ok(xml) => xml,
                Err(error) => {
                    // CSXCAD serialisation failure; native reports a failed
                    // SetupOperator with code 2 (openEMS/openems.cpp:1213-1217).
                    set_error_code("", &error.to_string(), 2);
                    return Some(0u8);
                }
            };
            solver.core.set_csx_xml(xml);
        }
        if solver.core.setup_fdtd() {
            Some(1u8)
        } else {
            set_error_code("", "", 2);
            Some(0u8)
        }
    });
    match result {
        Some(Some(ok)) => ok,
        _ => 0,
    }
}

/// Run the FDTD simulation.
///
/// Native `RunFDTD` returns void (`openEMS/openems.h:60`); the ABI reports the
/// core's outcome as a boolean. A call before a successful setup fails without
/// fabricating an error code, because native has no defined behaviour for it.
#[no_mangle]
pub extern "C" fn openems_run_fdtd(ptr: *mut c_void) -> u8 {
    clear_error();
    let result = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        solver.core.run_fdtd().then_some(1u8)
    });
    match result {
        Some(Some(ok)) => ok,
        _ => 0,
    }
}

/// Get the number of time steps executed so far.
///
/// ABI extension over the native engine accessor (`FDTD/engine.h:50`,
/// `GetNumberOfTimesteps`); the controller itself does not expose it.
#[no_mangle]
pub extern "C" fn openems_get_number_of_timesteps(ptr: *mut c_void) -> c_uint {
    clear_error();
    let result = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        Some(
            solver
                .core
                .fdtd_eng
                .as_ref()
                .map_or(0, |engine| engine.get_number_of_timesteps() as c_uint),
        )
    });
    match result {
        Some(Some(n)) => n,
        _ => 0,
    }
}

/// Get the boundary condition type of one face.
///
/// Native `Get_BC_Type` returns -1 for an index outside 0..5
/// (`openEMS/openems.cpp:817-822`); the ABI returns the same sentinel for an
/// out-of-range index and for an invalid handle.
#[no_mangle]
pub extern "C" fn openems_get_bc_type(ptr: *mut c_void, idx: c_uint) -> c_int {
    clear_error();
    let result = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        Some(if (idx as usize) < 6 {
            solver.core.get_bc_type(idx as usize)
        } else {
            NATIVE_INVALID_INT
        })
    });
    match result {
        Some(Some(value)) => value,
        _ => NATIVE_INVALID_INT,
    }
}

/// Get the PML size of one face.
///
/// Native `Get_PML_Size` returns -1 for an index outside 0..5 and for a face
/// that is not a PML (`openEMS/openems.cpp:832-839`). The native return type is
/// `int` (`openEMS/openEMS.pxd`, `int Get_PML_Size(int idx)`), so -1 crosses the
/// boundary as its two's-complement bit pattern; the constant
/// [`NATIVE_INVALID_INT`] cast to `c_uint` is the value to compare against.
#[no_mangle]
pub extern "C" fn openems_get_pml_size(ptr: *mut c_void, idx: c_uint) -> c_uint {
    clear_error();
    let result = catch_unwind(|| {
        let solver = solver_mut(ptr)?;
        let value = if (idx as usize) < 6 && solver.core.get_bc_type(idx as usize) == BC_PML {
            solver.core.get_pml_size(idx as usize) as c_int
        } else {
            NATIVE_INVALID_INT
        };
        Some(value as c_uint)
    });
    match result {
        Some(Some(value)) => value,
        _ => NATIVE_INVALID_INT as c_uint,
    }
}

/// Identifier of the error recorded by the last exported call, or null when the
/// last call succeeded. Free the returned string with [`openems_free_string`].
#[no_mangle]
pub extern "C" fn openems_last_error_id() -> *mut c_char {
    error_string(last_error().id)
}

/// Message of the error recorded by the last exported call, or null when the
/// last call succeeded. Free the returned string with [`openems_free_string`].
#[no_mangle]
pub extern "C" fn openems_last_error_message() -> *mut c_char {
    error_string(last_error().message)
}

/// Native numeric error code of the last exported call (0 when the native path
/// has no code or the call succeeded).
#[no_mangle]
pub extern "C" fn openems_last_error_code() -> c_int {
    last_error().code
}

/// Free a string returned by this ABI. A null pointer is a no-op.
#[no_mangle]
pub extern "C" fn openems_free_string(ptr: *mut c_char) {
    if ptr.is_null() {
        return;
    }
    let _ = catch_unwind(|| {
        // Safety: the pointer was produced by `CString::into_raw` in
        // `error_string`.
        unsafe { drop(CString::from_raw(ptr)) };
    });
}

#[cfg(test)]
mod abi_inventory;
#[cfg(test)]
mod contract_tests;
