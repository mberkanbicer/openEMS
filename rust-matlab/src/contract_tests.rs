//! Contract tests for the MATLAB/Octave C ABI (RM-094).
//!
//! The MATLAB delivery path is the unchanged M-file API plus this C ABI
//! (`libopenems_matlab`), which a MEX shim or an embedding host calls. Each test
//! below pins one behaviour the native sources define and asserts the real
//! outcome: the return value, the native error identifier and the native message
//! text. No assertion is relaxed to "does not crash".
//!
//! Native references are quoted per test. The shared rules are:
//! * `openEMS/openems.cpp:810-846` — the `Set_BC_*`/`Get_BC_*`/`Set_Mur_PhaseVel`
//!   face accessors, including the `-1` sentinel and the silent index no-op.
//! * `openEMS/openems.cpp:1163-1175` — `SetCSX` ownership transfer, `GetCSX`
//!   pointer identity.
//! * `openEMS/openems.cpp:1177-1234` — `SetupFDTD` error codes 2 and 3.
//! * `openEMS/matlab/SetBoundaryCond.m:51-71` — the six-entry boundary contract
//!   and its `openEMS:SetBoundaryCond` identifiers.
//! * `openEMS/python/openEMS/openEMS.pyx:169-179` — `SetCoordSystem` validation.
//! * `CSXCAD/python/CSXCAD/CSXCAD.pyx:118` — the deleted-wrapper message.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};

use csxcad_core::model::Structure;
use openems_core::{BC_MUR, BC_PEC, BC_PMC, BC_PML};

use crate::*;

/// Minimal geometry that the core operator accepts: four lines per axis at a
/// 1 mm delta unit (the shape used by the core's own setup tests,
/// `openEMS/rust/src/openems.rs:1987-2004`).
const SMALL_CSX: &str = r#"<ContinuousStructure CoordSystem="0">
    <RectilinearGrid DeltaUnit="0.001">
        <XLines>0,1,2,3</XLines>
        <YLines>0,1,2,3</YLines>
        <ZLines>0,1,2,3</ZLines>
    </RectilinearGrid>
    <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
    <Properties></Properties>
</ContinuousStructure>"#;

/// A geometry the operator rejects: a single line per axis cannot build a mesh
/// (`openEMS/rust/src/operator.rs`, `load_csx` "insufficient lines").
const ONE_LINE_CSX: &str = r#"<ContinuousStructure CoordSystem="0">
    <RectilinearGrid DeltaUnit="0.001">
        <XLines>0</XLines>
        <YLines>0</YLines>
        <ZLines>0</ZLines>
    </RectilinearGrid>
    <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
    <Properties></Properties>
</ContinuousStructure>"#;

/// Native message the ABI reports when no geometry is set
/// (`openEMS/openems.cpp:1186`).
const CSX_NOT_SET: &str = "openEMS::SetupFDTD: Error: CSXCAD is not set!";

fn take_string(ptr: *mut c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    // Safety: the pointer was produced by this ABI and freed exactly once here.
    let text = unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned();
    openems_free_string(ptr);
    text
}

/// Read and clear the native error triple recorded by the last ABI call.
fn last_error() -> (String, String, c_int) {
    let id = take_string(openems_last_error_id());
    let message = take_string(openems_last_error_message());
    let code = openems_last_error_code();
    (id, message, code)
}

fn assert_no_error(context: &str) {
    let (id, message, code) = last_error();
    assert!(id.is_empty(), "{context}: unexpected error id {id:?}");
    assert!(
        message.is_empty(),
        "{context}: unexpected error message {message:?}"
    );
    assert_eq!(code, 0, "{context}: unexpected error code");
}

/// The native "deleted object" rule: `CSXCAD/python/CSXCAD/CSXCAD.pyx:118`
/// raises `RuntimeError('wrapped C++ object of type {} has been deleted')`, and
/// `openEMS/python/openEMS/openEMS.pyx:115-117` states that destroying the
/// controller invalidates its wrapper. No native source gives this condition an
/// `openEMS:` identifier, so the identifier is empty.
fn assert_deleted_error(context: &str) {
    let (id, message, code) = last_error();
    assert!(
        id.is_empty(),
        "{context}: native deleted-object errors carry no identifier, got {id:?}"
    );
    assert_eq!(message, crate::ERR_DELETED_MESSAGE, "{context}");
    assert_eq!(code, 0, "{context}");
}

fn boxed_csx(xml: &str) -> *mut c_void {
    let structure = Structure::from_xml_str(xml).expect("fixture CSXCAD XML parses");
    Box::into_raw(Box::new(structure)) as *mut c_void
}

fn c_string(value: &str) -> CString {
    CString::new(value).expect("test string has no NUL")
}

// ---------------------------------------------------------------------------
// Handles, double free, use after free
// ---------------------------------------------------------------------------

#[test]
fn null_handle_is_rejected_everywhere() {
    let null: *mut c_void = std::ptr::null_mut();

    openems_set_number_of_time_steps(null, 10);
    assert_deleted_error("set_number_of_time_steps");
    openems_set_end_criteria(null, 1e-5);
    assert_deleted_error("set_end_criteria");
    openems_set_enable_dumps(null, 1);
    assert_deleted_error("set_enable_dumps");
    openems_set_cylinder_coords(null, 1);
    assert_deleted_error("set_cylinder_coords");
    openems_set_bc_type(null, 0, BC_PEC);
    assert_deleted_error("set_bc_type");
    openems_set_pml_size(null, 0, 8);
    assert_deleted_error("set_pml_size");
    openems_set_mur_phase_vel(null, 0, 3e8);
    assert_deleted_error("set_mur_phase_vel");
    openems_set_max_time(null, 1e-9);
    assert_deleted_error("set_max_time");
    openems_set_time_step_method(null, 3);
    assert_deleted_error("set_time_step_method");
    openems_set_time_step(null, 1e-12);
    assert_deleted_error("set_time_step");
    openems_set_time_step_factor(null, 0.5);
    assert_deleted_error("set_time_step_factor");
    openems_set_over_sampling(null, 4);
    assert_deleted_error("set_over_sampling");
    openems_set_cell_constant_material(null, 1);
    assert_deleted_error("set_cell_constant_material");
    openems_set_gauss_excite(null, 0.0, 1e9);
    assert_deleted_error("set_gauss_excite");
    openems_set_sinus_excite(null, 1e9);
    assert_deleted_error("set_sinus_excite");
    openems_set_dirac_excite(null, 1e9);
    assert_deleted_error("set_dirac_excite");
    openems_set_step_excite(null, 1e9);
    assert_deleted_error("set_step_excite");
    openems_set_csx_xml(null, std::ptr::null());
    assert_deleted_error("set_csx_xml");
    openems_set_boundary_cond(null, std::ptr::null(), 6);
    assert_deleted_error("set_boundary_cond");

    // Value-returning entry points report the native sentinel plus the error.
    assert_eq!(openems_set_coord_system(null, 0), 0);
    assert_deleted_error("set_coord_system");
    assert_eq!(openems_set_csx(null, std::ptr::null_mut()), 0);
    assert_deleted_error("set_csx");
    assert!(openems_get_csx(null).is_null());
    assert_deleted_error("get_csx");
    assert_eq!(openems_setup_fdtd(null), 0);
    assert_deleted_error("setup_fdtd");
    assert_eq!(openems_run_fdtd(null), 0);
    assert_deleted_error("run_fdtd");
    assert_eq!(openems_get_number_of_timesteps(null), 0);
    assert_deleted_error("get_number_of_timesteps");
    assert_eq!(openems_get_bc_type(null, 0), -1);
    assert_deleted_error("get_bc_type");
    assert_eq!(openems_get_pml_size(null, 0) as i32, -1);
    assert_deleted_error("get_pml_size");
    let mut out = [0 as c_int; 6];
    assert_eq!(openems_get_boundary_cond(null, out.as_mut_ptr(), 6), 0);
    assert_deleted_error("get_boundary_cond");

    // Freeing null is the native `delete NULL` no-op and records no error.
    openems_free(null);
    assert_no_error("free(null)");
}

#[test]
fn double_free_is_a_defined_no_op() {
    let solver = openems_new();
    assert!(!solver.is_null());
    assert!(crate::is_live(solver));

    openems_free(solver);
    assert!(!crate::is_live(solver));
    assert_no_error("first free");

    // Native `delete` on a stale pointer is undefined behaviour
    // (openEMS/openems.cpp:96-99); the ABI makes the second call a no-op instead
    // of a double free, and the handle stays dead.
    openems_free(solver);
    assert!(!crate::is_live(solver));
    assert_eq!(openems_get_bc_type(solver, 0), -1);
    assert_deleted_error("use after double free");
}

#[test]
fn use_after_free_reports_the_deleted_object_error() {
    let first = openems_new();
    let second = openems_new();
    assert!(!first.is_null() && !second.is_null());
    assert_ne!(first, second, "each solver gets its own handle");

    openems_set_bc_type(second, 0, BC_MUR);
    assert_eq!(openems_get_bc_type(second, 0), BC_MUR);

    openems_free(first);

    // The freed handle is reported, not dereferenced.
    assert_eq!(openems_get_bc_type(first, 0), -1);
    assert_deleted_error("use after free");

    // The other handle is unaffected.
    assert_eq!(openems_get_bc_type(second, 0), BC_MUR);
    assert_no_error("surviving handle");

    openems_free(second);
}

#[test]
fn successful_call_clears_the_previous_error() {
    let solver = openems_new();
    assert_eq!(openems_set_coord_system(solver, 7), 0);
    let (_, message, _) = last_error();
    assert_eq!(message, "Invalid coordinate system");

    openems_set_bc_type(solver, 0, BC_PMC);
    assert_no_error("a successful call clears the error channel");
    assert_eq!(openems_get_bc_type(solver, 0), BC_PMC);

    openems_free(solver);
}

// ---------------------------------------------------------------------------
// Face indices, PML and the native sentinels
// ---------------------------------------------------------------------------

#[test]
fn out_of_range_face_index_is_the_native_silent_no_op() {
    let solver = openems_new();
    let mut bc = [0 as c_int; 6];
    openems_set_boundary_cond(solver, bc.as_ptr(), 6);
    for (face, value) in bc.iter_mut().enumerate() {
        *value = (face % 3) as c_int;
        openems_set_bc_type(solver, face as u32, *value);
    }
    assert_no_error("in-range setters");

    // `openEMS/openems.cpp:812-813`: idx outside 0..5 returns without touching
    // any face and without raising.
    openems_set_bc_type(solver, 6, BC_PML);
    openems_set_bc_type(solver, 1000, BC_PML);
    assert_no_error("out-of-range set_bc_type");
    openems_set_pml_size(solver, 6, 50);
    assert_no_error("out-of-range set_pml_size");
    openems_set_mur_phase_vel(solver, 6, 1.0);
    assert_no_error("out-of-range set_mur_phase_vel");

    for (face, expected) in bc.iter().enumerate() {
        assert_eq!(
            openems_get_bc_type(solver, face as u32),
            *expected,
            "face {face} was changed by an out-of-range write"
        );
    }

    openems_free(solver);
}

#[test]
fn get_bc_type_returns_the_native_out_of_range_sentinel() {
    let solver = openems_new();
    // `openEMS/openems.cpp:817-822`: -1 for idx outside 0..5, no error state.
    assert_eq!(openems_get_bc_type(solver, 6), -1);
    assert_eq!(openems_get_bc_type(solver, u32::MAX), -1);
    assert_no_error("out-of-range read");
    openems_free(solver);
}

#[test]
fn calls_cross_the_boundary_as_native_index_sentinels() {
    // The C boundary carries the native `int` sentinels: a c_uint of
    // u32::MAX arrives as the native -1 (openEMS/openems.cpp:817-822, :832-839).
    let solver = openems_new();
    assert_eq!(openems_get_pml_size(solver, u32::MAX) as c_int, -1);
    assert_eq!(openems_get_bc_type(solver, u32::MAX), -1);
    openems_free(solver);
}

#[test]
fn pml_defaults_follow_the_native_reset() {
    let solver = openems_new();

    // `openEMS/openems.cpp:836-838`: a face that is not a PML reports -1 even
    // though its stored size is 8 (`openEMS/openems.cpp:130`).
    for face in 0..6u32 {
        assert_eq!(
            openems_get_pml_size(solver, face) as c_int,
            -1,
            "face {face} is PEC by default (openEMS/openems.cpp:127-132)"
        );
    }

    // Turning a face into a PML exposes the native default size 8.
    openems_set_bc_type(solver, 0, BC_PML);
    assert_eq!(openems_get_bc_type(solver, 0), BC_PML);
    assert_eq!(openems_get_pml_size(solver, 0), 8);

    // Switching back hides the size again.
    openems_set_bc_type(solver, 0, BC_PEC);
    assert_eq!(openems_get_pml_size(solver, 0) as c_int, -1);

    openems_free(solver);
}

#[test]
fn set_pml_size_also_forces_the_face_type_to_pml() {
    let solver = openems_new();

    // `openEMS/openems.cpp:824-830`: Set_BC_PML writes m_BC_type[idx] = 3.
    openems_set_pml_size(solver, 2, 12);
    assert_eq!(openems_get_bc_type(solver, 2), BC_PML);
    assert_eq!(openems_get_pml_size(solver, 2), 12);
    assert_no_error("set_pml_size");

    // Faces that were not addressed keep their state.
    assert_eq!(openems_get_bc_type(solver, 3), BC_PEC);
    assert_eq!(openems_get_pml_size(solver, 3) as c_int, -1);

    openems_free(solver);
}

// ---------------------------------------------------------------------------
// Argument validation that the native sources define
// ---------------------------------------------------------------------------

#[test]
fn coordinate_system_accepts_only_the_native_values() {
    let solver = openems_new();

    // `openEMS/python/openEMS/openEMS.pyx:174-179`.
    assert_eq!(openems_set_coord_system(solver, 0), 1);
    assert_no_error("cartesian");
    assert_eq!(openems_set_coord_system(solver, 1), 1);
    assert_no_error("cylindrical");

    // The native wrapper builds `SetCoordSystem: Invalid coordinate system`;
    // the `SetCoordSystem: ` prefix belongs to the wrapper, the ABI owns the
    // message and has no identifier for this path.
    for invalid in [2, -1, 3] {
        assert_eq!(openems_set_coord_system(solver, invalid), 0);
        assert_eq!(
            last_error(),
            (String::new(), "Invalid coordinate system".to_string(), 0)
        );
    }

    openems_free(solver);
}

#[test]
fn boundary_cond_array_requires_six_entries() {
    let solver = openems_new();
    let values = [1 as c_int; 7];

    // `openEMS/matlab/SetBoundaryCond.m:51-52` and
    // `openEMS/python/openEMS/openEMS.pyx:320-321`.
    for len in [0u32, 5, 7, 8] {
        assert_eq!(openems_set_boundary_cond(solver, values.as_ptr(), len), 0);
        assert_eq!(
            last_error(),
            (
                "openEMS:SetBoundaryCond".to_string(),
                "wrong number of boundary conditions".to_string(),
                0
            ),
            "len {len}"
        );
    }

    // A rejected call leaves every face untouched.
    for face in 0..6u32 {
        assert_eq!(openems_get_bc_type(solver, face), BC_PEC);
    }

    openems_free(solver);
}

#[test]
fn boundary_cond_array_is_zero_based_in_xmin_to_zmax_order() {
    let solver = openems_new();

    // `openEMS/matlab/SetBoundaryCond.m:55-61` maps BC(1..6) to xmin, xmax,
    // ymin, ymax, zmin, zmax; the Python loop uses the same order zero-based
    // (`openEMS/python/openEMS/openEMS.pyx:322-331`).
    let bc: [c_int; 6] = [BC_PMC, BC_PMC, BC_PEC, BC_PEC, BC_MUR, BC_PML];
    assert_eq!(openems_set_boundary_cond(solver, bc.as_ptr(), 6), 1);
    assert_no_error("valid boundary array");

    for (face, expected) in bc.iter().enumerate() {
        assert_eq!(
            openems_get_bc_type(solver, face as u32),
            *expected,
            "array entry {face} must land on face {face}"
        );
    }
    // Entry 3 is PEC, entry 5 became a PML with the native default size.
    assert_eq!(openems_get_pml_size(solver, 3) as c_int, -1);
    assert_eq!(openems_get_pml_size(solver, 5), 8);

    // The read-back writes exactly six ints and leaves the tail of the caller's
    // buffer untouched (it is written into a MATLAB-sized buffer here).
    let mut out: [c_int; 8] = [777; 8];
    assert_eq!(openems_get_boundary_cond(solver, out.as_mut_ptr(), 6), 1);
    assert_no_error("read back");
    assert_eq!(&out[..6], &bc[..]);
    assert_eq!(&out[6..], &[777, 777]);

    // A short read buffer is the same native shape error.
    assert_eq!(openems_get_boundary_cond(solver, out.as_mut_ptr(), 5), 0);
    assert_eq!(
        last_error(),
        (
            "openEMS:SetBoundaryCond".to_string(),
            "wrong number of boundary conditions".to_string(),
            0
        )
    );

    openems_free(solver);
}

#[test]
fn boundary_cond_null_array_is_the_native_unknown_type_error() {
    let solver = openems_new();

    // `openEMS/matlab/SetBoundaryCond.m:69-70` raises this identifier when the
    // argument is neither numeric nor a cell array; at the C boundary that is an
    // absent buffer.
    assert_eq!(openems_set_boundary_cond(solver, std::ptr::null(), 6), 0);
    assert_eq!(
        last_error(),
        (
            "openEMS:SetBoundaryCond".to_string(),
            "unknown boundary condition type".to_string(),
            0
        )
    );

    let mut out = [0 as c_int; 6];
    assert_eq!(openems_get_boundary_cond(solver, out.as_mut_ptr(), 6), 1);
    openems_free(solver);
}

#[test]
fn boundary_cond_values_are_forwarded_without_enum_validation() {
    let solver = openems_new();

    // Native stores the integer verbatim: `openEMS/openems.cpp:810-815` has no
    // enum check and `openEMS/python/openEMS/openEMS.pyx:322-325` forwards
    // built-in integers unchecked.
    for face in 0..6u32 {
        openems_set_bc_type(solver, face, 99);
        assert_eq!(openems_get_bc_type(solver, face), 99);
        assert_no_error("face 99");
    }

    openems_free(solver);
}

// ---------------------------------------------------------------------------
// Setup codes and CSX ownership
// ---------------------------------------------------------------------------

#[test]
fn setup_without_geometry_returns_the_native_error_code() {
    let solver = openems_new();
    assert!(openems_get_csx(solver).is_null());
    assert_no_error("fresh solver has no geometry");

    // `openEMS/openems.cpp:1184-1188`.
    assert_eq!(openems_setup_fdtd(solver), 0);
    assert_eq!(last_error(), (String::new(), CSX_NOT_SET.to_string(), 3));

    openems_free(solver);
}

#[test]
fn setup_with_unusable_geometry_returns_code_two() {
    let solver = openems_new();
    let xml = c_string(ONE_LINE_CSX);
    openems_set_csx_xml(solver, xml.as_ptr());
    assert_no_error("geometry accepted for later setup");

    // `openEMS/openems.cpp:1213-1217` reports a failed SetupOperator as code 2
    // and prints no message.
    assert_eq!(openems_setup_fdtd(solver), 0);
    assert_eq!(last_error(), (String::new(), String::new(), 2));

    openems_free(solver);
}

#[test]
fn setup_uses_a_transferred_structure() {
    let solver = openems_new();
    let csx = boxed_csx(SMALL_CSX);

    assert_eq!(openems_set_csx(solver, csx), 1);
    assert_no_error("set_csx");
    assert_eq!(openems_get_csx(solver), csx, "GetCSX pointer identity");

    // The transferred structure is the geometry the operator is built from.
    assert_eq!(openems_setup_fdtd(solver), 1);
    assert_no_error("setup with the owned structure");
    assert_eq!(
        openems_get_csx(solver),
        csx,
        "GetCSX is stable across setup"
    );
    assert_eq!(openems_get_number_of_timesteps(solver), 0);

    openems_free(solver);
}

#[test]
fn csx_ownership_round_trip_matches_native() {
    let solver = openems_new();
    assert!(openems_get_csx(solver).is_null());

    let first = boxed_csx(SMALL_CSX);
    assert_eq!(openems_set_csx(solver, first), 1);
    assert_eq!(openems_get_csx(solver), first);

    // `openEMS/openems.cpp:1165-1166`: the previous structure is destroyed and
    // the new one becomes the single geometry slot.
    let second = boxed_csx(SMALL_CSX);
    assert_eq!(openems_set_csx(solver, second), 1);
    assert_eq!(openems_get_csx(solver), second);
    assert_ne!(openems_get_csx(solver), first);

    // `openEMS/openems.cpp:1163-1170` with a null argument clears the geometry.
    assert_eq!(openems_set_csx(solver, std::ptr::null_mut()), 1);
    assert!(openems_get_csx(solver).is_null());
    assert_no_error("clear geometry");

    assert_eq!(openems_setup_fdtd(solver), 0);
    assert_eq!(last_error(), (String::new(), CSX_NOT_SET.to_string(), 3));

    openems_free(solver);
}

#[test]
fn setting_the_same_csx_twice_keeps_one_owner() {
    let solver = openems_new();
    let csx = boxed_csx(SMALL_CSX);

    assert_eq!(openems_set_csx(solver, csx), 1);
    // Native `SetCSX` deletes the stored pointer before assigning the new one
    // (`openEMS/openems.cpp:1165-1166`), so passing the same pointer twice would
    // double delete. The ABI keeps the existing owner and reports success.
    assert_eq!(openems_set_csx(solver, csx), 1);
    assert_no_error("idempotent set_csx");
    assert_eq!(openems_get_csx(solver), csx);
    assert_eq!(openems_setup_fdtd(solver), 1);

    // Dropping the solver releases it exactly once.
    openems_free(solver);
}

#[test]
fn geometry_has_a_single_slot_like_native() {
    let solver = openems_new();
    let csx = boxed_csx(SMALL_CSX);
    assert_eq!(openems_set_csx(solver, csx), 1);
    assert_eq!(openems_get_csx(solver), csx);

    // Native has one `m_CSX` slot (openEMS/openems.h:135), so handing over the
    // XML form replaces (and destroys) the owned structure.
    let xml = c_string(SMALL_CSX);
    openems_set_csx_xml(solver, xml.as_ptr());
    assert!(openems_get_csx(solver).is_null());
    assert_no_error("xml replaces the structure");
    assert_eq!(openems_setup_fdtd(solver), 1, "the XML geometry is used");

    // A null XML pointer is native `SetCSX(NULL)`.
    openems_set_csx_xml(solver, std::ptr::null());
    assert_eq!(openems_setup_fdtd(solver), 0);
    assert_eq!(last_error(), (String::new(), CSX_NOT_SET.to_string(), 3));

    openems_free(solver);
}

#[test]
fn run_before_setup_fails_without_fabricating_an_error() {
    let solver = openems_new();

    // Native `RunFDTD` returns void (openEMS/openems.h:60) and expects a prior
    // setup; the ABI reports the core's refusal as 0 and does not invent an
    // identifier, message or code for a path native leaves undefined.
    assert_eq!(openems_run_fdtd(solver), 0);
    assert_no_error("run before setup");

    openems_free(solver);
}

// ---------------------------------------------------------------------------
// Excitation setters (native contract: no wrapper-side validation)
// ---------------------------------------------------------------------------

#[test]
fn excitation_setters_have_no_native_validation() {
    let solver = openems_new();

    // The M-file wrappers only write FDTD attributes and the Python methods
    // forward directly (`openEMS/matlab/SetGaussExcite.m`,
    // `openEMS/python/openEMS/openEMS.pyx:257-292`); no setter can report an
    // error for any finite value.
    openems_set_gauss_excite(solver, 0.0, 10e9);
    assert_no_error("gauss");
    openems_set_sinus_excite(solver, 5e9);
    assert_no_error("sinus");
    openems_set_dirac_excite(solver, 20e9);
    assert_no_error("dirac");
    openems_set_step_excite(solver, 20e9);
    assert_no_error("step");
    openems_set_gauss_excite(solver, -1.0, -1.0);
    assert_no_error("negative gauss values are forwarded");

    openems_free(solver);
}
