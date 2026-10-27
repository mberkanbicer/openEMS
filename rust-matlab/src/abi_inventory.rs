//! Inventory of the MATLAB/Octave C ABI (RM-094).
//!
//! Every exported symbol is mapped to the native contract it implements, with
//! the file and line that defines it. The test below fails if an export is
//! removed, renamed, or added without a native contract, and the signature
//! table fails if one of the C signatures a MEX layer compiles against changes.
//!
//! Native sources:
//! * `openEMS/openems.h`, `openEMS/openems.cpp` — the C++ controller.
//! * `openEMS/python/openEMS/openEMS.pyx`, `openEMS.pxd` — the Python
//!   controller (the same controller surface the M-files configure).
//! * `openEMS/matlab/SetBoundaryCond.m` — the M-file boundary contract.
//! * `CSXCAD/python/CSXCAD/CSXCAD.pyx` — the invalid/deleted wrapper rule.

use std::os::raw::{c_char, c_double, c_int, c_uint, c_void};

use crate::*;

/// One exported entry point plus the native contract it implements.
struct AbiEntry {
    /// C symbol name.
    symbol: &'static str,
    /// Native API it corresponds to (`none` when the ABI has no native peer).
    native: &'static str,
    /// Defining native source line, or the reason the ABI owns the contract.
    citation: &'static str,
}

const ABI_INVENTORY: &[AbiEntry] = &[
    AbiEntry {
        symbol: "openems_new",
        native: "openEMS::openEMS() / openEMS::Reset()",
        citation: "openEMS/openems.cpp:69-133 (defaults incl. PML size 8 at :130); \
                   openEMS/python/openEMS/openEMS.pyx:76-113 (constructor defaults)",
    },
    AbiEntry {
        symbol: "openems_free",
        native: "openEMS::~openEMS()",
        citation: "openEMS/openems.cpp:96-111 (destructor resets and deletes m_CSX); \
                   openEMS/python/openEMS/openEMS.pyx:115-117",
    },
    AbiEntry {
        symbol: "openems_set_number_of_time_steps",
        native: "openEMS::SetNumberOfTimeSteps",
        citation: "openEMS/openems.h:64; openEMS/python/openEMS/openEMS.pyx:124-129",
    },
    AbiEntry {
        symbol: "openems_set_end_criteria",
        native: "openEMS::SetEndCriteria",
        citation: "openEMS/openems.h:66; openEMS/python/openEMS/openEMS.pyx:131-136",
    },
    AbiEntry {
        symbol: "openems_set_enable_dumps",
        native: "openEMS::SetEnableDumps",
        citation: "openEMS/openems.h:65 (no Python wrapper; CLI --disable-dumps)",
    },
    AbiEntry {
        symbol: "openems_set_cylinder_coords",
        native: "openEMS::SetCylinderCoords",
        citation: "openEMS/openems.h:70; openEMS/python/openEMS/openEMS.pyx:198-207",
    },
    AbiEntry {
        symbol: "openems_set_bc_type",
        native: "openEMS::Set_BC_Type",
        citation: "openEMS/openems.cpp:810-815 (index guard, stored verbatim); \
                   openEMS/python/openEMS/openEMS.pyx:322-327",
    },
    AbiEntry {
        symbol: "openems_set_pml_size",
        native: "openEMS::Set_BC_PML",
        citation: "openEMS/openems.cpp:824-830 (also forces the face type to PML); \
                   openEMS/python/openEMS/openEMS.pyx:329-331",
    },
    AbiEntry {
        symbol: "openems_set_mur_phase_vel",
        native: "openEMS::Set_Mur_PhaseVel",
        citation: "openEMS/openems.cpp:841-846; \
                   openEMS/matlab/SetBoundaryCond.m:23-26 (MUR_PhaseVelocity option)",
    },
    AbiEntry {
        symbol: "openems_set_max_time",
        native: "openEMS::SetMaxTime",
        citation: "openEMS/openems.h:77; openEMS/python/openEMS/openEMS.pyx:242-255",
    },
    AbiEntry {
        symbol: "openems_set_time_step_method",
        native: "openEMS::SetTimeStepMethod",
        citation: "openEMS/openems.h:74; openEMS/python/openEMS/openEMS.pyx:209-221",
    },
    AbiEntry {
        symbol: "openems_set_time_step",
        native: "openEMS::SetTimeStep",
        citation: "openEMS/openems.h:75; openEMS/python/openEMS/openEMS.pyx:223-231",
    },
    AbiEntry {
        symbol: "openems_set_time_step_factor",
        native: "openEMS::SetTimeStepFactor",
        citation: "openEMS/openems.h:76; openEMS/python/openEMS/openEMS.pyx:233-240",
    },
    AbiEntry {
        symbol: "openems_set_over_sampling",
        native: "openEMS::SetOverSampling",
        citation: "openEMS/openems.h:67; openEMS/python/openEMS/openEMS.pyx:138-146",
    },
    AbiEntry {
        symbol: "openems_set_cell_constant_material",
        native: "openEMS::SetCellConstantMaterial",
        citation: "openEMS/openems.h:68; openEMS/python/openEMS/openEMS.pyx:148-167",
    },
    AbiEntry {
        symbol: "openems_set_coord_system",
        native: "openEMS::SetCylinderCoords via openEMS.SetCoordSystem",
        citation: "openEMS/python/openEMS/openEMS.pyx:169-179 (0/1 accepted, \
                   'Invalid coordinate system' otherwise)",
    },
    AbiEntry {
        symbol: "openems_set_gauss_excite",
        native: "openEMS::SetGaussExcite",
        citation: "openEMS/openems.h:107; openEMS/matlab/SetGaussExcite.m:1-20",
    },
    AbiEntry {
        symbol: "openems_set_sinus_excite",
        native: "openEMS::SetSinusExcite",
        citation: "openEMS/openems.h:108; openEMS/matlab/SetSinusExcite.m:1-16",
    },
    AbiEntry {
        symbol: "openems_set_dirac_excite",
        native: "openEMS::SetDiracExcite",
        citation: "openEMS/openems.h:109; openEMS/matlab/SetDiracExcite.m:1-26",
    },
    AbiEntry {
        symbol: "openems_set_step_excite",
        native: "openEMS::SetStepExcite",
        citation: "openEMS/openems.h:110; openEMS/matlab/SetStepExcite.m:1-25",
    },
    AbiEntry {
        symbol: "openems_set_csx_xml",
        native: "openEMS::SetCSX (XML form)",
        citation: "openEMS/openems.cpp:1163-1170 (single geometry slot, NULL clears); \
                   ABI-specific transport of the same geometry",
    },
    AbiEntry {
        symbol: "openems_setup_fdtd",
        native: "openEMS::SetupFDTD",
        citation: "openEMS/openems.cpp:1177-1234 (codes 3 at :1188, 2 at :1216, \
                   3 at :1233); openEMS/python/openEMS/openEMS.pyx:693-699",
    },
    AbiEntry {
        symbol: "openems_run_fdtd",
        native: "openEMS::RunFDTD",
        citation: "openEMS/openems.h:60 (returns void); \
                   openEMS/python/openEMS/openEMS.pyx:700-701",
    },
    AbiEntry {
        symbol: "openems_get_number_of_timesteps",
        native: "Engine::GetNumberOfTimesteps",
        citation: "openEMS/FDTD/engine.h:50 (no controller setter exists: ABI extension)",
    },
    AbiEntry {
        symbol: "openems_get_bc_type",
        native: "openEMS::Get_BC_Type",
        citation: "openEMS/openems.cpp:817-822 (-1 outside 0..5); \
                   openEMS/python/openEMS/openEMS.pyx:488-489",
    },
    AbiEntry {
        symbol: "openems_get_pml_size",
        native: "openEMS::Get_PML_Size",
        citation: "openEMS/openems.cpp:832-839 (-1 outside 0..5 and for non-PML faces); \
                   openEMS/python/openEMS/openEMS.pyx:498-499",
    },
    AbiEntry {
        symbol: "openems_set_csx",
        native: "openEMS::SetCSX",
        citation: "openEMS/openems.h:115-119; openEMS/openems.cpp:1163-1170; \
                   openEMS/python/openEMS/openEMS.pyx:519-536",
    },
    AbiEntry {
        symbol: "openems_get_csx",
        native: "openEMS::GetCSX",
        citation: "openEMS/openems.h:118-119; openEMS/openems.cpp:1172-1175; \
                   openEMS/python/openEMS/openEMS.pyx:538-550",
    },
    AbiEntry {
        symbol: "openems_set_boundary_cond",
        native: "SetBoundaryCond / SetBoundaryCond(FDTD, BC)",
        citation: "openEMS/matlab/SetBoundaryCond.m:51-71 (six entries, 1-based order \
                   xmin..zmax); openEMS/python/openEMS/openEMS.pyx:306-333",
    },
    AbiEntry {
        symbol: "openems_get_boundary_cond",
        native: "six Get_BC_Type reads (CreateNF2FFBox)",
        citation: "openEMS/python/openEMS/openEMS.pyx:488-499 (reads faces 0..5 in order)",
    },
    AbiEntry {
        symbol: "openems_last_error_id",
        native: "none (ABI error channel)",
        citation: "Carries the M-file identifier form `error('openEMS:...', msg)`, \
                   e.g. openEMS/matlab/SetBoundaryCond.m:52",
    },
    AbiEntry {
        symbol: "openems_last_error_message",
        native: "none (ABI error channel)",
        citation: "Carries the native message text, quoted from the raising line",
    },
    AbiEntry {
        symbol: "openems_last_error_code",
        native: "int openEMS::SetupFDTD()",
        citation: "openEMS/openems.cpp:1188, :1216, :1233; \
                   openEMS/python/openEMS/openEMS.pyx:696-699",
    },
    AbiEntry {
        symbol: "openems_free_string",
        native: "none (ABI string ownership)",
        citation: "Mirrors CSXCAD/python delivery of owned strings \
                   (CSXCAD/rust-matlab/src/lib.rs csxcad_free_string precedent)",
    },
];

/// Number of exported entry points. Every symbol above is `pub extern "C"`.
const EXPORT_COUNT: usize = 34;

/// Native contract table must cover every export exactly once.
#[test]
fn inventory_covers_every_export_exactly_once() {
    assert_eq!(ABI_INVENTORY.len(), EXPORT_COUNT);
    for (index, entry) in ABI_INVENTORY.iter().enumerate() {
        assert!(
            !entry.symbol.is_empty() && !entry.native.is_empty() && !entry.citation.is_empty(),
            "entry {index} is incomplete"
        );
        for other in &ABI_INVENTORY[index + 1..] {
            assert_ne!(
                entry.symbol, other.symbol,
                "duplicate inventory entry for {}",
                entry.symbol
            );
        }
    }
}

/// The source must export exactly the inventoried symbols, so removing,
/// renaming or silently adding an entry point fails this test.
#[test]
fn exported_symbols_match_the_inventory() {
    let source = include_str!("lib.rs");
    let mut exported: Vec<&str> = source
        .lines()
        .filter_map(|line| line.trim().strip_prefix("pub extern \"C\" fn "))
        .map(|rest| rest.split('(').next().unwrap_or(rest).trim())
        .collect();
    exported.sort_unstable();

    let mut expected: Vec<&str> = ABI_INVENTORY.iter().map(|entry| entry.symbol).collect();
    expected.sort_unstable();

    assert_eq!(
        exported, expected,
        "the C ABI export set changed; map every new symbol to its native contract \
         in ABI_INVENTORY and keep the MEX-visible signatures stable"
    );
}

/// Pin the C signature of every export. A MEX layer compiles against these;
/// changing one is an ABI break and must be a deliberate, reported decision.
#[test]
fn c_signatures_are_pinned() {
    // Core lifetime.
    let _: extern "C" fn() -> *mut c_void = openems_new;
    let _: extern "C" fn(*mut c_void) = openems_free;

    // Solver settings.
    let _: extern "C" fn(*mut c_void, c_uint) = openems_set_number_of_time_steps;
    let _: extern "C" fn(*mut c_void, c_double) = openems_set_end_criteria;
    let _: extern "C" fn(*mut c_void, u8) = openems_set_enable_dumps;
    let _: extern "C" fn(*mut c_void, u8) = openems_set_cylinder_coords;
    let _: extern "C" fn(*mut c_void, c_uint, c_int) = openems_set_bc_type;
    let _: extern "C" fn(*mut c_void, c_uint, c_uint) = openems_set_pml_size;
    let _: extern "C" fn(*mut c_void, c_uint, c_double) = openems_set_mur_phase_vel;
    let _: extern "C" fn(*mut c_void, c_double) = openems_set_max_time;
    let _: extern "C" fn(*mut c_void, c_int) = openems_set_time_step_method;
    let _: extern "C" fn(*mut c_void, c_double) = openems_set_time_step;
    let _: extern "C" fn(*mut c_void, c_double) = openems_set_time_step_factor;
    let _: extern "C" fn(*mut c_void, c_uint) = openems_set_over_sampling;
    let _: extern "C" fn(*mut c_void, u8) = openems_set_cell_constant_material;
    let _: extern "C" fn(*mut c_void, c_int) -> u8 = openems_set_coord_system;

    // Excitations.
    let _: extern "C" fn(*mut c_void, c_double, c_double) = openems_set_gauss_excite;
    let _: extern "C" fn(*mut c_void, c_double) = openems_set_sinus_excite;
    let _: extern "C" fn(*mut c_void, c_double) = openems_set_dirac_excite;
    let _: extern "C" fn(*mut c_void, c_double) = openems_set_step_excite;

    // Geometry, setup, run.
    let _: extern "C" fn(*mut c_void, *const c_char) = openems_set_csx_xml;
    let _: extern "C" fn(*mut c_void, *mut c_void) -> u8 = openems_set_csx;
    let _: extern "C" fn(*mut c_void) -> *mut c_void = openems_get_csx;
    let _: extern "C" fn(*mut c_void) -> u8 = openems_setup_fdtd;
    let _: extern "C" fn(*mut c_void) -> u8 = openems_run_fdtd;
    let _: extern "C" fn(*mut c_void) -> c_uint = openems_get_number_of_timesteps;

    // Boundaries as arrays and per face.
    let _: extern "C" fn(*mut c_void, *const c_int, c_uint) -> u8 = openems_set_boundary_cond;
    let _: extern "C" fn(*mut c_void, *mut c_int, c_uint) -> u8 = openems_get_boundary_cond;
    let _: extern "C" fn(*mut c_void, c_uint) -> c_int = openems_get_bc_type;
    let _: extern "C" fn(*mut c_void, c_uint) -> c_uint = openems_get_pml_size;

    // Error channel and string ownership.
    let _: extern "C" fn() -> *mut c_char = openems_last_error_id;
    let _: extern "C" fn() -> *mut c_char = openems_last_error_message;
    let _: extern "C" fn() -> c_int = openems_last_error_code;
    let _: extern "C" fn(*mut c_char) = openems_free_string;
}
