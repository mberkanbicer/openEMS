//! Main solver orchestrator: ParseFDTDSetup, SetupFDTD, RunFDTD.
//! Mirrors `openems.h/cpp` in the C++ codebase.

use crate::engine::Engine;
use crate::excitation::Excitation;
use crate::operator::Operator;
use crate::operator_base::OperatorBase;
use crate::operator_extension::{
    ConductingSheetOperatorExtension, DebyeOperatorExtension, ExcitationOperatorExtension,
    LorentzOperatorExtension, LumpedRLCOperatorExtension, MurOperatorExtension,
    OperatorExtensionRegistry, SteadyStateOperatorExtension, TfsfOperatorExtension,
    UpmlOperatorExtension,
};
use crate::processing::{EngineInterface, EngineMeta, Processing, ProcessingArray};
use crate::processing_integral::{
    CurrentIntegral, FieldProbeIntegral, ModeMatchIntegral, ProcessIntegral, VoltageIntegral,
};
use std::sync::Arc;

/// Native end-criteria cadence (`openems.cpp:1461-1467`, `:1511-1526`).
///
/// `exact` (`--exact-endcriteria`) evaluates every excitation Nyquist period;
/// the default evaluates every four seconds of wall clock, measured from the
/// previous checkpoint (the first checkpoint is four seconds into the run).
const WALL_CLOCK_CHECK_INTERVAL: f64 = 4.0;

fn end_criteria_due(
    exact: bool,
    ts: u32,
    nyquist: u32,
    now_seconds: f64,
    last_checkpoint: &mut f64,
) -> bool {
    if exact {
        nyquist > 0 && ts.is_multiple_of(nyquist)
    } else if now_seconds - *last_checkpoint > WALL_CLOCK_CHECK_INTERVAL {
        *last_checkpoint = now_seconds;
        true
    } else {
        false
    }
}

/// Boundary condition types (mirroring C++).
pub const BC_PEC: i32 = 0;
pub const BC_PMC: i32 = 1;
pub const BC_MUR: i32 = 2;
pub const BC_PML: i32 = 3;

/// Statistics of one solver run, used by `--dump-statistics`.
#[derive(Clone, Debug, Default)]
pub struct RunStats {
    pub cells: u64,
    pub timesteps: u32,
    pub dt: f64,
    pub wall_seconds: f64,
    pub end_change: f64,
    pub end_criteria: f64,
}

/// The main openEMS solver class.
/// Orchestrates the FDTD simulation: parsing setup, building the operator,
/// creating the engine, and running the time-stepping loop.
pub struct OpenEMS {
    pub fdtd_op: Option<Box<Operator>>,
    pub fdtd_eng: Option<Box<Engine>>,
    pub excitation: Option<Box<Excitation>>,
    pub nr_ts: u32,
    pub end_crit: f64,
    pub over_sampling: u32,
    pub cell_constant_material: bool,
    pub enable_dumps: bool,
    pub cylinder_coords: bool,
    pub bc_type: [i32; 6],
    pub bc_type_set: bool,
    pub pml_size: [u32; 6],
    pub mur_v_ph: [f64; 6],
    /// Cylindrical multi-grid radii (native `m_CC_MultiGrid`).
    pub multi_grid: Vec<f64>,
    pub max_time: f64,
    /// Write field dumps in the legacy Octave/MATLAB HDF5 layout
    /// (native `--legacyHDF5Dumps` / `g_settings.GetLegacyHDF5Dumps`).
    pub legacy_hdf5_dumps: bool,
    /// Dump the native staggered field components instead of rotating the
    /// cylindrical `(rho, alpha, z)` components into `(x, y, z)`
    /// (native `--nativeFieldDumps`). Cartesian dumps are unchanged.
    pub native_field_dumps: bool,
    pub time_step_method: i32,
    pub time_step: f64,
    pub time_step_factor: f64,
    pub abort: bool,
    pub num_threads: i32,
    pub verbose_level: i32,
    /// Engine selection (`fastest`, `basic`, `sse`, `sse-compressed`,
    /// `multithreaded`). Accepted for CLI compatibility; the basic Rust kernel
    /// is used in all cases and the choice only affects the worker count.
    pub engine: String,
    /// Run setup only (`--no-simulation`).
    pub no_simulation: bool,
    /// Write `openEMS_stats.txt` and `openEMS_run_stats.txt`.
    pub dump_statistics: bool,
    /// Evaluate the energy end criterion on the Nyquist cadence instead of
    /// every 100 steps (`--exact-endcriteria`).
    pub exact_end_criteria: bool,
    /// Debug dumps requested by the native CLI.
    pub debug_material_dump: bool,
    pub debug_operator_dump: bool,
    pub debug_box_dump: bool,
    pub debug_pec_dump: bool,
    pub debug_csx_dump: bool,
    /// Statistics of the last run.
    pub last_run_stats: Option<RunStats>,
    /// CSXCAD XML geometry string. When set, `setup_fdtd()` parses it via
    /// `Operator::load_csx()` to populate the grid, materials, boundary
    /// conditions and per-cell materials.
    pub csx_xml: Option<String>,
    /// Post-processing array (probes, field dumps, SAR, etc.).
    /// Wired into the `run_fdtd()` time-stepping loop.
    pub processing_array: Option<ProcessingArray>,
}

impl OpenEMS {
    /// Create a new openEMS solver instance.
    pub fn new() -> Self {
        Self {
            fdtd_op: None,
            fdtd_eng: None,
            excitation: None,
            nr_ts: 0,
            end_crit: 1e-6,
            over_sampling: 4,
            cell_constant_material: false,
            enable_dumps: true,
            cylinder_coords: false,
            bc_type: [BC_PEC; 6],
            bc_type_set: false,
            pml_size: [0; 6],
            mur_v_ph: [0.0; 6],
            multi_grid: Vec::new(),
            max_time: 0.0,
            legacy_hdf5_dumps: false,
            native_field_dumps: false,
            // native `openEMS::openEMS()` default: 3 (Rennings_2), the largest
            // stable step on rectilinear meshes
            time_step_method: 3,
            time_step: 0.0,
            time_step_factor: 1.0,
            abort: false,
            num_threads: 1,
            verbose_level: 0,
            engine: "fastest".to_string(),
            no_simulation: false,
            dump_statistics: false,
            exact_end_criteria: false,
            debug_material_dump: false,
            debug_operator_dump: false,
            debug_box_dump: false,
            debug_pec_dump: false,
            debug_csx_dump: false,
            last_run_stats: None,
            csx_xml: None,
            processing_array: None,
        }
    }

    /// Set the number of time steps.
    pub fn set_number_of_time_steps(&mut self, val: u32) {
        self.nr_ts = val;
    }

    /// Set the end criteria.
    pub fn set_end_criteria(&mut self, val: f64) {
        // Native maps only `endCriteria == 0` to the 1e-6 default; a negative
        // value is kept and disables the criterion (`change > endCrit` always
        // holds).
        if val != 0.0 {
            self.end_crit = val;
        }
    }

    /// Enable or disable dumps.
    pub fn set_enable_dumps(&mut self, val: bool) {
        self.enable_dumps = val;
    }

    /// Set cylinder coordinates mode.
    pub fn set_cylinder_coords(&mut self, val: bool) {
        self.cylinder_coords = val;
    }

    /// Set boundary condition type for a given side (0..6).
    pub fn set_bc_type(&mut self, idx: usize, typ: i32) {
        if idx < 6 {
            self.bc_type[idx] = typ;
            self.bc_type_set = true;
        }
    }

    /// Get boundary condition type for a given side (0..6).
    pub fn get_bc_type(&self, idx: usize) -> i32 {
        if idx < 6 {
            self.bc_type[idx]
        } else {
            BC_PEC
        }
    }

    /// Set PML size for a given side.
    pub fn set_pml_size(&mut self, idx: usize, size: u32) {
        if idx < 6 {
            self.pml_size[idx] = size;
        }
    }

    /// Get PML size for a given side (0..6).
    pub fn get_pml_size(&self, idx: usize) -> u32 {
        if idx < 6 {
            self.pml_size[idx]
        } else {
            0
        }
    }

    /// Set MUR phase velocity for a given side.
    pub fn set_mur_phase_vel(&mut self, idx: usize, val: f64) {
        if idx < 6 {
            self.mur_v_ph[idx] = val;
        }
    }

    /// Set maximum simulation time (overrides nr_ts if reached first).
    pub fn set_max_time(&mut self, val: f64) {
        self.max_time = val;
    }

    /// Write field dumps in the legacy Octave/MATLAB HDF5 layout.
    pub fn set_legacy_hdf5_dumps(&mut self, val: bool) {
        self.legacy_hdf5_dumps = val;
    }

    pub fn get_legacy_hdf5_dumps(&self) -> bool {
        self.legacy_hdf5_dumps
    }

    /// Dump the native staggered field components (native `--nativeFieldDumps`).
    pub fn set_native_field_dumps(&mut self, val: bool) {
        self.native_field_dumps = val;
    }

    pub fn get_native_field_dumps(&self) -> bool {
        self.native_field_dumps
    }

    /// Set the timestep method.
    pub fn set_time_step_method(&mut self, val: i32) {
        self.time_step_method = val;
    }

    /// Set a fixed timestep (overrides CFL-computed value if > 0).
    pub fn set_time_step(&mut self, val: f64) {
        self.time_step = val;
    }

    /// Set the timestep factor (multiplier on CFL timestep).
    pub fn set_time_step_factor(&mut self, val: f64) {
        self.time_step_factor = val;
    }

    /// Set the oversampling factor for time-domain dumps/probes.
    pub fn set_over_sampling(&mut self, val: u32) {
        self.over_sampling = val.max(1);
    }

    /// Set cell constant material mode.
    pub fn set_cell_constant_material(&mut self, val: bool) {
        self.cell_constant_material = val;
    }

    /// Set the coordinate system (0=Cartesian, 1=Cylindrical).
    pub fn set_coord_system(&mut self, val: i32) -> Result<(), String> {
        if val != 0 && val != 1 {
            return Err("Invalid coordinate system".to_string());
        }
        self.cylinder_coords = val == 1;
        Ok(())
    }

    /// Set cylindrical multi-grid radii.
    pub fn set_multi_grid(&mut self, radii: &[f64]) -> Result<(), String> {
        if radii.is_empty() {
            return Err("Invalid multi grid definition".to_string());
        }
        // the radii are written back to the FDTD setup and used by the
        // cylindrical multi-grid operator during `setup_fdtd()`
        self.multi_grid = radii.to_vec();
        Ok(())
    }

    /// The native `<FDTD ...>` element for the current solver settings.
    pub fn write_fdtd_element(&self) -> String {
        use crate::fdtd_options::{FdtdExcitation, FdtdWriteSettings};
        let excitation = match self.excitation.as_deref() {
            Some(excitation) => match excitation.exc_type {
                crate::excitation::ExcitationType::Gauss => FdtdExcitation::Gauss {
                    f0: excitation.f0,
                    fc: excitation.fc,
                },
                crate::excitation::ExcitationType::Sinus => {
                    FdtdExcitation::Sinus { f0: excitation.f0 }
                }
                crate::excitation::ExcitationType::Dirac => FdtdExcitation::Dirac {
                    f_max: excitation.f_max,
                },
                crate::excitation::ExcitationType::Step => FdtdExcitation::Step {
                    f_max: excitation.f_max,
                },
                crate::excitation::ExcitationType::Custom => FdtdExcitation::Custom {
                    function: excitation.custom_function().unwrap_or_default().to_string(),
                    f0: excitation.f0,
                    f_max: excitation.f_max,
                },
            },
            None => FdtdExcitation::Gauss { f0: 0.0, fc: 0.0 },
        };
        crate::fdtd_options::write_fdtd_element(&FdtdWriteSettings {
            nr_ts: self.nr_ts,
            end_criteria: self.end_crit,
            max_time: self.max_time,
            over_sampling: self.over_sampling,
            cell_constant_material: self.cell_constant_material,
            cylinder_coords: self.cylinder_coords,
            multi_grid: self.multi_grid.clone(),
            time_step: self.time_step,
            time_step_factor: self.time_step_factor,
            time_step_method: self.time_step_method,
            bc_type: self.bc_type,
            pml_size: self.pml_size,
            mur_phase_velocity: std::array::from_fn(|index| {
                (self.mur_v_ph[index] > 0.0).then_some(self.mur_v_ph[index])
            }),
            excitation,
        })
    }

    /// Apply the options of a native `<FDTD>` element (native
    /// `Parse_XML_FDTDSetup`).
    pub fn apply_fdtd_options(&mut self, options: &crate::fdtd_options::FdtdOptions) {
        if let Some(nr_ts) = options.nr_ts {
            self.nr_ts = nr_ts;
        }
        if let Some(end_criteria) = options.end_criteria {
            self.end_crit = end_criteria;
        }
        if let Some(max_time) = options.max_time {
            self.max_time = max_time;
        }
        if let Some(over_sampling) = options.over_sampling {
            if over_sampling > 1 {
                self.over_sampling = over_sampling;
            }
        }
        if let Some(cell_constant_material) = options.cell_constant_material {
            self.cell_constant_material = cell_constant_material;
        }
        if let Some(time_step) = options.time_step {
            self.time_step = time_step;
        }
        if let Some(time_step_factor) = options.time_step_factor {
            self.time_step_factor = time_step_factor;
        }
        if let Some(time_step_method) = options.time_step_method {
            self.time_step_method = time_step_method;
        }
        if let Some(coord_system) = options.coord_system {
            self.cylinder_coords = coord_system == 1;
        }
        if let Some(cylinder_coords) = options.cylinder_coords {
            self.cylinder_coords = cylinder_coords;
        }
        if let Some(multi_grid) = &options.multi_grid {
            self.multi_grid = multi_grid.clone();
        }
        if let Some(bc_type) = options.bc_type {
            self.bc_type = bc_type;
            self.bc_type_set = true;
        }
        if let Some(pml_size) = options.pml_size {
            self.pml_size = pml_size;
        }
        for (index, velocity) in options.mur_phase_velocity.iter().enumerate() {
            if let Some(velocity) = velocity {
                self.mur_v_ph[index] = *velocity;
            }
        }
        match &options.excitation {
            Some(crate::fdtd_options::FdtdExcitation::Gauss { f0, fc }) => {
                self.set_gauss_excite(*f0, *fc);
            }
            Some(crate::fdtd_options::FdtdExcitation::Sinus { f0 }) => {
                self.set_sinus_excite(*f0);
            }
            Some(crate::fdtd_options::FdtdExcitation::Dirac { f_max }) => {
                self.set_dirac_excite(*f_max);
            }
            Some(crate::fdtd_options::FdtdExcitation::Step { f_max }) => {
                self.set_step_excite(*f_max);
            }
            Some(crate::fdtd_options::FdtdExcitation::Custom {
                function,
                f0,
                f_max,
            }) => {
                self.set_custom_excite(function, *f0, *f_max);
            }
            None => {}
        }
    }

    /// Apply the `<Excitation>` child of `<FDTD>` if the document has one.
    ///
    /// Native `openEMS::ParseFDTDSetup` reads the excitation from the input
    /// file; only an input file without any excitation falls back to the
    /// default pulse.
    pub fn apply_excitation_option(
        &mut self,
        excitation: Option<&crate::fdtd_options::FdtdExcitation>,
    ) {
        match excitation {
            Some(crate::fdtd_options::FdtdExcitation::Gauss { f0, fc }) => {
                self.set_gauss_excite(*f0, *fc);
            }
            Some(crate::fdtd_options::FdtdExcitation::Sinus { f0 }) => {
                self.set_sinus_excite(*f0);
            }
            Some(crate::fdtd_options::FdtdExcitation::Dirac { f_max }) => {
                self.set_dirac_excite(*f_max);
            }
            Some(crate::fdtd_options::FdtdExcitation::Step { f_max }) => {
                self.set_step_excite(*f_max);
            }
            Some(crate::fdtd_options::FdtdExcitation::Custom {
                function,
                f0,
                f_max,
            }) => {
                self.set_custom_excite(function, *f0, *f_max);
            }
            None => {}
        }
    }

    /// Set a Gaussian pulse excitation.
    pub fn set_gauss_excite(&mut self, f0: f64, fc: f64) {
        self.excitation = Some(Box::new(Excitation::gauss(f0, fc)));
    }

    /// Set a sinusoidal excitation.
    pub fn set_sinus_excite(&mut self, f0: f64) {
        self.excitation = Some(Box::new(Excitation::sinus(f0)));
    }

    /// Set a Dirac pulse excitation.
    pub fn set_dirac_excite(&mut self, f_max: f64) {
        self.excitation = Some(Box::new(Excitation::dirac(f_max)));
    }

    /// Set a step function excitation.
    pub fn set_step_excite(&mut self, f_max: f64) {
        self.excitation = Some(Box::new(Excitation::step(f_max)));
    }

    /// Set a custom function excitation (parsed by fparser).
    pub fn set_custom_excite(&mut self, _str: &str, f0: f64, fmax: f64) {
        self.excitation = Some(Box::new(Excitation::custom(_str, f0, fmax)));
    }

    /// Set the abort flag (requests graceful shutdown).
    pub fn set_abort(&mut self, val: bool) {
        self.abort = val;
    }

    /// Set the CSXCAD XML geometry string.
    /// When set, `setup_fdtd()` parses it via `Operator::load_csx()` to
    /// populate the grid, materials, boundary conditions and per-cell materials.
    pub fn set_csx_xml(&mut self, xml: String) {
        self.csx_xml = Some(xml);
    }

    /// Clear the CSXCAD XML geometry (falls back to the default grid).
    pub fn clear_csx_xml(&mut self) {
        self.csx_xml = None;
    }

    /// Add a processing object (probe, field dump, SAR, etc.) to the
    /// post-processing array. The array is wired into the `run_fdtd()`
    /// time-stepping loop.
    pub fn add_processing(&mut self, proc: Box<dyn Processing>) {
        if self.processing_array.is_none() {
            self.processing_array = Some(ProcessingArray::new(u32::MAX));
        }
        self.processing_array.as_mut().unwrap().add_processing(proc);
    }

    /// Set the number of threads for multithreaded engine.
    pub fn set_number_of_threads(&mut self, val: i32) {
        self.num_threads = val;
        if self.verbose_level > 0 {
            println!("[openEMS] Number of threads set to {}", val);
        }
    }

    /// Set the verbose level for console output.
    pub fn set_verbose_level(&mut self, level: i32) {
        self.verbose_level = level;
        if level > 0 {
            println!("[openEMS] Verbose level set to {}", level);
        }
    }

    /// Select the engine (`fastest`, `basic`, `sse`, `sse-compressed`,
    /// `multithreaded`). Unknown names keep the previous selection.
    pub fn set_engine(&mut self, engine: &str) {
        let normalized = engine.to_ascii_lowercase();
        if matches!(
            normalized.as_str(),
            "fastest" | "basic" | "sse" | "sse-compressed" | "multithreaded"
        ) {
            self.engine = normalized;
        }
    }

    /// Run preprocessing only (`--no-simulation`).
    pub fn set_no_simulation(&mut self, val: bool) {
        self.no_simulation = val;
    }

    /// Write the statistics files after a run.
    pub fn set_dump_statistics(&mut self, val: bool) {
        self.dump_statistics = val;
    }

    /// Evaluate the energy end criterion on the Nyquist cadence.
    pub fn set_exact_end_criteria(&mut self, val: bool) {
        self.exact_end_criteria = val;
    }

    /// Request the native debug dumps.
    pub fn set_debug_dumps(
        &mut self,
        material: bool,
        operator: bool,
        boxes: bool,
        pec: bool,
        csx: bool,
    ) {
        self.debug_material_dump = material;
        self.debug_operator_dump = operator;
        self.debug_box_dump = boxes;
        self.debug_pec_dump = pec;
        self.debug_csx_dump = csx;
    }

    /// Enable debug material output.
    pub fn debug_material(&mut self) {
        if let Some(ref eng) = self.fdtd_eng {
            let op = &eng.op;
            println!("[openEMS debug] Material:");
            println!("  background eps_r: {}", op.get_background_eps_r());
            println!("  background mue_r: {}", op.get_background_mue_r());
            println!("  background kappa: {}", op.get_background_kappa());
            println!("  background sigma: {}", op.get_background_sigma());
            println!("  background density: {}", op.get_background_density());
        } else {
            println!("[openEMS debug] Material: engine not initialized");
        }
    }

    /// Enable debug operator output.
    pub fn debug_operator(&mut self) {
        if let Some(ref eng) = self.fdtd_eng {
            let op = &eng.op;
            println!("[openEMS debug] Operator:");
            println!("  grid delta: {} m", op.get_grid_delta());
            println!("  timestep: {} s", op.get_timestep());
            println!("  number of cells: {}", op.get_number_cells());
            for n in 0..3 {
                println!(
                    "  lines[{}]: {} ({})",
                    n,
                    op.get_number_of_lines(n as i32, true),
                    op.get_dir_name(n as i32)
                );
            }
        } else {
            println!("[openEMS debug] Operator: engine not initialized");
        }
    }

    /// Enable debug box output.
    pub fn debug_box(&mut self) {
        if let Some(ref eng) = self.fdtd_eng {
            let op = &eng.op;
            println!("[openEMS debug] Box/Grid:");
            for n in 0..3 {
                println!(
                    "  {} direction: {} lines, delta={}",
                    op.get_dir_name(n as i32),
                    op.get_number_of_lines(n as i32, true),
                    op.get_grid_delta()
                );
            }
        } else {
            println!("[openEMS debug] Box: engine not initialized");
        }
    }

    /// Enable debug PEC output.
    pub fn debug_pec(&mut self) {
        if let Some(ref eng) = self.fdtd_eng {
            let op = &eng.op;
            println!("[openEMS debug] PEC/Boundary Conditions:");
            for i in 0..6 {
                let bc = match self.bc_type[i] {
                    BC_PEC => "PEC",
                    BC_PMC => "PMC",
                    BC_MUR => "MUR",
                    BC_PML => "PML",
                    _ => "unknown",
                };
                println!("  face {}: {}", i, bc);
            }
        } else {
            println!("[openEMS debug] PEC: engine not initialized");
        }
    }

    /// Enable debug CSX output.
    pub fn debug_csx(&mut self) {
        if let Some(ref xml) = self.csx_xml {
            println!("[openEMS debug] CSXCAD XML ({} bytes):", xml.len());
            for line in xml.lines().take(20) {
                println!("  {}", line);
            }
            if xml.lines().count() > 20 {
                println!("  ... ({} more lines)", xml.lines().count() - 20);
            }
        } else {
            println!("[openEMS debug] CSX: no XML geometry loaded");
        }
    }

    /// Check abort conditions.
    pub fn check_abort_cond(&self) -> bool {
        self.abort
    }

    /// Write the native debug dumps from the finished operator.
    fn write_debug_dumps(&self, op: &Operator) {
        use crate::debug_dump;

        let counts = op.base.num_lines;
        let dims = [counts[0] as usize, counts[1] as usize, counts[2] as usize];
        let cells = dims[0] * dims[1] * dims[2];
        let mut disc: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for axis in 0..3 {
            for index in 0..dims[axis] {
                disc[axis]
                    .push(op.get_disc_line(axis as i32, index as u32, false) * op.base.grid_delta);
            }
        }
        let lines: [&[f64]; 3] = [&disc[0], &disc[1], &disc[2]];

        if self.debug_material_dump {
            if let Some(grid) = &op.material_grid {
                let mut values = vec![0.0; cells];
                for index in 0..cells {
                    values[index] = grid[index].eps_r;
                }
                match debug_dump::write_scalar_vtk(
                    "material_dump.vtk",
                    "eps_r",
                    dims,
                    lines,
                    &values,
                ) {
                    Ok(()) => println!("[openEMS] material written to 'material_dump.vtk'"),
                    Err(error) => eprintln!("[openEMS] material_dump.vtk failed: {error}"),
                }
            }
        }

        if self.debug_operator_dump {
            // Native `--debug-operator` writes `vv`, `vi`, `ii` and `iv` as
            // three-component vectors (`VTK_File_Writer::SetNativeDump`), so
            // the dump carries the full operator, not just one scalar.
            let bc = op.base.m_bc;
            let mut fields: Vec<(&str, [Vec<f64>; 3])> = Vec::new();
            for (name, array) in [
                ("vv", &op.vv_ptr),
                ("vi", &op.vi_ptr),
                ("ii", &op.ii_ptr),
                ("iv", &op.iv_ptr),
            ] {
                let Some(values) = array.as_ref() else {
                    continue;
                };
                let mut components = [vec![0.0; cells], vec![0.0; cells], vec![0.0; cells]];
                for (n, component) in components.iter_mut().enumerate() {
                    for i in 0..dims[0] {
                        for j in 0..dims[1] {
                            for k in 0..dims[2] {
                                let index = (i * dims[1] + j) * dims[2] + k;
                                // Native zeroes these edges in the operator
                                // itself (`ApplyElectricBC` with every face
                                // that is not "none", `FDTD/operator.cpp:1042`):
                                // the normal component on the high line and
                                // both tangential components on every face.
                                // The solver never integrates them, so the port
                                // keeps the bulk value and reports the native
                                // value here.
                                let pos = [i, j, k];
                                // Only the electric arrays get the boundary
                                // mask: native applies the magnetic BC only
                                // to PMC faces.
                                let boundary = (name == "vv" || name == "vi")
                                    && (0..3).any(|axis| {
                                        if bc[2 * axis] == -1 && bc[2 * axis + 1] == -1 {
                                            return false;
                                        }
                                        if axis == n {
                                            pos[axis] + 1 == dims[axis]
                                        } else {
                                            pos[axis] == 0 || pos[axis] + 1 == dims[axis]
                                        }
                                    });
                                component[index] = if boundary {
                                    0.0
                                } else {
                                    *values.get(n, i, j, k) as f64
                                };
                            }
                        }
                    }
                }
                fields.push((name, components));
            }
            if !fields.is_empty() {
                let refs: Vec<(&str, [&[f64]; 3])> = fields
                    .iter()
                    .map(|(name, components)| {
                        (
                            *name,
                            [
                                components[0].as_slice(),
                                components[1].as_slice(),
                                components[2].as_slice(),
                            ],
                        )
                    })
                    .collect();
                match debug_dump::write_operator_vtk("operator_dump.vtk", dims, lines, &refs) {
                    Ok(()) => println!("[openEMS] operator written to 'operator_dump.vtk'"),
                    Err(error) => eprintln!("[openEMS] operator_dump.vtk failed: {error}"),
                }
            }
        }

        if self.debug_pec_dump {
            if let (Some(vv), Some(vi)) = (op.vv_ptr.as_ref(), op.vi_ptr.as_ref()) {
                let mut values = vec![0.0; cells];
                for i in 0..dims[0] {
                    for j in 0..dims[1] {
                        for k in 0..dims[2] {
                            let index = (i * dims[1] + j) * dims[2] + k;
                            if *vv.get(0, i, j, k) == 0.0 && *vi.get(0, i, j, k) == 0.0 {
                                values[index] = 1.0;
                            }
                        }
                    }
                }
                match debug_dump::write_scalar_vtk("PEC_dump.vtk", "PEC", dims, lines, &values) {
                    Ok(()) => println!("[openEMS] PEC written to 'PEC_dump.vtk'"),
                    Err(error) => eprintln!("[openEMS] PEC_dump.vtk failed: {error}"),
                }
            }
        }

        if self.debug_box_dump {
            if let Some(structure) = op.structure.as_ref() {
                let params = structure.parameters();
                let scale = op.base.grid_delta;
                let mut boxes = Vec::new();
                for index in 0..structure.property_count() {
                    let Some(property) = structure.property_by_index(index) else {
                        continue;
                    };
                    for primitive in property.primitives() {
                        if let Ok(bounds) = primitive.bounding_box(Some(params), false) {
                            boxes.push([
                                bounds[0] * scale,
                                bounds[2] * scale,
                                bounds[4] * scale,
                                bounds[1] * scale,
                                bounds[3] * scale,
                                bounds[5] * scale,
                            ]);
                        }
                    }
                }
                match debug_dump::write_boxes_vtk("box_dump_0.vtk", "boxes", &boxes) {
                    Ok(()) => println!("[openEMS] boxes written to 'box_dump_0.vtk'"),
                    Err(error) => eprintln!("[openEMS] box_dump_0.vtk failed: {error}"),
                }
            }
        }
    }

    /// Native `Excitation::DumpVoltageExcite` / `DumpCurrentExcite`: the
    /// sampled source waveforms are written as `et` and `ht` next to the other
    /// solver output, before the time loop.
    fn dump_excitation_signals(&self, dt: f64) {
        let Some(signal) = self.excitation.as_ref() else {
            return;
        };
        let length = signal.signal_length() as usize;
        let mut voltage = String::with_capacity(length * 24);
        let mut current = String::with_capacity(length * 24);
        for n in 0..length {
            voltage.push_str(&format!("{}\t{}\n", n as f64 * dt, signal.voltage(n)));
            current.push_str(&format!(
                "{}\t{}\n",
                n as f64 * dt + 0.5 * dt,
                signal.current(n)
            ));
        }
        for (name, contents) in [("et", voltage), ("ht", current)] {
            if let Err(error) = std::fs::write(name, contents) {
                eprintln!("[openEMS] Could not write '{name}': {error}");
            }
        }
    }

    /// Initialize the FDTD operator from a CSX geometry.
    /// Mirrors C++ `SetupFDTD` (operator setup portion).
    ///
    /// A CSXCAD geometry is required. There is no fabricated fallback grid.
    pub fn setup_fdtd(&mut self) -> bool {
        let Some(xml) = self.csx_xml.clone() else {
            eprintln!("[openEMS] Setup failed: no CSXCAD geometry was provided");
            return false;
        };
        // Native `ParseFDTDSetup`: solver settings live in the `<FDTD>` element.
        let options = crate::fdtd_options::parse_fdtd_options(&xml);
        if let Some(value) = options.nr_ts {
            self.nr_ts = value;
        }
        if let Some(value) = options.end_criteria {
            self.end_crit = value;
        }
        if let Some(value) = options.max_time {
            self.max_time = value;
        }
        if let Some(value) = options.over_sampling {
            self.over_sampling = value;
        }
        if let Some(value) = options.cell_constant_material {
            self.cell_constant_material = value;
        }
        if let Some(value) = options.time_step {
            self.time_step = value;
        }
        if let Some(value) = options.time_step_factor {
            self.time_step_factor = value;
        }
        if let Some(value) = options.time_step_method {
            self.time_step_method = value;
        }
        if let Some(value) = options.cylinder_coords {
            self.cylinder_coords = value;
        }
        if let Some(value) = options.coord_system {
            self.cylinder_coords = value == 1;
        }
        // The `<Excitation>` child of `<FDTD>` defines the source waveform
        // (native `ParseFDTDSetup`); without one the documented default pulse
        // below applies.
        self.apply_excitation_option(options.excitation.as_ref());
        if std::env::var_os("OPENEMS_DEBUG_PROBES").is_some() {
            eprintln!("[debug] parsed excitation: {:?}", options.excitation);
        }
        // `<BoundaryCond>` of `<FDTD>` sets the six faces and their PML
        // thicknesses (native `ParseFDTDSetup` + boundary setup).
        if let Some(bc) = options.bc_type {
            self.bc_type = bc;
            self.bc_type_set = true;
        }
        if let Some(pml) = options.pml_size {
            self.pml_size = pml;
        }
        let mut op = Operator::new();
        if let Err(error) = op.load_csx(&xml) {
            eprintln!("[openEMS] Failed to load CSXCAD geometry: {error}");
            return false;
        }
        if self.verbose_level > 0 {
            println!(
                "[openEMS] CSXCAD geometry loaded: {} lines, delta={:.6}",
                op.base.num_lines.iter().map(|&v| v as f64).sum::<f64>() / 3.0,
                op.base.grid_delta
            );
        }

        // An explicit boundary-condition setting overrides the CSX properties.
        if self.bc_type_set {
            op.base.set_boundary_condition(self.bc_type);
        }
        // Mirror the native PML setup: a PML face defaults to 8 cells when the
        // caller did not give an explicit size.
        for face in 0..6 {
            if op.base.m_bc[face] == BC_PML {
                if self.pml_size[face] == 0 {
                    self.pml_size[face] = 8;
                }
            } else {
                self.pml_size[face] = 0;
            }
        }
        self.bc_type = op.base.m_bc;

        // Ensure EC coefficient arrays are allocated.
        if op.vv_ptr.is_none() || op.vi_ptr.is_none() || op.ii_ptr.is_none() || op.iv_ptr.is_none()
        {
            let num_lines_usize: [usize; 3] = op.base.num_lines.map(|v| v as usize);
            op.init_data_storage(num_lines_usize);
        }

        // Set up excitation (default Gaussian pulse if none specified).
        if self.excitation.is_none() {
            let exc = Excitation::gauss(1.0e9, 5.0e8);
            self.excitation = Some(Box::new(exc));
        }

        // Compute the stable timestep (CFL condition) unless fixed explicitly.
        // This must happen before the EC coefficients are built because the
        // voltage/current update factors depend on dt.
        if !matches!(self.time_step_method, 0 | 1 | 3) {
            eprintln!(
                "[openEMS] Unsupported timestep method: {}",
                self.time_step_method
            );
            return false;
        }
        op.base.m_time_step_var = if self.time_step_method == 3 { 3 } else { 1 };
        if self.time_step > 0.0 {
            op.base.d_t = self.time_step;
        }
        // Native order (`FDTD/operator.cpp:986-1012`): `Calc_EC()` fills
        // EC_C/EC_G/EC_L/EC_R first, then `CalcTimestep()` runs on those
        // coefficients, and only afterwards the time-step factor is applied.
        // Var3 reads EC_C/EC_L, so computing the step before the operator left
        // it with empty arrays and a 13% larger step on a 1 mm mesh.
        op.calc_ec_operator();
        if self.time_step <= 0.0 {
            op.calc_timestep();
        }
        if self.time_step_factor <= 0.0 || !self.time_step_factor.is_finite() {
            eprintln!(
                "[openEMS] Invalid timestep factor: {}",
                self.time_step_factor
            );
            return false;
        }
        op.base.d_t *= self.time_step_factor;

        // Native `CalcECOperator` (`FDTD/operator.cpp:1014-1020`) shrinks the
        // step to an integer number of steps per signal period, so a sinusoid
        // repeats exactly; the sampled excitation and the steady-state detector
        // both rely on it.
        if let Some(signal) = self.excitation.as_deref() {
            let period = signal.signal_period_seconds();
            if period > 0.0 {
                let steps = (period / op.base.d_t).ceil();
                if steps >= 1.0 {
                    op.base.d_t = period / steps;
                }
            }
        }

        // Compute EC coefficients (vv, vi, ii, iv) for all cells with the final
        // timestep; the first pass above only needed the material coefficients.
        op.calc_ec_operator();

        // Build the excitation waveform now that the timestep is known.
        // Native openEMS samples the signal once instead of evaluating an
        // analytic formula inside the time loop.
        if let Some(excitation) = self.excitation.as_mut() {
            if let Err(error) = excitation.build(op.base.d_t, self.nr_ts) {
                eprintln!("[openEMS] Failed to build excitation signal: {}", error);
                return false;
            }
        }

        // Apply boundary conditions.
        op.apply_electric_bc();
        op.apply_magnetic_bc();

        // Native `Operator_Ext_Cylinder::BuildExtension` runs after the
        // electric BC (which zeroes the axis edges) and installs the summed
        // rho = 0 coefficients.
        op.build_r0_coefficients();

        // Native debug dumps (`--debug-*`).
        if self.debug_csx_dump {
            match std::fs::write("debugCSX.xml", &xml) {
                Ok(()) => println!("[openEMS] CSX geometry written to 'debugCSX.xml'"),
                Err(error) => eprintln!("[openEMS] Could not write 'debugCSX.xml': {error}"),
            }
        }
        if self.debug_material_dump
            || self.debug_operator_dump
            || self.debug_box_dump
            || self.debug_pec_dump
        {
            self.write_debug_dumps(&op);
        }

        // Build every operator-side extension through one ordered registry,
        // mirroring `Operator::AddExtension` + `BuildExtension`. Physics that
        // needs to rewrite the operator happens here, after the boundary
        // conditions and before the operator is moved into the engine.
        let mut registry = OperatorExtensionRegistry::new();
        if let Some(signal) = self.excitation.as_deref() {
            let period = signal.signal_period();
            if period > 0 {
                registry.add(Box::new(SteadyStateOperatorExtension { period_ts: period }));
            }
            registry.add(Box::new(ExcitationOperatorExtension {
                signal: Arc::new(signal.clone()),
            }));
            registry.add(Box::new(TfsfOperatorExtension {
                signal: Arc::new(signal.clone()),
            }));
        }
        if op.base.m_bc.iter().any(|&boundary| boundary == BC_PML) {
            registry.add(Box::new(UpmlOperatorExtension {
                pml_size: self.pml_size,
            }));
        }
        // Conducting sheets need the boundary-layer sizes: edges inside a Mur
        // or PML layer fall back to PEC, as in the native operator.
        let has_conducting_sheet = op.structure.as_ref().is_some_and(|structure| {
            (0..structure.property_count()).any(|index| {
                structure
                    .property_by_index(index)
                    .is_some_and(|property| property.conducting_sheet().is_some())
            })
        });
        if has_conducting_sheet {
            let mut boundary_size = [0u32; 6];
            for face in 0..6 {
                if op.base.m_bc[face] == BC_PML {
                    boundary_size[face] = self.pml_size[face];
                } else if op.base.m_bc[face] == BC_MUR {
                    boundary_size[face] = 1;
                }
            }
            let f_max = self
                .excitation
                .as_ref()
                .map(|signal| signal.f_max)
                .unwrap_or(0.0);
            registry.add(Box::new(ConductingSheetOperatorExtension {
                f_max,
                boundary_size,
            }));
        }
        // Lorentz and Debye materials need the finished EC coefficients to size
        // their Drude/Lorentz elements. Native adds Lorentz first.
        let has_lorentz = op.structure.as_ref().is_some_and(|structure| {
            (0..structure.property_count()).any(|index| {
                structure
                    .property_by_index(index)
                    .is_some_and(|property| property.lorentz_material().is_some())
            })
        });
        if has_lorentz {
            registry.add(Box::new(LorentzOperatorExtension));
        }
        let has_debye = op.structure.as_ref().is_some_and(|structure| {
            (0..structure.property_count()).any(|index| {
                structure
                    .property_by_index(index)
                    .is_some_and(|property| property.debye_material().is_some())
            })
        });
        if has_debye {
            registry.add(Box::new(DebyeOperatorExtension));
        }
        // Lumped RLC loads rewrite the EC capacitance/conductance and add an
        // ADE that runs after the sheet model, as in the native setup order.
        let has_lumped_element = op.structure.as_ref().is_some_and(|structure| {
            (0..structure.property_count()).any(|index| {
                structure
                    .property_by_index(index)
                    .is_some_and(|property| property.lumped_element().is_some())
            })
        });
        if has_lumped_element {
            let f_max = self
                .excitation
                .as_ref()
                .map(|signal| signal.f_max)
                .unwrap_or(0.0);
            registry.add(Box::new(LumpedRLCOperatorExtension { f_max }));
        }
        registry.add(Box::new(MurOperatorExtension {
            phase_velocity: self.mur_v_ph,
        }));
        let mut engine_extensions = match registry.build(&mut op) {
            Ok(extensions) => extensions,
            Err(error) => {
                eprintln!("[openEMS] Failed to build an operator extension: {error}");
                return false;
            }
        };
        // Native `Operator_Ext_Cylinder`: a closed azimuthal mesh is periodic.
        if let Some(cylinder) = op.build_cylinder_extension() {
            engine_extensions.push(Box::new(cylinder));
        }
        if op.is_cylindrical() && self.verbose_level > 0 {
            println!(
                "[openEMS] cylindrical mesh: rho = {:?}..{:?}, alpha closed = {}, r0 axis = {}",
                op.base.disc_lines[0].as_ref().and_then(|l| l.first()),
                op.base.disc_lines[0].as_ref().and_then(|l| l.last()),
                op.closed_alpha(),
                op.r0_included
            );
        }

        // Box as trait object for the engine (unsizing coercion).
        let op_dyn: Box<dyn OperatorBase> = Box::new(op);
        let mut engine = Engine::new(op_dyn);
        engine.set_number_of_threads(self.num_threads);
        for extension in engine_extensions {
            engine.add_extension(extension);
        }
        self.fdtd_eng = Some(Box::new(engine));
        self.fdtd_op = None; // operator is owned by the engine

        // Native `openems.cpp` writes the sampled excitation waveforms (`et`,
        // `ht`) before `SetupProcessing`; a probe that reuses those names then
        // truncates the file when it is initialised.
        if self.enable_dumps {
            let dt = self
                .fdtd_eng
                .as_ref()
                .expect("engine checked above")
                .op
                .get_timestep();
            self.dump_excitation_signals(dt);
        }

        // Native `SetupProcessing`: every `ProbeBox` and `DumpBox` property
        // becomes a processing object that streams its data to a file.
        self.setup_csx_probes(&xml);
        self.setup_csx_dumps(&xml);

        true
    }

    /// Create the field-dump processings from the CSX `DumpBox` properties.
    ///
    /// Mirrors `openEMS::SetupProcessing` for dump boxes: time-domain dumps
    /// (`DumpType` 0..5) use the global oversampling, H- and B-field dumps the
    /// dual mesh and the half timestep, and both are cropped to the box.
    fn setup_csx_dumps(&mut self, xml: &str) {
        let Ok(structure) = csxcad_core::model::Structure::from_xml_str(xml) else {
            return;
        };
        let Some(engine) = self.fdtd_eng.as_ref() else {
            return;
        };
        let dt = engine.op.get_timestep();
        let grid_delta = engine.op.get_grid_delta();
        let f_max = self
            .excitation
            .as_ref()
            .map(|signal| signal.f_max)
            .unwrap_or(0.0);
        let nyquist = if f_max > 0.0 && dt > 0.0 {
            (1.0 / f_max / 2.0 / dt).floor() as u32
        } else {
            0
        };
        let interval = if nyquist > 0 {
            (nyquist / self.over_sampling.max(1)).max(1)
        } else {
            1
        };

        let cylindrical = matches!(
            engine.op.mesh_type(),
            crate::operator::MeshType::Cylindrical { .. }
        );
        let mesh_type = if cylindrical {
            crate::processing::MeshType::Cylindrical
        } else {
            crate::processing::MeshType::Cartesian
        };

        let mut array = self
            .processing_array
            .take()
            .unwrap_or_else(|| ProcessingArray::new(u32::MAX));
        let mut created = 0usize;
        for property in structure.properties_by_type(csxcad_core::model::PropertyType::DumpBox) {
            let Some(dump) = property.dump_box() else {
                continue;
            };
            let multiple = property.primitive_count() > 1;
            for (index, primitive) in property.primitives().iter().enumerate() {
                let name = if multiple {
                    format!("{}_{}", property.name(), index)
                } else {
                    property.name().to_string()
                };
                let Ok(bounds) = primitive.bounding_box(None, true) else {
                    continue;
                };
                // native `SnapBox2Mesh` receives drawing-unit coordinates; the
                // cylindrical azimuth is an angle and never scales.
                let scale = |axis: usize| -> f64 {
                    if axis == 1 && cylindrical {
                        1.0
                    } else {
                        grid_delta
                    }
                };
                let start = [
                    bounds[0] * scale(0),
                    bounds[2] * scale(1),
                    bounds[4] * scale(2),
                ];
                let stop = [
                    bounds[1] * scale(0),
                    bounds[3] * scale(1),
                    bounds[5] * scale(2),
                ];
                let dump_type = dump.dump_type();
                match dump_type {
                    0..=5 => {
                        let mut processing =
                            crate::processing::ProcessFieldsTD::new(&name, start, stop);
                        processing.set_bounds(start, stop);
                        processing.set_dump_type(dump_type);
                        processing.set_file_type(dump.file_type());
                        processing.set_dump_mode(dump.dump_mode());
                        processing.set_legacy_format(self.legacy_hdf5_dumps);
                        processing.set_native_field_dumps(self.native_field_dumps);
                        processing.set_process_interval(interval);
                        if cylindrical {
                            processing.set_mesh_type(mesh_type);
                        }
                        if dump.sub_sampling_enabled() {
                            for (axis, value) in dump.sub_sampling().iter().enumerate() {
                                processing.set_sub_sampling(*value as i32, axis as i32);
                            }
                        }
                        if dump.optimal_resolution_enabled() {
                            for (axis, value) in dump.optimal_resolution().iter().enumerate() {
                                processing.set_opt_resolution(*value, axis as i32);
                            }
                        }
                        if dump_type == 1 || dump_type == 5 {
                            processing.set_dual_time(true);
                            processing.set_dual_mesh(true);
                        }
                        // native `SetDumpMode` overrides the H dual-mesh default
                        match dump.dump_mode() {
                            1 => processing.set_dual_mesh(false),
                            2 => processing.set_dual_mesh(true),
                            _ => {}
                        }
                        array.add_processing(Box::new(processing));
                        created += 1;
                    }
                    10..=15 => {
                        let base_type = dump_type - 10;
                        let mut processing =
                            crate::processing::ProcessFieldsFD::new(&name, start, stop);
                        processing.set_bounds(start, stop);
                        processing.set_dump_type(base_type);
                        processing.set_file_type(dump.file_type());
                        processing.set_dump_mode_value(dump.dump_mode());
                        processing.set_legacy_format(self.legacy_hdf5_dumps);
                        processing.set_native_field_dumps(self.native_field_dumps);
                        processing.set_process_interval(interval);
                        processing.set_fd_nyquist(nyquist);
                        // native: FD dumps default to the plain Nyquist rate
                        processing.set_fd_oversampling(Self::dump_oversampling(property));
                        if cylindrical {
                            processing.set_mesh_type(mesh_type);
                        }
                        if dump.sub_sampling_enabled() {
                            for (axis, value) in dump.sub_sampling().iter().enumerate() {
                                processing.set_sub_sampling_value(*value as i32, axis as i32);
                            }
                        }
                        if dump.optimal_resolution_enabled() {
                            for (axis, value) in dump.optimal_resolution().iter().enumerate() {
                                processing.set_opt_resolution_value(*value, axis as i32);
                            }
                        }
                        if base_type == 1 || base_type == 5 {
                            processing.set_dual_time(true);
                            processing.set_dual_mesh(true);
                        }
                        match dump.dump_mode() {
                            1 => processing.set_dual_mesh(false),
                            2 => processing.set_dual_mesh(true),
                            _ => {}
                        }
                        for frequency in dump.fd_samples() {
                            processing.add_frequency(*frequency);
                        }
                        if processing.get_frequencies().is_empty() {
                            eprintln!(
                                "[openEMS] SetupFDTD: Warning: frequency-domain dump '{}' has no <Frequency> samples... skipping",
                                property.name()
                            );
                            continue;
                        }
                        array.add_processing(Box::new(processing));
                        created += 1;
                    }
                    29 => {
                        let mut processing = crate::processing::ProcessFieldsSAR::new(&name, 29);
                        if cylindrical {
                            processing.set_mesh_type(mesh_type);
                        }
                        processing.set_bounds(start, stop);
                        processing.set_cell_data(Self::build_sar_cell_data(&structure, engine));
                        processing.set_file_type(dump.file_type());
                        processing.set_dump_mode_value(dump.dump_mode());
                        processing.set_sar_method(dump.sar_method());
                        processing.set_legacy_format(self.legacy_hdf5_dumps);
                        processing.set_process_interval(interval);
                        processing.set_fd_nyquist(nyquist);
                        processing.set_fd_oversampling(Self::dump_oversampling(property));
                        for frequency in dump.fd_samples() {
                            processing.add_frequency(*frequency);
                        }
                        if processing.get_frequencies().is_empty() {
                            eprintln!(
                                "[openEMS] SetupFDTD: Warning: SAR dump '{}' has no <FD_Samples> frequencies... skipping",
                                property.name()
                            );
                            continue;
                        }
                        if dump.file_type() != 1 {
                            eprintln!(
                                "[openEMS] SetupFDTD: Warning: raw SAR dump '{}' requires the HDF5 file type... skipping",
                                property.name()
                            );
                            continue;
                        }
                        array.add_processing(Box::new(processing));
                        created += 1;
                    }
                    20..=22 => {
                        // Native `ProcessFieldsSAR` result dumps: the same
                        // in-solver DFT as the raw dump, post-processed by the
                        // shared `SAR_Calculation` writer.
                        if dump.file_type() != 1 {
                            eprintln!(
                                "[openEMS] SetupFDTD: Warning: SAR result dump '{}' requires the HDF5 file type... skipping",
                                property.name()
                            );
                            continue;
                        }
                        let mut processing =
                            crate::processing::ProcessFieldsSAR::new(&name, dump_type);
                        if cylindrical {
                            processing.set_mesh_type(mesh_type);
                        }
                        processing.set_bounds(start, stop);
                        processing.set_cell_data(Self::build_sar_cell_data(&structure, engine));
                        processing.set_file_type(dump.file_type());
                        processing.set_dump_mode_value(dump.dump_mode());
                        processing.set_sar_method(dump.sar_method());
                        processing.set_legacy_format(self.legacy_hdf5_dumps);
                        processing.set_process_interval(interval);
                        processing.set_fd_nyquist(nyquist);
                        processing.set_fd_oversampling(Self::dump_oversampling(property));
                        for frequency in dump.fd_samples() {
                            processing.add_frequency(*frequency);
                        }
                        if processing.get_frequencies().is_empty() {
                            eprintln!(
                                "[openEMS] SetupFDTD: Warning: SAR dump '{}' has no <FD_Samples> frequencies... skipping",
                                property.name()
                            );
                            continue;
                        }
                        array.add_processing(Box::new(processing));
                        created += 1;
                    }
                    other => {
                        eprintln!(
                            "[openEMS] SetupFDTD: unknown dump box type {} of property '{}'... skipping",
                            other,
                            property.name()
                        );
                    }
                }
            }
        }
        self.processing_array = Some(array);
        if created > 0 && self.verbose_level > 0 {
            println!("[openEMS] {} field dump(s) configured", created);
        }
    }

    /// Native `CSPropProbeBox::GetOverSampling` of a dump box: a dump box
    /// inherits the probe's oversampling override, `1` (plain Nyquist) when it
    /// is unset.
    fn dump_oversampling(property: &csxcad_core::model::Property) -> u32 {
        property
            .probe()
            .map(|probe| probe.oversampling())
            .filter(|value| *value >= 0)
            .map(|value| value.max(1) as u32)
            .unwrap_or(1)
    }

    /// Per-cell volume, mass density and conductivity for the SAR dumps.
    ///
    /// Mirrors `ProcessFieldsSAR::DumpFDData`: the volume from the mesh, the mass
    /// density and conductivity from the material at the cell centre
    /// (`GetCellCenterMaterialAvgCoord` + `GetPropertyByCoordPriority`).
    fn build_sar_cell_data(
        structure: &csxcad_core::model::Structure,
        engine: &Engine,
    ) -> crate::processing::SarCellData {
        let params = structure.parameters();
        let num_lines = engine.num_lines;
        let closed_alpha = matches!(
            engine.op.mesh_type(),
            crate::operator::MeshType::Cylindrical { .. }
        );
        let line = |axis: usize, index: usize| -> f64 {
            engine.op.get_disc_line(axis as i32, index as u32, false)
        };
        let centre = |axis: usize, index: usize| -> f64 {
            let count = num_lines[axis];
            if index + 1 < count {
                0.5 * (line(axis, index) + line(axis, index + 1))
            } else if index > 0 {
                line(axis, index) + 0.5 * (line(axis, index) - line(axis, index - 1))
            } else {
                line(axis, 0)
            }
        };

        let cells = num_lines[0] * num_lines[1] * num_lines[2];
        let mut data = crate::processing::SarCellData {
            volume: vec![0.0; cells],
            density: vec![0.0; cells],
            conductivity: vec![0.0; cells],
        };
        for i in 0..num_lines[0] {
            for j in 0..num_lines[1] {
                for k in 0..num_lines[2] {
                    let index = (i * num_lines[1] + j) * num_lines[2] + k;
                    // native `GetCellVolume`: the operator metric of the cell, so
                    // a cylindrical mesh yields annulus segments and the
                    // `rho = 0` axis cells their wedge volumes
                    data.volume[index] = engine
                        .op
                        .get_cell_volume([i as u32, j as u32, k as u32], false);
                    // the material lookup uses drawing units, like the native
                    // GetPropertyByCoordPriority; on a closed azimuth the seam and
                    // padding lines map back into the ring
                    // (native `GetCellCenterMaterialAvgCoord`)
                    let j_eff = if closed_alpha && j + 2 >= num_lines[1] {
                        j - (num_lines[1] - 2)
                    } else {
                        j
                    };
                    let coordinate = [centre(0, i), centre(1, j_eff), centre(2, k)];
                    let mut winner: Option<&csxcad_core::material::MaterialProperties> = None;
                    let mut winner_priority = i32::MIN;
                    for property in 0..structure.property_count() {
                        let Some(property) = structure.property_by_index(property) else {
                            continue;
                        };
                        let Some(material) = property.material() else {
                            continue;
                        };
                        for primitive in property.primitives() {
                            if winner.is_some() && primitive.priority() <= winner_priority {
                                continue;
                            }
                            if primitive
                                .is_inside(coordinate, Some(params))
                                .unwrap_or(false)
                            {
                                winner = Some(material);
                                winner_priority = primitive.priority();
                            }
                        }
                    }
                    if let Some(material) = winner {
                        data.density[index] = material
                            .weighted_density(
                                Some(params),
                                coordinate,
                                csxcad_core::grid::CoordinateSystem::Cartesian,
                            )
                            .unwrap_or(0.0);
                        data.conductivity[index] = material
                            .weighted_value(
                                csxcad_core::material::MaterialField::Kappa,
                                0,
                                Some(params),
                                coordinate,
                                csxcad_core::grid::CoordinateSystem::Cartesian,
                            )
                            .ok()
                            .flatten()
                            .unwrap_or(0.0);
                    }
                }
            }
        }
        if std::env::var_os("OPENEMS_DEBUG_PROBES").is_some() {
            let nonzero = data.density.iter().filter(|value| **value > 0.0).count();
            eprintln!(
                "[debug] SAR cell data: {nonzero} of {cells} cells with density, {} properties",
                structure.property_count()
            );
        }
        data
    }

    /// Create the probe processings from the CSX `ProbeBox` properties.
    ///
    /// Mirrors `openEMS::SetupProcessing` for probe boxes: voltage probes use a
    /// line integral, current probes a surface integral, field probes all three
    /// components at one cell, and mode-match probes an overlap integral.
    fn setup_csx_probes(&mut self, xml: &str) {
        let debug = std::env::var_os("OPENEMS_DEBUG_PROBES").is_some();
        let structure = match csxcad_core::model::Structure::from_xml_str(xml) {
            Ok(structure) => structure,
            Err(error) => {
                if debug {
                    eprintln!("[debug] probe setup: CSX parse failed: {error}");
                }
                return;
            }
        };
        if debug {
            let count = structure
                .properties_by_type(csxcad_core::model::PropertyType::ProbeBox)
                .count();
            eprintln!("[debug] probe setup: {count} probe box properties");
        }
        let Some(engine) = self.fdtd_eng.as_ref() else {
            return;
        };
        let dt = engine.op.get_timestep();
        let grid_delta = engine.op.get_grid_delta();
        let probe_cylindrical = matches!(
            engine.op.mesh_type(),
            crate::operator::MeshType::Cylindrical { .. }
        );
        let f_max = self
            .excitation
            .as_ref()
            .map(|signal| signal.f_max)
            .unwrap_or(0.0);
        let nyquist = if f_max > 0.0 && dt > 0.0 {
            (1.0 / f_max / 2.0 / dt).floor() as u32
        } else {
            0
        };
        let interval = if nyquist > 0 {
            (nyquist / self.over_sampling.max(1)).max(1)
        } else {
            1
        };
        if std::env::var_os("OPENEMS_DEBUG_PROBES").is_some() {
            eprintln!(
                "[debug] setup probes: dt={dt:e} f_max={f_max:e} f0={:?} fc={:?} nyquist={nyquist} over={} interval={interval}",
                self.excitation.as_ref().map(|s| s.f0),
                self.excitation.as_ref().map(|s| s.fc),
                self.over_sampling
            );
        }

        let mut array = self
            .processing_array
            .take()
            .unwrap_or_else(|| ProcessingArray::new(u32::MAX));
        let mut created = 0usize;
        for property in structure.properties_by_type(csxcad_core::model::PropertyType::ProbeBox) {
            let Some(probe) = property.probe() else {
                continue;
            };
            let multiple = property.primitive_count() > 1;
            for (index, primitive) in property.primitives().iter().enumerate() {
                let name = if multiple {
                    format!("{}_{}", property.name(), index)
                } else {
                    property.name().to_string()
                };
                let Ok(bounds) = primitive.bounding_box(None, true) else {
                    continue;
                };
                // like the dump boxes, the azimuth coordinate is an angle and
                // never scales by the grid delta
                let probe_scale = |axis: usize| -> f64 {
                    if axis == 1 && probe_cylindrical {
                        1.0
                    } else {
                        grid_delta
                    }
                };
                let start = [
                    bounds[0] * probe_scale(0),
                    bounds[2] * probe_scale(1),
                    bounds[4] * probe_scale(2),
                ];
                let stop = [
                    bounds[1] * probe_scale(0),
                    bounds[3] * probe_scale(1),
                    bounds[5] * probe_scale(2),
                ];

                let calculator: Box<dyn crate::processing_integral::IntegralCalculator> =
                    match probe.probe_type() {
                        0 => {
                            let Ok(dimension) = primitive.dimension(None) else {
                                continue;
                            };
                            let Some(shape) = primitive.box_shape() else {
                                eprintln!(
                                    "[openEMS] SetupProcessing: probe '{}' is not a box, skipping",
                                    property.name()
                                );
                                continue;
                            };
                            if dimension != 1 {
                                eprintln!(
                                    "[openEMS] SetupProcessing: probe '{}' is not a line, skipping",
                                    property.name()
                                );
                                continue;
                            }
                            let Ok(coordinates) = shape.evaluated_coordinates(None) else {
                                continue;
                            };
                            let line_start = coordinates[0].map(|value| value * grid_delta);
                            let line_stop = coordinates[1].map(|value| value * grid_delta);
                            Box::new(VoltageIntegral::new(line_start, line_stop))
                        }
                        1 => Box::new(CurrentIntegral::new(start, stop, probe.normal_direction())),
                        2 | 3 => {
                            let centre = [
                                0.5 * (start[0] + stop[0]),
                                0.5 * (start[1] + stop[1]),
                                0.5 * (start[2] + stop[2]),
                            ];
                            Box::new(FieldProbeIntegral::new(
                                centre,
                                usize::from(probe.probe_type() == 3),
                            ))
                        }
                        10 | 11 => {
                            let mut integral = ModeMatchIntegral::new(start, stop);
                            integral.set_field_type(usize::from(probe.probe_type() == 11));
                            for component in 0..3 {
                                if let Some(function) = probe.mode_function(component) {
                                    integral.set_weight_function(component, function);
                                }
                            }
                            integral.set_weight_origin(probe.mode_origin());
                            Box::new(integral)
                        }
                        other => {
                            eprintln!(
                                "[openEMS] SetupFDTD: Warning: Probe type {} of property '{}' is unknown...",
                                other,
                                property.name()
                            );
                            continue;
                        }
                    };

                let mut processing = ProcessIntegral::new(&name, &name, start, stop, calculator);
                if probe.probe_type() == 1 || probe.probe_type() == 3 {
                    processing.set_dual_time(true);
                }
                processing.set_weight(probe.weight());
                processing.set_normal_dir(probe.normal_direction());
                processing.set_process_interval(interval);
                processing.set_fd_nyquist(nyquist);
                processing.add_frequencies(probe.fd_samples());
                let oversampling = probe.oversampling();
                if oversampling > 0 {
                    processing.set_fd_oversampling(oversampling as u32);
                }
                if probe.start_time() > 0.0 || probe.stop_time() > 0.0 {
                    processing.set_process_start_stop_time(
                        probe.start_time(),
                        probe.stop_time(),
                        dt,
                    );
                }
                processing.init_process();
                array.add_processing(Box::new(processing));
                created += 1;
            }
        }
        self.processing_array = Some(array);
        if created > 0 && self.verbose_level > 0 {
            println!("[openEMS] {} probe processing(s) configured", created);
        }
    }

    /// Run the FDTD simulation for the configured number of time steps.
    /// Mirrors C++ `RunFDTD`.
    pub fn run_fdtd(&mut self) -> bool {
        if self.fdtd_eng.is_none() {
            return false;
        }
        if self.no_simulation {
            println!("[openEMS] preprocessing only (--no-simulation)");
            return true;
        }
        let started = std::time::Instant::now();

        // Invoke PreProcess on all registered processings. The array is taken
        // out so that the field interface can borrow the engine immutably while
        // the processing state is mutable.
        let mut processing = self.processing_array.take();
        if let Some(ref mut pa) = processing {
            // Native global oversampling: probes and dumps without an explicit
            // interval sample at Nyquist/oversampling.
            let dt = self
                .fdtd_eng
                .as_ref()
                .expect("engine checked above")
                .op
                .get_timestep();
            let f_max = self
                .excitation
                .as_ref()
                .map(|signal| signal.f_max)
                .unwrap_or(0.0);
            let nyquist = if f_max > 0.0 && dt > 0.0 {
                (1.0 / f_max / 2.0 / dt).floor() as u32
            } else {
                0
            };
            if nyquist > 0 {
                let interval = nyquist / self.over_sampling.max(1);
                if interval > 0 {
                    pa.set_global_interval(interval);
                }
            }
            pa.pre_process();
        }

        let nr_ts = self.nr_ts;
        let end_crit = self.end_crit;
        let dt = self
            .fdtd_eng
            .as_ref()
            .expect("engine checked above")
            .op
            .get_timestep();
        let engine_meta = {
            let engine = self.fdtd_eng.as_ref().expect("engine checked above");
            let num_lines = engine.num_lines;
            let mesh_type = engine.op.mesh_type();
            let mut disc_lines: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
            for axis in 0..3 {
                // native `GetDiscLine` keeps the drawing units; the cylindrical
                // azimuth is an angle, so it never scales by the grid delta.
                let scale = if axis == 1
                    && matches!(mesh_type, crate::operator::MeshType::Cylindrical { .. })
                {
                    1.0
                } else {
                    engine.op.get_grid_delta()
                };
                for index in 0..num_lines[axis] {
                    disc_lines[axis]
                        .push(engine.op.get_disc_line(axis as i32, index as u32, false) * scale);
                }
            }
            EngineMeta {
                num_lines,
                disc_lines,
                dt,
                materials: engine
                    .op
                    .material_grid_data()
                    .map(|grid| std::sync::Arc::new(grid.to_vec())),
                mesh_type: engine.op.mesh_type(),
                grid_delta: engine.op.get_grid_delta(),
            }
        };
        let max_ts_time = if self.max_time > 0.0 && dt > 0.0 {
            (self.max_time / dt).floor().max(0.0) as u32
        } else {
            u32::MAX
        };
        let ts_limit = if nr_ts == 0 {
            max_ts_time
        } else {
            nr_ts.min(max_ts_time)
        };

        // Native `Process` runs once at ts=0 before the first field update, so
        // every stream starts with its t=0 sample.
        if let Some(ref mut pa) = processing {
            let engine = self.fdtd_eng.as_ref().expect("engine checked above");
            let interface = EngineInterface {
                meta: &engine_meta,
                volt: &engine.volt_ptr,
                curr: &engine.curr_ptr,
            };
            pa.process_fields(0, &interface);
        }

        // Native end-criteria cadence (`openems.cpp:1456-1467`, `:1511-1526`):
        // the default evaluates the rough energy every four seconds of wall
        // clock; `--exact-endcriteria` evaluates every excitation Nyquist
        // period. `maxE` is seeded at the maximum-excitation timestep even in
        // the default mode, where that sample does not update `change`.
        let exact = self.exact_end_criteria;
        let nyquist = self
            .excitation
            .as_ref()
            .map(|signal| signal.nyquist_num())
            .unwrap_or(0);
        let max_excite_ts = self
            .excitation
            .as_ref()
            .map(|signal| signal.max_timestep() as u32);
        if exact {
            println!(
                "Exact-endcriteria: evaluating the end criteria every {} timestep(s) (this excitation's Nyquist rate)",
                nyquist.max(1)
            );
        }

        let mut ts = 0u32;
        let mut change: f64 = 1.0;
        let mut max_energy: f64 = 0.0;
        let started = std::time::Instant::now();
        // `prevTime = startTime` in native: the first checkpoint is ~4 s in.
        let mut last_checkpoint = 0.0f64;
        let mut max_excite_sampled = false;
        if max_excite_ts == Some(0) {
            max_energy = self
                .fdtd_eng
                .as_ref()
                .map(|engine| engine.fast_energy())
                .unwrap_or(0.0);
            max_excite_sampled = true;
        }

        while ts < ts_limit && change > end_crit && !self.check_abort_cond() {
            // Execute one timestep (voltage + current updates with extensions).
            self.fdtd_eng.as_mut().unwrap().iterate_ts(1);
            ts = self.fdtd_eng.as_ref().unwrap().get_number_of_timesteps();

            // Invoke Process on all registered processings with field access.
            if let Some(ref mut pa) = processing {
                let engine = self.fdtd_eng.as_ref().expect("engine checked above");
                let interface = EngineInterface {
                    meta: &engine_meta,
                    volt: &engine.volt_ptr,
                    curr: &engine.curr_ptr,
                };
                pa.process_fields(ts, &interface);
            }

            // Steady-state end criterion: the extension metric when a periodic
            // signal provides one, otherwise the native energy fallback
            // (`currE / maxE`).
            match self.fdtd_eng.as_ref().unwrap().get_steady_state_diff() {
                Some(diff) => change = diff as f64,
                None => {
                    let seed = !max_excite_sampled && max_excite_ts == Some(ts);
                    let due = end_criteria_due(
                        exact,
                        ts,
                        nyquist,
                        if exact {
                            0.0
                        } else {
                            started.elapsed().as_secs_f64()
                        },
                        &mut last_checkpoint,
                    );
                    if seed || due {
                        let energy = self.fdtd_eng.as_ref().unwrap().fast_energy();
                        if energy > max_energy {
                            max_energy = energy;
                        }
                        // default mode: only the wall-clock checkpoint updates
                        // `change`; exact mode also updates it at the seed
                        let update_change = if exact { true } else { due };
                        if update_change && max_energy > 0.0 {
                            change = energy / max_energy;
                        }
                    }
                    if seed {
                        max_excite_sampled = true;
                    }
                }
            }

            // Progress reporting every 100 steps.
            if ts.is_multiple_of(100) {
                if std::env::var_os("OPENEMS_DEBUG_ENERGY").is_some() {
                    println!(
                        "[openEMS] Timestep: {ts} || Energy change: {change:.3e} ({:.2} dB)",
                        -10.0 * change.max(1e-300).log10()
                    );
                } else {
                    println!("[openEMS] Timestep: {}", ts);
                }
            }
        }

        // Invoke PostProcess on all registered processings.
        if let Some(ref mut pa) = processing {
            pa.post_process();
        }
        self.processing_array = processing;

        if change > end_crit {
            eprintln!(
                "[openEMS] Warning: Max. number of timesteps reached before end-criteria of {:.1} dB.",
                -10.0 * end_crit.log10()
            );
        } else {
            println!(
                "[openEMS] End-criteria of {:.1} dB reached after {} timesteps.",
                -10.0 * change.log10(),
                ts
            );
        }

        println!("[openEMS] FDTD simulation completed: {} timesteps.", ts);

        let cells = self
            .fdtd_eng
            .as_ref()
            .map(|engine| engine.op.get_number_cells() as u64)
            .unwrap_or(0);
        let stats = RunStats {
            cells,
            timesteps: ts,
            dt,
            wall_seconds: started.elapsed().as_secs_f64(),
            end_change: change,
            end_criteria: end_crit,
        };
        if self.dump_statistics {
            self.write_statistics(&stats);
        }
        self.last_run_stats = Some(stats);

        true
    }

    /// Write `openEMS_stats.txt` and `openEMS_run_stats.txt`.
    fn write_statistics(&self, stats: &RunStats) {
        let mut setup = String::new();
        setup.push_str("# openEMS simulation statistics\n");
        setup.push_str(&format!("cells = {}\n", stats.cells));
        setup.push_str(&format!("dt = {:.12e}\n", stats.dt));
        for face in 0..6 {
            setup.push_str(&format!(
                "boundary[{}] = {}\n",
                face,
                match self.bc_type[face] {
                    BC_PEC => "PEC",
                    BC_PMC => "PMC",
                    BC_MUR => "MUR",
                    BC_PML => "PML",
                    _ => "unknown",
                }
            ));
            setup.push_str(&format!("pml_size[{}] = {}\n", face, self.pml_size[face]));
        }
        setup.push_str(&format!("over_sampling = {}\n", self.over_sampling));
        setup.push_str(&format!("engine = {}\n", self.engine));

        let mut run = String::new();
        run.push_str("# openEMS run statistics\n");
        run.push_str(&format!("timesteps = {}\n", stats.timesteps));
        run.push_str(&format!("wall_seconds = {:.6}\n", stats.wall_seconds));
        run.push_str(&format!("end_change = {:.12e}\n", stats.end_change));
        run.push_str(&format!("end_criteria = {:.12e}\n", stats.end_criteria));
        run.push_str(&format!(
            "end_criteria_dB = {:.3}\n",
            -10.0 * stats.end_change.max(f64::MIN_POSITIVE).log10()
        ));

        for (name, contents) in [("openEMS_stats.txt", setup), ("openEMS_run_stats.txt", run)] {
            match std::fs::write(name, contents) {
                Ok(()) => println!("[openEMS] statistics written to '{name}'"),
                Err(error) => eprintln!("[openEMS] could not write '{name}': {error}"),
            }
        }
    }
}

impl Default for OpenEMS {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openems_creation() {
        let sim = OpenEMS::new();
        assert_eq!(sim.nr_ts, 0);
        assert!((sim.end_crit - 1e-6).abs() < 1e-12);
        assert!(sim.enable_dumps);
        assert!(!sim.cylinder_coords);
        assert_eq!(sim.bc_type, [BC_PEC; 6]);
    }

    #[test]
    fn test_set_number_of_time_steps() {
        let mut sim = OpenEMS::new();
        sim.set_number_of_time_steps(1000);
        assert_eq!(sim.nr_ts, 1000);
    }

    #[test]
    fn test_set_end_criteria() {
        let mut sim = OpenEMS::new();
        sim.set_end_criteria(1e-4);
        assert!((sim.end_crit - 1e-4).abs() < 1e-12);
    }

    #[test]
    fn test_set_bc_type() {
        let mut sim = OpenEMS::new();
        sim.set_bc_type(0, BC_MUR);
        assert_eq!(sim.bc_type[0], BC_MUR);
        assert_eq!(sim.bc_type[1], BC_PEC);
    }

    #[test]
    fn test_set_pml_size() {
        let mut sim = OpenEMS::new();
        sim.set_pml_size(0, 10);
        assert_eq!(sim.pml_size[0], 10);
    }

    #[test]
    fn test_set_cylinder_coords() {
        let mut sim = OpenEMS::new();
        sim.set_cylinder_coords(true);
        assert!(sim.cylinder_coords);
    }

    #[test]
    fn test_setup_and_run() {
        let mut sim = OpenEMS::new();
        sim.set_number_of_time_steps(10);
        sim.set_csx_xml(
            r#"<ContinuousStructure CoordSystem="0">
                <RectilinearGrid DeltaUnit="0.001">
                    <XLines>0,1,2,3</XLines>
                    <YLines>0,1,2,3</YLines>
                    <ZLines>0,1,2,3</ZLines>
                </RectilinearGrid>
                <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
                <Properties></Properties>
            </ContinuousStructure>"#
                .to_string(),
        );
        assert!(sim.setup_fdtd());
        assert!(sim.run_fdtd());
        assert_eq!(sim.fdtd_eng.as_ref().unwrap().get_number_of_timesteps(), 10);
    }

    const SMALL_CSX: &str = r#"<ContinuousStructure CoordSystem="0">
        <RectilinearGrid DeltaUnit="0.001">
            <XLines>0,1,2,3</XLines>
            <YLines>0,1,2,3</YLines>
            <ZLines>0,1,2,3</ZLines>
        </RectilinearGrid>
        <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
        <Properties></Properties>
    </ContinuousStructure>"#;

    #[test]
    fn timestep_factor_scales_the_cfl_step() {
        let mut reference = OpenEMS::new();
        reference.set_number_of_time_steps(1);
        reference.set_csx_xml(SMALL_CSX.to_string());
        assert!(reference.setup_fdtd());
        let dt_reference = reference.fdtd_eng.as_ref().unwrap().op.get_timestep();

        let mut halved = OpenEMS::new();
        halved.set_number_of_time_steps(1);
        halved.set_time_step_factor(0.5);
        halved.set_csx_xml(SMALL_CSX.to_string());
        assert!(halved.setup_fdtd());
        let dt_halved = halved.fdtd_eng.as_ref().unwrap().op.get_timestep();
        assert!((dt_halved / dt_reference - 0.5).abs() < 1e-9);
    }

    #[test]
    fn max_time_limits_the_run() {
        let mut sim = OpenEMS::new();
        sim.set_number_of_time_steps(1_000);
        sim.set_time_step(1.0e-12);
        sim.set_max_time(5.0e-12);
        sim.set_csx_xml(SMALL_CSX.to_string());
        assert!(sim.setup_fdtd());
        assert!(sim.run_fdtd());
        assert_eq!(sim.fdtd_eng.as_ref().unwrap().get_number_of_timesteps(), 5);
    }

    #[test]
    fn test_run_without_setup() {
        let mut sim = OpenEMS::new();
        assert!(!sim.run_fdtd());
    }

    /// Native `SetupFDTD` registers one UPML extension per PML face and keeps
    /// that order (`OperatorExtensionRegistry::build` sorts by the native
    /// priorities). A Gaussian source in a PML box with no CSX excitation
    /// property registers exactly those six regions: the steady-state detector
    /// needs a periodic signal, TFSF needs a plane wave, the excitation
    /// extension needs an `<Excitation>` primitive, and Mur/sheets/lumped/
    /// Debye/Lorentz need their own properties. Native prints the same
    /// "no excitation properties found" warning and builds six PML regions.
    #[test]
    fn setup_registers_extensions_for_a_pml_fixture() {
        let lines = (0..21)
            .map(|index| index.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let xml = format!(
            r#"<ContinuousStructure CoordSystem="0">
                <RectilinearGrid DeltaUnit="0.001">
                    <XLines>{lines}</XLines>
                    <YLines>{lines}</YLines>
                    <ZLines>{lines}</ZLines>
                </RectilinearGrid>
                <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
                <Properties></Properties>
            </ContinuousStructure>"#
        );
        let mut sim = OpenEMS::new();
        sim.set_csx_xml(xml);
        sim.set_gauss_excite(5.0e9, 5.0e9);
        for face in 0..6 {
            sim.set_bc_type(face, 3); // PML
        }
        sim.set_number_of_time_steps(1);
        assert!(sim.setup_fdtd(), "PML setup must succeed");
        let names = sim.fdtd_eng.as_ref().unwrap().extension_names();
        assert_eq!(
            names,
            vec!["UPML"; 6],
            "six PML regions, one per face: {names:?}"
        );
    }

    /// Native `SetupFDTD` aborts with code 3 when no CSX geometry is present;
    /// the port must fail instead of fabricating a grid.
    #[test]
    fn setup_requires_a_csx_geometry() {
        let mut sim = OpenEMS::new();
        assert!(!sim.setup_fdtd(), "setup must fail without a CSX geometry");
        assert!(
            sim.fdtd_eng.is_none(),
            "no engine or fallback grid may be created"
        );
    }

    #[test]
    fn test_add_processing() {
        use crate::processing::FieldProbe;
        let mut sim = OpenEMS::new();
        sim.add_processing(Box::new(FieldProbe::new("probe1", [0.0, 0.0, 0.0])));
        assert!(sim.processing_array.is_some());
    }

    #[test]
    fn excitation_property_injects_field_end_to_end() {
        let xml = r#"<ContinuousStructure CoordSystem="0">
            <RectilinearGrid DeltaUnit="0.001">
                <XLines>0,1,2,3,4,5,6,7,8,9,10</XLines>
                <YLines>0,1,2,3,4,5,6,7,8,9,10</YLines>
                <ZLines>0,1,2,3,4,5,6,7,8,9,10</ZLines>
            </RectilinearGrid>
            <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
            <Properties>
                <Excitation Name="exc" Number="0" Enabled="1" Frequency="1e9" Delay="0" Type="0" Excite="1,0,0">
                    <Primitives>
                        <Box Priority="1"><P1 X="5" Y="5" Z="5"/><P2 X="6" Y="6" Z="6"/></Box>
                    </Primitives>
                </Excitation>
            </Properties>
        </ContinuousStructure>"#;
        let mut sim = OpenEMS::new();
        sim.set_number_of_time_steps(400);
        sim.set_gauss_excite(1.0e9, 5.0e8);
        sim.set_csx_xml(xml.to_string());
        assert!(sim.setup_fdtd(), "setup with excitation CSX failed");

        {
            let engine = sim.fdtd_eng.as_ref().expect("engine");
            assert_eq!(
                engine.extensions.len(),
                1,
                "the excitation extension must be registered"
            );
        }

        assert!(sim.run_fdtd());
        let engine = sim.fdtd_eng.as_ref().expect("engine");
        let [nx, ny, nz] = engine.num_lines;
        let mut max_abs = 0.0f32;
        for n in 0..3 {
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        let value = *engine.volt_ptr.get(n, i, j, k);
                        max_abs = max_abs.max(value.abs());
                    }
                }
            }
        }
        assert!(
            max_abs > 0.0,
            "the excitation property did not inject any field"
        );
        // The update factors must be non-zero for the wave to leave the source
        // cell. This guards the timestep-before-EC-coefficient ordering.
        let neighbor = engine.volt_ptr.get(0, 6, 5, 5).abs();
        assert!(
            neighbor > 0.0,
            "the excited field did not propagate to the neighbouring cell"
        );
    }
}

#[cfg(test)]
mod processing_wiring_tests {
    use super::*;
    use crate::processing::{EngineInterface, Processing};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct Counter {
        calls: Arc<AtomicUsize>,
    }

    impl Processing for Counter {
        fn get_name(&self) -> String {
            "counter".to_string()
        }
        fn set_name(&mut self, _val: &str) {}
        fn get_processing_name(&self) -> String {
            "counter".to_string()
        }
        fn set_process_interval(&mut self, _interval: u32) {}
        fn get_process_interval(&self) -> u32 {
            0
        }
        fn process_fields(&mut self, _num_ts: u32, _interface: &EngineInterface<'_>) -> i32 {
            self.calls.fetch_add(1, Ordering::Relaxed);
            0
        }
    }

    #[test]
    fn run_fdtd_drives_processing_fields() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut sim = OpenEMS::new();
        sim.set_number_of_time_steps(10);
        sim.set_csx_xml(
            r#"<ContinuousStructure CoordSystem="0">
                <RectilinearGrid DeltaUnit="0.001">
                    <XLines>0,1,2,3</XLines>
                    <YLines>0,1,2,3</YLines>
                    <ZLines>0,1,2,3</ZLines>
                </RectilinearGrid>
                <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
                <Properties></Properties>
            </ContinuousStructure>"#
                .to_string(),
        );
        sim.add_processing(Box::new(Counter {
            calls: calls.clone(),
        }));
        assert!(sim.setup_fdtd());
        assert!(sim.run_fdtd());
        // ten timesteps plus the native t=0 sample
        assert_eq!(calls.load(Ordering::Relaxed), 11);
    }
}

#[cfg(test)]
mod upml_absorption_tests {
    use crate::engine::Engine;
    use crate::excitation::Excitation;
    use crate::extensions::EngineExtExcitation;
    use crate::openems::{BC_PEC, BC_PML};
    use crate::operator::Operator;
    use std::sync::Arc;

    fn uniform_xml(nx: usize, ny: usize, nz: usize, delta: f64) -> String {
        let lines = |count: usize| {
            (0..count)
                .map(|index| index.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        format!(
            r#"<ContinuousStructure CoordSystem="0">
                <RectilinearGrid DeltaUnit="{delta}">
                    <XLines>{}</XLines>
                    <YLines>{}</YLines>
                    <ZLines>{}</ZLines>
                </RectilinearGrid>
                <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
                <Properties></Properties>
            </ContinuousStructure>"#,
            lines(nx),
            lines(ny),
            lines(nz)
        )
    }

    fn run_pulse(pml: bool) -> Vec<f64> {
        run_pulse_faces([pml; 6])
    }

    /// `pml_faces` selects which of the six faces get a PML; the rest are PEC.
    fn run_pulse_faces(pml_faces: [bool; 6]) -> Vec<f64> {
        let (nx, ny, nz) = (41usize, 21usize, 21usize);
        let delta = 1.0e-3;
        let mut op = Operator::new();
        op.load_csx(&uniform_xml(nx, ny, nz, delta))
            .expect("uniform CSX should load");
        op.base.m_bc = std::array::from_fn(|face| if pml_faces[face] { BC_PML } else { BC_PEC });
        op.calc_timestep();
        let dt = op.base.d_t;
        op.calc_ec_operator();
        op.apply_electric_bc();
        op.apply_magnetic_bc();

        let sizes: [u32; 6] = std::array::from_fn(|face| if pml_faces[face] { 8 } else { 0 });
        let extensions = if pml_faces.iter().any(|&face| face) {
            op.build_upml_extensions(sizes).expect("UPML should build")
        } else {
            Vec::new()
        };

        let mut engine = Engine::new(Box::new(op));
        for extension in extensions {
            engine.add_extension(Box::new(extension));
        }

        // Hard E_z source at the centre.
        let source = [nx / 2, ny / 2, nz / 2];
        let signal = Excitation::gauss(6.0e9, 3.0e9);
        let mut excitation = EngineExtExcitation::new();
        let mut signal = signal;
        signal.build(dt, 800).expect("signal should build");
        let signal = Arc::new(signal);
        excitation.add_voltage_excitation(2, source, 0, 1.0, signal);
        engine.add_extension(Box::new(excitation));

        let mut history = Vec::new();
        for step in 0..900u32 {
            engine.iterate_ts(1);
            if step % 10 == 0 {
                let mut energy = 0.0f64;
                for n in 0..3 {
                    for i in 0..nx {
                        for j in 0..ny {
                            for k in 0..nz {
                                let value = *engine.volt_ptr.get(n, i, j, k) as f64;
                                energy += value * value;
                            }
                        }
                    }
                }
                history.push(energy);
            }
        }
        history
    }

    #[test]
    fn upml_absorbs_a_gaussian_pulse() {
        let with_pml = run_pulse(true);
        let with_pec = run_pulse(false);
        let peak = with_pml.iter().cloned().fold(0.0f64, f64::max);
        let tail = with_pml[with_pml.len() - 10..]
            .iter()
            .cloned()
            .fold(0.0f64, f64::max);
        let pec_tail = with_pec[with_pec.len() - 10..]
            .iter()
            .cloned()
            .fold(0.0f64, f64::max);
        println!(
            "PML peak={peak:e} tail={tail:e} ratio={:e}; PEC tail={pec_tail:e}",
            tail / peak
        );
        assert!(peak > 0.0, "the pulse never launched");
        assert!(
            tail < peak * 1.0e-3,
            "UPML did not absorb the pulse: tail {tail:e} vs peak {peak:e}"
        );
        assert!(
            tail < pec_tail * 0.1,
            "UPML tail {tail:e} should be far below PEC tail {pec_tail:e}"
        );
    }
}

#[cfg(test)]
mod pml_setup_tests {
    use super::*;

    fn mesh_xml() -> String {
        let lines = |count: usize| {
            (0..count)
                .map(|index| index.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        format!(
            r#"<ContinuousStructure CoordSystem="0">
                <RectilinearGrid DeltaUnit="0.001">
                    <XLines>{}</XLines>
                    <YLines>{}</YLines>
                    <ZLines>{}</ZLines>
                </RectilinearGrid>
                <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
                <Properties></Properties>
            </ContinuousStructure>"#,
            lines(31),
            lines(21),
            lines(21)
        )
    }

    #[test]
    fn setup_fdtd_accepts_pml_and_defaults_to_eight_cells() {
        let mut sim = OpenEMS::new();
        sim.set_csx_xml(mesh_xml());
        for face in 0..6 {
            sim.set_bc_type(face, BC_PML);
        }
        sim.set_number_of_time_steps(5);
        assert!(sim.setup_fdtd(), "PML setup must succeed");
        for face in 0..6 {
            assert_eq!(sim.get_pml_size(face), 8);
        }
        assert!(sim.run_fdtd());
    }

    #[test]
    fn setup_fdtd_rejects_a_pml_larger_than_the_mesh() {
        let mut sim = OpenEMS::new();
        sim.set_csx_xml(mesh_xml());
        for face in 0..6 {
            sim.set_bc_type(face, BC_PML);
            sim.set_pml_size(face, 40);
        }
        assert!(!sim.setup_fdtd());
    }
}

#[cfg(test)]
mod steady_state_wiring_tests {
    use super::*;

    fn mesh_xml() -> String {
        let lines = |count: usize| {
            (0..count)
                .map(|index| index.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        format!(
            r#"<ContinuousStructure CoordSystem="0">
                <RectilinearGrid DeltaUnit="0.001">
                    <XLines>{}</XLines>
                    <YLines>{}</YLines>
                    <ZLines>{}</ZLines>
                </RectilinearGrid>
                <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
                <Properties>
                    <Excitation Number="0" Enabled="1" Frequency="5e9" Delay="0" Type="0" Excite="1,0,0" PropDir="0,0,0">
                        <Primitives>
                            <Box Priority="0"><P1 X="10" Y="10" Z="10"/><P2 X="11" Y="11" Z="11"/></Box>
                        </Primitives>
                    </Excitation>
                </Properties>
            </ContinuousStructure>"#,
            lines(21),
            lines(21),
            lines(21)
        )
    }

    #[test]
    fn periodic_excitation_registers_steady_state_detection() {
        let mut sim = OpenEMS::new();
        sim.set_csx_xml(mesh_xml());
        sim.set_sinus_excite(5.0e9);
        for face in 0..6 {
            sim.set_bc_type(face, BC_PML);
            sim.set_pml_size(face, 4);
        }
        sim.set_number_of_time_steps(20);
        assert!(sim.setup_fdtd());
        assert!(sim.run_fdtd());
        assert!(
            sim.fdtd_eng
                .as_ref()
                .unwrap()
                .get_steady_state_diff()
                .is_some(),
            "the periodic excitation must register the steady-state metric"
        );
    }

    #[test]
    fn steady_state_end_criteria_stops_a_periodic_run_early() {
        let mut sim = OpenEMS::new();
        sim.set_csx_xml(mesh_xml());
        sim.set_sinus_excite(5.0e9);
        for face in 0..6 {
            sim.set_bc_type(face, BC_PML);
            sim.set_pml_size(face, 4);
        }
        sim.set_number_of_time_steps(4_000);
        // A loose criterion so the test does not depend on the exact decay.
        sim.set_end_criteria(0.5);
        assert!(sim.setup_fdtd());
        assert!(sim.run_fdtd());
        let steps = sim.fdtd_eng.as_ref().unwrap().get_number_of_timesteps();
        assert!(
            steps < 4_000,
            "steady-state criterion did not stop the run ({steps} steps)"
        );
    }

    #[test]
    fn a_fast_default_run_reaches_the_timestep_limit() {
        // Native evaluates the energy criterion only after four seconds of
        // wall clock, so a small fast run must reach NrTS. The old hard-coded
        // 100-step cadence stopped such runs early (RM-146).
        let mut sim = OpenEMS::new();
        sim.set_csx_xml(mesh_xml());
        sim.set_gauss_excite(6.0e9, 6.0e9);
        for face in 0..6 {
            sim.set_bc_type(face, BC_PML);
            sim.set_pml_size(face, 4);
        }
        sim.set_number_of_time_steps(30);
        sim.set_end_criteria(1.0e-12);
        assert!(sim.setup_fdtd());
        assert!(sim.run_fdtd());
        assert_eq!(
            sim.fdtd_eng.as_ref().unwrap().get_number_of_timesteps(),
            30,
            "a sub-four-second run must not be stopped by the default cadence"
        );
    }

    #[test]
    fn the_end_criteria_cadence_matches_native() {
        // exact mode: every excitation Nyquist period
        let mut last = 0.0;
        assert!(end_criteria_due(true, 21, 21, 0.0, &mut last));
        assert!(end_criteria_due(true, 42, 21, 0.0, &mut last));
        assert!(!end_criteria_due(true, 43, 21, 0.0, &mut last));
        // Dirac/Step excitations have a Nyquist of one, i.e. every step
        assert!(end_criteria_due(true, 7, 1, 0.0, &mut last));
        // default mode: the first check is four seconds into the run, then
        // every four seconds of wall clock
        let mut last = 0.0;
        assert!(!end_criteria_due(false, 100, 21, 3.9, &mut last));
        assert!(end_criteria_due(false, 100, 21, 4.1, &mut last));
        assert!(!end_criteria_due(false, 200, 21, 7.0, &mut last));
        assert!(end_criteria_due(false, 300, 21, 8.5, &mut last));
        // the constant, not the loop, owns the interval
        assert_eq!(WALL_CLOCK_CHECK_INTERVAL, 4.0);
    }

    #[test]
    fn a_negative_end_criteria_disables_the_criterion() {
        let mut sim = OpenEMS::new();
        sim.set_end_criteria(-1.0);
        assert_eq!(sim.end_crit, -1.0, "native keeps negative criteria");
    }

    #[test]
    fn aperiodic_excitation_uses_the_energy_end_criteria() {
        let mut sim = OpenEMS::new();
        sim.set_csx_xml(mesh_xml());
        sim.set_gauss_excite(6.0e9, 6.0e9);
        for face in 0..6 {
            sim.set_bc_type(face, BC_PML);
            sim.set_pml_size(face, 4);
        }
        sim.set_number_of_time_steps(400);
        assert!(sim.setup_fdtd());
        assert!(sim.run_fdtd());
        // No steady-state extension for a pulsed signal.
        assert!(sim
            .fdtd_eng
            .as_ref()
            .unwrap()
            .get_steady_state_diff()
            .is_none());
    }
}

#[cfg(test)]
mod lumped_rlc_wiring_tests {
    use super::*;

    fn mesh_with_element() -> String {
        let lines = |count: usize| {
            (0..count)
                .map(|index| index.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        format!(
            r#"<ContinuousStructure CoordSystem="0">
                <RectilinearGrid DeltaUnit="0.001">
                    <XLines>{}</XLines>
                    <YLines>{}</YLines>
                    <ZLines>{}</ZLines>
                </RectilinearGrid>
                <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
                <Properties>
                    <Excitation Name="src" Number="0" Enabled="1" Frequency="5e9" Delay="0" Type="0" Excite="0,1,0">
                        <Primitives><Box Priority="0"><P1 X="10" Y="10" Z="10"/><P2 X="11" Y="11" Z="11"/></Box></Primitives>
                    </Excitation>
                    <LumpedElement Name="load" Direction="0" LEtype="1" R="50" L="1e-9" C="1e-12" Caps="0">
                        <Primitives><Box Priority="1"><P1 X="5" Y="5" Z="5"/><P2 X="10" Y="6" Z="6"/></Box></Primitives>
                    </LumpedElement>
                </Properties>
            </ContinuousStructure>"#,
            lines(21),
            lines(21),
            lines(21)
        )
    }

    #[test]
    fn lumped_element_is_registered_and_the_run_stays_finite() {
        let mut sim = OpenEMS::new();
        sim.set_csx_xml(mesh_with_element());
        sim.set_gauss_excite(5.0e9, 5.0e9);
        for face in 0..6 {
            sim.set_bc_type(face, BC_MUR);
        }
        sim.set_number_of_time_steps(30);
        assert!(sim.setup_fdtd(), "setup with a lumped element must succeed");
        let names = sim
            .fdtd_eng
            .as_ref()
            .expect("engine built")
            .extension_names();
        assert!(
            names.contains(&"lumped_rlc"),
            "lumped RLC extension is missing: {names:?}"
        );
        assert!(sim.run_fdtd());
        let energy = sim.fdtd_eng.as_ref().unwrap().fast_energy();
        assert!(energy.is_finite(), "energy is not finite: {energy}");
    }
}

#[cfg(test)]
mod tfsf_wiring_tests {
    use super::*;
    use crate::constants::C0;

    fn plane_wave_mesh(frequency: f64) -> String {
        let lines = |count: usize| {
            (0..count)
                .map(|index| index.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        format!(
            r#"<ContinuousStructure CoordSystem="0">
                <RectilinearGrid DeltaUnit="0.001">
                    <XLines>{}</XLines>
                    <YLines>{}</YLines>
                    <ZLines>{}</ZLines>
                </RectilinearGrid>
                <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
                <Properties>
                    <Excitation Name="plane" Number="0" Enabled="1" Type="10" Frequency="{frequency}" Excite="0,1,0" PropDir="1,0,0">
                        <Primitives><Box Priority="0"><P1 X="5" Y="5" Z="5"/><P2 X="35" Y="15" Z="15"/></Box></Primitives>
                    </Excitation>
                </Properties>
            </ContinuousStructure>"#,
            lines(41),
            lines(21),
            lines(21)
        )
    }

    #[test]
    fn plane_wave_tfsf_propagates_at_the_numeric_phase_velocity() {
        let frequency = 20.0e9;
        let mut sim = OpenEMS::new();
        sim.set_csx_xml(plane_wave_mesh(frequency));
        sim.set_sinus_excite(frequency);
        for face in 0..6 {
            sim.set_bc_type(face, BC_MUR);
        }
        sim.set_number_of_time_steps(2_000);
        assert!(sim.setup_fdtd(), "plane-wave setup must succeed");
        let names = sim.fdtd_eng.as_ref().unwrap().extension_names();
        assert!(
            names.contains(&"TFSF"),
            "TFSF extension is missing: {names:?}"
        );
        assert!(sim.run_fdtd());

        let engine = sim.fdtd_eng.as_ref().unwrap();
        // Spatial snapshot along x at the cross-section centre.
        let mut samples = Vec::new();
        for x in 6..=34usize {
            samples.push((x as f64 * 1.0e-3, *engine.volt_ptr.get(1, x, 10, 10) as f64));
        }

        // Fit E(x) = a sin(k x) + b cos(k x) by scanning k around the
        // analytic free-space value.
        let k0 = std::f64::consts::TAU * frequency / C0;
        let mut best = (f64::INFINITY, k0);
        let mut k = 0.70 * k0;
        while k <= 1.30 * k0 {
            let mut ss = 0.0;
            let mut cc = 0.0;
            let mut sc = 0.0;
            let mut ys = 0.0;
            let mut yc = 0.0;
            for &(x, value) in &samples {
                let (s, c) = ((k * x).sin(), (k * x).cos());
                ss += s * s;
                cc += c * c;
                sc += s * c;
                ys += value * s;
                yc += value * c;
            }
            let determinant = ss * cc - sc * sc;
            if determinant.abs() > 1.0e-12 {
                let a = (ys * cc - yc * sc) / determinant;
                let b = (yc * ss - ys * sc) / determinant;
                let mut residual = 0.0;
                for &(x, value) in &samples {
                    let model = a * (k * x).sin() + b * (k * x).cos();
                    residual += (value - model) * (value - model);
                }
                if residual < best.0 {
                    best = (residual, k);
                }
            }
            k += 0.002 * k0;
        }
        let measured_velocity = std::f64::consts::TAU * frequency / best.1;
        // At 20 GHz and 1 mm cells the numerical phase velocity is ~0.994 c.
        let relative = (measured_velocity - C0).abs() / C0;
        assert!(
            relative < 0.03,
            "phase velocity {measured_velocity:e} m/s (relative {relative:e})"
        );

        // Amplitude inside the total-field box and in the scattered region.
        let incident = samples
            .iter()
            .map(|(_, value)| value.abs())
            .fold(0.0, f64::max);
        assert!(incident > 0.0, "no incident field inside the TFSF box");
        let scattered = (2..5)
            .map(|x| (*engine.volt_ptr.get(1, x, 10, 10) as f64).abs())
            .fold(0.0, f64::max);
        assert!(
            scattered < 0.1 * incident,
            "scattered field {scattered:e} is too large vs incident {incident:e}"
        );
    }
}

#[cfg(test)]
mod option_wiring_tests {
    use super::*;
    use crate::processing::FieldProbe;

    fn mesh() -> String {
        let lines = |count: usize| {
            (0..count)
                .map(|index| index.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        format!(
            r#"<ContinuousStructure CoordSystem="0">
                <RectilinearGrid DeltaUnit="0.001">
                    <XLines>{}</XLines>
                    <YLines>{}</YLines>
                    <ZLines>{}</ZLines>
                </RectilinearGrid>
                <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
                <Properties>
                    <Excitation Number="0" Enabled="1" Frequency="2e9" Delay="0" Type="0" Excite="0,1,0">
                        <Primitives><Box Priority="0"><P1 X="10" Y="10" Z="10"/><P2 X="11" Y="11" Z="11"/></Box></Primitives>
                    </Excitation>
                </Properties>
            </ContinuousStructure>"#,
            lines(21),
            lines(21),
            lines(21)
        )
    }

    fn configure(threads: i32) -> OpenEMS {
        let mut sim = OpenEMS::new();
        sim.set_csx_xml(mesh());
        sim.set_gauss_excite(2.0e9, 1.0e9);
        for face in 0..6 {
            sim.set_bc_type(face, BC_MUR);
        }
        sim.set_number_of_threads(threads);
        sim.set_number_of_time_steps(20);
        assert!(sim.setup_fdtd());
        sim
    }

    #[test]
    fn threaded_update_matches_the_serial_kernel_bit_for_bit() {
        let mut serial = configure(1);
        let mut threaded = configure(4);
        assert_eq!(threaded.fdtd_eng.as_ref().unwrap().num_threads, 4);
        assert!(serial.run_fdtd());
        assert!(threaded.run_fdtd());
        let serial_engine = serial.fdtd_eng.as_ref().unwrap();
        let threaded_engine = threaded.fdtd_eng.as_ref().unwrap();
        for n in 0..3 {
            for i in 0..serial_engine.num_lines[0] {
                for j in 0..serial_engine.num_lines[1] {
                    for k in 0..serial_engine.num_lines[2] {
                        assert_eq!(
                            serial_engine.volt_ptr.get(n, i, j, k).to_bits(),
                            threaded_engine.volt_ptr.get(n, i, j, k).to_bits(),
                            "volt mismatch at ({n},{i},{j},{k})"
                        );
                        assert_eq!(
                            serial_engine.curr_ptr.get(n, i, j, k).to_bits(),
                            threaded_engine.curr_ptr.get(n, i, j, k).to_bits(),
                            "curr mismatch at ({n},{i},{j},{k})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn over_sampling_sets_the_default_probe_interval() {
        let mut sim = configure(1);
        sim.set_over_sampling(2);
        sim.add_processing(Box::new(FieldProbe::new("probe", [0.010, 0.010, 0.010])));
        let dt = sim.fdtd_eng.as_ref().unwrap().op.get_timestep();
        let f_max = sim.excitation.as_ref().unwrap().f_max;
        let nyquist = (1.0 / f_max / 2.0 / dt).floor() as u32;
        assert!(sim.run_fdtd());
        let intervals = sim.processing_array.as_ref().unwrap().process_intervals();
        assert_eq!(intervals.len(), 1);
        assert_eq!(intervals[0], nyquist / 2);
        assert!(intervals[0] > 0);
    }

    #[test]
    fn max_time_stops_the_run_before_the_requested_timesteps() {
        let mut sim = configure(1);
        let dt = sim.fdtd_eng.as_ref().unwrap().op.get_timestep();
        sim.set_number_of_time_steps(1000);
        sim.set_max_time(10.0 * dt);
        assert!(sim.run_fdtd());
        assert_eq!(sim.fdtd_eng.as_ref().unwrap().get_number_of_timesteps(), 10);
    }

    #[test]
    fn abort_before_the_run_executes_no_timestep() {
        let mut sim = configure(1);
        sim.set_abort(true);
        assert!(sim.run_fdtd());
        assert_eq!(sim.fdtd_eng.as_ref().unwrap().get_number_of_timesteps(), 0);
    }

    #[test]
    fn over_sampling_must_be_positive() {
        let mut sim = OpenEMS::new();
        sim.set_over_sampling(0);
        assert_eq!(sim.over_sampling, 1);
    }
}
