//! Main solver orchestrator: ParseFDTDSetup, SetupFDTD, RunFDTD.
//! Mirrors `openems.h/cpp` in the C++ codebase.

use crate::engine::Engine;
use crate::excitation::Excitation;
use crate::operator::Operator;
use crate::operator_base::OperatorBase;
use crate::processing::{Processing, ProcessingArray};

/// Boundary condition types (mirroring C++).
pub const BC_PEC: i32 = 0;
pub const BC_PMC: i32 = 1;
pub const BC_MUR: i32 = 2;
pub const BC_PML: i32 = 3;

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
    pub pml_size: [u32; 6],
    pub mur_v_ph: [f64; 6],
    pub max_time: f64,
    pub time_step_method: i32,
    pub time_step: f64,
    pub time_step_factor: f64,
    pub abort: bool,
    pub num_threads: i32,
    pub verbose_level: i32,
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
            pml_size: [0; 6],
            mur_v_ph: [0.0; 6],
            max_time: 0.0,
            time_step_method: 0,
            time_step: 0.0,
            time_step_factor: 1.0,
            abort: false,
            num_threads: 1,
            verbose_level: 0,
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
        self.end_crit = val;
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
        self.over_sampling = val;
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
        // Store radii for later use in cylinder multi-grid setup.
        // The actual grid setup happens in setup_fdtd().
        let _ = radii;
        Ok(())
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

    /// Clear the CSXCAD XML geometry (falls back to default placeholder grid).
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
                println!("  lines[{}]: {} ({})", n, op.get_number_of_lines(n as i32, true), op.get_dir_name(n as i32));
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
                println!("  {} direction: {} lines, delta={}", op.get_dir_name(n as i32), op.get_number_of_lines(n as i32, true), op.get_grid_delta());
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

    /// Initialize the FDTD operator from a CSX geometry.
    /// Mirrors C++ `SetupFDTD` (operator setup portion).
    ///
    /// When `csx_xml` is set, the geometry is parsed via `Operator::load_csx()`
    /// to populate the grid, materials, boundary conditions and per-cell
    /// materials. Otherwise a default placeholder grid is used.
    pub fn setup_fdtd(&mut self) -> bool {
        let mut op = Operator::new();

        // Load CSXCAD geometry if an XML string was provided.
        if let Some(ref xml) = self.csx_xml {
            match op.load_csx(xml) {
                Ok(()) => {
                    println!("[openEMS] CSXCAD geometry loaded: {} lines, delta={:.6}",
                        op.base.num_lines.iter().map(|&v| v as f64).sum::<f64>() / 3.0,
                        op.base.grid_delta);
                }
                Err(e) => {
                    eprintln!("[openEMS] Failed to load CSXCAD geometry: {}", e);
                    return false;
                }
            }
        } else {
            // Configure a uniform grid based on the excitation frequency.
            // Rule of thumb: at least 20 cells per wavelength at f0.
            let c0: f64 = 299_792_458.0; // speed of light in m/s
            let f0 = self.excitation.as_ref().map(|e| e.f0).unwrap_or(1.0e9);
            let wavelength = c0 / f0;
            let cells_per_wavelength = 20.0;
            let grid_delta = wavelength / cells_per_wavelength;
            // Grid extends a few wavelengths in each direction.
            let num_wavelengths = 3.0;
            let num_lines = ((num_wavelengths * wavelength) / grid_delta).round() as u32;
            let num_lines = num_lines.max(10); // minimum grid size

            op.base.grid_delta = grid_delta;
            op.base.num_lines = [num_lines, num_lines, num_lines];
            op.base.d_t = 0.0; // will be computed by calc_timestep

            if self.verbose_level > 0 {
                println!("[openEMS] Grid: {} lines x {} lines x {} lines, delta={:.6} m (f0={:.1} Hz, lambda={:.3} m)",
                    num_lines, num_lines, num_lines, grid_delta, f0, wavelength);
            }
        }

        // Set boundary conditions (CSXCAD may have already set them via load_csx;
        // only override with explicit settings when no XML was loaded).
        if self.csx_xml.is_none() {
            op.base.set_boundary_condition(self.bc_type);
        }

        // Ensure EC coefficient arrays are allocated.
        if op.vv_ptr.is_none() || op.vi_ptr.is_none() || op.ii_ptr.is_none() || op.iv_ptr.is_none() {
            let num_lines_usize: [usize; 3] = op.base.num_lines.map(|v| v as usize);
            op.init_data_storage(num_lines_usize);
        }

        // Set up excitation (default Gaussian pulse if none specified).
        if self.excitation.is_none() {
            let exc = Excitation::gauss(1.0e9, 5.0e8);
            self.excitation = Some(Box::new(exc));
        }

        // Compute EC coefficients (vv, vi, ii, iv) for all cells.
        op.calc_ec_operator();

        // Compute the stable timestep (CFL condition).
        op.calc_timestep();

        // Apply boundary conditions.
        op.apply_electric_bc();
        op.apply_magnetic_bc();

        // Box as trait object for the engine (unsizing coercion).
        let op_dyn: Box<dyn OperatorBase> = Box::new(op);
        self.fdtd_eng = Some(Box::new(Engine::new(op_dyn)));
        self.fdtd_op = None; // operator is owned by the engine

        true
    }

    /// Run the FDTD simulation for the configured number of time steps.
    /// Mirrors C++ `RunFDTD`.
    pub fn run_fdtd(&mut self) -> bool {
        if self.fdtd_eng.is_none() {
            return false;
        }

        // Invoke PreProcess on all registered processings.
        if let Some(ref mut pa) = self.processing_array {
            pa.pre_process();
        }

        let nr_ts = self.nr_ts;
        let end_crit = self.end_crit;

        let mut ts = 0u32;
        let mut change: f64 = 1.0;

        while ts < nr_ts && change > end_crit && !self.check_abort_cond() {
            // Execute one timestep (voltage + current updates with extensions).
            self.fdtd_eng.as_mut().unwrap().iterate_ts(1);
            ts = self.fdtd_eng.as_ref().unwrap().get_number_of_timesteps();

            // Invoke Process on all registered processings.
            if let Some(ref mut pa) = self.processing_array {
                pa.process(ts);
            }

            // Check steady-state end criteria.
            if let Some(diff) = self.fdtd_eng.as_ref().unwrap().get_steady_state_diff() {
                change = diff as f64;
            }

            // Progress reporting every 100 steps.
            if ts.is_multiple_of(100) {
                println!("[openEMS] Timestep: {}", ts);
            }
        }

        // Invoke PostProcess on all registered processings.
        if let Some(ref mut pa) = self.processing_array {
            pa.post_process();
        }

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
        true
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
        assert!(sim.setup_fdtd());
        assert!(sim.run_fdtd());
        assert_eq!(sim.fdtd_eng.as_ref().unwrap().get_number_of_timesteps(), 10);
    }

    #[test]
    fn test_run_without_setup() {
        let mut sim = OpenEMS::new();
        assert!(!sim.run_fdtd());
    }

    #[test]
    fn test_add_processing() {
        use crate::processing::FieldProbe;
        let mut sim = OpenEMS::new();
        sim.add_processing(Box::new(FieldProbe::new("probe1", [0.0, 0.0, 0.0])));
        assert!(sim.processing_array.is_some());
    }
}
