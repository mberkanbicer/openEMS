//! Main solver orchestrator: ParseFDTDSetup, SetupFDTD, RunFDTD.
//! Mirrors `openems.h/cpp` in the C++ codebase.

use crate::engine::Engine;
use crate::excitation::Excitation;
use crate::operator::Operator;
use crate::operator_base::OperatorBase;

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

    /// Set PML size for a given side.
    pub fn set_pml_size(&mut self, idx: usize, size: u32) {
        if idx < 6 {
            self.pml_size[idx] = size;
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

    /// Set the abort flag (requests graceful shutdown).
    pub fn set_abort(&mut self, val: bool) {
        self.abort = val;
    }

    /// Check abort conditions.
    pub fn check_abort_cond(&self) -> bool {
        self.abort
    }

    /// Initialize the FDTD operator from a CSX geometry.
    /// Mirrors C++ `SetupFDTD` (operator setup portion).
    pub fn setup_fdtd(&mut self) -> bool {
        let mut op = Operator::new();

        // Configure a uniform grid (placeholder values; CSXCAD integration will set these).
        op.base.grid_delta = 1.0e-3; // 1 mm
        op.base.num_lines = [10, 10, 10];
        op.base.d_t = 0.0; // will be computed by calc_timestep

        // Set boundary conditions.
        op.base.set_boundary_condition(self.bc_type);

        // Allocate EC coefficient arrays (convert u32 -> usize).
        let num_lines_usize: [usize; 3] = op.base.num_lines.map(|v| v as usize);
        op.init_data_storage(num_lines_usize);

        // Set up excitation (default Gaussian pulse).
        let exc = Excitation::gauss(1.0e9, 5.0e8);
        self.excitation = Some(Box::new(exc));

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

        let nr_ts = self.nr_ts;
        let end_crit = self.end_crit;

        let mut ts = 0u32;
        let mut change: f64 = 1.0;

        while ts < nr_ts && change > end_crit && !self.check_abort_cond() {
            // Execute one timestep (voltage + current updates with extensions).
            self.fdtd_eng.as_mut().unwrap().iterate_ts(1);
            ts = self.fdtd_eng.as_ref().unwrap().get_number_of_timesteps();

            // Check steady-state end criteria.
            if let Some(diff) = self.fdtd_eng.as_ref().unwrap().get_steady_state_diff() {
                change = diff as f64;
            }

            // Progress reporting every 100 steps.
            if ts.is_multiple_of(100) {
                println!("[openEMS] Timestep: {}", ts);
            }
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
}
