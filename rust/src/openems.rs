//! Main solver orchestrator: ParseFDTDSetup, SetupFDTD, RunFDTD.
//! Mirrors `openems.h/cpp` in the C++ codebase.

use crate::engine::Engine;
use crate::excitation::Excitation;
use crate::operator::Operator;

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
}

impl OpenEMS {
    /// Create a new openEMS solver instance.
    pub fn new() -> Self {
        Self {
            fdtd_op: None,
            fdtd_eng: None,
            excitation: None,
            nr_ts: 0,
            end_crit: 0.0,
            enable_dumps: false,
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

    /// Initialize the FDTD operator from a CSX geometry.
    pub fn setup_fdtd(&mut self) -> bool {
        // Placeholder: full implementation requires CSXCAD integration.
        // Uses a uniform Cartesian grid with background material.
        let mut op = Operator::new();

        // Configure a uniform grid (placeholder values; CSXCAD integration will set these).
        op.base.grid_delta = 1.0e-3; // 1 mm
        op.base.num_lines = [10, 10, 10];
        op.base.d_t = 0.0; // will be computed by calc_timestep

        // Allocate EC coefficient arrays (convert u32 -> usize).
        let num_lines_usize: [usize; 3] = op.base.num_lines.map(|v| v as usize);
        op.init_data_storage(num_lines_usize);

        // Compute EC coefficients (vv, vi, ii, iv) for all cells.
        op.calc_ec_operator();

        // Compute the stable timestep (CFL condition).
        op.calc_timestep();

        // Apply boundary conditions.
        op.apply_electric_bc();
        op.apply_magnetic_bc();

        // Box as trait object for the engine (unsizing coercion).
        let op_dyn: Box<dyn crate::operator_base::OperatorBase> = Box::new(op);
        self.fdtd_eng = Some(Box::new(Engine::new(op_dyn)));
        self.fdtd_op = None; // operator is owned by the engine
        true
    }

    /// Run the FDTD simulation for the configured number of time steps.
    pub fn run_fdtd(&mut self) -> bool {
        if self.fdtd_eng.is_none() {
            return false;
        }
        // Placeholder: full time-stepping loop
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
        assert_eq!(sim.end_crit, 0.0);
        assert!(!sim.enable_dumps);
    }

    #[test]
    fn test_set_number_of_time_steps() {
        let mut sim = OpenEMS::new();
        sim.set_number_of_time_steps(1000);
        assert_eq!(sim.nr_ts, 1000);
    }

    #[test]
    fn test_setup_and_run() {
        let mut sim = OpenEMS::new();
        assert!(sim.setup_fdtd());
        assert!(sim.run_fdtd());
    }
}
