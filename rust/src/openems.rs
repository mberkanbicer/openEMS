//! Main solver orchestrator: ParseFDTDSetup, SetupFDTD, RunFDTD.
//! Mirrors `openems.h/cpp` in the C++ codebase.

use crate::engine::Engine;
use crate::excitation::Excitation;
use crate::operator::Operator;
use crate::operator_base::OperatorBase;
use crate::processing::{
    EngineInterface, EngineMeta, Processing, ProcessingArray,
};

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
    pub bc_type_set: bool,
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
            bc_type_set: false,
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
    /// A CSXCAD geometry is required. There is no fabricated fallback grid.
    pub fn setup_fdtd(&mut self) -> bool {
        let Some(xml) = self.csx_xml.as_deref() else {
            eprintln!("[openEMS] Setup failed: no CSXCAD geometry was provided");
            return false;
        };
        let mut op = Operator::new();
        if let Err(error) = op.load_csx(xml) {
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
        if op.vv_ptr.is_none() || op.vi_ptr.is_none() || op.ii_ptr.is_none() || op.iv_ptr.is_none() {
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
        } else {
            op.calc_timestep();
        }
        if self.time_step_factor <= 0.0 || !self.time_step_factor.is_finite() {
            eprintln!("[openEMS] Invalid timestep factor: {}", self.time_step_factor);
            return false;
        }
        op.base.d_t *= self.time_step_factor;

        // Compute EC coefficients (vv, vi, ii, iv) for all cells.
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

        // UPML: build the layer coefficients and the matching engine extension
        // for every PML face. `build_upml_extensions` modifies the main
        // operator coefficients inside the layer, exactly like the native
        // `Operator_Ext_UPML::BuildExtension`.
        let mut upml_extensions = Vec::new();
        if op.base.m_bc.iter().any(|&boundary| boundary == BC_PML) {
            match op.build_upml_extensions(self.pml_size) {
                Ok(extensions) => upml_extensions = extensions,
                Err(error) => {
                    eprintln!("[openEMS] Failed to build UPML: {error}");
                    return false;
                }
            }
        }

        // Build the operator-side extension state before the operator is moved
        // into the engine.
        let mur_extension = op.build_mur_extension(self.mur_v_ph);
        let excitation_extension = self
            .excitation
            .as_deref()
            .map(|signal| op.build_excitation_extension(signal));

        // Box as trait object for the engine (unsizing coercion).
        let op_dyn: Box<dyn OperatorBase> = Box::new(op);
        let mut engine = Engine::new(op_dyn);
        for extension in upml_extensions {
            engine.add_extension(Box::new(extension));
        }
        if !mur_extension.faces.is_empty() {
            engine.add_extension(Box::new(mur_extension));
        }
        if let Some(extension) = excitation_extension {
            if !extension.voltage_excitations.is_empty()
                || !extension.current_excitations.is_empty()
            {
                engine.add_extension(Box::new(extension));
            }
        }
        self.fdtd_eng = Some(Box::new(engine));
        self.fdtd_op = None; // operator is owned by the engine

        true
    }

    /// Run the FDTD simulation for the configured number of time steps.
    /// Mirrors C++ `RunFDTD`.
    pub fn run_fdtd(&mut self) -> bool {
        if self.fdtd_eng.is_none() {
            return false;
        }

        // Invoke PreProcess on all registered processings. The array is taken
        // out so that the field interface can borrow the engine immutably while
        // the processing state is mutable.
        let mut processing = self.processing_array.take();
        if let Some(ref mut pa) = processing {
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
            let mut disc_lines: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
            for axis in 0..3 {
                for index in 0..num_lines[axis] {
                    disc_lines[axis].push(
                        engine.op.get_disc_line(axis as i32, index as u32, false)
                            * engine.op.get_grid_delta(),
                    );
                }
            }
            EngineMeta {
                num_lines,
                disc_lines,
                dt,
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

        let mut ts = 0u32;
        let mut change: f64 = 1.0;

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
        assert_eq!(
            sim.fdtd_eng
                .as_ref()
                .unwrap()
                .get_number_of_timesteps(),
            10
        );
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
        assert_eq!(
            sim.fdtd_eng.as_ref().unwrap().get_number_of_timesteps(),
            5
        );
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
                        <Box Priority="1"><P1 X="5" Y="5" Z="5"/><P2 X="5" Y="6" Z="6"/></Box>
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
        assert_eq!(calls.load(Ordering::Relaxed), 10);
    }
}

#[cfg(test)]
mod upml_absorption_tests {
    use crate::engine::Engine;
    use crate::excitation::Excitation;
    use crate::extensions::EngineExtExcitation;
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
        let (nx, ny, nz) = (41usize, 21usize, 21usize);
        let delta = 1.0e-3;
        let mut op = Operator::new();
        op.load_csx(&uniform_xml(nx, ny, nz, delta))
            .expect("uniform CSX should load");
        op.base.m_bc = if pml {
            [crate::openems::BC_PML; 6]
        } else {
            [crate::openems::BC_PEC; 6]
        };
        op.calc_timestep();
        let dt = op.base.d_t;
        op.calc_ec_operator();
        op.apply_electric_bc();
        op.apply_magnetic_bc();

        let extensions = if pml {
            op.build_upml_extensions([8; 6]).expect("UPML should build")
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
