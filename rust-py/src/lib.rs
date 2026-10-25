//! Python bindings for the openEMS FDTD solver.
//! Mirrors the Cython `openEMS.pyx` interface using PyO3.

use pyo3::exceptions::{PyAssertionError, PyValueError};
use pyo3::prelude::{PyAny, PyModule, PyResult, Python};
use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods, PyList, PyListMethods, PyModuleMethods};
use pyo3::{pyclass, pymethods, pymodule, Bound, FromPyObject, Py};

use openems_core::openems::{BC_PEC, BC_PMC, BC_MUR, BC_PML, OpenEMS};

/// Helper: extract a Rust type from a Python object using FromPyObject.
fn extract<'a, 'py, T: FromPyObject<'a, 'py>>(val: &'a Bound<'py, PyAny>) -> PyResult<T> {
    val.extract::<T>().map_err(Into::into)
}

fn extract_str<'py>(val: &Bound<'py, PyAny>) -> PyResult<String> {
    extract(val)
}

fn extract_f64<'py>(val: &Bound<'py, PyAny>) -> PyResult<f64> {
    extract(val)
}

fn extract_u32<'py>(val: &Bound<'py, PyAny>) -> PyResult<u32> {
    if let Ok(value) = val.extract::<u32>() {
        return Ok(value);
    }
    let value = val.extract::<f64>()?;
    if !value.is_finite() || value < 0.0 {
        return Err(PyValueError::new_err(format!(
            "expected a non-negative integer, got {value}"
        )));
    }
    Ok(value as u32)
}

fn extract_i32<'py>(val: &Bound<'py, PyAny>) -> PyResult<i32> {
    if let Ok(value) = val.extract::<i32>() {
        return Ok(value);
    }
    let value = val.extract::<f64>()?;
    if !value.is_finite() {
        return Err(PyValueError::new_err("expected an integer"));
    }
    Ok(value as i32)
}

fn extract_bool<'py>(val: &Bound<'py, PyAny>) -> PyResult<bool> {
    if let Ok(value) = val.extract::<bool>() {
        return Ok(value);
    }
    Ok(val.extract::<i64>()? != 0)
}

fn extract_vec_f64<'py>(val: &Bound<'py, PyAny>) -> PyResult<Vec<f64>> {
    extract(val)
}

/// Python wrapper around the Rust `OpenEMS` solver.
///
/// Mirrors the Cython `cdef class openEMS` from `openEMS.pyx`.
#[pyclass(unsendable, subclass)]
pub struct PyOpenEMS {
    inner: OpenEMS,
    csx: Option<Py<PyAny>>,
}

#[pymethods]
impl PyOpenEMS {
    /// Create a new openEMS solver instance.
    #[new]
    #[pyo3(signature = (**kwargs))]
    fn py_new(_py: Python, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut sim = OpenEMS::new();

        if let Some(kw) = kwargs {
            for (key, val) in kw.iter() {
                let key_str = extract_str(&key)?;
                match key_str.as_str() {
                    "NrTS" => {
                        let v = extract_u32(&val)?;
                        sim.set_number_of_time_steps(v);
                    }
                    "EndCriteria" => {
                        let v = extract_f64(&val)?;
                        sim.set_end_criteria(v);
                    }
                    "MaxTime" => {
                        let v = extract_f64(&val)?;
                        sim.set_max_time(v);
                    }
                    "OverSampling" => {
                        let v = extract_u32(&val)?;
                        sim.set_over_sampling(v);
                    }
                    "CoordSystem" => {
                        let v = extract_i32(&val)?;
                        sim.set_coord_system(v).map_err(|e| {
                            PyValueError::new_err(format!("SetCoordSystem: {}", e))
                        })?;
                    }
                    "TimeStep" => {
                        let v = extract_f64(&val)?;
                        sim.set_time_step(v);
                    }
                    "TimeStepFactor" => {
                        let v = extract_f64(&val)?;
                        sim.set_time_step_factor(v);
                    }
                    "TimeStepMethod" => {
                        let v = extract_i32(&val)?;
                        sim.set_time_step_method(v);
                    }
                    "CellConstantMaterial" => {
                        let v = extract_bool(&val)?;
                        sim.set_cell_constant_material(v);
                    }
                    "MultiGrid" => {
                        let v = extract_vec_f64(&val)?;
                        sim.set_multi_grid(&v).map_err(|e| {
                            PyValueError::new_err(format!("SetMultiGrid: {}", e))
                        })?;
                    }
                    _ => {
                        return Err(PyAssertionError::new_err(format!(
                            "Unknown keyword argument: {}",
                            key_str
                        )));
                    }
                }
            }
        }

        Ok(PyOpenEMS {
            inner: sim,
            csx: None,
        })
    }

    /// Show the openEMS welcome screen.
    #[staticmethod]
    fn welcome_screen() {
        println!("================================================");
        println!("  openEMS - Finite-Difference Time-Domain Solver");
        println!("  Rust port (openems-py)");
        println!("================================================");
    }

    /// Reset the solver state.
    fn reset(&mut self) {
        self.csx = None;
        self.inner = OpenEMS::new();
    }

    /// Set the number of time steps.
    fn set_number_of_time_steps(&mut self, val: u32) {
        self.inner.set_number_of_time_steps(val);
    }

    /// Set the end criteria value.
    fn set_end_criteria(&mut self, val: f64) {
        self.inner.set_end_criteria(val);
    }

    /// Set the oversampling factor for time-domain dumps/probes.
    fn set_over_sampling(&mut self, val: u32) {
        self.inner.set_over_sampling(val);
    }

    /// Set cell constant material mode.
    fn set_cell_constant_material(&mut self, val: bool) {
        self.inner.set_cell_constant_material(val);
    }

    /// Enable or disable field dumps.
    fn set_enable_dumps(&mut self, val: bool) {
        self.inner.set_enable_dumps(val);
    }

    /// Set MUR phase velocity for a given boundary side (0..6).
    fn set_mur_phase_vel(&mut self, idx: usize, val: f64) -> PyResult<()> {
        if idx >= 6 {
            return Err(PyValueError::new_err("Boundary index must be 0..6"));
        }
        self.inner.set_mur_phase_vel(idx, val);
        Ok(())
    }

    /// Check abort conditions.
    fn check_abort_cond(&self) -> bool {
        self.inner.check_abort_cond()
    }

    /// Set the coordinate system (0=Cartesian, 1=Cylindrical).
    fn set_coord_system(&mut self, val: i32) -> PyResult<()> {
        self.inner.set_coord_system(val).map_err(|e| {
            PyValueError::new_err(format!("SetCoordSystem: {}", e))
        })
    }

    /// Set cylindrical multi-grid radii.
    fn set_multi_grid(&mut self, radii: Vec<f64>) -> PyResult<()> {
        self.inner.set_multi_grid(&radii).map_err(|e| {
            PyValueError::new_err(format!("SetMultiGrid: {}", e))
        })
    }

    /// Enable cylindrical coordinates.
    fn set_cylinder_coords(&mut self) {
        self.inner.set_cylinder_coords(true);
    }

    /// Set the timestep calculation method (1=CFL, 3=Rennings).
    fn set_time_step_method(&mut self, val: i32) {
        self.inner.set_time_step_method(val);
    }

    /// Force a fixed timestep.
    fn set_time_step(&mut self, val: f64) {
        self.inner.set_time_step(val);
    }

    /// Set the timestep factor (multiplier on CFL timestep).
    fn set_time_step_factor(&mut self, val: f64) {
        self.inner.set_time_step_factor(val);
    }

    /// Set the maximum simulated time in seconds.
    fn set_max_time(&mut self, val: f64) {
        self.inner.set_max_time(val);
    }

    /// Set a Gaussian pulse excitation.
    fn set_gauss_excite(&mut self, f0: f64, fc: f64) {
        self.inner.set_gauss_excite(f0, fc);
    }

    /// Set a sinusoidal excitation.
    fn set_sinus_excite(&mut self, f0: f64) {
        self.inner.set_sinus_excite(f0);
    }

    /// Set a Dirac pulse excitation.
    fn set_dirac_excite(&mut self, f_max: f64) {
        self.inner.set_dirac_excite(f_max);
    }

    /// Set a step function excitation.
    fn set_step_excite(&mut self, f_max: f64) {
        self.inner.set_step_excite(f_max);
    }

    /// Set a custom function excitation (parsed by fparser).
    fn set_custom_excite(&mut self, _str: &str, f0: f64, fmax: f64) {
        self.inner.set_custom_excite(_str, f0, fmax);
    }

    /// Set boundary conditions for all six FDTD directions.
    fn set_boundary_cond(&mut self, bc: Bound<'_, PyList>) -> PyResult<()> {
        let len: usize = bc.len();
        if len != 6 {
            return Err(PyValueError::new_err(
                "Invalid boundary condition size! Expected 6 elements.",
            ));
        }

        for n in 0..6 {
            let item = bc.get_item(n)?;
            // Try integer first
            if let Ok(val) = extract_i32(&item) {
                self.inner.set_bc_type(n, val);
                continue;
            }
            // Try string
            let s = extract_str(&item)?;
            match s.as_str() {
                "PEC" => self.inner.set_bc_type(n, BC_PEC),
                "PMC" => self.inner.set_bc_type(n, BC_PMC),
                "MUR" => self.inner.set_bc_type(n, BC_MUR),
                s if s.starts_with("PML_") => {
                    let size: u32 = s.strip_prefix("PML_").unwrap().parse().map_err(|_| {
                        PyValueError::new_err(format!("Invalid PML size in: {}", s))
                    })?;
                    self.inner.set_pml_size(n, size);
                    self.inner.set_bc_type(n, BC_PML);
                }
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "Unknown boundary condition: {}",
                        s
                    )));
                }
            }
        }

        Ok(())
    }

    /// Get the boundary condition type for a given side (0..6).
    fn get_bc_type(&self, idx: usize) -> PyResult<i32> {
        if idx >= 6 {
            return Err(PyValueError::new_err("Boundary index must be 0..6"));
        }
        Ok(self.inner.get_bc_type(idx))
    }

    /// Get the PML size for a given side (0..6).
    fn get_pml_size(&self, idx: usize) -> PyResult<u32> {
        if idx >= 6 {
            return Err(PyValueError::new_err("Boundary index must be 0..6"));
        }
        Ok(self.inner.get_pml_size(idx))
    }

    /// Set the CSXCAD ContinuousStructure for CAD data handling.
    fn set_csx(&mut self, csx: Py<PyAny>) {
        self.csx = Some(csx);
    }

    /// Get the CSXCAD ContinuousStructure.
    fn get_csx(&self, py: Python) -> Option<Py<PyAny>> {
        self.csx.as_ref().map(|c| c.clone_ref(py))
    }

    /// Set the CSXCAD XML geometry string.
    /// When set, `run()` parses it via `Operator::load_csx()` to populate
    /// the grid, materials, boundary conditions and per-cell materials.
    fn set_csx_xml(&mut self, xml: &str) {
        self.inner.set_csx_xml(xml.to_string());
    }

    /// Run the openEMS FDTD simulation.
    ///
    /// Returns the native exit code: 0 on success, 1 when the FDTD setup fails.
    /// The Python layer owns the simulation path, the cleanup and the option
    /// mapping, exactly like the native `Run` wrapper.
    fn run(&mut self) -> PyResult<i32> {
        if !self.inner.setup_fdtd() {
            return Ok(1);
        }
        if !self.inner.run_fdtd() {
            return Ok(1);
        }
        Ok(0)
    }

    /// Skip the simulation, only run the setup (native `--no-simulation`).
    fn set_no_simulation(&mut self, val: bool) {
        self.inner.set_no_simulation(val);
    }

    /// Write the native statistics files at the end of a run.
    fn set_dump_statistics(&mut self, val: bool) {
        self.inner.set_dump_statistics(val);
    }

    /// Use the exact energy end criterion (native `--exact-endcriteria`).
    fn set_exact_end_criteria(&mut self, val: bool) {
        self.inner.set_exact_end_criteria(val);
    }

    /// Set the abort flag.
    fn set_abort(&mut self, val: bool) {
        self.inner.set_abort(val);
    }

    /// Set the number of threads for multithreaded engine.
    fn set_number_of_threads(&mut self, val: i32) {
        self.inner.set_number_of_threads(val);
    }

    /// Set the verbose level for console output.
    fn set_verbose_level(&mut self, level: i32) {
        self.inner.set_verbose_level(level);
    }

    /// Enable debug material output.
    fn debug_material(&mut self) {
        self.inner.debug_material();
    }

    /// Enable debug operator output.
    fn debug_operator(&mut self) {
        self.inner.debug_operator();
    }

    /// Enable debug box output.
    fn debug_box(&mut self) {
        self.inner.debug_box();
    }

    /// Enable debug PEC output.
    fn debug_pec(&mut self) {
        self.inner.debug_pec();
    }

    /// Enable debug CSX output.
    fn debug_csx(&mut self) {
        self.inner.debug_csx();
    }

    /// The native `<FDTD ...>` element for the current solver settings.
    fn fdtd_xml(&self) -> String {
        self.inner.write_fdtd_element()
    }

    /// Apply the options of a native `<FDTD>` element (attributes plus the
    /// `Excitation` and `BoundaryCond` children).
    fn apply_fdtd_xml(&mut self, xml: &str) {
        let options = openems_core::fdtd_options::parse_fdtd_options(xml);
        self.inner.apply_fdtd_options(&options);
    }

    /// Write the FDTD setup and CSX data to an XML file.
    fn write2xml(&self, file: &str) -> PyResult<bool> {
        let xml = self
            .inner
            .csx_xml
            .as_ref()
            .ok_or_else(|| {
                PyValueError::new_err("Write2XML: No CSXCAD XML geometry set")
            })?;
        match std::fs::write(file, xml) {
            Ok(()) => Ok(true),
            Err(e) => Err(PyValueError::new_err(format!(
                "Write2XML: Failed to write '{}': {}",
                file, e
            ))),
        }
    }

    /// Read the FDTD setup and CSX data from an XML file.
    fn read_from_xml(&mut self, file: &str) -> PyResult<bool> {
        let xml = match std::fs::read_to_string(file) {
            Ok(v) => v,
            Err(e) => {
                return Err(PyValueError::new_err(format!(
                    "ReadFromXML: Failed to read '{}': {}",
                    file, e
                )))
            }
        };
        self.inner.set_csx_xml(xml);
        Ok(true)
    }

}

/// Python wrapper around the native `SAR_Calculation`.
#[pyclass(unsendable, subclass, name = "PySarCalculation")]
pub struct PySarCalculation {
    inner: openems_core::sar_hdf5::SarRunner,
}

#[pymethods]
impl PySarCalculation {
    #[new]
    #[pyo3(signature = (**kwargs))]
    fn py_new(kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        // the Python subclass applies the native keyword arguments in __init__
        let _ = kwargs;
        Self {
            inner: openems_core::sar_hdf5::SarRunner::new(),
        }
    }

    fn set_debug_level(&mut self, level: i32) {
        self.inner.debug_level = level;
    }

    fn enable_progress(&mut self, enable: bool) {
        self.inner.progress = enable;
    }

    /// Averaging mass in kg (the Python layer converts from grams).
    fn set_averaging_mass(&mut self, mass: f64) {
        self.inner.averaging_mass = mass;
    }

    #[pyo3(signature = (method, silent = true))]
    fn set_averaging_method(&mut self, method: &str, silent: bool) -> bool {
        let _ = silent;
        self.inner.set_averaging_method(method)
    }

    fn enable_cube_stats(&mut self) {
        self.inner.record_cube_stats = true;
    }

    fn enable_auto_range(&mut self, db: f64) {
        self.inner.auto_range = db;
    }

    /// Read the raw SAR data, calculate and write the result.
    ///
    /// `num_threads` is accepted for API compatibility; the calculation is
    /// deterministic and produces the same result for every worker count.
    fn calc_from_hdf5(
        &mut self,
        input: &str,
        output: &str,
        num_threads: u32,
    ) -> PyResult<bool> {
        let _ = num_threads;
        self.inner
            .calc_from_hdf5(input, output)
            .map_err(|error| PyValueError::new_err(format!("CalcFromHDF5: {error}")))
    }
}

/// Python module definition.
#[pymodule]
fn _openems(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOpenEMS>()?;
    m.add_class::<PySarCalculation>()?;
    m.add("BC_PEC", BC_PEC)?;
    m.add("BC_PMC", BC_PMC)?;
    m.add("BC_MUR", BC_MUR)?;
    m.add("BC_PML", BC_PML)?;
    Ok(())
}
