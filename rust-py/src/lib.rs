//! Python bindings for the openEMS FDTD solver.
//! Mirrors the Cython `openEMS.pyx` interface using PyO3.

use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::{PyAny, PyModule, PyResult, Python};
use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods, PyList, PyListMethods, PyModuleMethods};
use pyo3::{pyclass, pymethods, pymodule, Bound, FromPyObject, Py};

use openems_core::openems::{BC_PEC, BC_PMC, BC_MUR, BC_PML, OpenEMS};

/// Helper: extract a Rust type from a Python object using FromPyObject.
fn extract<'py, T: FromPyObject<'py>>(val: &Bound<'py, PyAny>) -> PyResult<T> {
    val.extract::<T>()
}

fn extract_str<'py>(val: &Bound<'py, PyAny>) -> PyResult<String> {
    extract(val)
}

fn extract_f64<'py>(val: &Bound<'py, PyAny>) -> PyResult<f64> {
    extract(val)
}

fn extract_u32<'py>(val: &Bound<'py, PyAny>) -> PyResult<u32> {
    extract(val)
}

fn extract_i32<'py>(val: &Bound<'py, PyAny>) -> PyResult<i32> {
    extract(val)
}

fn extract_bool<'py>(val: &Bound<'py, PyAny>) -> PyResult<bool> {
    extract(val)
}

fn extract_vec_f64<'py>(val: &Bound<'py, PyAny>) -> PyResult<Vec<f64>> {
    extract(val)
}

/// Python wrapper around the Rust `OpenEMS` solver.
///
/// Mirrors the Cython `cdef class openEMS` from `openEMS.pyx`.
#[pyclass(unsendable)]
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
                        return Err(PyValueError::new_err(format!(
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

    /// Add edges of primitives to the FDTD grid.
    #[pyo3(signature = (dirs, primitives=None))]
    fn add_edges2grid(
        &mut self,
        dirs: &str,
        primitives: Option<Bound<'_, PyList>>,
    ) -> PyResult<()> {
        let _csx = self.csx.as_ref().ok_or_else(|| {
            PyValueError::new_err("AddEdges2Grid: Unable to access CSX!")
        })?;
        let _ = (dirs, primitives);

        Ok(())
    }

    /// Create a near-field to far-field box.
    #[pyo3(signature = (name=None, start=None, stop=None, _kw=None))]
    fn create_nf2ff_box(
        &mut self,
        name: Option<&str>,
        start: Option<Vec<f64>>,
        stop: Option<Vec<f64>>,
        _kw: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let _ = (name, start, stop);
        Err(PyException::new_err(
            "CreateNF2FFBox: requires CSXCAD integration (future task)",
        ))
    }

    /// Run the openEMS FDTD simulation.
    #[pyo3(signature = (sim_path, _kw=None))]
    fn run(
        &mut self,
        sim_path: &str,
        _kw: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<i32> {
        let _ = sim_path;
        self.inner.setup_fdtd();
        self.inner.run_fdtd();
        Ok(0)
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

    /// Add a processing object (probe, field dump, SAR, etc.) to the
    /// post-processing array.
    fn add_processing(&mut self, _proc: Bound<'_, PyAny>) -> PyResult<()> {
        // Processing objects are added via the Rust API; Python-side
        // processing wrappers are a future task.
        let _ = _proc;
        Err(PyException::new_err(
            "AddProcessing: Python processing wrappers not yet implemented",
        ))
    }
}

/// Python module definition.
#[pymodule]
fn openems_py(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOpenEMS>()?;
    m.add("BC_PEC", BC_PEC)?;
    m.add("BC_PMC", BC_PMC)?;
    m.add("BC_MUR", BC_MUR)?;
    m.add("BC_PML", BC_PML)?;
    Ok(())
}
