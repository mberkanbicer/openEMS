//! Processing: post-processing (probes, dumps, SAR, NF2FF interface).
//! Mirrors `Common/processing.h/cpp` and subclasses in the C++ codebase.

use crate::array_nijk::FieldArray;
use crate::hdf5::{Hdf5Error, Hdf5File, OPENEMS_HDF5_VERSION};

/// Native file-type enum (`ProcessFields::FileType`).
pub const VTK_FILETYPE: i32 = 0;
/// Native file-type enum (`ProcessFields::FileType`).
pub const HDF5_FILETYPE: i32 = 1;

/// Field name for a native `DumpType` value
/// (`ProcessFields::GetFieldNameByType`).
pub fn field_name_by_type(type_: i32) -> String {
    match type_ {
        0 => "E-Field".to_string(),
        1 => "H-Field".to_string(),
        2 => "J-Field".to_string(),
        3 => "RotH-Field".to_string(),
        4 => "D-Field".to_string(),
        5 => "B-Field".to_string(),
        20 => "SAR-local".to_string(),
        21 => "SAR_1g".to_string(),
        22 => "SAR_10g".to_string(),
        29 => "SAR_raw_data".to_string(),
        _ => "unknown field".to_string(),
    }
}

/// A complex accumulator for frequency-domain sampling.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ComplexPoint {
    pub re: f64,
    pub im: f64,
}

impl ComplexPoint {
    pub fn magnitude(&self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }
}

/// Mesh and timestep metadata needed by processing objects to map coordinates
/// to field indices.
#[derive(Debug, Clone)]
pub struct EngineMeta {
    pub num_lines: [usize; 3],
    /// Mesh line coordinates in metres.
    pub disc_lines: [Vec<f64>; 3],
    pub dt: f64,
}

impl EngineMeta {
    /// Map a physical coordinate (metres) to the nearest mesh index.
    pub fn index_from_coordinates(&self, coordinates: [f64; 3]) -> Option<[usize; 3]> {
        let mut index = [0usize; 3];
        for axis in 0..3 {
            let lines = &self.disc_lines[axis];
            if lines.len() < 2 {
                return None;
            }
            if coordinates[axis] < lines[0] || coordinates[axis] > lines[lines.len() - 1] {
                return None;
            }
            let mut best = 0usize;
            let mut best_distance = f64::INFINITY;
            for (candidate, line) in lines.iter().enumerate() {
                let distance = (line - coordinates[axis]).abs();
                if distance < best_distance {
                    best_distance = distance;
                    best = candidate;
                }
            }
            index[axis] = best.min(self.num_lines[axis].saturating_sub(1));
        }
        Some(index)
    }

    pub fn coordinate(&self, axis: usize, index: usize) -> f64 {
        self.disc_lines
            .get(axis)
            .and_then(|lines| lines.get(index))
            .copied()
            .unwrap_or(0.0)
    }
}

/// Read-only view of the running engine passed to processing objects.
pub struct EngineInterface<'a> {
    pub meta: &'a EngineMeta,
    pub volt: &'a FieldArray,
    pub curr: &'a FieldArray,
}

/// Mesh type for processing (mirrors C++ `Processing::MeshType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshType {
    Cartesian,
    Cylindrical,
}

/// Base trait for processing objects that sample the running simulation.
/// Mirrors the C++ `Processing` abstract class.
///
/// Processing objects are invoked by [`ProcessingArray`] during the FDTD
/// time-stepping loop. They can sample fields at specific timesteps,
/// accumulate frequency-domain data, and write output files.
pub trait Processing {
    /// Get the name of this processing.
    fn get_name(&self) -> String;
    /// Set the name of this processing.
    fn set_name(&mut self, val: &str);
    /// Get the processing name used in file descriptions.
    fn get_processing_name(&self) -> String;
    /// Initialize processing before the simulation run.
    fn init_process(&mut self) {}
    /// Reset processing state (e.g. step counters).
    fn reset(&mut self) {}
    /// Define start/stop coordinates snapped to the mesh.
    fn define_start_stop_coord(&mut self, _start: &[f64; 3], _stop: &[f64; 3]) {}
    /// Show snapped coordinates (debug output).
    fn show_snapped_coords(&self) {}
    /// Set the process interval in timesteps.
    fn set_process_interval(&mut self, interval: u32);
    /// Get the process interval.
    fn get_process_interval(&self) -> u32;
    /// Set process start/stop time in seconds.
    fn set_process_start_stop_time(&mut self, _start: f64, _stop: f64) {}
    /// Add a processing step (timestep at which to process).
    fn add_step(&mut self, _step: u32) {}
    /// Add multiple processing steps.
    fn add_steps(&mut self, _steps: &[u32]) {}
    /// Set frequency domain oversampling factor.
    fn set_fd_oversampling(&mut self, _factor: u32) {}
    /// Add a frequency for frequency-domain sampling.
    fn add_frequency(&mut self, _freq: f64) {}
    /// Add multiple frequencies.
    fn add_frequencies(&mut self, _freqs: &[f64]) {}
    /// Check if the current timestep should be processed.
    /// Advances the internal step pointer on match (mirrors C++ behavior).
    fn check_timestep(&mut self, _num_ts: u32) -> bool {
        false
    }
    /// Get the next processing interval (timesteps until next process).
    fn get_next_interval(&self, _num_ts: u32) -> i32 {
        i32::MAX
    }
    /// Process data prior to the simulation run.
    fn pre_process(&mut self) {}
    /// Process data during simulation run. Returns next interval.
    fn process(&mut self, _num_ts: u32) -> i32 {
        0
    }
    /// Process data with access to the running engine fields.
    /// The default forwards to [`Processing::process`].
    fn process_fields(&mut self, num_ts: u32, _interface: &EngineInterface<'_>) -> i32 {
        self.process(num_ts)
    }
    /// Process data after simulation has finished.
    fn post_process(&mut self) {}
    /// Enable or disable processing.
    fn set_enable(&mut self, _val: bool) {}
    /// Get whether processing is enabled.
    fn get_enable(&self) -> bool {
        true
    }
    /// Set the weight for this processing.
    fn set_weight(&mut self, _weight: f64) {}
    /// Get the weight.
    fn get_weight(&self) -> f64 {
        1.0
    }
    /// Invoke this flag to flush all stored data to disk.
    fn flush_next(&mut self) {}
    /// Flush stored data to disk.
    fn flush_data(&mut self) {}
    /// Set mesh type.
    fn set_mesh_type(&mut self, _mesh_type: MeshType) {}
    /// Get mesh type.
    fn get_mesh_type(&self) -> MeshType {
        MeshType::Cartesian
    }
    /// Set dump precision.
    fn set_precision(&mut self, _val: u32) {}
    /// Get dump precision.
    fn get_precision(&self) -> u32 {
        12
    }
    /// Dump probe geometry to VTK file.
    fn dump_box_to_file(&self, _vtk_filename_prefix: &str, _dual_mesh: bool) {}
    /// Set dual mesh flag.
    fn set_dual_mesh(&mut self, _val: bool) {}
    /// Set dual time flag.
    fn set_dual_time(&mut self, _val: bool) {}
}

/// Manages a collection of [`Processing`] objects.
/// Mirrors the C++ `ProcessingArray` class.
///
/// The array invokes `PreProcess`, `Process`, and `PostProcess` on all
/// registered processings, and returns the smallest next processing interval
/// so the caller knows when the next processing is due.
pub struct ProcessingArray {
    max_interval: u32,
    processings: Vec<Box<dyn Processing>>,
}

impl ProcessingArray {
    /// Create a new `ProcessingArray` with the given maximum interval.
    pub fn new(max_interval: u32) -> Self {
        Self {
            max_interval,
            processings: Vec::new(),
        }
    }

    /// Add a processing to the array.
    pub fn add_processing(&mut self, proc: Box<dyn Processing>) {
        self.processings.push(proc);
    }

    /// Initialize all processings.
    pub fn init_all(&mut self) {
        for proc in self.processings.iter_mut() {
            proc.init_process();
        }
    }

    /// Invoke `flush_next` on all processings.
    pub fn flush_next(&mut self) {
        for proc in self.processings.iter_mut() {
            proc.flush_next();
        }
    }

    /// Reset all processings.
    pub fn reset(&mut self) {
        for proc in self.processings.iter_mut() {
            proc.reset();
        }
    }

    /// Invoke `pre_process` on all processings.
    pub fn pre_process(&mut self) {
        for proc in self.processings.iter_mut() {
            proc.pre_process();
        }
    }

    /// Invoke `process` on all processings. Returns the smallest next
    /// interval (in timesteps) until any processing is due.
    pub fn process(&mut self, num_ts: u32) -> i32 {
        let mut next_process = self.max_interval as i32;
        for proc in self.processings.iter_mut() {
            let step = proc.process(num_ts);
            if step > 0 && step < next_process {
                next_process = step;
            }
        }
        next_process
    }

    /// Invoke `process_fields` on all processings with engine access.
    pub fn process_fields(&mut self, num_ts: u32, interface: &EngineInterface<'_>) -> i32 {
        let mut next_process = self.max_interval as i32;
        for proc in self.processings.iter_mut() {
            let step = proc.process_fields(num_ts, interface);
            if step > 0 && step < next_process {
                next_process = step;
            }
        }
        next_process
    }

    /// Invoke `post_process` on all processings.
    pub fn post_process(&mut self) {
        for proc in self.processings.iter_mut() {
            proc.post_process();
        }
    }

    /// Dump boxes to VTK file for all processings.
    pub fn dump_boxes_to_file(&self, vtk_filename_prefix: &str) {
        for proc in self.processings.iter() {
            proc.dump_box_to_file(vtk_filename_prefix, false);
        }
    }

    /// Get the number of processings.
    pub fn get_number_of_processings(&self) -> usize {
        self.processings.len()
    }

    /// Get a reference to a processing by index.
    pub fn get_processing(&self, number: usize) -> Option<&dyn Processing> {
        self.processings.get(number).map(|p| p.as_ref())
    }
}

/// A field probe that samples E/H fields at a point during the simulation.
/// Mirrors the C++ `FieldProbe` subclass of `Processing`.
#[allow(dead_code)]
pub struct FieldProbe {
    name: String,
    processing_name: String,
    position: [f64; 3],
    pub component: usize,
    pub samples: Vec<f32>,
    pub frequencies: Vec<f64>,
    pub dft: Vec<ComplexPoint>,
    fd_oversampling: u32,
    process_interval: u32,
    start_ts: u32,
    stop_ts: u32,
    process_steps: Vec<u32>,
    ps_pos: usize,
    enabled: bool,
    weight: f64,
    mesh_type: MeshType,
    precision: u32,
    dual_mesh: bool,
    dual_time: bool,
    start: [u32; 3],
    stop: [u32; 3],
    m_start_inside: [bool; 3],
    m_stop_inside: [bool; 3],
}

impl FieldProbe {
    /// Create a new field probe.
    pub fn new(name: &str, position: [f64; 3]) -> Self {        Self {
            name: name.to_string(),
            processing_name: name.to_string(),
            position,
            component: 0,
            samples: Vec::new(),
            frequencies: Vec::new(),
            dft: Vec::new(),
            fd_oversampling: 1,
            process_interval: 0,
            start_ts: 0,
            stop_ts: u32::MAX,
            process_steps: Vec::new(),
            ps_pos: 0,
            enabled: true,
            weight: 1.0,
            mesh_type: MeshType::Cartesian,
            precision: 12,
            dual_mesh: false,
            dual_time: false,
            start: [0, 0, 0],
            stop: [0, 0, 0],
            m_start_inside: [false, false, false],
            m_stop_inside: [false, false, false],
        }
    }

    /// Select the field component (0=Ex, 1=Ey, 2=Ez for voltage probes).
    pub fn set_component(&mut self, component: usize) {
        if component < 3 {
            self.component = component;
        }
    }

    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    pub fn frequencies(&self) -> &[f64] {
        &self.frequencies
    }

    pub fn dft(&self) -> &[ComplexPoint] {
        &self.dft
    }

    pub fn fd_oversampling(&self) -> u32 {
        self.fd_oversampling
    }
}

impl Processing for FieldProbe {
    fn get_name(&self) -> String {
        self.name.clone()
    }

    fn set_name(&mut self, val: &str) {
        self.name = val.to_string();
    }

    fn get_processing_name(&self) -> String {
        self.processing_name.clone()
    }

    fn set_process_interval(&mut self, interval: u32) {
        self.process_interval = interval;
    }

    fn get_process_interval(&self) -> u32 {
        self.process_interval
    }

    fn set_process_start_stop_time(&mut self, start: f64, stop: f64) {
        // Placeholder timestep (1 ps). In the full integration this should
        // come from the engine/operator via `get_timestep()`.
        let d_t: f64 = 1.0e-12;
        self.start_ts = if start > 0.0 {
            (start / d_t).floor() as u32
        } else {
            0
        };
        self.stop_ts = if stop > 0.0 {
            (stop / d_t).ceil() as u32
        } else {
            u32::MAX
        };
        if self.stop_ts <= self.start_ts {
            self.start_ts = 0;
            self.stop_ts = u32::MAX;
        }
    }

    fn add_step(&mut self, step: u32) {
        if self.process_steps.is_empty() || !self.process_steps.contains(&step) {
            self.process_steps.push(step);
        }
    }

    fn add_steps(&mut self, steps: &[u32]) {
        for &step in steps {
            self.add_step(step);
        }
    }

    fn set_fd_oversampling(&mut self, factor: u32) {
        if factor > 0 {
            self.fd_oversampling = factor;
        }
    }

    fn add_frequency(&mut self, freq: f64) {
        if freq > 0.0 && !self.frequencies.contains(&freq) {
            self.frequencies.push(freq);
            self.dft.push(ComplexPoint::default());
        }
    }

    fn add_frequencies(&mut self, freqs: &[f64]) {
        for &freq in freqs {
            self.add_frequency(freq);
        }
    }

    fn check_timestep(&mut self, num_ts: u32) -> bool {
        if num_ts < self.start_ts || num_ts > self.stop_ts {
            return false;
        }
        if self.ps_pos < self.process_steps.len() && self.process_steps[self.ps_pos] == num_ts {
            self.ps_pos += 1;
            return true;
        }
        if self.process_interval > 0 && num_ts.is_multiple_of(self.process_interval) {
            return true;
        }
        false
    }

    fn get_next_interval(&self, num_ts: u32) -> i32 {
        if !self.enabled {
            return -1;
        }
        let mut next = i32::MAX;
        if self.ps_pos < self.process_steps.len() {
            next = self.process_steps[self.ps_pos] as i32 - num_ts as i32;
        }
        if self.process_interval != 0 {
            let next_interval =
                self.process_interval as i32 - (num_ts % self.process_interval) as i32;
            if next_interval < next {
                next = next_interval;
            }
        }
        next
    }

    fn pre_process(&mut self) {
        // A point probe needs no pre-processing.
    }

    fn process(&mut self, num_ts: u32) -> i32 {
        if !self.enabled {
            return -1;
        }
        // Legacy path without engine access; see `process_fields`.
        self.get_next_interval(num_ts)
    }

    fn process_fields(&mut self, num_ts: u32, interface: &EngineInterface<'_>) -> i32 {
        if !self.enabled {
            return -1;
        }
        if self.check_timestep(num_ts)
            || (self.process_interval == 0 && self.process_steps.is_empty())
        {
            if let Some(index) = interface.meta.index_from_coordinates(self.position) {
                let sample =
                    *interface.volt.get(self.component, index[0], index[1], index[2]);
                self.samples.push(sample);
                let time = num_ts as f64 * interface.meta.dt;
                for (frequency, accumulator) in self.frequencies.iter().zip(self.dft.iter_mut()) {
                    let argument = -2.0 * std::f64::consts::PI * frequency * time;
                    accumulator.re += sample as f64 * argument.cos();
                    accumulator.im += sample as f64 * argument.sin();
                }
            }
        }
        self.get_next_interval(num_ts)
    }

    fn post_process(&mut self) {
        // A point probe needs no post-processing.
    }

    fn reset(&mut self) {
        self.samples.clear();
        for accumulator in &mut self.dft {
            *accumulator = ComplexPoint::default();
        }
        self.ps_pos = 0;
    }

    fn set_enable(&mut self, val: bool) {
        self.enabled = val;
    }

    fn get_enable(&self) -> bool {
        self.enabled
    }

    fn set_weight(&mut self, weight: f64) {
        self.weight = weight;
    }

    fn get_weight(&self) -> f64 {
        self.weight
    }

    fn set_mesh_type(&mut self, mesh_type: MeshType) {
        self.mesh_type = mesh_type;
    }

    fn get_mesh_type(&self) -> MeshType {
        self.mesh_type
    }

    fn set_precision(&mut self, val: u32) {
        self.precision = val;
    }

    fn get_precision(&self) -> u32 {
        self.precision
    }

    fn set_dual_mesh(&mut self, val: bool) {
        self.dual_mesh = val;
    }

    fn set_dual_time(&mut self, val: bool) {
        self.dual_time = val;
    }
}

impl Default for FieldProbe {
    fn default() -> Self {
        Self::new("probe", [0.0, 0.0, 0.0])
    }
}

/// Base struct for integral-based processing (voltage, current, etc.).
/// Mirrors the C++ `ProcessIntegral` subclass of `Processing`.
#[allow(dead_code)]
pub struct ProcessIntegral {
    name: String,
    processing_name: String,
    position: [f64; 3],
    normal_dir: i32,
    normal_vector: [f64; 3],
}

impl ProcessIntegral {
    /// Create a new integral processing. `normal_dir` is a direction vector
    /// like [1.0, 0.0, 0.0]; it is converted to an axis index 0/1/2.
    pub fn new(position: [f64; 3], normal_dir: [f64; 3]) -> Self {
        let axis = normal_dir
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.abs()
                    .partial_cmp(&b.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i as i32)
            .unwrap_or(0);
        Self {
            name: "integral".to_string(),
            processing_name: "integral".to_string(),
            position,
            normal_dir: axis,
            normal_vector: normal_dir,
        }
    }

    pub fn get_processing_name(&self) -> String {
        self.processing_name.clone()
    }

    pub fn get_integral_name(&self, _index: usize) -> String {
        "integral".to_string()
    }

    pub fn get_number_of_integrals(&self) -> usize {
        1
    }

    pub fn get_normal_dir(&self) -> i32 {
        self.normal_dir
    }

    pub fn set_normal_dir(&mut self, dir: i32) {
        self.normal_dir = dir;
    }

    pub fn calc_integral(&self) -> f64 {
        0.0
    }

    pub fn calc_multiple_integrals(&self) -> Vec<f64> {
        vec![0.0]
    }
}

impl Processing for ProcessIntegral {
    fn get_name(&self) -> String {
        self.name.clone()
    }

    fn set_name(&mut self, val: &str) {
        self.name = val.to_string();
    }

    fn get_processing_name(&self) -> String {
        self.processing_name.clone()
    }

    fn set_process_interval(&mut self, _interval: u32) {}

    fn get_process_interval(&self) -> u32 {
        0
    }
}

/// Voltage integration processing.
#[allow(dead_code)]
pub struct ProcessVoltage {
    name: String,
    processing_name: String,
    position: [f64; 3],
    normal_dir: i32,
    normal_vector: [f64; 3],
}

impl ProcessVoltage {
    pub fn new(position: [f64; 3], normal_dir: [f64; 3]) -> Self {
        let axis = normal_dir
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.abs()
                    .partial_cmp(&b.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i as i32)
            .unwrap_or(0);
        Self {
            name: "voltage".to_string(),
            processing_name: "voltage integration".to_string(),
            position,
            normal_dir: axis,
            normal_vector: normal_dir,
        }
    }

    pub fn get_processing_name(&self) -> String {
        self.processing_name.clone()
    }

    pub fn get_integral_name(&self, _index: usize) -> String {
        "voltage".to_string()
    }

    pub fn get_number_of_integrals(&self) -> usize {
        1
    }

    pub fn get_normal_dir(&self) -> i32 {
        self.normal_dir
    }

    pub fn set_normal_dir(&mut self, dir: i32) {
        self.normal_dir = dir;
    }

    pub fn calc_integral(&self) -> f64 {
        0.0
    }

    pub fn calc_multiple_integrals(&self) -> Vec<f64> {
        vec![0.0]
    }
}

impl Processing for ProcessVoltage {
    fn get_name(&self) -> String {
        self.name.clone()
    }

    fn set_name(&mut self, val: &str) {
        self.name = val.to_string();
    }

    fn get_processing_name(&self) -> String {
        self.processing_name.clone()
    }

    fn set_process_interval(&mut self, _interval: u32) {}

    fn get_process_interval(&self) -> u32 {
        0
    }
}

/// Current integration processing.
#[allow(dead_code)]
pub struct ProcessCurrent {
    name: String,
    processing_name: String,
    position: [f64; 3],
    normal_dir: i32,
    normal_vector: [f64; 3],
}

impl ProcessCurrent {
    pub fn new(position: [f64; 3], normal_dir: [f64; 3]) -> Self {
        let axis = normal_dir
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.abs()
                    .partial_cmp(&b.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i as i32)
            .unwrap_or(0);
        Self {
            name: "current".to_string(),
            processing_name: "current integration".to_string(),
            position,
            normal_dir: axis,
            normal_vector: normal_dir,
        }
    }

    pub fn get_processing_name(&self) -> String {
        self.processing_name.clone()
    }

    pub fn get_integral_name(&self, _index: usize) -> String {
        "current".to_string()
    }

    pub fn get_number_of_integrals(&self) -> usize {
        1
    }

    pub fn get_normal_dir(&self) -> i32 {
        self.normal_dir
    }

    pub fn set_normal_dir(&mut self, dir: i32) {
        self.normal_dir = dir;
    }

    pub fn calc_integral(&self) -> f64 {
        0.0
    }

    pub fn calc_multiple_integrals(&self) -> Vec<f64> {
        vec![0.0]
    }
}

impl Processing for ProcessCurrent {
    fn get_name(&self) -> String {
        self.name.clone()
    }

    fn set_name(&mut self, val: &str) {
        self.name = val.to_string();
    }

    fn get_processing_name(&self) -> String {
        self.processing_name.clone()
    }

    fn set_process_interval(&mut self, _interval: u32) {}

    fn get_process_interval(&self) -> u32 {
        0
    }
}

/// Common field processing (E/H/J/D/B field dumps, SAR, etc.).
/// Mirrors the C++ `ProcessFields` subclass of `Processing`.
#[allow(dead_code)]
pub struct ProcessFields {
    name: String,
    processing_name: String,
    start: [f64; 3],
    stop: [f64; 3],
    dump_type: i32,
    file_type: i32,
    sub_sampling: [i32; 3],
    opt_resolution: [f64; 3],
    dump_mode: i32,
}

impl ProcessFields {
    pub fn new(name: &str, start: [f64; 3], stop: [f64; 3]) -> Self {
        Self {
            name: name.to_string(),
            processing_name: "common field processing".to_string(),
            start,
            stop,
            dump_type: 0, // E_FIELD_DUMP
            file_type: VTK_FILETYPE,
            sub_sampling: [1, 1, 1],
            opt_resolution: [0.0, 0.0, 0.0],
            dump_mode: 0,
        }
    }

    pub fn get_processing_name(&self) -> String {
        self.processing_name.clone()
    }

    pub fn get_dump_type(&self) -> i32 {
        self.dump_type
    }

    pub fn get_file_type(&self) -> i32 {
        self.file_type
    }

    pub fn set_dump_type(&mut self, val: i32) {
        self.dump_type = val;
    }

    pub fn set_file_type(&mut self, val: i32) {
        self.file_type = val;
    }

    pub fn set_sub_sampling(&mut self, val: i32, _: i32) {
        self.sub_sampling[0] = val;
    }

    pub fn set_opt_resolution(&mut self, val: f64, _: i32) {
        self.opt_resolution[0] = val;
    }

    pub fn set_dump_mode(&mut self, val: i32) {
        self.dump_mode = val;
    }

    pub fn set_dump_mode_2_node(&mut self) {
        self.dump_mode = 0;
    }

    pub fn set_dump_mode_2_cell(&mut self) {
        self.dump_mode = 1;
    }

    pub fn need_conductivity(&self) -> bool {
        true
    }

    pub fn need_permittivity(&self) -> bool {
        true
    }

    pub fn need_permeability(&self) -> bool {
        true
    }

    pub fn calc_total_energy_estimate(&self) -> f64 {
        0.0
    }

    pub fn get_field_name_by_type(type_: i32) -> String {
        field_name_by_type(type_)
    }
}

impl Processing for ProcessFields {
    fn get_name(&self) -> String {
        self.name.clone()
    }

    fn set_name(&mut self, val: &str) {
        self.name = val.to_string();
    }

    fn get_processing_name(&self) -> String {
        self.processing_name.clone()
    }

    fn set_process_interval(&mut self, _interval: u32) {}

    fn get_process_interval(&self) -> u32 {
        0
    }
}

/// Time-domain field dump processing.
/// Mirrors the C++ `ProcessFieldsTD` concept (time-domain E/H/J/D/B field dumps).
#[allow(dead_code)]
pub struct ProcessFieldsTD {
    name: String,
    processing_name: String,
    start: [f64; 3],
    stop: [f64; 3],
    dump_type: i32,
    file_type: i32,
    sub_sampling: [i32; 3],
    opt_resolution: [f64; 3],
    dump_mode: i32,
    process_interval: u32,
    process_steps: Vec<u32>,
    ps_pos: usize,
    initialized: bool,
}

impl ProcessFieldsTD {
    pub fn new(name: &str, start: [f64; 3], stop: [f64; 3]) -> Self {
        Self {
            name: name.to_string(),
            processing_name: "time-domain field dump".to_string(),
            start,
            stop,
            dump_type: 0, // E_FIELD_DUMP
            file_type: VTK_FILETYPE,
            sub_sampling: [1, 1, 1],
            opt_resolution: [0.0, 0.0, 0.0],
            dump_mode: 0,
            process_interval: 1,
            process_steps: Vec::new(),
            ps_pos: 0,
            initialized: false,
        }
    }

    pub fn get_processing_name(&self) -> String {
        self.processing_name.clone()
    }

    pub fn get_dump_type(&self) -> i32 {
        self.dump_type
    }

    pub fn get_file_type(&self) -> i32 {
        self.file_type
    }

    pub fn set_dump_type(&mut self, val: i32) {
        self.dump_type = val;
    }

    pub fn set_file_type(&mut self, val: i32) {
        self.file_type = val;
    }

    pub fn set_sub_sampling(&mut self, val: i32, dir: i32) {
        if val < 1 {
            return;
        }
        if (0..=2).contains(&dir) {
            self.sub_sampling[dir as usize] = val;
        } else {
            self.sub_sampling = [val; 3];
        }
        self.opt_resolution = [0.0; 3];
    }

    pub fn set_opt_resolution(&mut self, val: f64, dir: i32) {
        if !(val > 0.0) {
            return;
        }
        if (0..=2).contains(&dir) {
            self.opt_resolution[dir as usize] = val;
        } else {
            self.opt_resolution = [val; 3];
        }
        self.sub_sampling = [1; 3];
    }

    pub fn set_dump_mode(&mut self, val: i32) {
        self.dump_mode = val;
    }

    /// Mesh indices written by this dump, mirroring `ProcessFields::CalcMeshPos`.
    /// A positive optical resolution wins over the integer sub-sampling rate.
    pub fn dump_indices(&self, axis: usize, lines: &[f64]) -> Vec<usize> {
        let count = lines.len();
        if count == 0 {
            return Vec::new();
        }
        if self.opt_resolution[axis] > 0.0 && count > 1 {
            let resolution = self.opt_resolution[axis];
            let mut indices = Vec::new();
            let mut last = f64::NEG_INFINITY;
            for (index, &coordinate) in lines.iter().enumerate() {
                if indices.is_empty() || (coordinate - last).abs() >= resolution - 1e-15 {
                    indices.push(index);
                    last = coordinate;
                }
            }
            indices
        } else {
            let stride = self.sub_sampling[axis].max(1) as usize;
            (0..count).step_by(stride).collect()
        }
    }

    pub fn set_dump_mode_2_node(&mut self) {
        self.dump_mode = 0;
    }

    pub fn set_dump_mode_2_cell(&mut self) {
        self.dump_mode = 1;
    }

    pub fn need_conductivity(&self) -> bool {
        true
    }

    pub fn need_permittivity(&self) -> bool {
        true
    }

    pub fn need_permeability(&self) -> bool {
        true
    }

    pub fn calc_total_energy_estimate(&self) -> f64 {
        0.0
    }

    pub fn get_field_name_by_type(type_: i32) -> String {
        field_name_by_type(type_)
    }
}

impl Processing for ProcessFieldsTD {
    fn get_name(&self) -> String {
        self.name.clone()
    }

    fn set_name(&mut self, val: &str) {
        self.name = val.to_string();
    }

    fn get_processing_name(&self) -> String {
        self.processing_name.clone()
    }

    fn set_process_interval(&mut self, interval: u32) {
        self.process_interval = interval;
    }

    fn get_process_interval(&self) -> u32 {
        self.process_interval
    }

    fn add_step(&mut self, step: u32) {
        if !self.process_steps.contains(&step) {
            self.process_steps.push(step);
        }
    }

    fn add_steps(&mut self, steps: &[u32]) {
        for &step in steps {
            self.add_step(step);
        }
    }

    fn check_timestep(&mut self, num_ts: u32) -> bool {
        if self.ps_pos < self.process_steps.len() && self.process_steps[self.ps_pos] == num_ts {
            self.ps_pos += 1;
            return true;
        }
        self.process_interval > 0 && num_ts.is_multiple_of(self.process_interval)
    }

    fn process_fields(&mut self, num_ts: u32, interface: &EngineInterface<'_>) -> i32 {
        if !self.check_timestep(num_ts) {
            return 0;
        }
        let field = if self.dump_type == 1 {
            interface.curr
        } else {
            interface.volt
        };
        let field_name = Self::get_field_name_by_type(self.dump_type);

        // Dump mesh: native `CalcMeshPos` reduces the written grid by
        // sub-sampling or optical resolution.
        let indices = [
            self.dump_indices(0, &interface.meta.disc_lines[0]),
            self.dump_indices(1, &interface.meta.disc_lines[1]),
            self.dump_indices(2, &interface.meta.disc_lines[2]),
        ];
        let dims = [indices[0].len(), indices[1].len(), indices[2].len()];
        let coordinates: [Vec<f64>; 3] = [
            indices[0]
                .iter()
                .map(|&index| interface.meta.disc_lines[0][index])
                .collect(),
            indices[1]
                .iter()
                .map(|&index| interface.meta.disc_lines[1][index])
                .collect(),
            indices[2]
                .iter()
                .map(|&index| interface.meta.disc_lines[2][index])
                .collect(),
        ];
        let coordinate_refs: [&[f64]; 3] = [&coordinates[0], &coordinates[1], &coordinates[2]];
        let reduced = extract_field(field, &indices);

        if self.file_type == HDF5_FILETYPE {
            // Native: /FieldData/TD/<8-digit timestep> + a `time` attribute.
            let dataset = format!("{num_ts:08}");
            let writer = HDF5FileWriter::new(&format!("{}.h5", self.name));
            let result = (|| -> Result<(), Hdf5Error> {
                if !self.initialized {
                    writer.create(self.dump_type, false, true)?;
                    writer.write_rect_mesh(dims, coordinate_refs, 0, 1.0)?;
                    self.initialized = true;
                }
                writer.write_vector_field("/FieldData/TD", &dataset, &reduced, dims, true)?;
                writer.write_time_attribute(
                    &format!("/FieldData/TD/{dataset}"),
                    (num_ts as f64 * interface.meta.dt) as f32,
                )
            })();
            if let Err(error) = result {
                eprintln!(
                    "[openEMS] Failed to write time-domain HDF5 dump '{}': {error}",
                    self.name
                );
            }
        } else {
            // Native VTK writer default pad length is 10.
            let prefix = format!("{}_{num_ts:010}", self.name);
            let writer = VTKFileWriter::new(&prefix);
            if let Err(error) =
                writer.write_structured_grid(&field_name, dims, coordinate_refs, &reduced)
            {
                eprintln!("[openEMS] Failed to write time-domain dump '{prefix}': {error}");
            }
        }
        0
    }

    fn reset(&mut self) {
        self.ps_pos = 0;
        self.initialized = false;
    }
}

/// Frequency-domain field dump processing.
/// Mirrors the C++ `ProcessFieldsFD` concept (frequency-domain field dumps via FD oversampling).
#[allow(dead_code)]
pub struct ProcessFieldsFD {
    name: String,
    processing_name: String,
    start: [f64; 3],
    stop: [f64; 3],
    dump_type: i32,
    file_type: i32,
    fd_oversampling: u32,
    frequencies: Vec<f64>,
    fd_interval: u32,
    sample_count: u32,
    process_interval: u32,
    dims: Option<[usize; 3]>,
    disc_lines: Option<[Vec<f64>; 3]>,
    acc_re: Vec<Vec<f32>>,
    acc_im: Vec<Vec<f32>>,
    initialized: bool,
}

impl ProcessFieldsFD {
    pub fn new(name: &str, start: [f64; 3], stop: [f64; 3]) -> Self {
        Self {
            name: name.to_string(),
            processing_name: "frequency-domain field dump".to_string(),
            start,
            stop,
            dump_type: 0, // E_FIELD_DUMP
            file_type: VTK_FILETYPE,
            fd_oversampling: 4,
            frequencies: Vec::new(),
            fd_interval: 0,
            sample_count: 0,
            process_interval: 1,
            dims: None,
            disc_lines: None,
            acc_re: Vec::new(),
            acc_im: Vec::new(),
            initialized: false,
        }
    }

    pub fn get_processing_name(&self) -> String {
        self.processing_name.clone()
    }

    pub fn get_field_name_by_type(type_: i32) -> String {
        field_name_by_type(type_)
    }

    pub fn get_dump_type(&self) -> i32 {
        self.dump_type
    }

    pub fn get_file_type(&self) -> i32 {
        self.file_type
    }

    pub fn get_fd_oversampling(&self) -> u32 {
        self.fd_oversampling
    }

    pub fn set_fd_oversampling(&mut self, factor: u32) {
        self.fd_oversampling = factor;
    }

    pub fn add_frequency(&mut self, freq: f64) {
        if freq > 0.0 && !self.frequencies.contains(&freq) {
            self.frequencies.push(freq);
        }
    }

    pub fn get_frequencies(&self) -> &[f64] {
        &self.frequencies
    }

    /// Sampling interval in timesteps used for the running DFT.
    pub fn fd_interval(&self) -> u32 {
        self.fd_interval
    }

    /// Number of accumulated frequency-domain samples.
    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    /// Accumulated real part for a frequency index in `(n, i, j, k)` order.
    pub fn real_part(&self, index: usize) -> Option<&[f32]> {
        self.acc_re.get(index).map(|values| values.as_slice())
    }

    /// Accumulated imaginary part for a frequency index in `(n, i, j, k)` order.
    pub fn imag_part(&self, index: usize) -> Option<&[f32]> {
        self.acc_im.get(index).map(|values| values.as_slice())
    }

    /// Compute the sampling interval from the timestep and the highest requested
    /// frequency, mirroring `Processing::AddFrequency`'s Nyquist logic.
    fn update_interval(&mut self, dt: f64) {
        let highest = self.frequencies.iter().copied().fold(0.0f64, f64::max);
        if highest <= 0.0 || dt <= 0.0 {
            self.fd_interval = 1;
            return;
        }
        let nyquist_steps = 1.0 / (2.0 * highest * dt);
        let oversampling = self.fd_oversampling.max(1) as f64;
        self.fd_interval = ((nyquist_steps / oversampling).round() as u32).max(1);
    }

    fn ensure_accumulators(&mut self, dims: [usize; 3]) {
        let cells = 3 * dims[0] * dims[1] * dims[2];
        if self.dims != Some(dims) || self.acc_re.len() != self.frequencies.len() {
            self.dims = Some(dims);
            self.acc_re = vec![vec![0.0f32; cells]; self.frequencies.len()];
            self.acc_im = vec![vec![0.0f32; cells]; self.frequencies.len()];
            self.sample_count = 0;
        }
    }

    pub fn set_dump_type(&mut self, val: i32) {
        self.dump_type = val;
    }

    pub fn set_file_type(&mut self, val: i32) {
        self.file_type = val;
    }

    pub fn need_conductivity(&self) -> bool {
        true
    }

    pub fn need_permittivity(&self) -> bool {
        true
    }

    pub fn need_permeability(&self) -> bool {
        true
    }

    pub fn calc_total_energy_estimate(&self) -> f64 {
        0.0
    }
}

impl Processing for ProcessFieldsFD {
    fn get_name(&self) -> String {
        self.name.clone()
    }

    fn set_name(&mut self, val: &str) {
        self.name = val.to_string();
    }

    fn get_processing_name(&self) -> String {
        self.processing_name.clone()
    }

    fn set_process_interval(&mut self, interval: u32) {
        self.process_interval = interval;
    }

    fn get_process_interval(&self) -> u32 {
        self.process_interval
    }

    fn set_fd_oversampling(&mut self, factor: u32) {
        self.fd_oversampling = factor.max(1);
    }

    fn add_frequency(&mut self, freq: f64) {
        ProcessFieldsFD::add_frequency(self, freq);
    }

    fn add_frequencies(&mut self, freqs: &[f64]) {
        for &freq in freqs {
            ProcessFieldsFD::add_frequency(self, freq);
        }
    }

    fn process_fields(&mut self, num_ts: u32, interface: &EngineInterface<'_>) -> i32 {
        if self.frequencies.is_empty() {
            return 0;
        }
        if self.fd_interval == 0 {
            self.update_interval(interface.meta.dt);
        }
        let interval = self.fd_interval.max(1);
        if !num_ts.is_multiple_of(interval) {
            return 0;
        }
        let dims = interface.meta.num_lines;
        self.ensure_accumulators(dims);
        if self.disc_lines.is_none() {
            self.disc_lines = Some(interface.meta.disc_lines.clone());
        }
        let field = if self.dump_type == 1 {
            interface.curr
        } else {
            interface.volt
        };
        let [nx, ny, nz] = dims;
        let time = num_ts as f64 * interface.meta.dt;
        let scale = 2.0 * interface.meta.dt * interval as f64;
        for (frequency, (re, im)) in self
            .frequencies
            .iter()
            .zip(self.acc_re.iter_mut().zip(self.acc_im.iter_mut()))
        {
            let argument = -2.0 * std::f64::consts::PI * frequency * time;
            let cosine = argument.cos() * scale;
            let sine = argument.sin() * scale;
            let mut index = 0usize;
            for n in 0..3 {
                for i in 0..nx {
                    for j in 0..ny {
                        for k in 0..nz {
                            let value = *field.get(n, i, j, k) as f64;
                            re[index] += (value * cosine) as f32;
                            im[index] += (value * sine) as f32;
                            index += 1;
                        }
                    }
                }
            }
        }
        self.sample_count += 1;
        self.get_next_interval(num_ts)
    }

    fn get_next_interval(&self, num_ts: u32) -> i32 {
        let interval = self.fd_interval.max(1);
        let next = interval as i32 - (num_ts % interval) as i32;
        next.max(1)
    }

    fn check_timestep(&mut self, num_ts: u32) -> bool {
        num_ts.is_multiple_of(self.fd_interval.max(1))
    }

    fn post_process(&mut self) {
        if self.frequencies.is_empty() || self.acc_re.is_empty() {
            return;
        }
        let Some(dims) = self.dims else {
            return;
        };
        if self.file_type == HDF5_FILETYPE {
            self.dump_hdf5(dims);
        } else {
            self.dump_vtk(dims);
        }
    }

    fn reset(&mut self) {
        for values in &mut self.acc_re {
            values.fill(0.0);
        }
        for values in &mut self.acc_im {
            values.fill(0.0);
        }
        self.sample_count = 0;
        self.initialized = false;
        self.fd_interval = 0;
    }
}

impl ProcessFieldsFD {
    /// Format a frequency the way the native dumpers do: integral values are
    /// written without a fractional part.
    pub fn frequency_label(frequency: f64) -> String {
        if frequency.fract() == 0.0 {
            format!("{}", frequency as i64)
        } else {
            format!("{frequency}")
        }
    }

    fn dump_hdf5(&mut self, dims: [usize; 3]) {
        let writer = HDF5FileWriter::new(&format!("{}.h5", self.name));
        let result = (|| -> Result<(), Hdf5Error> {
            writer.create(self.dump_type, false, true)?;
            if let Some(lines) = &self.disc_lines {
                let references: [&[f64]; 3] = [&lines[0], &lines[1], &lines[2]];
                writer.write_rect_mesh(dims, references, 0, 1.0)?;
            }
            writer.write_frequency_list(&self.frequencies)?;
            for (index, frequency) in self.frequencies.iter().enumerate() {
                writer.write_complex_field(
                    "/FieldData/FD",
                    &format!("f{index}"),
                    &self.acc_re[index],
                    &self.acc_im[index],
                    dims,
                    *frequency,
                )?;
            }
            Ok(())
        })();
        if let Err(error) = result {
            eprintln!(
                "[openEMS] Failed to write frequency-domain HDF5 dump '{}': {error}",
                self.name
            );
        }
    }

    fn dump_vtk(&mut self, dims: [usize; 3]) {
        let field_name = Self::get_field_name_by_type(self.dump_type);
        let [nx, ny, nz] = dims;
        let coordinates: [Vec<f64>; 3] = [
            (0..nx).map(|index| index as f64).collect(),
            (0..ny).map(|index| index as f64).collect(),
            (0..nz).map(|index| index as f64).collect(),
        ];
        let coordinate_refs: [&[f64]; 3] = [&coordinates[0], &coordinates[1], &coordinates[2]];
        for (index, frequency) in self.frequencies.iter().enumerate() {
            let re = &self.acc_re[index];
            let im = &self.acc_im[index];
            let label = Self::frequency_label(*frequency);
            // Magnitude and phase dumps.
            let mut magnitude = FieldArray::new("magnitude", dims);
            let mut phase = FieldArray::new("phase", dims);
            for n in 0..3 {
                for i in 0..nx {
                    for j in 0..ny {
                        for k in 0..nz {
                            let position = n * nx * ny * nz + i * ny * nz + j * nz + k;
                            let real = re[position] as f64;
                            let imag = im[position] as f64;
                            *magnitude.get_mut(n, i, j, k) = real.hypot(imag) as f32;
                            *phase.get_mut(n, i, j, k) = imag.atan2(real) as f32;
                        }
                    }
                }
            }
            let _ = VTKFileWriter::new(&format!("{}_f={label}_abs", self.name))
                .write_structured_grid(&field_name, dims, coordinate_refs, &magnitude);
            let _ = VTKFileWriter::new(&format!("{}_f={label}_arg", self.name))
                .write_structured_grid(&field_name, dims, coordinate_refs, &phase);

            // 21 phase snapshots, matching the native VTK FD dump.
            for step in 0..21 {
                let angle = 2.0 * std::f64::consts::PI * step as f64 / 21.0;
                let cosine = angle.cos();
                let sine = angle.sin();
                let mut snapshot = FieldArray::new("phase-snapshot", dims);
                for n in 0..3 {
                    for i in 0..nx {
                        for j in 0..ny {
                            for k in 0..nz {
                                let position = n * nx * ny * nz + i * ny * nz + j * nz + k;
                                let real = re[position] as f64;
                                let imag = im[position] as f64;
                                *snapshot.get_mut(n, i, j, k) =
                                    (real * cosine - imag * sine) as f32;
                            }
                        }
                    }
                }
                let degrees = (angle * 180.0 / std::f64::consts::PI).round() as i32;
                let _ = VTKFileWriter::new(&format!(
                    "{}_f={label}_p={degrees:03}",
                    self.name
                ))
                .write_structured_grid(&field_name, dims, coordinate_refs, &snapshot);
            }
        }
    }
}

/// SAR (Specific Absorption Rate) field processing.
/// Mirrors the C++ `ProcessFieldsSAR` concept (SAR-Local, SAR-1G, SAR-10G dumps).
#[allow(dead_code)]
pub struct ProcessFieldsSAR {
    name: String,
    processing_name: String,
    start: [f64; 3],
    stop: [f64; 3],
    dump_type: i32, // SAR_LOCAL_DUMP=20, SAR_1G_DUMP=21, SAR_10G_DUMP=22
    file_type: i32,
    mass_density: f64,
}

impl ProcessFieldsSAR {
    pub fn new(name: &str, start: [f64; 3], stop: [f64; 3]) -> Self {
        Self {
            name: name.to_string(),
            processing_name: "SAR field processing".to_string(),
            start,
            stop,
            dump_type: 20, // SAR_LOCAL_DUMP
            file_type: VTK_FILETYPE,
            mass_density: 1000.0, // kg/m^3 (water)
        }
    }

    pub fn get_processing_name(&self) -> String {
        self.processing_name.clone()
    }

    pub fn get_dump_type(&self) -> i32 {
        self.dump_type
    }

    pub fn get_file_type(&self) -> i32 {
        self.file_type
    }

    pub fn get_mass_density(&self) -> f64 {
        self.mass_density
    }

    pub fn set_dump_type(&mut self, val: i32) {
        self.dump_type = val;
    }

    pub fn set_file_type(&mut self, val: i32) {
        self.file_type = val;
    }

    pub fn set_mass_density(&mut self, val: f64) {
        self.mass_density = val;
    }

    pub fn need_conductivity(&self) -> bool {
        true
    }

    pub fn need_permittivity(&self) -> bool {
        false
    }

    pub fn need_permeability(&self) -> bool {
        false
    }

    pub fn calc_total_energy_estimate(&self) -> f64 {
        0.0
    }
}

impl Processing for ProcessFieldsSAR {
    fn get_name(&self) -> String {
        self.name.clone()
    }

    fn set_name(&mut self, val: &str) {
        self.name = val.to_string();
    }

    fn get_processing_name(&self) -> String {
        self.processing_name.clone()
    }

    fn set_process_interval(&mut self, _interval: u32) {}

    fn get_process_interval(&self) -> u32 {
        0
    }
}

/// Mode matching processing.
/// Mirrors the C++ `ProcessModeMatch` concept (modal decomposition at ports).
#[allow(dead_code)]
pub struct ProcessModeMatch {
    name: String,
    processing_name: String,
    start: [f64; 3],
    stop: [f64; 3],
    normal_dir: i32,
    num_modes: u32,
    frequencies: Vec<f64>,
}

impl ProcessModeMatch {
    pub fn new(name: &str, start: [f64; 3], stop: [f64; 3], normal_dir: i32) -> Self {
        Self {
            name: name.to_string(),
            processing_name: "mode match".to_string(),
            start,
            stop,
            normal_dir,
            num_modes: 1,
            frequencies: Vec::new(),
        }
    }

    pub fn get_processing_name(&self) -> String {
        self.processing_name.clone()
    }

    pub fn get_normal_dir(&self) -> i32 {
        self.normal_dir
    }

    pub fn get_num_modes(&self) -> u32 {
        self.num_modes
    }

    pub fn set_num_modes(&mut self, n: u32) {
        self.num_modes = n;
    }

    pub fn add_frequency(&mut self, freq: f64) {
        self.frequencies.push(freq);
    }

    pub fn get_frequencies(&self) -> &[f64] {
        &self.frequencies
    }
}

impl Processing for ProcessModeMatch {
    fn get_name(&self) -> String {
        self.name.clone()
    }

    fn set_name(&mut self, val: &str) {
        self.name = val.to_string();
    }

    fn get_processing_name(&self) -> String {
        self.processing_name.clone()
    }

    fn set_process_interval(&mut self, _interval: u32) {}

    fn get_process_interval(&self) -> u32 {
        0
    }
}

/// Near-field to far-field transformation.
/// Mirrors the C++ NF2FF post-processing concept.
#[allow(dead_code)]
pub struct NF2FF {
    name: String,
    processing_name: String,
    start: [f64; 3],
    stop: [f64; 3],
    center: [f64; 3],
    radius: f64,
    frequencies: Vec<f64>,
}

impl NF2FF {
    pub fn new(name: &str, start: [f64; 3], stop: [f64; 3]) -> Self {
        Self {
            name: name.to_string(),
            processing_name: "near-field to far-field".to_string(),
            start,
            stop,
            center: [0.0, 0.0, 0.0],
            radius: 0.0,
            frequencies: Vec::new(),
        }
    }

    pub fn get_processing_name(&self) -> String {
        self.processing_name.clone()
    }

    pub fn set_center(&mut self, center: [f64; 3]) {
        self.center = center;
    }

    pub fn set_radius(&mut self, radius: f64) {
        self.radius = radius;
    }

    pub fn add_frequency(&mut self, freq: f64) {
        self.frequencies.push(freq);
    }

    pub fn get_frequencies(&self) -> &[f64] {
        &self.frequencies
    }
}

impl Processing for NF2FF {
    fn get_name(&self) -> String {
        self.name.clone()
    }

    fn set_name(&mut self, val: &str) {
        self.name = val.to_string();
    }

    fn get_processing_name(&self) -> String {
        self.processing_name.clone()
    }

    fn set_process_interval(&mut self, _interval: u32) {}

    fn get_process_interval(&self) -> u32 {
        0
    }
}

/// SAR calculation processing (standalone SAR, not field-dump based).
/// Mirrors the C++ SAR post-processing concept.
#[allow(dead_code)]
pub struct SAR {
    name: String,
    processing_name: String,
    start: [f64; 3],
    stop: [f64; 3],
    mass_density: f64,
    averaging_cube: f64, // 1g or 10g averaging cube side length
}

impl SAR {
    pub fn new(name: &str, start: [f64; 3], stop: [f64; 3]) -> Self {
        Self {
            name: name.to_string(),
            processing_name: "SAR calculation".to_string(),
            start,
            stop,
            mass_density: 1000.0,
            averaging_cube: 0.0,
        }
    }

    pub fn get_processing_name(&self) -> String {
        self.processing_name.clone()
    }

    pub fn get_mass_density(&self) -> f64 {
        self.mass_density
    }

    pub fn get_averaging_cube(&self) -> f64 {
        self.averaging_cube
    }

    pub fn set_mass_density(&mut self, val: f64) {
        self.mass_density = val;
    }

    pub fn set_averaging_cube(&mut self, val: f64) {
        self.averaging_cube = val;
    }

    pub fn need_conductivity(&self) -> bool {
        true
    }

    pub fn need_permittivity(&self) -> bool {
        false
    }

    pub fn need_permeability(&self) -> bool {
        false
    }
}

impl Processing for SAR {
    fn get_name(&self) -> String {
        self.name.clone()
    }

    fn set_name(&mut self, val: &str) {
        self.name = val.to_string();
    }

    fn get_processing_name(&self) -> String {
        self.processing_name.clone()
    }

    fn set_process_interval(&mut self, _interval: u32) {}

    fn get_process_interval(&self) -> u32 {
        0
    }
}

/// HDF5 file writer for field dumps.
/// Mirrors the C++ `HDF5_File_Writer` class.
///
/// Every write goes through [`crate::hdf5::Hdf5File`], which stores the data in
/// the native openEMS layout. Without the `hdf5` feature the methods return
/// [`Hdf5Error::Unsupported`]; no substitute format is produced.
pub struct HDF5FileWriter {
    filename: String,
    precision: u32,
}

impl HDF5FileWriter {
    pub fn new(filename: &str) -> Self {
        Self {
            filename: filename.to_string(),
            precision: 12,
        }
    }

    pub fn get_filename(&self) -> &str {
        &self.filename
    }

    pub fn set_precision(&mut self, val: u32) {
        self.precision = val;
    }

    pub fn get_precision(&self) -> u32 {
        self.precision
    }

    /// Create the file and write the root metadata attributes.
    ///
    /// `dump_type` follows the native `DumpType` enum, `dual_mesh` records the
    /// interpolation mesh and `legacy_fmt` selects the Octave/MATLAB ordering.
    pub fn create(&self, dump_type: i32, dual_mesh: bool, legacy_fmt: bool) -> Result<(), Hdf5Error> {
        let file = Hdf5File::create(&self.filename)?;
        file.write_f64_attribute("/", "openEMS_HDF5_version", OPENEMS_HDF5_VERSION)?;
        file.write_i32_attribute("/", "dump_type", dump_type)?;
        file.write_bool_attribute("/", "dual_mesh", dual_mesh)?;
        if legacy_fmt {
            file.write_bool_attribute("/", "legacy_fmt", true)?;
        }
        Ok(())
    }

    /// Write the rectilinear mesh exactly like `HDF5_File_Writer::WriteRectMesh`.
    pub fn write_rect_mesh(
        &self,
        num_lines: [usize; 3],
        disc_lines: [&[f64]; 3],
        mesh_type: i32,
        scaling: f64,
    ) -> Result<(), Hdf5Error> {
        let file = Hdf5File::open(&self.filename, true)?;
        file.create_group("/Mesh")?;
        file.write_i32_attribute("/Mesh", "mesh_type", mesh_type)?;
        file.write_f64_attribute("/Mesh", "mesh_scaling", scaling)?;
        let names = ["x", "y", "z"];
        for axis in 0..3 {
            let mut values = Vec::with_capacity(num_lines[axis]);
            for &line in disc_lines[axis].iter().take(num_lines[axis]) {
                values.push(line * scaling);
            }
            file.write_f64_dataset("/Mesh", names[axis], &values)?;
        }
        Ok(())
    }

    /// Write a scalar real field as a 1-D dataset.
    pub fn write_field(&self, field_name: &str, data: &[f64]) -> Result<(), Hdf5Error> {
        let file = Hdf5File::open(&self.filename, true)?;
        file.write_f64_dataset("/", field_name, data)
    }

    /// Write a vector field in native `NXYZ` or legacy `NZYX` order.
    pub fn write_vector_field(
        &self,
        group: &str,
        dataset: &str,
        values: &FieldArray,
        dims: [usize; 3],
        legacy_fmt: bool,
    ) -> Result<(), Hdf5Error> {
        let file = Hdf5File::open(&self.filename, true)?;
        let raw = field_values(values, dims);
        if legacy_fmt {
            let ordered = crate::hdf5::field_to_nzyx(&raw, dims);
            file.write_f32_dataset_nd(
                group,
                dataset,
                &[3, dims[2], dims[1], dims[0]],
                &ordered,
                Some("NZYX"),
            )
        } else {
            let ordered = crate::hdf5::field_to_nxyz(&raw, dims);
            file.write_f32_dataset_nd(
                group,
                dataset,
                &[3, dims[0], dims[1], dims[2]],
                &ordered,
                Some("NXYZ"),
            )
        }
    }

    /// Write the real and imaginary parts of a complex field in legacy order.
    pub fn write_complex_field(
        &self,
        group: &str,
        dataset: &str,
        re: &[f32],
        im: &[f32],
        dims: [usize; 3],
        frequency: f64,
    ) -> Result<(), Hdf5Error> {
        let file = Hdf5File::open(&self.filename, true)?;
        let real = crate::hdf5::field_to_nzyx(re, dims);
        let imag = crate::hdf5::field_to_nzyx(im, dims);
        let shape = [3, dims[2], dims[1], dims[0]];
        let real_name = format!("{dataset}_real");
        let imag_name = format!("{dataset}_imag");
        file.write_f32_dataset_nd(group, &real_name, &shape, &real, Some("NZYX"))?;
        file.write_f32_dataset_nd(group, &imag_name, &shape, &imag, Some("NZYX"))?;
        let real_path = format!("{group}/{real_name}");
        let imag_path = format!("{group}/{imag_name}");
        file.write_f64_attribute(&real_path, "frequency", frequency)?;
        file.write_f64_attribute(&imag_path, "frequency", frequency)?;
        Ok(())
    }

    /// Write a dataset attribute with the native time stamp.
    pub fn write_time_attribute(&self, path: &str, time: f32) -> Result<(), Hdf5Error> {
        let file = Hdf5File::open(&self.filename, true)?;
        file.write_f32_attribute_vector(path, "time", &[time])
    }

    /// Write the frequency list attribute of a frequency-domain dump.
    pub fn write_frequency_list(&self, frequencies: &[f64]) -> Result<(), Hdf5Error> {
        let file = Hdf5File::open(&self.filename, true)?;
        let group = "/FieldData/FD";
        file.create_group(group)?;
        let values: Vec<f32> = frequencies.iter().map(|&value| value as f32).collect();
        file.write_f32_attribute_vector(group, "frequency", &values)
    }
}

/// Flatten a field array into `(n, i, j, k)` linear order.
/// Copy the selected mesh indices out of a full field array, preserving the
/// `(n, i, j, k)` component-major layout used by the writers.
fn extract_field(values: &FieldArray, indices: &[Vec<usize>; 3]) -> FieldArray {
    let dims = [indices[0].len(), indices[1].len(), indices[2].len()];
    let mut out = FieldArray::new("dump", dims);
    for n in 0..3 {
        for (ii, &i) in indices[0].iter().enumerate() {
            for (jj, &j) in indices[1].iter().enumerate() {
                for (kk, &k) in indices[2].iter().enumerate() {
                    *out.get_mut(n, ii, jj, kk) = *values.get(n, i, j, k);
                }
            }
        }
    }
    out
}
fn field_values(values: &FieldArray, dims: [usize; 3]) -> Vec<f32> {
    let [nx, ny, nz] = dims;
    let mut out = Vec::with_capacity(3 * nx * ny * nz);
    for n in 0..3 {
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    out.push(*values.get(n, i, j, k));
                }
            }
        }
    }
    out
}

/// VTK file writer for field dumps.
/// Mirrors the C++ `VTK_File_Writer` concept.
#[allow(dead_code)]
pub struct VTKFileWriter {
    filename_prefix: String,
    precision: u32,
}

impl VTKFileWriter {
    pub fn new(filename_prefix: &str) -> Self {
        Self {
            filename_prefix: filename_prefix.to_string(),
            precision: 12,
        }
    }

    pub fn get_filename_prefix(&self) -> &str {
        &self.filename_prefix
    }

    pub fn set_precision(&mut self, val: u32) {
        self.precision = val;
    }

    pub fn get_precision(&self) -> u32 {
        self.precision
    }

    /// Write field data to VTK legacy format file.
    /// Writes a STRUCTURED_POINTS dataset with the field as point data.
    pub fn write_field(&self, field_name: &str, data: &[f64]) {
        use std::io::Write;
        let n = data.len();
        let filename = format!("{}.vtk", self.filename_prefix);
        let mut file = std::fs::File::create(&filename).expect("failed to create VTK file");
        let mut w = |s: &str| file.write_all(s.as_bytes()).unwrap();
        w("# vtk DataFile Version 3.0\n");
        w("Field data from openEMS\n");
        w("ASCII\n");
        w("DATASET STRUCTURED_POINTS\n");
        w(&format!("DIMENSIONS {} 1 1\n", n));
        w("ORIGIN 0 0 0\n");
        w("SPACING 1 1 1\n");
        w(&format!("POINT_DATA {}\n", n));
        w("FIELD FieldData 1\n");
        w(&format!("{} 1 {} double\n", field_name, n));
        for (i, &v) in data.iter().enumerate() {
            if i > 0 {
                w(" ");
            }
            w(&format!("{:.precision$}", v, precision = self.precision as usize));
        }
        w("\n");
    }

    /// Write a real 3-D vector field on a rectilinear structured grid.
    /// Supports non-uniform meshes by emitting explicit point coordinates.
    pub fn write_structured_grid(
        &self,
        field_name: &str,
        dims: [usize; 3],
        coordinates: [&[f64]; 3],
        values: &FieldArray,
    ) -> std::io::Result<()> {
        use std::io::Write;
        let [nx, ny, nz] = dims;
        let points = nx.saturating_mul(ny).saturating_mul(nz);
        let filename = format!("{}.vtk", self.filename_prefix);
        let mut file = std::fs::File::create(&filename)?;
        let precision = self.precision as usize;
        writeln!(file, "# vtk DataFile Version 3.0")?;
        writeln!(file, "openEMS field data")?;
        writeln!(file, "ASCII")?;
        writeln!(file, "DATASET STRUCTURED_GRID")?;
        writeln!(file, "DIMENSIONS {nx} {ny} {nz}")?;
        writeln!(file, "POINTS {points} double")?;
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    writeln!(
                        file,
                        "{:.precision$} {:.precision$} {:.precision$}",
                        coordinates[0].get(i).copied().unwrap_or(0.0),
                        coordinates[1].get(j).copied().unwrap_or(0.0),
                        coordinates[2].get(k).copied().unwrap_or(0.0),
                        precision = precision
                    )?;
                }
            }
        }
        writeln!(file, "POINT_DATA {points}")?;
        writeln!(file, "VECTORS {field_name} double")?;
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    writeln!(
                        file,
                        "{:.precision$} {:.precision$} {:.precision$}",
                        *values.get(0, i, j, k),
                        *values.get(1, i, j, k),
                        *values.get(2, i, j, k),
                        precision = precision
                    )?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_probe_creation() {
        let probe = FieldProbe::new("probe1", [0.0, 0.0, 0.0]);
        assert_eq!(probe.get_name(), "probe1");
        assert_eq!(probe.get_processing_name(), "probe1");
        assert!(probe.get_enable());
        assert_eq!(probe.get_process_interval(), 0);
        assert_eq!(probe.get_weight(), 1.0);
        assert_eq!(probe.get_precision(), 12);
        assert_eq!(probe.get_mesh_type(), MeshType::Cartesian);
    }

    #[test]
    fn test_field_probe_process_no_panic() {
        let mut probe = FieldProbe::new("probe1", [0.0, 0.0, 0.0]);
        probe.process(0); // should not panic
        probe.process(100);
    }

    #[test]
    fn test_field_probe_set_name() {
        let mut probe = FieldProbe::new("probe1", [0.0, 0.0, 0.0]);
        probe.set_name("renamed");
        assert_eq!(probe.get_name(), "renamed");
    }

    #[test]
    fn test_field_probe_process_interval() {
        let mut probe = FieldProbe::new("probe1", [0.0, 0.0, 0.0]);
        probe.set_process_interval(10);
        assert_eq!(probe.get_process_interval(), 10);
        assert!(probe.check_timestep(10));
        assert!(!probe.check_timestep(11));
        assert!(!probe.check_timestep(9));
    }

    #[test]
    fn test_field_probe_process_interval_zero_disabled() {
        let mut probe = FieldProbe::new("probe1", [0.0, 0.0, 0.0]);
        probe.set_process_interval(0);
        // Interval 0 means interval-based processing is disabled.
        assert!(!probe.check_timestep(10));
    }

    #[test]
    fn test_field_probe_enable_disable() {
        let mut probe = FieldProbe::new("probe1", [0.0, 0.0, 0.0]);
        assert!(probe.get_enable());
        probe.set_enable(false);
        assert!(!probe.get_enable());
        assert_eq!(probe.process(0), -1);
        probe.set_enable(true);
        assert!(probe.get_enable());
    }

    #[test]
    fn test_field_probe_add_step() {
        let mut probe = FieldProbe::new("probe1", [0.0, 0.0, 0.0]);
        probe.add_step(5);
        probe.add_step(10);
        // check_timestep advances ps_pos on match (mirrors C++ behavior).
        assert!(probe.check_timestep(5));
        assert!(probe.check_timestep(10));
        assert!(!probe.check_timestep(7));
    }

    #[test]
    fn test_field_probe_add_steps_duplicate_ignored() {
        let mut probe = FieldProbe::new("probe1", [0.0, 0.0, 0.0]);
        probe.add_step(5);
        probe.add_step(5); // duplicate, should be ignored
        assert!(probe.check_timestep(5));
    }

    #[test]
    fn test_field_probe_add_steps_batch() {
        let mut probe = FieldProbe::new("probe1", [0.0, 0.0, 0.0]);
        probe.add_steps(&[3, 6, 9]);
        assert!(probe.check_timestep(3));
        assert!(probe.check_timestep(6));
        assert!(probe.check_timestep(9));
        assert!(!probe.check_timestep(4));
    }

    #[test]
    fn test_field_probe_start_stop_time() {
        let mut probe = FieldProbe::new("probe1", [0.0, 0.0, 0.0]);
        // With placeholder d_t = 1e-12: start_ts = 1000, stop_ts = 5000
        probe.set_process_start_stop_time(1e-9, 5e-9);
        // With no steps and no interval, check_timestep returns false.
        // Set an interval to enable time-based processing.
        probe.set_process_interval(100);
        assert!(!probe.check_timestep(500));
        assert!(probe.check_timestep(1000));
        assert!(probe.check_timestep(5000));
        assert!(!probe.check_timestep(5001));
    }

    #[test]
    fn test_field_probe_start_stop_invalid() {
        let mut probe = FieldProbe::new("probe1", [0.0, 0.0, 0.0]);
        // stop <= start should disable processing
        probe.set_process_start_stop_time(5e-9, 1e-9);
        assert_eq!(probe.start_ts, 0);
        assert_eq!(probe.stop_ts, u32::MAX);
    }

    #[test]
    fn test_field_probe_weight() {
        let mut probe = FieldProbe::new("probe1", [0.0, 0.0, 0.0]);
        assert!((probe.get_weight() - 1.0).abs() < 1e-12);
        probe.set_weight(2.5);
        assert!((probe.get_weight() - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_field_probe_precision() {
        let mut probe = FieldProbe::new("probe1", [0.0, 0.0, 0.0]);
        assert_eq!(probe.get_precision(), 12);
        probe.set_precision(6);
        assert_eq!(probe.get_precision(), 6);
    }

    #[test]
    fn test_field_probe_mesh_type() {
        let mut probe = FieldProbe::new("probe1", [0.0, 0.0, 0.0]);
        assert_eq!(probe.get_mesh_type(), MeshType::Cartesian);
        probe.set_mesh_type(MeshType::Cylindrical);
        assert_eq!(probe.get_mesh_type(), MeshType::Cylindrical);
    }

    #[test]
    fn test_field_probe_dual_mesh_dual_time() {
        let mut probe = FieldProbe::new("probe1", [0.0, 0.0, 0.0]);
        probe.set_dual_mesh(true);
        probe.set_dual_time(true);
        // No getters for these yet; just ensure no panic.
    }

    #[test]
    fn test_field_probe_dump_box_to_file() {
        let probe = FieldProbe::new("probe1", [0.0, 0.0, 0.0]);
        // Placeholder implementation should not panic.
        probe.dump_box_to_file("/tmp/test_probe", false);
        probe.dump_box_to_file("/tmp/test_probe_dual", true);
    }

    #[test]
    fn test_field_probe_flush_next() {
        let mut probe = FieldProbe::new("probe1", [0.0, 0.0, 0.0]);
        probe.flush_next(); // should not panic
        probe.flush_data(); // should not panic
    }

    #[test]
    fn test_processing_array_creation() {
        let array = ProcessingArray::new(100);
        assert_eq!(array.get_number_of_processings(), 0);
    }

    #[test]
    fn test_process_integral_creation() {
        let voltage = ProcessVoltage::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert_eq!(voltage.get_processing_name(), "voltage integration");
        assert_eq!(voltage.get_integral_name(0), "voltage");
        assert_eq!(voltage.get_number_of_integrals(), 1);
        assert_eq!(voltage.get_normal_dir(), 0);
    }

    #[test]
    fn test_process_integral_calc() {
        let voltage = ProcessVoltage::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert_eq!(voltage.calc_integral(), 0.0);
        let results = voltage.calc_multiple_integrals();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_process_integral_normal_dir() {
        let mut voltage = ProcessVoltage::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        voltage.set_normal_dir(1);
        assert_eq!(voltage.get_normal_dir(), 1);
    }

    #[test]
    fn test_process_current_creation() {
        let current = ProcessCurrent::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert_eq!(current.get_processing_name(), "current integration");
        assert_eq!(current.get_integral_name(0), "current");
        assert_eq!(current.get_number_of_integrals(), 1);
    }

    #[test]
    fn test_process_fields_creation() {
        let fields = ProcessFields::new("dump", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert_eq!(fields.get_processing_name(), "common field processing");
        assert_eq!(fields.get_dump_type(), 0); // E_FIELD_DUMP
        assert_eq!(fields.get_file_type(), VTK_FILETYPE);
    }

    #[test]
    fn test_process_fields_dump_type() {
        let mut fields = ProcessFields::new("dump", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        fields.set_dump_type(1); // H_FIELD_DUMP
        assert_eq!(fields.get_dump_type(), 1);
    }

    #[test]
    fn test_process_fields_file_type() {
        let mut fields = ProcessFields::new("dump", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        fields.set_file_type(HDF5_FILETYPE);
        assert_eq!(fields.get_file_type(), HDF5_FILETYPE);
    }

    #[test]
    fn test_process_fields_sub_sampling() {
        let mut fields = ProcessFields::new("dump", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        fields.set_sub_sampling(2, -1);
        fields.set_sub_sampling(3, 0);
        fields.set_opt_resolution(0.5, -1);
    }

    #[test]
    fn test_process_fields_dump_mode() {
        let mut fields = ProcessFields::new("dump", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        fields.set_dump_mode(0); // NODE_INTERPOLATE
        fields.set_dump_mode(1); // CELL_INTERPOLATE
        fields.set_dump_mode_2_node();
        fields.set_dump_mode_2_cell();
    }

    #[test]
    fn test_process_fields_conductivity_needs() {
        let fields = ProcessFields::new("dump", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!(fields.need_conductivity());
        assert!(fields.need_permittivity());
        assert!(fields.need_permeability());
    }

    #[test]
    fn test_process_fields_calc_total_energy() {
        let fields = ProcessFields::new("dump", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert_eq!(fields.calc_total_energy_estimate(), 0.0);
    }

    #[test]
    fn test_process_fields_field_name_by_type() {
        assert_eq!(ProcessFields::get_field_name_by_type(0), "E-Field");
        assert_eq!(ProcessFields::get_field_name_by_type(1), "H-Field");
        assert_eq!(ProcessFields::get_field_name_by_type(2), "J-Field");
        assert_eq!(ProcessFields::get_field_name_by_type(3), "RotH-Field");
        assert_eq!(ProcessFields::get_field_name_by_type(4), "D-Field");
        assert_eq!(ProcessFields::get_field_name_by_type(5), "B-Field");
        assert_eq!(ProcessFields::get_field_name_by_type(20), "SAR-local");
        assert_eq!(ProcessFields::get_field_name_by_type(21), "SAR_1g");
        assert_eq!(ProcessFields::get_field_name_by_type(22), "SAR_10g");
        assert_eq!(ProcessFields::get_field_name_by_type(29), "SAR_raw_data");
        assert_eq!(ProcessFields::get_field_name_by_type(99), "unknown field");
    }

    #[test]
    fn test_processing_array_add_processing() {
        let mut array = ProcessingArray::new(100);
        let probe = Box::new(FieldProbe::new("probe1", [0.0, 0.0, 0.0]));
        array.add_processing(probe);
        assert_eq!(array.get_number_of_processings(), 1);
    }

    #[test]
    fn test_processing_array_init_pre_post() {
        let mut array = ProcessingArray::new(100);
        let probe = Box::new(FieldProbe::new("probe1", [0.0, 0.0, 0.0]));
        array.add_processing(probe);
        array.init_all();
        array.pre_process();
        array.post_process(); // should not panic
    }

    #[test]
    fn test_processing_array_process() {
        let mut array = ProcessingArray::new(100);
        let probe = Box::new(FieldProbe::new("probe1", [0.0, 0.0, 0.0]));
        array.add_processing(probe);
        let next = array.process(0);
        // With no interval and no steps, returns max_interval (100).
        assert_eq!(next, 100);
    }

    #[test]
    fn test_processing_array_process_with_interval() {
        let mut array = ProcessingArray::new(1000);
        let mut probe = FieldProbe::new("probe1", [0.0, 0.0, 0.0]);
        probe.set_process_interval(10);
        array.add_processing(Box::new(probe));
        let next = array.process(5);
        // Next interval should be 10 - 5%10 = 5.
        assert_eq!(next, 5);
    }

    #[test]
    fn test_processing_array_flush_next_reset() {
        let mut array = ProcessingArray::new(100);
        let probe = Box::new(FieldProbe::new("probe1", [0.0, 0.0, 0.0]));
        array.add_processing(probe);
        array.flush_next(); // should not panic
        array.reset(); // should not panic
    }

    #[test]
    fn test_processing_array_dump_boxes_to_file() {
        let mut array = ProcessingArray::new(100);
        let probe = Box::new(FieldProbe::new("probe1", [0.0, 0.0, 0.0]));
        array.add_processing(probe);
        array.dump_boxes_to_file("/tmp/test_array"); // should not panic
    }

    #[test]
    fn test_processing_array_get_processing() {
        let mut array = ProcessingArray::new(100);
        let probe = Box::new(FieldProbe::new("probe1", [0.0, 0.0, 0.0]));
        array.add_processing(probe);
        let retrieved = array.get_processing(0);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().get_name(), "probe1");
        assert!(array.get_processing(1).is_none());
    }

    #[test]
    fn test_field_probe_default() {
        let probe = FieldProbe::default();
        assert_eq!(probe.get_name(), "probe");
        assert_eq!(probe.position, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_processing_trait_default_methods() {
        // Verify that a minimal impl only needs the required methods.
        struct MinimalProc;
        impl Processing for MinimalProc {
            fn get_name(&self) -> String {
                "minimal".to_string()
            }
            fn set_name(&mut self, _val: &str) {}
            fn get_processing_name(&self) -> String {
                "minimal".to_string()
            }
            fn set_process_interval(&mut self, _interval: u32) {}
            fn get_process_interval(&self) -> u32 {
                0
            }
        }
        let proc = MinimalProc;
        assert_eq!(proc.get_name(), "minimal");
        assert!(proc.get_enable());
        assert_eq!(proc.get_weight(), 1.0);
        assert_eq!(proc.get_precision(), 12);
        assert_eq!(proc.get_mesh_type(), MeshType::Cartesian);
    }

    #[test]
    fn test_process_fields_td_creation() {
        let td = ProcessFieldsTD::new("td_dump", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert_eq!(td.get_processing_name(), "time-domain field dump");
        assert_eq!(td.get_dump_type(), 0); // E_FIELD_DUMP
        assert_eq!(td.get_file_type(), VTK_FILETYPE);
        assert!(td.need_conductivity());
        assert!(td.need_permittivity());
        assert!(td.need_permeability());
    }

    #[test]
    fn test_process_fields_td_dump_type() {
        let mut td = ProcessFieldsTD::new("td_dump", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        td.set_dump_type(1); // H_FIELD_DUMP
        assert_eq!(td.get_dump_type(), 1);
    }

    #[test]
    fn test_process_fields_fd_creation() {
        let fd = ProcessFieldsFD::new("fd_dump", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert_eq!(fd.get_processing_name(), "frequency-domain field dump");
        assert_eq!(fd.get_fd_oversampling(), 4);
        assert_eq!(fd.get_frequencies().len(), 0);
    }

    #[test]
    fn test_process_fields_fd_frequency() {
        let mut fd = ProcessFieldsFD::new("fd_dump", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        fd.add_frequency(1.0e9);
        fd.add_frequency(2.0e9);
        assert_eq!(fd.get_frequencies(), &[1.0e9, 2.0e9]);
        fd.set_fd_oversampling(8);
        assert_eq!(fd.get_fd_oversampling(), 8);
    }

    #[test]
    fn test_process_fields_sar_creation() {
        let sar = ProcessFieldsSAR::new("sar_dump", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert_eq!(sar.get_processing_name(), "SAR field processing");
        assert_eq!(sar.get_dump_type(), 20); // SAR_LOCAL_DUMP
        assert!((sar.get_mass_density() - 1000.0).abs() < 1e-12);
        assert!(sar.need_conductivity());
        assert!(!sar.need_permittivity());
        assert!(!sar.need_permeability());
    }

    #[test]
    fn test_process_fields_sar_density() {
        let mut sar = ProcessFieldsSAR::new("sar_dump", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        sar.set_mass_density(1050.0);
        assert!((sar.get_mass_density() - 1050.0).abs() < 1e-12);
    }

    #[test]
    fn test_process_mode_match_creation() {
        let mm = ProcessModeMatch::new("mm", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0);
        assert_eq!(mm.get_processing_name(), "mode match");
        assert_eq!(mm.get_normal_dir(), 0);
        assert_eq!(mm.get_num_modes(), 1);
    }

    #[test]
    fn test_process_mode_match_modes() {
        let mut mm = ProcessModeMatch::new("mm", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0);
        mm.set_num_modes(3);
        mm.add_frequency(1.0e9);
        mm.add_frequency(2.0e9);
        assert_eq!(mm.get_num_modes(), 3);
        assert_eq!(mm.get_frequencies(), &[1.0e9, 2.0e9]);
    }

    #[test]
    fn test_nf2ff_creation() {
        let nf2ff = NF2FF::new("nf2ff", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert_eq!(nf2ff.get_processing_name(), "near-field to far-field");
        assert_eq!(nf2ff.center, [0.0, 0.0, 0.0]);
        assert_eq!(nf2ff.radius, 0.0);
    }

    #[test]
    fn test_nf2ff_center_radius() {
        let mut nf2ff = NF2FF::new("nf2ff", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        nf2ff.set_center([0.5, 0.5, 0.5]);
        nf2ff.set_radius(2.0);
        nf2ff.add_frequency(1.0e9);
        assert_eq!(nf2ff.center, [0.5, 0.5, 0.5]);
        assert_eq!(nf2ff.radius, 2.0);
        assert_eq!(nf2ff.get_frequencies(), &[1.0e9]);
    }

    #[test]
    fn test_sar_creation() {
        let sar = SAR::new("sar", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert_eq!(sar.get_processing_name(), "SAR calculation");
        assert!((sar.get_mass_density() - 1000.0).abs() < 1e-12);
        assert_eq!(sar.get_averaging_cube(), 0.0);
        assert!(sar.need_conductivity());
        assert!(!sar.need_permittivity());
        assert!(!sar.need_permeability());
    }

    #[test]
    fn test_sar_averaging_cube() {
        let mut sar = SAR::new("sar", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        sar.set_averaging_cube(0.022); // 1g cube ~22mm
        assert!((sar.get_averaging_cube() - 0.022).abs() < 1e-12);
    }

    #[test]
    fn test_hdf5_file_writer() {
        let mut writer = HDF5FileWriter::new("output.h5");
        assert_eq!(writer.get_filename(), "output.h5");
        assert_eq!(writer.get_precision(), 12);
        writer.set_precision(6);
        assert_eq!(writer.get_precision(), 6);
        // Without the `hdf5` feature every write reports the same explicit error
        // instead of silently producing a private format.
        #[cfg(not(feature = "hdf5"))]
        assert_eq!(
            writer.write_field("E-Field", &[1.0, 2.0, 3.0]),
            Err(Hdf5Error::Unsupported)
        );
    }

    #[test]
    fn test_vtk_file_writer() {
        let mut writer = VTKFileWriter::new("output");
        assert_eq!(writer.get_filename_prefix(), "output");
        assert_eq!(writer.get_precision(), 12);
        writer.set_precision(6);
        assert_eq!(writer.get_precision(), 6);
        // write_field should not panic (placeholder).
        writer.write_field("E-Field", &[1.0, 2.0, 3.0]);
    }
}

#[cfg(test)]
mod engine_interface_tests {
    use super::*;

    #[test]
    fn field_probe_samples_engine_fields() {
        let lines = vec![0.0, 1.0, 2.0];
        let meta = EngineMeta {
            num_lines: [3, 3, 3],
            disc_lines: [lines.clone(), lines.clone(), lines],
            dt: 1.0e-12,
        };
        let mut volt = FieldArray::new("volt", [3, 3, 3]);
        *volt.get_mut(0, 1, 1, 1) = 3.5;
        let curr = FieldArray::new("curr", [3, 3, 3]);
        let mut probe = FieldProbe::new("probe", [1.0, 1.0, 1.0]);
        let interface = EngineInterface {
            meta: &meta,
            volt: &volt,
            curr: &curr,
        };
        probe.process_fields(1, &interface);
        assert_eq!(probe.samples(), &[3.5]);
        // A position outside the mesh records nothing.
        probe.set_name("outside");
        let mut outside = FieldProbe::new("outside", [99.0, 0.0, 0.0]);
        outside.process_fields(2, &interface);
        assert!(outside.samples().is_empty());
    }

    #[test]
    fn field_probe_accumulates_dft() {
        let lines = vec![0.0, 1.0, 2.0];
        let meta = EngineMeta {
            num_lines: [3, 3, 3],
            disc_lines: [lines.clone(), lines.clone(), lines],
            dt: 1.0e-12,
        };
        let mut volt = FieldArray::new("volt", [3, 3, 3]);
        let curr = FieldArray::new("curr", [3, 3, 3]);
        let mut probe = FieldProbe::new("probe", [1.0, 1.0, 1.0]);
        probe.add_frequency(10.0e9);
        probe.add_frequency(5.0e9);
        probe.add_frequency(10.0e9); // duplicate ignored
        assert_eq!(probe.frequencies().len(), 2);
        assert_eq!(probe.fd_oversampling(), 1);

        let signal_frequency = 10.0e9;
        let steps = 1_000usize; // 10 periods at dt = 1 ps
        for step in 0..steps {
            let time = step as f64 * meta.dt;
            *volt.get_mut(0, 1, 1, 1) =
                (2.0 * std::f64::consts::PI * signal_frequency * time).sin() as f32;
            let interface = EngineInterface {
                meta: &meta,
                volt: &volt,
                curr: &curr,
            };
            probe.process_fields(step as u32, &interface);
        }

        assert_eq!(probe.samples().len(), steps);
        let on_bin = probe.dft()[0].magnitude();
        let off_bin = probe.dft()[1].magnitude();
        assert!(
            on_bin > 100.0 * off_bin,
            "DFT peak not at the signal frequency: on={on_bin:e}, off={off_bin:e}"
        );
        let expected = steps as f64 / 2.0;
        assert!(
            (on_bin - expected).abs() / expected < 1.0e-3,
            "DFT amplitude {on_bin:e} vs expected {expected:e}"
        );

        probe.reset();
        assert!(probe.samples().is_empty());
        assert_eq!(probe.dft()[0], ComplexPoint::default());
        assert_eq!(probe.frequencies().len(), 2);
    }
}

#[cfg(test)]
mod vtk_tests {
    use super::*;

    #[test]
    fn process_fields_td_writes_real_vtk() {
        let prefix = std::env::temp_dir().join(format!("oems_td_{}", std::process::id()));
        let prefix = prefix.to_string_lossy().to_string();
        let lines = vec![0.0, 1.0, 2.0];
        let meta = EngineMeta {
            num_lines: [3, 3, 3],
            disc_lines: [lines.clone(), lines.clone(), lines],
            dt: 1.0e-12,
        };
        let mut volt = FieldArray::new("volt", [3, 3, 3]);
        *volt.get_mut(0, 1, 1, 1) = 5.0;
        *volt.get_mut(1, 1, 1, 1) = 6.0;
        *volt.get_mut(2, 1, 1, 1) = 7.0;
        let curr = FieldArray::new("curr", [3, 3, 3]);
        let interface = EngineInterface {
            meta: &meta,
            volt: &volt,
            curr: &curr,
        };
        let mut dump = ProcessFieldsTD::new(&prefix, [0.0; 3], [2.0; 3]);
        dump.process_fields(1, &interface);

        let filename = format!("{prefix}_0000000001.vtk");
        let content = std::fs::read_to_string(&filename).expect("VTK dump should exist");
        assert!(content.contains("DATASET STRUCTURED_GRID"));
        assert!(content.contains("DIMENSIONS 3 3 3"));
        assert!(content.contains("POINTS 27 double"));
        assert!(content.contains("VECTORS E-Field double"));
        assert!(content.contains("5.000000000000"));
        assert!(content.contains("7.000000000000"));
        let _ = std::fs::remove_file(filename);
    }
}

#[cfg(test)]
mod fd_dump_tests {
    use super::*;

    fn meta(dt: f64) -> EngineMeta {
        let lines = vec![0.0, 1.0, 2.0];
        EngineMeta {
            num_lines: [3, 3, 3],
            disc_lines: [lines.clone(), lines.clone(), lines],
            dt,
        }
    }

    #[test]
    fn frequency_domain_accumulates_a_known_tone() {
        let meta = meta(1.0e-12);
        let mut volt = FieldArray::new("volt", [3, 3, 3]);
        let curr = FieldArray::new("curr", [3, 3, 3]);
        let mut dump = ProcessFieldsFD::new("fd", [0.0; 3], [2.0; 3]);
        dump.add_frequency(10.0e9);
        dump.add_frequency(5.0e9);
        dump.add_frequency(10.0e9); // duplicate ignored
        assert_eq!(dump.get_frequencies().len(), 2);

        let signal = 10.0e9;
        let steps = 2_000u32;
        for step in 0..steps {
            let time = step as f64 * meta.dt;
            *volt.get_mut(0, 1, 1, 1) =
                (2.0 * std::f64::consts::PI * signal * time).sin() as f32;
            let interface = EngineInterface {
                meta: &meta,
                volt: &volt,
                curr: &curr,
            };
            dump.process_fields(step, &interface);
        }

        let samples = dump.sample_count();
        assert!(samples > 100, "too few samples: {samples}");
        let interval = dump.fd_interval() as f64;
        let expected = (samples as f64) * meta.dt * interval;
        let re = dump.real_part(0).expect("real");
        let im = dump.imag_part(0).expect("imag");
        let position = 1 * 3 * 3 + 1 * 3 + 1;
        // Only the (1,1,1) cell is non-zero.
        assert!(re[0].abs() < 1e-20);
        let magnitude = (re[position] as f64).hypot(im[position] as f64);
        assert!(
            (magnitude - expected).abs() / expected < 1.0e-3,
            "magnitude {magnitude:e} vs expected {expected:e}"
        );
        // The off-frequency bin is far smaller.
        let off_re = dump.real_part(1).unwrap();
        let off_im = dump.imag_part(1).unwrap();
        let off = (off_re[position] as f64).hypot(off_im[position] as f64);
        assert!(off < magnitude * 1.0e-2, "off bin {off:e} vs {magnitude:e}");
    }

    #[test]
    fn fd_interval_follows_nyquist_and_oversampling() {
        let mut dump = ProcessFieldsFD::new("fd", [0.0; 3], [1.0; 3]);
        dump.set_fd_oversampling(4);
        dump.add_frequency(10.0e9);
        let meta = meta(1.0e-12);
        let volt = FieldArray::new("volt", [3, 3, 3]);
        let curr = FieldArray::new("curr", [3, 3, 3]);
        let interface = EngineInterface {
            meta: &meta,
            volt: &volt,
            curr: &curr,
        };
        dump.process_fields(0, &interface);
        // nyquist = 1/(2*10e9*1e-12) = 50 steps; /4 = 12.5 -> 13
        assert_eq!(dump.fd_interval(), 13);
        dump.reset();
        assert_eq!(dump.sample_count(), 0);
        assert_eq!(dump.fd_interval(), 0);
    }

    #[test]
    fn frequency_labels_match_native_formatting() {
        assert_eq!(ProcessFieldsFD::frequency_label(1.0e9), "1000000000");
        assert_eq!(ProcessFieldsFD::frequency_label(2.5e9), "2500000000");
    }

    #[test]
    fn fd_vtk_dump_writes_magnitude_phase_and_snapshots() {
        let directory = std::env::temp_dir().join(format!("oems_fd_{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let prefix = directory.join("fd");
        let prefix = prefix.to_string_lossy().to_string();
        let meta = meta(1.0e-12);
        let mut volt = FieldArray::new("volt", [3, 3, 3]);
        let curr = FieldArray::new("curr", [3, 3, 3]);
        let mut dump = ProcessFieldsFD::new(&prefix, [0.0; 3], [2.0; 3]);
        dump.add_frequency(1.0e9);
        for step in 0..200u32 {
            *volt.get_mut(0, 1, 1, 1) = (step as f32 * 0.1).sin();
            let interface = EngineInterface {
                meta: &meta,
                volt: &volt,
                curr: &curr,
            };
            dump.process_fields(step, &interface);
        }
        dump.post_process();
        let abs = format!("{prefix}_f=1000000000_abs.vtk");
        let arg = format!("{prefix}_f=1000000000_arg.vtk");
        let phase = format!("{prefix}_f=1000000000_p=000.vtk");
        for path in [&abs, &arg, &phase] {
            assert!(std::path::Path::new(path).exists(), "missing {path}");
            let text = std::fs::read_to_string(path).unwrap();
            assert!(text.contains("DATASET STRUCTURED_GRID"));
            assert!(text.contains("VECTORS E-Field double"));
            let _ = std::fs::remove_file(path);
        }
        let _ = std::fs::remove_dir_all(&directory);
    }
}

#[cfg(all(test, feature = "hdf5"))]
mod hdf5_dump_tests {
    use super::*;

    fn meta(dt: f64) -> EngineMeta {
        let lines = vec![0.0, 1.0, 2.0];
        EngineMeta {
            num_lines: [3, 3, 3],
            disc_lines: [lines.clone(), lines.clone(), lines],
            dt,
        }
    }

    fn temp_prefix(name: &str) -> String {
        std::env::temp_dir()
            .join(format!("oems_{name}_{}", std::process::id()))
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn time_domain_dump_writes_native_hdf5_layout() {
        let prefix = temp_prefix("td_h5");
        let meta = meta(1.0e-12);
        let mut volt = FieldArray::new("volt", [3, 3, 3]);
        let curr = FieldArray::new("curr", [3, 3, 3]);
        for n in 0..3 {
            *volt.get_mut(n, 1, 1, 1) = n as f32 + 1.0;
        }
        let mut dump = ProcessFieldsTD::new(&prefix, [0.0; 3], [2.0; 3]);
        dump.set_file_type(HDF5_FILETYPE);
        let interface = EngineInterface {
            meta: &meta,
            volt: &volt,
            curr: &curr,
        };
        dump.process_fields(1, &interface);
        dump.process_fields(2, &interface);

        let path = format!("{prefix}.h5");
        let file = Hdf5File::open(&path, false).expect("open dump");
        assert_eq!(
            file.read_f64_attribute("/", "openEMS_HDF5_version").unwrap(),
            OPENEMS_HDF5_VERSION
        );
        assert_eq!(file.read_i32_attribute("/", "dump_type").unwrap(), 0);
        assert!(file.read_bool_attribute("/", "legacy_fmt").unwrap());
        let (dims, x) = file.read_f64_dataset("/Mesh/x").unwrap();
        assert_eq!(dims, vec![3]);
        assert_eq!(x, vec![0.0, 1.0, 2.0]);
        assert_eq!(file.read_i32_attribute("/Mesh", "mesh_type").unwrap(), 0);

        let (dims, values) = file.read_f32_dataset("/FieldData/TD/00000001").unwrap();
        // legacy NZYX ordering: [3][nz][ny][nx]
        assert_eq!(dims, vec![3, 3, 3, 3]);
        // The (1,1,1) cell of component 0 holds 1.0; index = n*27 + k*9 + j*3 + i
        let index = 0 * 27 + 1 * 9 + 1 * 3 + 1;
        assert_eq!(values[index], 1.0);
        let component_y = 1 * 27 + 1 * 9 + 1 * 3 + 1;
        assert_eq!(values[component_y], 2.0);
        let component_z = 2 * 27 + 1 * 9 + 1 * 3 + 1;
        assert_eq!(values[component_z], 3.0);
        assert_eq!(
            file.read_string_attribute("/FieldData/TD/00000001", "d_order").unwrap(),
            "NZYX"
        );
        let time = file
            .read_f32_attribute_vector("/FieldData/TD/00000001", "time")
            .unwrap();
        assert!((time[0] - 1.0e-12f32).abs() < 1.0e-18);
        assert!(file.exists("/FieldData/TD/00000002"));
        drop(file);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn frequency_domain_dump_writes_native_hdf5_layout() {
        let prefix = temp_prefix("fd_h5");
        let meta = meta(1.0e-12);
        let mut volt = FieldArray::new("volt", [3, 3, 3]);
        let curr = FieldArray::new("curr", [3, 3, 3]);
        let mut dump = ProcessFieldsFD::new(&prefix, [0.0; 3], [2.0; 3]);
        dump.set_file_type(HDF5_FILETYPE);
        dump.add_frequency(1.0e9);
        for step in 0..400u32 {
            let time = step as f64 * meta.dt;
            *volt.get_mut(2, 2, 1, 1) =
                (2.0 * std::f64::consts::PI * 1.0e9 * time).sin() as f32;
            let interface = EngineInterface {
                meta: &meta,
                volt: &volt,
                curr: &curr,
            };
            dump.process_fields(step, &interface);
        }
        dump.post_process();

        let path = format!("{prefix}.h5");
        let file = Hdf5File::open(&path, false).expect("open dump");
        let frequencies = file
            .read_f32_attribute_vector("/FieldData/FD", "frequency")
            .unwrap();
        assert_eq!(frequencies, vec![1.0e9]);
        let (dims, re) = file
            .read_f32_dataset("/FieldData/FD/f0_real")
            .expect("legacy real dataset");
        assert_eq!(dims, vec![3, 3, 3, 3]);
        assert!(re.iter().all(|value| value.abs() < 1.0e-6));
        let (imag_dims, im) = file
            .read_f32_dataset("/FieldData/FD/f0_imag")
            .expect("legacy imaginary dataset");
        assert_eq!(imag_dims, dims);
        assert_eq!(im.len(), re.len());
        assert!(
            im.iter().any(|value| value.abs() > 0.0),
            "the sine tone must accumulate a non-zero imaginary part"
        );
        assert_eq!(
            file.read_f64_attribute("/FieldData/FD/f0_real", "frequency")
                .unwrap(),
            1.0e9
        );
        drop(file);
        let _ = std::fs::remove_file(path);
    }
}

#[cfg(test)]
mod dump_scheduling_tests {
    use super::*;

    fn meta() -> EngineMeta {
        let lines = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        EngineMeta {
            num_lines: [5, 5, 5],
            disc_lines: [lines.clone(), lines.clone(), lines],
            dt: 1.0e-12,
        }
    }

    fn prefix(name: &str) -> String {
        std::env::temp_dir()
            .join(format!("oems_{name}_{}", std::process::id()))
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn sub_sampling_and_optical_resolution_reduce_the_dump_mesh() {
        let lines = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let mut dump = ProcessFieldsTD::new("d", [0.0; 3], [4.0; 3]);
        assert_eq!(dump.dump_indices(0, &lines), vec![0, 1, 2, 3, 4]);

        dump.set_sub_sampling(2, -1);
        assert_eq!(dump.dump_indices(0, &lines), vec![0, 2, 4]);
        dump.set_sub_sampling(3, 1);
        assert_eq!(dump.dump_indices(1, &lines), vec![0, 3]);
        assert_eq!(dump.dump_indices(0, &lines), vec![0, 2, 4]);

        dump.set_opt_resolution(2.5, -1);
        assert_eq!(dump.dump_indices(0, &lines), vec![0, 3]);
        assert_eq!(dump.sub_sampling, [1, 1, 1]);
    }

    #[test]
    fn time_domain_dump_writes_the_native_timestep_sequence() {
        let prefix = prefix("td_seq");
        let meta = meta();
        let mut volt = FieldArray::new("volt", [5, 5, 5]);
        *volt.get_mut(2, 2, 2, 2) = 4.0;
        let curr = FieldArray::new("curr", [5, 5, 5]);
        let interface = EngineInterface {
            meta: &meta,
            volt: &volt,
            curr: &curr,
        };
        let mut dump = ProcessFieldsTD::new(&prefix, [0.0; 3], [4.0; 3]);
        dump.set_process_interval(2);
        dump.process_fields(1, &interface);
        dump.process_fields(2, &interface);
        dump.process_fields(3, &interface);

        let eleven = format!("{prefix}_0000000002.vtk");
        assert!(std::path::Path::new(&eleven).exists(), "missing {eleven}");
        let four = format!("{prefix}_0000000004.vtk");
        assert!(!std::path::Path::new(&four).exists());
        let _ = std::fs::remove_file(eleven);
    }

    #[test]
    fn time_domain_dump_sub_sampling_changes_the_written_grid() {
        let prefix = prefix("td_sub");
        let meta = meta();
        let mut volt = FieldArray::new("volt", [5, 5, 5]);
        *volt.get_mut(2, 4, 4, 4) = 9.0;
        let curr = FieldArray::new("curr", [5, 5, 5]);
        let interface = EngineInterface {
            meta: &meta,
            volt: &volt,
            curr: &curr,
        };
        let mut dump = ProcessFieldsTD::new(&prefix, [0.0; 3], [4.0; 3]);
        dump.set_sub_sampling(2, -1);
        dump.process_fields(1, &interface);

        let filename = format!("{prefix}_0000000001.vtk");
        let text = std::fs::read_to_string(&filename).expect("dump exists");
        assert!(text.contains("DIMENSIONS 3 3 3"), "{text}");
        assert!(text.contains("POINTS 27 double"));
        assert!(text.contains("9.000000000000"));
        let _ = std::fs::remove_file(filename);
    }
}
