//! Processing: post-processing (probes, dumps, SAR, NF2FF interface).
//! Mirrors `Common/processing.h/cpp` and subclasses in the C++ codebase.

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
    pub fn new(name: &str, position: [f64; 3]) -> Self {
        Self {
            name: name.to_string(),
            processing_name: name.to_string(),
            position,
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

    fn set_fd_oversampling(&mut self, _factor: u32) {
        // Placeholder: frequency-domain sampling not yet implemented.
    }

    fn add_frequency(&mut self, _freq: f64) {
        // Placeholder: frequency-domain sampling not yet implemented.
    }

    fn add_frequencies(&mut self, _freqs: &[f64]) {
        // Placeholder: frequency-domain sampling not yet implemented.
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
        // Placeholder: no pre-processing needed for a point probe.
    }

    fn process(&mut self, num_ts: u32) -> i32 {
        if !self.enabled {
            return -1;
        }
        // Placeholder: sample fields at position.
        // Field access will be wired up when the engine exposes field getters.
        self.get_next_interval(num_ts)
    }

    fn post_process(&mut self) {
        // Placeholder: no post-processing needed for a point probe.
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
            file_type: 1, // VTK_FILETYPE
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
        match type_ {
            0 => "E-Field".to_string(),
            1 => "H-Field".to_string(),
            2 => "J-Field".to_string(),
            3 => "D-Field".to_string(),
            4 => "B-Field".to_string(),
            5 => "B-Field".to_string(),
            20 => "SAR-Local".to_string(),
            _ => "Unknown".to_string(),
        }
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
}

impl ProcessFieldsTD {
    pub fn new(name: &str, start: [f64; 3], stop: [f64; 3]) -> Self {
        Self {
            name: name.to_string(),
            processing_name: "time-domain field dump".to_string(),
            start,
            stop,
            dump_type: 0, // E_FIELD_DUMP
            file_type: 1, // VTK_FILETYPE
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
        match type_ {
            0 => "E-Field".to_string(),
            1 => "H-Field".to_string(),
            2 => "J-Field".to_string(),
            3 => "D-Field".to_string(),
            4 => "B-Field".to_string(),
            5 => "B-Field".to_string(),
            20 => "SAR-Local".to_string(),
            _ => "Unknown".to_string(),
        }
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

    fn set_process_interval(&mut self, _interval: u32) {}

    fn get_process_interval(&self) -> u32 {
        0
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
}

impl ProcessFieldsFD {
    pub fn new(name: &str, start: [f64; 3], stop: [f64; 3]) -> Self {
        Self {
            name: name.to_string(),
            processing_name: "frequency-domain field dump".to_string(),
            start,
            stop,
            dump_type: 0, // E_FIELD_DUMP
            file_type: 1, // VTK_FILETYPE
            fd_oversampling: 4,
            frequencies: Vec::new(),
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

    pub fn get_fd_oversampling(&self) -> u32 {
        self.fd_oversampling
    }

    pub fn set_fd_oversampling(&mut self, factor: u32) {
        self.fd_oversampling = factor;
    }

    pub fn add_frequency(&mut self, freq: f64) {
        self.frequencies.push(freq);
    }

    pub fn get_frequencies(&self) -> &[f64] {
        &self.frequencies
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

    fn set_process_interval(&mut self, _interval: u32) {}

    fn get_process_interval(&self) -> u32 {
        0
    }

    fn set_fd_oversampling(&mut self, factor: u32) {
        self.fd_oversampling = factor;
    }

    fn add_frequency(&mut self, freq: f64) {
        self.frequencies.push(freq);
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
            file_type: 1,  // VTK_FILETYPE
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
/// Mirrors the C++ `HDF5_File_Writer` concept.
#[allow(dead_code)]
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

    /// Write field data to HDF5 file (placeholder: no-op).
    pub fn write_field(&self, _field_name: &str, _data: &[f64]) {
        // Placeholder: HDF5 backend not yet linked.
    }
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

    /// Write field data to VTK file (placeholder: no-op).
    pub fn write_field(&self, _field_name: &str, _data: &[f64]) {
        // Placeholder: VTK backend not yet linked.
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
        assert_eq!(fields.get_file_type(), 1); // VTK_FILETYPE
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
        fields.set_file_type(0); // HDF5_FILETYPE
        assert_eq!(fields.get_file_type(), 0);
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
        assert_eq!(ProcessFields::get_field_name_by_type(5), "B-Field");
        assert_eq!(ProcessFields::get_field_name_by_type(20), "SAR-Local");
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
        assert_eq!(td.get_file_type(), 1); // VTK_FILETYPE
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
        // write_field should not panic (placeholder).
        writer.write_field("E-Field", &[1.0, 2.0, 3.0]);
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
