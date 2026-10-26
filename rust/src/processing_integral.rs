//! Time- and frequency-domain integral probes.
//!
//! Mirrors `Common/processintegral.{h,cpp}`. Concrete integrals implement
//! [`IntegralCalculator`] (native `CalcMultipleIntegrals`).

use crate::processing::{ComplexPoint, EngineInterface, EngineMeta, Processing};

/// Format a sample the way C++ `setprecision(p)` does (general format, `p`
/// significant digits).
///
/// Rust's `Display` for floats always uses fixed notation, which would quantise
/// a 1.44e-10 time step to 1 ps and break the uniform sampling grid the
/// Octave/MATLAB FFT assumes.
fn format_sample(value: f64, precision: usize) -> String {
    if value == 0.0 {
        return "0".to_string();
    }
    format!("{:.*e}", precision.saturating_sub(1), value)
}

/// One integral quantity computed from the running fields.
///
/// Mirrors the `ProcessIntegral::CalcMultipleIntegrals` override point in the
/// native code: each concrete integral contributes one or more rows.
pub trait IntegralCalculator: Send {
    /// Number of scalar results produced per sample.
    fn number_of_integrals(&self) -> usize;

    /// Column name written to the TD/FD file header.
    fn integral_name(&self, row: usize) -> String;

    /// Compute the current value of every integral.
    fn calc(&mut self, num_ts: u32, interface: &EngineInterface<'_>) -> Vec<f64>;
}

/// Time- and frequency-domain integral probe.
///
/// Mirrors `Common/processintegral.{h,cpp}`. The time-domain rows are streamed to
/// `<filename>`, the running DFT is written to `<filename>_FD` in `post_process`.
pub struct ProcessIntegral {
    name: String,
    processing_name: String,
    filename: String,
    start: [f64; 3],
    stop: [f64; 3],
    weight: f64,
    precision: usize,
    process_interval: u32,
    fd_interval: u32,
    fd_oversampling: u32,
    fd_samples: Vec<f64>,
    fd_results: Vec<Vec<ComplexPoint>>,
    results: Vec<f64>,
    sample_count: u32,
    norm_dir: i32,
    enabled: bool,
    header_written: bool,
    /// Sample the current at the half timestep (native `SetDualTime`).
    dual_time: bool,
    /// Nyquist step count of the highest excited frequency (native
    /// `m_FD_Nyquist`); caps the per-frequency DFT interval.
    fd_nyquist: u32,
    calculator: Box<dyn IntegralCalculator>,
}

impl ProcessIntegral {
    pub fn new(
        name: &str,
        filename: &str,
        start: [f64; 3],
        stop: [f64; 3],
        calculator: Box<dyn IntegralCalculator>,
    ) -> Self {
        let rows = calculator.number_of_integrals();
        Self {
            name: name.to_string(),
            processing_name: "integral".to_string(),
            filename: filename.to_string(),
            start,
            stop,
            weight: 1.0,
            precision: 12,
            process_interval: 1,
            fd_interval: 0,
            // Native probeFDOverSampling defaults to 1 (plain Nyquist rate),
            // independent of the time-domain OverSampling.
            fd_oversampling: 1,
            fd_samples: Vec::new(),
            fd_results: Vec::new(),
            results: vec![0.0; rows],
            sample_count: 0,
            norm_dir: -1,
            enabled: true,
            header_written: false,
            dual_time: false,
            fd_nyquist: 0,
            calculator,
        }
    }

    /// Cap the DFT interval at the excitation Nyquist (native
    /// `Processing::AddFrequency`).
    pub fn set_fd_nyquist(&mut self, nyquist: u32) {
        self.fd_nyquist = nyquist;
    }

    pub fn get_filename(&self) -> &str {
        &self.filename
    }

    pub fn set_filename(&mut self, val: &str) {
        self.filename = val.to_string();
    }

    pub fn set_weight(&mut self, val: f64) {
        self.weight = val;
    }

    pub fn get_weight(&self) -> f64 {
        self.weight
    }

    pub fn set_precision(&mut self, val: usize) {
        self.precision = val.max(1);
    }

    pub fn get_precision(&self) -> usize {
        self.precision
    }

    pub fn set_normal_dir(&mut self, dir: i32) {
        self.norm_dir = dir;
    }

    pub fn get_normal_dir(&self) -> i32 {
        self.norm_dir
    }

    /// Sample at the half timestep (native `SetDualTime`).
    pub fn set_dual_time(&mut self, val: bool) {
        self.dual_time = val;
    }

    pub fn get_dual_time(&self) -> bool {
        self.dual_time
    }

    /// Latest integral values (before the weight is applied).
    pub fn results(&self) -> &[f64] {
        &self.results
    }

    /// Running DFT for one row: `(frequency, accumulator)`.
    pub fn fd_results(&self, row: usize) -> Option<&[ComplexPoint]> {
        self.fd_results.get(row).map(|values| values.as_slice())
    }

    pub fn fd_interval(&self) -> u32 {
        self.fd_interval
    }

    /// Sampling interval from the Nyquist rate of the highest frequency and the
    /// oversampling factor, as in `Processing::AddFrequency`.
    fn update_fd_interval(&mut self, dt: f64) {
        let highest = self.fd_samples.iter().copied().fold(0.0f64, f64::max);
        if highest <= 0.0 || dt <= 0.0 {
            self.fd_interval = 1;
            return;
        }
        let frequency_nyquist = (1.0 / (2.0 * highest * dt)).floor() as u32;
        let nyquist = if self.fd_nyquist > 0 {
            frequency_nyquist.min(self.fd_nyquist)
        } else {
            frequency_nyquist
        };
        let oversampling = self.fd_oversampling.max(1);
        self.fd_interval = (nyquist / oversampling).max(1);
    }

    fn write_header(&mut self) {
        let mut text = String::new();
        text.push_str(&format!(
            "% time-domain {} '{}' by openEMS (Rust)\n",
            self.processing_name, self.name
        ));
        text.push_str(&format!(
            "% start-coordinates: ({}, {}, {}) m\n",
            self.start[0], self.start[1], self.start[2]
        ));
        text.push_str(&format!(
            "% stop-coordinates: ({}, {}, {}) m\n",
            self.stop[0], self.stop[1], self.stop[2]
        ));
        text.push_str("% t/s");
        for row in 0..self.calculator.number_of_integrals() {
            text.push('\t');
            text.push_str(&self.calculator.integral_name(row));
        }
        text.push('\n');
        if let Err(error) = std::fs::write(&self.filename, text) {
            eprintln!(
                "[openEMS] Failed to create integral file '{}': {error}",
                self.filename
            );
        } else {
            self.header_written = true;
        }
    }

    fn append_td_row(&self, time: f64) {
        use std::io::Write;
        let Ok(mut file) = std::fs::OpenOptions::new()
            .append(true)
            .open(&self.filename)
        else {
            eprintln!(
                "[openEMS] Failed to append to integral file '{}'",
                self.filename
            );
            return;
        };
        let mut line = format_sample(time, self.precision);
        for value in &self.results {
            line.push('\t');
            line.push_str(&format_sample(value * self.weight, self.precision));
        }
        line.push('\n');
        if let Err(error) = file.write_all(line.as_bytes()) {
            eprintln!(
                "[openEMS] Failed to write integral row to '{}': {error}",
                self.filename
            );
        }
    }

    /// Write the frequency-domain table, mirroring `Dump_FD_Data`.
    pub fn dump_fd_data(&self, factor: f64) {
        if self.fd_samples.is_empty() {
            return;
        }
        let filename = format!("{}_FD", self.filename);
        let mut text = String::new();
        text.push_str(&format!(
            "% frequency-domain {} '{}' by openEMS (Rust)\n",
            self.processing_name, self.name
        ));
        text.push_str("% f/Hz");
        for row in 0..self.results.len() {
            text.push('\t');
            text.push_str(&self.calculator.integral_name(row));
            text.push('\t');
        }
        text.push('\n');
        text.push('%');
        for _ in 0..self.results.len() {
            text.push_str("\treal\timag");
        }
        text.push('\n');
        for (index, frequency) in self.fd_samples.iter().enumerate() {
            text.push_str(&format_sample(*frequency, self.precision));
            for row in 0..self.results.len() {
                let value = self
                    .fd_results
                    .get(row)
                    .and_then(|values| values.get(index))
                    .copied()
                    .unwrap_or_default();
                text.push('\t');
                text.push_str(&format_sample(value.re * factor, self.precision));
                text.push('\t');
                text.push_str(&format_sample(value.im * factor, self.precision));
            }
            text.push('\n');
        }
        if let Err(error) = std::fs::write(&filename, text) {
            eprintln!("[openEMS] Failed to write FD integral file '{filename}': {error}");
        }
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

    fn set_process_interval(&mut self, interval: u32) {
        self.process_interval = interval;
    }

    fn get_process_interval(&self) -> u32 {
        self.process_interval
    }

    fn set_enable(&mut self, val: bool) {
        self.enabled = val;
    }

    fn get_enable(&self) -> bool {
        self.enabled
    }

    fn set_fd_oversampling(&mut self, factor: u32) {
        self.fd_oversampling = factor.max(1);
    }

    fn add_frequency(&mut self, freq: f64) {
        if freq > 0.0 && !self.fd_samples.contains(&freq) {
            self.fd_samples.push(freq);
            for row in &mut self.fd_results {
                row.push(ComplexPoint::default());
            }
            while self.fd_results.len() < self.results.len() {
                self.fd_results
                    .push(vec![ComplexPoint::default(); self.fd_samples.len()]);
            }
        }
    }

    fn add_frequencies(&mut self, freqs: &[f64]) {
        for &freq in freqs {
            self.add_frequency(freq);
        }
    }

    fn init_process(&mut self) {
        self.header_written = false;
        if self.enabled {
            self.write_header();
        }
    }

    fn check_timestep(&mut self, num_ts: u32) -> bool {
        let td = self.process_interval > 0 && num_ts.is_multiple_of(self.process_interval);
        let fd = self.fd_interval > 0 && num_ts.is_multiple_of(self.fd_interval);
        td || fd
    }

    fn process_fields(&mut self, num_ts: u32, interface: &EngineInterface<'_>) -> i32 {
        if !self.enabled {
            return -1;
        }
        if !self.header_written {
            self.write_header();
        }
        self.results = self.calculator.calc(num_ts, interface);

        let time = (num_ts as f64 + if self.dual_time { 0.5 } else { 0.0 }) * interface.meta.dt;
        if self.process_interval > 0 && num_ts.is_multiple_of(self.process_interval) {
            self.append_td_row(time);
        }

        if !self.fd_samples.is_empty() {
            if self.fd_interval == 0 {
                self.update_fd_interval(interface.meta.dt);
            }
            if num_ts.is_multiple_of(self.fd_interval.max(1)) {
                if self.fd_results.is_empty() {
                    self.fd_results = vec![
                        vec![ComplexPoint::default(); self.fd_samples.len()];
                        self.results.len()
                    ];
                }
                let scale = 2.0 * interface.meta.dt * self.fd_interval.max(1) as f64;
                for (row, value) in self.results.iter().enumerate() {
                    if let Some(accumulators) = self.fd_results.get_mut(row) {
                        for (frequency, accumulator) in
                            self.fd_samples.iter().zip(accumulators.iter_mut())
                        {
                            let argument =
                                -2.0 * std::f64::consts::PI * frequency * time;
                            accumulator.re += value * self.weight * scale * argument.cos();
                            accumulator.im += value * self.weight * scale * argument.sin();
                        }
                    }
                }
                self.sample_count += 1;
            }
        }

        self.get_next_interval(num_ts)
    }

    fn get_next_interval(&self, num_ts: u32) -> i32 {
        let interval = self.process_interval.max(1);
        (interval as i32 - (num_ts % interval) as i32).max(1)
    }

    fn post_process(&mut self) {
        self.dump_fd_data(1.0);
    }

    fn reset(&mut self) {
        for row in &mut self.fd_results {
            row.fill(ComplexPoint::default());
        }
        self.sample_count = 0;
        self.header_written = false;
        self.fd_interval = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array_nijk::FieldArray;

    struct Sine {
        frequency: f64,
        dt: f64,
    }

    impl IntegralCalculator for Sine {
        fn number_of_integrals(&self) -> usize {
            1
        }

        fn integral_name(&self, _row: usize) -> String {
            "sin".to_string()
        }

        fn calc(&mut self, num_ts: u32, _interface: &EngineInterface<'_>) -> Vec<f64> {
            let time = num_ts as f64 * self.dt;
            vec![(2.0 * std::f64::consts::PI * self.frequency * time).sin()]
        }
    }

    fn meta(dt: f64) -> EngineMeta {
        let lines = vec![0.0, 1.0, 2.0];
        EngineMeta {
            num_lines: [3, 3, 3],
            disc_lines: [lines.clone(), lines.clone(), lines],
            dt,
            materials: None,
            mesh_type: crate::operator::MeshType::Cartesian,
        }
    }

    #[test]
    fn integral_streams_analytic_td_rows_and_matches_the_dft_sum() {
        let dt = 1.0e-12;
        let frequency = 10.0e9;
        let steps = 1_000u32;
        let filename = std::env::temp_dir()
            .join(format!("oems_integral_{}.txt", std::process::id()))
            .to_string_lossy()
            .into_owned();
        let calculator = Box::new(Sine { frequency, dt });
        let mut integral = ProcessIntegral::new("probe", &filename, [0.0; 3], [2.0; 3], calculator);
        integral.set_weight(2.0);
        integral.add_frequency(frequency);
        integral.add_frequency(frequency); // duplicate ignored
        integral.init_process();

        let meta = meta(dt);
        let volt = FieldArray::new("volt", [3, 3, 3]);
        let curr = FieldArray::new("curr", [3, 3, 3]);
        let interface = EngineInterface {
            meta: &meta,
            volt: &volt,
            curr: &curr,
        };
        for step in 1..=steps {
            integral.process_fields(step, &interface);
        }

        // Time-domain file: header plus one row per step.
        let text = std::fs::read_to_string(&filename).expect("TD file");
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines[0].contains("time-domain integral"));
        assert!(lines.iter().any(|line| line.starts_with("% t/s")));
        let data = &lines[4..];
        assert_eq!(data.len(), steps as usize);
        let first: Vec<f64> = data[0]
            .split_whitespace()
            .map(|value| value.parse().unwrap())
            .collect();
        assert!((first[0] - dt).abs() < 1e-18);
        let expected = 2.0 * (2.0 * std::f64::consts::PI * frequency * dt).sin();
        assert!((first[1] - expected).abs() < 1e-12, "{} vs {expected}", first[1]);

        // Frequency-domain accumulator compared with an independent sum.
        let interval = integral.fd_interval() as f64;
        let mut re = 0.0;
        let mut im = 0.0;
        for step in 1..=steps {
            if (step as u32) % interval as u32 != 0 {
                continue;
            }
            let time = step as f64 * dt;
            let value = 2.0 * (2.0 * std::f64::consts::PI * frequency * time).sin();
            let argument = -2.0 * std::f64::consts::PI * frequency * time;
            let scale = 2.0 * dt * interval;
            re += value * scale * argument.cos();
            im += value * scale * argument.sin();
        }
        let accumulator = integral.fd_results(0).expect("fd row")[0];
        assert!(
            (accumulator.re - re).abs() <= 1.0e-12 * re.abs().max(1.0),
            "re {} vs {re}",
            accumulator.re
        );
        assert!(
            (accumulator.im - im).abs() <= 1.0e-12 * im.abs().max(1.0),
            "im {} vs {im}",
            accumulator.im
        );

        integral.post_process();
        let fd_file = format!("{filename}_FD");
        let fd_text = std::fs::read_to_string(&fd_file).expect("FD file");
        assert!(fd_text.contains("frequency-domain integral"));
        let first_data = fd_text
            .lines()
            .skip(3)
            .next()
            .expect("FD data row")
            .split_whitespace()
            .next()
            .expect("frequency column")
            .parse::<f64>()
            .expect("numeric frequency");
        assert!((first_data - 10.0e9).abs() <= 1.0e-3);

        let _ = std::fs::remove_file(&filename);
        let _ = std::fs::remove_file(fd_file);
    }
}

/// Line integral of the electric voltage, `V = sum E_n dl_n`.
///
/// Mirrors `ProcessVoltage` / `Engine_Interface_FDTD::CalcVoltageIntegral`: the
/// start and stop points must differ on exactly one axis.
pub struct VoltageIntegral {
    pub start: [f64; 3],
    pub stop: [f64; 3],
    snapped: Option<([usize; 3], [usize; 3])>,
}

impl VoltageIntegral {
    pub fn new(start: [f64; 3], stop: [f64; 3]) -> Self {
        Self {
            start,
            stop,
            snapped: None,
        }
    }

    fn indices(&mut self, meta: &EngineMeta) -> Option<([usize; 3], [usize; 3])> {
        if let Some(indices) = self.snapped {
            return Some(indices);
        }
        let start = meta.snap_to_mesh(self.start, false, false);
        let stop = meta.snap_to_mesh(self.stop, false, false);
        self.snapped = Some((start, stop));
        Some((start, stop))
    }
}

impl IntegralCalculator for VoltageIntegral {
    fn number_of_integrals(&self) -> usize {
        1
    }

    fn integral_name(&self, _row: usize) -> String {
        "voltage".to_string()
    }

    fn calc(&mut self, _num_ts: u32, interface: &EngineInterface<'_>) -> Vec<f64> {
        let Some((start, stop)) = self.indices(interface.meta) else {
            return vec![0.0];
        };
        let differing = (0..3).filter(|&axis| start[axis] != stop[axis]).count();
        if differing != 1 {
            return vec![0.0];
        }
        let mut result = 0.0;
        for axis in 0..3 {
            if start[axis] < stop[axis] {
                let mut pos = start;
                while pos[axis] < stop[axis] {
                    result += *interface.volt.get(axis, pos[0], pos[1], pos[2]) as f64;
                    pos[axis] += 1;
                }
            } else if start[axis] > stop[axis] {
                let mut pos = stop;
                while pos[axis] < start[axis] {
                    result -= *interface.volt.get(axis, pos[0], pos[1], pos[2]) as f64;
                    pos[axis] += 1;
                }
            }
        }
        vec![result]
    }
}

/// Closed-loop integral of the current around a surface, `I = loop(H)`.
///
/// Mirrors `ProcessCurrent`: the start/stop box must be flat along `normal_dir`
/// (auto-detected when not given) and the loop runs around the perimeter.
pub struct CurrentIntegral {
    pub start: [f64; 3],
    pub stop: [f64; 3],
    pub normal_dir: i32,
    snapped: Option<([usize; 3], [usize; 3])>,
}

impl CurrentIntegral {
    pub fn new(start: [f64; 3], stop: [f64; 3], normal_dir: i32) -> Self {
        Self {
            start,
            stop,
            normal_dir,
            snapped: None,
        }
    }

    fn indices(&mut self, meta: &EngineMeta) -> Option<([usize; 3], [usize; 3])> {
        if let Some(indices) = self.snapped {
            return Some(indices);
        }
        // current probes live on the dual mesh (native `SetDualMesh`)
        let start = meta.snap_to_mesh(self.start, true, false);
        let stop = meta.snap_to_mesh(self.stop, true, false);
        self.snapped = Some((start, stop));
        Some((start, stop))
    }
}

impl IntegralCalculator for CurrentIntegral {
    fn number_of_integrals(&self) -> usize {
        1
    }

    fn integral_name(&self, _row: usize) -> String {
        "current".to_string()
    }

    fn calc(&mut self, _num_ts: u32, interface: &EngineInterface<'_>) -> Vec<f64> {
        let Some((start, stop)) = self.indices(interface.meta) else {
            return vec![0.0];
        };
        let limits = interface.meta.num_lines;
        let normal = if (0..3).contains(&self.normal_dir) {
            self.normal_dir as usize
        } else {
            match (0..3).find(|&axis| start[axis] == stop[axis]) {
                Some(axis) => axis,
                None => return vec![0.0],
            }
        };
        if start[normal] != stop[normal] {
            return vec![0.0];
        }
        let current = |component: usize, i: usize, j: usize, k: usize| -> f64 {
            if i >= limits[0] || j >= limits[1] || k >= limits[2] {
                0.0
            } else {
                *interface.curr.get(component, i, j, k) as f64
            }
        };
        let [s0, s1, s2] = start;
        let [e0, e1, e2] = stop;
        let mut total = 0.0;
        match normal {
            0 => {
                for i in (s1 + 1)..=e1 {
                    total += current(1, e0, i, s2);
                }
                for i in (s2 + 1)..=e2 {
                    total += current(2, e0, e1, i);
                }
                for i in (s1 + 1)..=e1 {
                    total -= current(1, s0, i, e2);
                }
                for i in (s2 + 1)..=e2 {
                    total -= current(2, s0, s1, i);
                }
            }
            1 => {
                for i in (s2 + 1)..=e2 {
                    total += current(2, s0, s1, i);
                }
                for i in (s0 + 1)..=e0 {
                    total += current(0, i, e1, e2);
                }
                for i in (s2 + 1)..=e2 {
                    total -= current(2, e0, e1, i);
                }
                for i in (s0 + 1)..=e0 {
                    total -= current(0, i, s1, s2);
                }
            }
            _ => {
                for i in (s0 + 1)..=e0 {
                    total += current(0, i, s1, s2);
                }
                for i in (s1 + 1)..=e1 {
                    total += current(1, e0, i, s2);
                }
                for i in (s0 + 1)..=e0 {
                    total -= current(0, i, e1, e2);
                }
                for i in (s1 + 1)..=e1 {
                    total -= current(1, s0, i, e2);
                }
            }
        }
        vec![total]
    }
}

#[cfg(test)]
mod port_integral_tests {
    use super::*;
    use crate::array_nijk::FieldArray;

    fn meta() -> EngineMeta {
        let lines = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        EngineMeta {
            num_lines: [5, 5, 5],
            disc_lines: [lines.clone(), lines.clone(), lines],
            dt: 1.0e-12,
            materials: None,
            mesh_type: crate::operator::MeshType::Cartesian,
        }
    }

    /// The native `SnapToMeshLine` on the Coax mesh.
    ///
    /// The reference values are taken from the native openEMS 0.37.0 probe
    /// headers for `combinedtests/Coax` (5 mm grid starting at -252.5 mm).
    #[test]
    fn snapping_reproduces_the_native_coax_indices() {
        let x: Vec<f64> = (0..101).map(|index| (-252.5 + 5.0 * index as f64) * 1e-3).collect();
        let z: Vec<f64> = (0..201).map(|index| (5.0 * index as f64) * 1e-3).collect();
        let meta = EngineMeta {
            num_lines: [101, 101, 201],
            disc_lines: [x.clone(), x, z],
            dt: 9.629166e-12,
                materials: None,
            mesh_type: crate::operator::MeshType::Cartesian,
            };
        // native voltage probe: (0.0975,-0.0025,0.5) -> [70,50,100] and
        // (0.2275,-0.0025,0.5) -> [96,50,100]
        assert_eq!(
            meta.snap_to_mesh([0.1, 0.0, 0.5], false, false),
            [70, 50, 100]
        );
        assert_eq!(
            meta.snap_to_mesh([0.23, 0.0, 0.5], false, false),
            [96, 50, 100]
        );
        // native current probe: (-0.1175,-0.1175,0.495) -> [27,27,99] and
        // (0.1125,0.1125,0.495) -> [73,73,99]
        assert_eq!(
            meta.snap_to_mesh([-0.115, -0.115, 0.5], true, false),
            [27, 27, 99]
        );
        assert_eq!(
            meta.snap_to_mesh([0.115, 0.115, 0.5], true, false),
            [73, 73, 99]
        );
        // dumps use the full mesh, so the last line is reachable
        assert_eq!(
            meta.snap_to_mesh([0.2525, 0.2525, 1.0], false, true),
            [100, 100, 200]
        );
    }

    #[test]
    fn voltage_integral_sums_the_line_voltages() {
        let meta = meta();
        let mut volt = FieldArray::new("volt", [5, 5, 5]);
        let curr = FieldArray::new("curr", [5, 5, 5]);
        let start = meta.snap_to_mesh([2.0, 1.0, 1.0], false, false);
        let stop = meta.snap_to_mesh([3.5, 1.0, 1.0], false, false);
        assert_eq!(start, [2, 1, 1]);
        assert_eq!(stop, [3, 1, 1]);
        for i in start[0]..stop[0] {
            *volt.get_mut(0, i, start[1], start[2]) = 5.0;
        }
        let interface = EngineInterface {
            meta: &meta,
            volt: &volt,
            curr: &curr,
        };
        // the native integral uses [start, stop)
        let mut voltage = VoltageIntegral::new([2.0, 1.0, 1.0], [3.5, 1.0, 1.0]);
        assert_eq!(voltage.calc(1, &interface), vec![5.0]);
        // reversed direction flips the sign
        let mut reversed = VoltageIntegral::new([3.5, 1.0, 1.0], [2.0, 1.0, 1.0]);
        assert_eq!(reversed.calc(1, &interface), vec![-5.0]);
    }

    #[test]
    fn current_integral_walks_the_surface_perimeter() {
        let meta = meta();
        let volt = FieldArray::new("volt", [5, 5, 5]);
        let start = meta.snap_to_mesh([2.0, 1.0, 1.0], true, false);
        let stop = meta.snap_to_mesh([2.0, 3.0, 3.0], true, false);
        let mut curr = FieldArray::new("curr", [5, 5, 5]);
        // y-currents on the z-low edge (positive) and z-high edge (negative)
        for y in (start[1] + 1)..=stop[1] {
            *curr.get_mut(1, stop[0], y, start[2]) = 2.0;
            *curr.get_mut(1, stop[0], y, stop[2]) = 2.0;
        }
        // z-currents on the y-high edge (positive) and y-low edge (negative)
        for z in (start[2] + 1)..=stop[2] {
            *curr.get_mut(2, stop[0], stop[1], z) = 3.0;
            *curr.get_mut(2, stop[0], start[1], z) = 3.0;
        }
        let interface = EngineInterface {
            meta: &meta,
            volt: &volt,
            curr: &curr,
        };
        let mut current = CurrentIntegral::new([2.0, 1.0, 1.0], [2.0, 3.0, 3.0], 0);
        assert_eq!(current.calc(1, &interface), vec![0.0]);

        // only the positive edges remain: 4 + 6 = 10
        let mut curr2 = FieldArray::new("curr", [5, 5, 5]);
        for y in (start[1] + 1)..=stop[1] {
            *curr2.get_mut(1, stop[0], y, start[2]) = 2.0;
        }
        for z in (start[2] + 1)..=stop[2] {
            *curr2.get_mut(2, stop[0], stop[1], z) = 3.0;
        }
        let interface = EngineInterface {
            meta: &meta,
            volt: &volt,
            curr: &curr2,
        };
        assert_eq!(current.calc(1, &interface), vec![10.0]);
    }
}
/// Point field probe (`ProcessFieldProbe`): all three components at one cell.
pub struct FieldProbeIntegral {
    pub position: [f64; 3],
    /// 0 = electric field, 1 = magnetic field.
    pub field_type: usize,
    snapped: Option<[usize; 3]>,
}

impl FieldProbeIntegral {
    pub fn new(position: [f64; 3], field_type: usize) -> Self {
        Self {
            position,
            field_type: field_type.min(1),
            snapped: None,
        }
    }

    fn index(&mut self, meta: &EngineMeta) -> Option<[usize; 3]> {
        if let Some(index) = self.snapped {
            return Some(index);
        }
        // native `SetDualMesh` for H-field probes
        let index = meta.snap_to_mesh(self.position, self.field_type == 1, false);
        self.snapped = Some(index);
        Some(index)
    }
}

impl IntegralCalculator for FieldProbeIntegral {
    fn number_of_integrals(&self) -> usize {
        3
    }

    fn integral_name(&self, row: usize) -> String {
        let names = if self.field_type == 0 {
            ["Ex/(V/m)", "Ey/(V/m)", "Ez/(V/m)"]
        } else {
            ["Hx/(A/m)", "Hy/(A/m)", "Hz/(A/m)"]
        };
        names.get(row).copied().unwrap_or("unknown").to_string()
    }

    fn calc(&mut self, _num_ts: u32, interface: &EngineInterface<'_>) -> Vec<f64> {
        let Some(index) = self.index(interface.meta) else {
            return vec![0.0; 3];
        };
        // native `GetEField` / `GetHField`: the EC voltage/current divided by
        // the edge length (dual mesh for H).
        let field = if self.field_type == 0 {
            interface.volt
        } else {
            interface.curr
        };
        (0..3)
            .map(|component| {
                let raw = *field.get(component, index[0], index[1], index[2]) as f64;
                let length = if self.field_type == 0 {
                    interface.meta.edge_length(component, index[component])
                } else {
                    interface.meta.dual_edge_length(component, index[component])
                };
                if length > 0.0 {
                    raw / length
                } else {
                    0.0
                }
            })
            .collect()
    }
}

/// Mode-matching surface integral (`ProcessModeMatch`).
///
/// Computes the overlap of the running E- or H-field on a plane with a
/// user-given mode template `(E_nP, E_nPP)` and reports both the modal
/// amplitude (row 0) and the mode purity `value^2 / ∫|field|^2 dA` (row 1).
pub struct ModeMatchIntegral {
    pub start: [f64; 3],
    pub stop: [f64; 3],
    /// 0 = electric field, 1 = magnetic field.
    pub field_type: usize,
    /// Mode template expressions, indexed by field component.
    pub weight_functions: [String; 3],
    /// Local coordinate origin applied before evaluating the template.
    pub weight_origin: [f64; 3],
    snapped: Option<SnappedSurface>,
}

struct SnappedSurface {
    normal: usize,
    start: [usize; 3],
    stop: [usize; 3],
    num: [usize; 2],
    /// Normalized mode values per component: `[component][posP][posPP]`.
    mode: [Vec<f64>; 2],
}

impl ModeMatchIntegral {
    /// Create a mode-matching surface between `start` and `stop` (metres).
    pub fn new(start: [f64; 3], stop: [f64; 3]) -> Self {
        Self {
            start,
            stop,
            field_type: 0,
            weight_functions: std::array::from_fn(|_| "0".to_string()),
            weight_origin: [0.0; 3],
            snapped: None,
        }
    }

    /// Set one weight function (`x, y, z, rho, a, r, t` variables).
    pub fn set_weight_function(&mut self, component: usize, function: &str) {
        if component < 3 {
            self.weight_functions[component] = function.to_string();
            self.snapped = None;
        }
    }

    /// Set the field type: 0 = E, 1 = H.
    pub fn set_field_type(&mut self, field_type: usize) {
        self.field_type = field_type.min(1);
        self.snapped = None;
    }

    /// Set the local coordinate origin used by the template.
    pub fn set_weight_origin(&mut self, origin: [f64; 3]) {
        self.weight_origin = origin;
        self.snapped = None;
    }

    fn snap(&mut self, meta: &EngineMeta) -> Option<&SnappedSurface> {
        if self.snapped.is_some() {
            return self.snapped.as_ref();
        }
        let mut lower = self.start;
        let mut upper = self.stop;
        for axis in 0..3 {
            if lower[axis] > upper[axis] {
                std::mem::swap(&mut lower[axis], &mut upper[axis]);
            }
        }
        let start = meta.index_from_coordinates(lower)?;
        let stop = meta.index_from_coordinates(upper)?;
        let mut start = start;
        let mut stop = stop;
        let dual = self.field_type == 1;
        let mut normal = None;
        let mut dimension = 0;
        for axis in 0..3 {
            if start[axis] > stop[axis] {
                std::mem::swap(&mut start[axis], &mut stop[axis]);
            }
            if start[axis] == 0 {
                start[axis] += 1;
            }
            if stop[axis] == meta.num_lines[axis].saturating_sub(1) {
                stop[axis] = stop[axis].saturating_sub(1);
            } else if dual && stop[axis] > start[axis] + 1 {
                stop[axis] -= 1;
            }
            if stop[axis] != start[axis] {
                dimension += 1;
            } else {
                normal = Some(axis);
            }
        }
        let normal = normal?;
        if dimension != 2 {
            return None;
        }
        let n_p = (normal + 1) % 3;
        let n_pp = (normal + 2) % 3;
        let num = [
            stop[n_p] - start[n_p] + 1,
            stop[n_pp] - start[n_pp] + 1,
        ];

        // Parse the two transverse templates.
        let mut parsers = [fparser::Parser::new(), fparser::Parser::new()];
        for (index, parser) in parsers.iter_mut().enumerate() {
            let component = (normal + index + 1) % 3;
            if parser
                .parse(&self.weight_functions[component], "x,y,z,rho,a,r,t", false)
                .is_err()
            {
                return None;
            }
        }

        let mut mode = [
            vec![0.0; num[0] * num[1]],
            vec![0.0; num[0] * num[1]],
        ];
        let mut norm = 0.0f64;
        for pos_p in 0..num[0] {
            for pos_pp in 0..num[1] {
                let mut pos = [0usize; 3];
                pos[normal] = start[normal];
                pos[n_p] = start[n_p] + pos_p;
                pos[n_pp] = start[n_pp] + pos_pp;
                let x = disc_line(meta, 0, pos[0], if normal == 0 { dual } else { false });
                let y = disc_line(meta, 1, pos[1], if normal == 1 { dual } else { false });
                let z = disc_line(meta, 2, pos[2], if normal == 2 { dual } else { false });
                let lx = x - self.weight_origin[0];
                let ly = y - self.weight_origin[1];
                let lz = z - self.weight_origin[2];
                let rho = (lx * lx + ly * ly).sqrt();
                let angle = ly.atan2(lx);
                let r = (lx * lx + ly * ly + lz * lz).sqrt();
                let theta = if rho != 0.0 {
                    std::f64::consts::FRAC_PI_2 - (lz / rho).atan()
                } else {
                    0.0
                };
                let values = [lx, ly, lz, rho, angle, r, theta];
                let area = node_area(meta, normal, pos, dual);
                for component in 0..2 {
                    let value = parsers[component].eval(&values).unwrap_or(0.0);
                    let value = if value.is_finite() { value } else { 0.0 };
                    mode[component][pos_p * num[1] + pos_pp] = value;
                    norm += value * value * area;
                }
            }
        }
        let norm = norm.sqrt();
        if norm > 0.0 {
            for component in 0..2 {
                for value in &mut mode[component] {
                    *value /= norm;
                }
            }
        }
        self.snapped = Some(SnappedSurface {
            normal,
            start,
            stop,
            num,
            mode,
        });
        self.snapped.as_ref()
    }
}

impl IntegralCalculator for ModeMatchIntegral {
    fn number_of_integrals(&self) -> usize {
        2
    }

    fn integral_name(&self, row: usize) -> String {
        match (row, self.field_type) {
            (0, 0) => "voltage".to_string(),
            (0, _) => "current".to_string(),
            _ => "mode_purity".to_string(),
        }
    }

    fn calc(&mut self, _num_ts: u32, interface: &EngineInterface<'_>) -> Vec<f64> {
        let field_type = self.field_type;
        let dual = field_type == 1;
        let Some(surface) = self.snap(interface.meta) else {
            return vec![0.0, 0.0];
        };
        let normal = surface.normal;
        let n_p = (normal + 1) % 3;
        let n_pp = (normal + 2) % 3;
        let start = surface.start;
        let num = surface.num;
        let mut value = 0.0f64;
        let mut purity = 0.0f64;
        for pos_p in 0..num[0] {
            for pos_pp in 0..num[1] {
                let mut pos = [0usize; 3];
                pos[normal] = start[normal];
                pos[n_p] = start[n_p] + pos_p;
                pos[n_pp] = start[n_pp] + pos_pp;
                let area = node_area(interface.meta, normal, pos, dual);
                let index = pos_p * num[1] + pos_pp;
                for component in 0..2 {
                    let field_component = (normal + component + 1) % 3;
                    let field = if field_type == 0 {
                        *interface.volt.get(field_component, pos[0], pos[1], pos[2]) as f64
                    } else {
                        *interface.curr.get(field_component, pos[0], pos[1], pos[2]) as f64
                    };
                    value += field * surface.mode[component][index] * area;
                    purity += field * field * area;
                }
            }
        }
        let mode_purity = if purity != 0.0 {
            value * value / purity
        } else {
            0.0
        };
        vec![value, mode_purity]
    }
}

/// Mesh line in metres, with the native dual-node extension at the upper end.
fn disc_line(meta: &EngineMeta, axis: usize, pos: usize, dual: bool) -> f64 {
    let lines = &meta.disc_lines[axis];
    let count = lines.len();
    if count == 0 {
        return 0.0;
    }
    let index = pos.min(count - 1);
    if !dual {
        return lines[index];
    }
    if index + 1 < count {
        0.5 * (lines[index] + lines[index + 1])
    } else if index > 0 {
        lines[index] + 0.5 * (lines[index] - lines[index - 1])
    } else {
        lines[index]
    }
}

/// Native `GetDiscDelta` in metres.
fn disc_delta(meta: &EngineMeta, axis: usize, pos: usize, dual: bool) -> f64 {
    if dual {
        if pos > 0 {
            disc_line(meta, axis, pos, true) - disc_line(meta, axis, pos - 1, true)
        } else {
            disc_line(meta, axis, 1, false) - disc_line(meta, axis, 0, false)
        }
    } else if pos + 1 < meta.num_lines[axis] {
        disc_line(meta, axis, pos + 1, false) - disc_line(meta, axis, pos, false)
    } else if pos > 0 {
        disc_line(meta, axis, pos, false) - disc_line(meta, axis, pos - 1, false)
    } else {
        0.0
    }
}

/// Native `GetNodeWidth`: `GetEdgeLength(axis, pos, !dual)`.
fn node_width(meta: &EngineMeta, axis: usize, pos: [usize; 3], dual: bool) -> f64 {
    disc_delta(meta, axis, pos[axis], !dual)
}

/// Native `GetNodeArea`.
fn node_area(meta: &EngineMeta, normal: usize, pos: [usize; 3], dual: bool) -> f64 {
    let n_p = (normal + 1) % 3;
    let n_pp = (normal + 2) % 3;
    node_width(meta, n_p, pos, dual) * node_width(meta, n_pp, pos, dual)
}



#[cfg(test)]
mod mode_match_tests {
    use super::*;
    use crate::array_nijk::FieldArray;

    fn meta() -> EngineMeta {
        let lines: Vec<f64> = (0..5).map(|index| index as f64 * 1.0e-3).collect();
        EngineMeta {
            num_lines: [5, 5, 5],
            disc_lines: [lines.clone(), lines.clone(), lines],
            dt: 1.0e-12,
            materials: None,
            mesh_type: crate::operator::MeshType::Cartesian,
        }
    }

    #[test]
    fn a_field_proportional_to_the_mode_has_unit_purity() {
        let meta = meta();
        let mut volt = FieldArray::new("volt", [5, 5, 5]);
        let mut integral = ModeMatchIntegral::new([0.0, 0.0, 2.0e-3], [4.0e-3, 4.0e-3, 2.0e-3]);
        // Constant E_x mode on a z-normal surface.
        integral.set_weight_function(0, "1");
        integral.set_weight_function(1, "0");
        let interface = EngineInterface {
            meta: &meta,
            volt: &volt,
            curr: &FieldArray::new("curr", [5, 5, 5]),
        };
        let results = integral.calc(0, &interface);
        // Normalize the mode by evaluating the table first.
        assert_eq!(results[0], 0.0);

        // Fill the field with 3 times the normalized mode.
        let surface = integral.snap(&meta).expect("surface");
        let normal = surface.normal;
        let n_p = (normal + 1) % 3;
        let n_pp = (normal + 2) % 3;
        for pos_p in 0..surface.num[0] {
            for pos_pp in 0..surface.num[1] {
                let mut pos = [0usize; 3];
                pos[normal] = surface.start[normal];
                pos[n_p] = surface.start[n_p] + pos_p;
                pos[n_pp] = surface.start[n_pp] + pos_pp;
                let index = pos_p * surface.num[1] + pos_pp;
                *volt.get_mut(n_p, pos[0], pos[1], pos[2]) =
                    (3.0 * surface.mode[0][index]) as f32;
            }
        }
        let interface = EngineInterface {
            meta: &meta,
            volt: &volt,
            curr: &FieldArray::new("curr", [5, 5, 5]),
        };
        let results = integral.calc(0, &interface);
        assert!((results[0] - 3.0).abs() < 1.0e-6, "amplitude {}", results[0]);
        assert!(
            (results[1] - 1.0).abs() < 1.0e-6,
            "purity {}",
            results[1]
        );
    }

    #[test]
    fn a_box_without_a_flat_surface_is_disabled() {
        let mut meta = meta();
        meta.num_lines = [5, 5, 5];
        let mut integral = ModeMatchIntegral::new([1.0e-3, 1.0e-3, 1.0e-3], [3.0e-3, 3.0e-3, 3.0e-3]);
        integral.set_weight_function(0, "1");
        let empty = FieldArray::new("volt", [5, 5, 5]);
        let interface = EngineInterface {
            meta: &meta,
            volt: &empty,
            curr: &FieldArray::new("curr", [5, 5, 5]),
        };
        assert_eq!(integral.calc(0, &interface), vec![0.0, 0.0]);
    }
}
