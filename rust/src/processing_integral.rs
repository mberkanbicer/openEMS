//! Time- and frequency-domain integral probes.
//!
//! Mirrors `Common/processintegral.{h,cpp}`. Concrete integrals implement
//! [`IntegralCalculator`] (native `CalcMultipleIntegrals`).

use crate::processing::{ComplexPoint, EngineInterface, EngineMeta, Processing};

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
            fd_oversampling: 4,
            fd_samples: Vec::new(),
            fd_results: Vec::new(),
            results: vec![0.0; rows],
            sample_count: 0,
            norm_dir: -1,
            enabled: true,
            header_written: false,
            calculator,
        }
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
        let nyquist_steps = 1.0 / (2.0 * highest * dt);
        let oversampling = self.fd_oversampling.max(1) as f64;
        self.fd_interval = ((nyquist_steps / oversampling).round() as u32).max(1);
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
        let mut line = format!("{:.precision$}", time, precision = self.precision);
        for value in &self.results {
            line.push('\t');
            line.push_str(&format!(
                "{:.precision$}",
                value * self.weight,
                precision = self.precision
            ));
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
            text.push_str(&format!("{frequency:.precision$}", precision = self.precision));
            for row in 0..self.results.len() {
                let value = self
                    .fd_results
                    .get(row)
                    .and_then(|values| values.get(index))
                    .copied()
                    .unwrap_or_default();
                text.push_str(&format!(
                    "\t{:.precision$}\t{:.precision$}",
                    value.re * factor,
                    value.im * factor,
                    precision = self.precision
                ));
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

        let time = num_ts as f64 * interface.meta.dt;
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
        assert!(fd_text.contains("10000000000"));

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
        let start = meta.index_from_coordinates(self.start)?;
        let stop = meta.index_from_coordinates(self.stop)?;
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
        let start = meta.index_from_coordinates(self.start)?;
        let stop = meta.index_from_coordinates(self.stop)?;
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
        }
    }

    #[test]
    fn voltage_integral_sums_the_line_voltages() {
        let mut volt = FieldArray::new("volt", [5, 5, 5]);
        let curr = FieldArray::new("curr", [5, 5, 5]);
        for i in 2..5 {
            *volt.get_mut(0, i, 1, 1) = 1.0;
        }
        let meta = meta();
        let interface = EngineInterface {
            meta: &meta,
            volt: &volt,
            curr: &curr,
        };
        // Coordinates map to indices 2..4; the native integral uses [start, stop).
        let mut voltage = VoltageIntegral::new([0.0, 1.0, 1.0], [4.0, 1.0, 1.0]);
        assert_eq!(voltage.calc(1, &interface), vec![2.0]);
        // Reversed direction flips the sign.
        let mut reversed = VoltageIntegral::new([4.0, 1.0, 1.0], [0.0, 1.0, 1.0]);
        assert_eq!(reversed.calc(1, &interface), vec![-2.0]);
    }

    #[test]
    fn current_integral_walks_the_surface_perimeter() {
        let volt = FieldArray::new("volt", [5, 5, 5]);
        let mut curr = FieldArray::new("curr", [5, 5, 5]);
        // x-normal surface at x=2, y in 1..=3, z in 1..=3.
        // y-currents on the z-low edge (positive) and z-high edge (negative).
        for y in 2..=3 {
            *curr.get_mut(1, 2, y, 1) = 2.0;
            *curr.get_mut(1, 2, y, 3) = 2.0;
        }
        // z-currents on the y-high edge (positive) and y-low edge (negative).
        for z in 2..=3 {
            *curr.get_mut(2, 2, 3, z) = 3.0;
            *curr.get_mut(2, 2, 1, z) = 3.0;
        }
        let meta = meta();
        let interface = EngineInterface {
            meta: &meta,
            volt: &volt,
            curr: &curr,
        };
        let mut current = CurrentIntegral::new([2.0, 1.0, 1.0], [2.0, 3.0, 3.0], 0);
        // +2*2 (y, z-low) + 3*2 (z, y-high) - 2*2 (y, z-high) - 3*2 (z, y-low)
        assert_eq!(current.calc(1, &interface), vec![0.0]);

        // Remove the negative edges: the loop becomes 4 + 6 = 10.
        let mut curr2 = FieldArray::new("curr", [5, 5, 5]);
        for y in 2..=3 {
            *curr2.get_mut(1, 2, y, 1) = 2.0;
        }
        for z in 2..=3 {
            *curr2.get_mut(2, 2, 3, z) = 3.0;
        }
        let interface = EngineInterface {
            meta: &meta,
            volt: &volt,
            curr: &curr2,
        };
        assert_eq!(current.calc(1, &interface), vec![10.0]);
    }
}
