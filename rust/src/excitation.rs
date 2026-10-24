//! Excitation: time-domain source signals.
//! Mirrors `FDTD/excitation.h/cpp` in the C++ codebase.
//!
//! Native openEMS does not evaluate an analytic formula during the time loop:
//! it builds a voltage array sampled at `t = n*dt` and a current array sampled
//! at `t = (n + 1/2)*dt` once the timestep is known. This module reproduces
//! that contract so the engine and the TFSF extension index the same arrays.

use std::f64::consts::PI;
use std::fmt;

/// Types of excitation signals supported by the solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExcitationType {
    Gauss,
    Sinus,
    Dirac,
    Step,
    Custom,
}

/// Errors raised while building an excitation waveform.
#[derive(Debug, Clone, PartialEq)]
pub enum ExcitationError {
    /// The engine timestep must be positive before a signal can be built.
    InvalidTimestep(f64),
    /// A required frequency is missing or not positive.
    InvalidFrequency { field: &'static str, value: f64 },
    /// The requested signal would have no samples.
    EmptySignal,
    /// The custom excitation expression could not be parsed.
    Parse(String),
    /// The custom excitation expression failed during evaluation.
    Eval(String),
}

impl fmt::Display for ExcitationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExcitationError::InvalidTimestep(dt) => {
                write!(f, "invalid excitation timestep: {dt} (must be > 0)")
            }
            ExcitationError::InvalidFrequency { field, value } => {
                write!(
                    f,
                    "invalid excitation frequency '{field}': {value} (must be > 0)"
                )
            }
            ExcitationError::EmptySignal => {
                write!(f, "excitation signal would contain no samples")
            }
            ExcitationError::Parse(message) => {
                write!(f, "excitation expression parse error: {message}")
            }
            ExcitationError::Eval(message) => {
                write!(f, "excitation expression evaluation error: {message}")
            }
        }
    }
}

impl std::error::Error for ExcitationError {}

/// An excitation time signal.
///
/// The waveform arrays are produced by [`Excitation::build`]. Before `build`
/// they are empty and [`Excitation::voltage`] / [`Excitation::current`] return
/// zero, matching the native "no signal before setup" behaviour.
#[derive(Debug, Clone)]
pub struct Excitation {
    pub exc_type: ExcitationType,
    pub f0: f64,
    pub fc: f64,
    pub f_max: f64,
    /// Timestep used to sample the signal, in seconds.
    pub dt: f64,
    /// Voltage samples at `t = n*dt`.
    pub signal_volt: Vec<f32>,
    /// Current samples at `t = (n + 1/2)*dt`.
    pub signal_curr: Vec<f32>,
    /// Native Nyquist timestep count for the highest modelled frequency.
    pub nyquist_ts: u32,
    custom_expr: Option<String>,
}

impl Excitation {
    /// Create a Gaussian excitation.
    pub fn gauss(f0: f64, fc: f64) -> Self {
        Self {
            exc_type: ExcitationType::Gauss,
            f0,
            fc,
            f_max: 0.0,
            dt: 0.0,
            signal_volt: Vec::new(),
            signal_curr: Vec::new(),
            nyquist_ts: 0,
            custom_expr: None,
        }
    }

    /// Create a sinusoidal excitation.
    pub fn sinus(f0: f64) -> Self {
        Self {
            exc_type: ExcitationType::Sinus,
            f0,
            fc: 0.0,
            f_max: 0.0,
            dt: 0.0,
            signal_volt: Vec::new(),
            signal_curr: Vec::new(),
            nyquist_ts: 0,
            custom_expr: None,
        }
    }

    /// Create a Dirac excitation.
    pub fn dirac(f_max: f64) -> Self {
        Self {
            exc_type: ExcitationType::Dirac,
            f0: 0.0,
            fc: 0.0,
            f_max,
            dt: 0.0,
            signal_volt: Vec::new(),
            signal_curr: Vec::new(),
            nyquist_ts: 0,
            custom_expr: None,
        }
    }

    /// Create a step excitation.
    pub fn step(f_max: f64) -> Self {
        Self {
            exc_type: ExcitationType::Step,
            f0: 0.0,
            fc: 0.0,
            f_max,
            dt: 0.0,
            signal_volt: Vec::new(),
            signal_curr: Vec::new(),
            nyquist_ts: 0,
            custom_expr: None,
        }
    }

    /// Create a custom function excitation (parsed by fparser at build time).
    pub fn custom(func_str: &str, f0: f64, fmax: f64) -> Self {
        Self {
            exc_type: ExcitationType::Custom,
            f0,
            fc: 0.0,
            f_max: fmax,
            dt: 0.0,
            signal_volt: Vec::new(),
            signal_curr: Vec::new(),
            nyquist_ts: 0,
            custom_expr: Some(func_str.to_string()),
        }
    }

    /// Build the sample arrays exactly as `Excitation::buildExcitationSignal`.
    ///
    /// `max_ts` is the maximum number of timesteps the solver will run. It caps
    /// the Gaussian pulse and defines the custom-expression length, matching the
    /// native `nTS` argument.
    pub fn build(&mut self, dt: f64, max_ts: u32) -> Result<(), ExcitationError> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(ExcitationError::InvalidTimestep(dt));
        }
        self.dt = dt;

        match self.exc_type {
            ExcitationType::Gauss => self.build_gauss(dt, max_ts),
            ExcitationType::Sinus => self.build_sinus(dt),
            ExcitationType::Dirac => {
                self.signal_volt = vec![0.0, 1.0];
                self.signal_curr = vec![0.0, 1.0];
                self.f_max = 0.0;
                self.nyquist_ts = 1;
                Ok(())
            }
            ExcitationType::Step => {
                self.signal_volt = vec![1.0, 1.0];
                self.signal_curr = vec![1.0, 1.0];
                self.f_max = 0.0;
                self.nyquist_ts = 1;
                Ok(())
            }
            ExcitationType::Custom => self.build_custom(dt, max_ts),
        }
    }

    fn build_gauss(&mut self, dt: f64, max_ts: u32) -> Result<(), ExcitationError> {
        if !self.fc.is_finite() || self.fc <= 0.0 {
            return Err(ExcitationError::InvalidFrequency {
                field: "fc",
                value: self.fc,
            });
        }
        if !self.f0.is_finite() || self.f0 < 0.0 {
            return Err(ExcitationError::InvalidFrequency {
                field: "f0",
                value: self.f0,
            });
        }

        // Native: m_length = ceil(2*9/(2*pi*fc)/dt), capped at maxTS.
        let delay = 9.0 / (2.0 * PI * self.fc);
        let length = (2.0 * delay / dt).ceil();
        let mut length = if length.is_finite() && length > 0.0 {
            length as usize
        } else {
            return Err(ExcitationError::EmptySignal);
        };
        if max_ts > 0 {
            length = length.min(max_ts as usize);
        }
        if length == 0 {
            return Err(ExcitationError::EmptySignal);
        }

        let mut volt = vec![0.0f32; length];
        let mut curr = vec![0.0f32; length];
        for n in 1..length {
            let t = n as f64 * dt;
            volt[n] = gauss_sample(self.f0, self.fc, t, delay) as f32;
            curr[n] = gauss_sample(self.f0, self.fc, t + 0.5 * dt, delay) as f32;
        }
        self.signal_volt = volt;
        self.signal_curr = curr;
        self.f_max = self.f0 + self.fc;
        self.nyquist_ts = calc_nyquist_num(self.f_max, dt);
        Ok(())
    }

    fn build_sinus(&mut self, dt: f64) -> Result<(), ExcitationError> {
        if !self.f0.is_finite() || self.f0 <= 0.0 {
            return Err(ExcitationError::InvalidFrequency {
                field: "f0",
                value: self.f0,
            });
        }
        // Native: m_length = round(2/f0/dt).
        let length = (2.0 / self.f0 / dt).round();
        let length = if length.is_finite() && length >= 2.0 {
            length as usize
        } else {
            return Err(ExcitationError::EmptySignal);
        };

        let mut volt = vec![0.0f32; length];
        let mut curr = vec![0.0f32; length];
        for n in 1..length {
            let t = n as f64 * dt;
            volt[n] = (2.0 * PI * self.f0 * t).sin() as f32;
            curr[n] = (2.0 * PI * self.f0 * (t + 0.5 * dt)).sin() as f32;
        }
        self.signal_volt = volt;
        self.signal_curr = curr;
        self.f_max = self.f0;
        self.nyquist_ts = calc_nyquist_num(self.f_max, dt);
        Ok(())
    }

    fn build_custom(&mut self, dt: f64, max_ts: u32) -> Result<(), ExcitationError> {
        let expression = self
            .custom_expr
            .as_deref()
            .ok_or_else(|| ExcitationError::Parse("empty custom expression".to_string()))?;
        if max_ts == 0 {
            return Err(ExcitationError::EmptySignal);
        }

        let mut parser = fparser::Parser::new();
        parser
            .parse(expression, "t", false)
            .map_err(|error| ExcitationError::Parse(error.to_string()))?;

        let length = max_ts as usize;
        let mut volt = vec![0.0f32; length];
        let mut curr = vec![0.0f32; length];
        for (n, (v, c)) in volt.iter_mut().zip(curr.iter_mut()).enumerate() {
            let t = n as f64 * dt;
            *v = parser
                .eval(&[t])
                .map_err(|error| ExcitationError::Eval(error.to_string()))? as f32;
            *c = parser
                .eval(&[t + 0.5 * dt])
                .map_err(|error| ExcitationError::Eval(error.to_string()))? as f32;
        }
        self.signal_volt = volt;
        self.signal_curr = curr;
        self.f_max = self.f0;
        self.nyquist_ts = calc_nyquist_num(self.f_max, dt);
        Ok(())
    }

    /// Whether the waveform arrays have been built.
    pub fn is_built(&self) -> bool {
        !self.signal_volt.is_empty()
    }

    /// Voltage sample at timestep `ts`; zero outside the built signal.
    pub fn voltage(&self, ts: usize) -> f32 {
        self.signal_volt.get(ts).copied().unwrap_or(0.0)
    }

    /// Current sample at timestep `ts`; zero outside the built signal.
    pub fn current(&self, ts: usize) -> f32 {
        self.signal_curr.get(ts).copied().unwrap_or(0.0)
    }

    /// Number of samples in the built signal.
    pub fn signal_length(&self) -> u32 {
        self.signal_volt.len() as u32
    }

    /// Samples per period for a periodic signal, or zero for aperiodic ones.
    pub fn signal_period(&self) -> u32 {
        if self.exc_type == ExcitationType::Sinus && self.dt > 0.0 && self.f0 > 0.0 {
            (1.0 / (self.f0 * self.dt)).round() as u32
        } else {
            0
        }
    }

    /// Index of the largest absolute voltage sample.
    pub fn max_timestep(&self) -> usize {
        let mut max_idx = 0usize;
        let mut max_abs = 0.0f32;
        for (index, value) in self.signal_volt.iter().enumerate() {
            let abs = value.abs();
            if abs > max_abs {
                max_abs = abs;
                max_idx = index;
            }
        }
        max_idx
    }

    /// Nyquist timestep count for the highest modelled frequency.
    pub fn nyquist_num(&self) -> u32 {
        self.nyquist_ts
    }
}

impl Default for Excitation {
    fn default() -> Self {
        Self::gauss(1.0e9, 5.0e8)
    }
}

/// Native Gaussian sample at time `t` with the fixed `9/(2*pi*fc)` delay.
fn gauss_sample(f0: f64, fc: f64, t: f64, delay: f64) -> f64 {
    let envelope_arg = 2.0 * PI * fc * t / 3.0 - 3.0;
    (2.0 * PI * f0 * (t - delay)).cos() * (-(envelope_arg * envelope_arg)).exp()
}

/// Native `CalcNyquistNum`: floor(1/fmax/2/dt), `u32::MAX` when fmax is zero.
fn calc_nyquist_num(fmax: f64, dt: f64) -> u32 {
    if fmax == 0.0 {
        return u32::MAX;
    }
    if dt == 0.0 {
        return 1;
    }
    let value = (1.0 / fmax / 2.0 / dt).floor();
    if !value.is_finite() || value <= 0.0 {
        0
    } else if value >= u32::MAX as f64 {
        u32::MAX
    } else {
        value as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f32, expected: f64, tolerance: f64) {
        let difference = (actual as f64 - expected).abs();
        assert!(
            difference <= tolerance,
            "actual {actual:e} vs expected {expected:e} (|diff|={difference:e} > {tolerance:e})"
        );
    }

    #[test]
    fn gaussian_waveform_matches_native_formula() {
        let f0 = 1.0e9;
        let fc = 5.0e8;
        let dt = 1.0e-12;
        let mut excitation = Excitation::gauss(f0, fc);
        excitation.build(dt, 10_000).unwrap();

        let delay = 9.0 / (2.0 * PI * fc);
        let expected_length = (2.0 * delay / dt).ceil() as usize;
        assert_eq!(excitation.signal_length() as usize, expected_length);
        assert_eq!(excitation.voltage(0), 0.0);
        assert_eq!(excitation.current(0), 0.0);

        for n in [1usize, 10, 100, 5729] {
            let t = n as f64 * dt;
            let expected_volt = gauss_sample(f0, fc, t, delay);
            let expected_curr = gauss_sample(f0, fc, t + 0.5 * dt, delay);
            assert_close(excitation.voltage(n), expected_volt, 1.0e-6);
            assert_close(excitation.current(n), expected_curr, 1.0e-6);
        }
    }

    #[test]
    fn gaussian_respects_max_timestep_cap() {
        let mut excitation = Excitation::gauss(1.0e9, 5.0e8);
        excitation.build(1.0e-12, 10).unwrap();
        assert_eq!(excitation.signal_length(), 10);
    }

    #[test]
    fn sinusoidal_waveform_matches_native_formula() {
        let f0 = 1.0e9;
        let dt = 1.0e-12;
        let mut excitation = Excitation::sinus(f0);
        excitation.build(dt, 0).unwrap();

        assert_eq!(excitation.signal_length(), 2000);
        assert_eq!(excitation.voltage(0), 0.0);
        for n in [1usize, 250, 999] {
            let expected = (2.0 * PI * f0 * n as f64 * dt).sin();
            assert_close(excitation.voltage(n), expected, 1.0e-6);
        }
        assert_eq!(excitation.signal_period(), 1000);
    }

    #[test]
    fn dirac_and_step_follow_native_samples() {
        let mut dirac = Excitation::dirac(1.0e9);
        dirac.build(1.0e-12, 100).unwrap();
        assert_eq!(dirac.signal_volt, vec![0.0, 1.0]);
        assert_eq!(dirac.signal_curr, vec![0.0, 1.0]);
        assert_eq!(dirac.nyquist_num(), 1);

        let mut step = Excitation::step(1.0e9);
        step.build(1.0e-12, 100).unwrap();
        assert_eq!(step.signal_volt, vec![1.0, 1.0]);
        assert_eq!(step.signal_curr, vec![1.0, 1.0]);
        assert_eq!(step.nyquist_num(), 1);
    }

    #[test]
    fn custom_expression_is_sampled_at_both_phases() {
        let dt = 1.0e-12;
        let mut excitation = Excitation::custom("2*t", 1.0e9, 2.0e9);
        excitation.build(dt, 16).unwrap();
        assert_eq!(excitation.signal_length(), 16);
        assert_close(excitation.voltage(0), 0.0, 0.0);
        assert_close(excitation.voltage(4), 8.0 * dt, 1.0e-12);
        assert_close(excitation.current(4), 9.0 * dt, 1.0e-12);
    }

    #[test]
    fn nyquist_matches_native_formula() {
        // floor(1/(2 * 1.5e9 * 1e-12)) = floor(333.33) = 333
        let mut excitation = Excitation::gauss(1.0e9, 5.0e8);
        excitation.build(1.0e-12, 0).unwrap();
        assert_eq!(excitation.nyquist_num(), 333);
    }

    #[test]
    fn invalid_inputs_are_reported() {
        let mut gauss = Excitation::gauss(1.0e9, 0.0);
        assert!(matches!(
            gauss.build(1.0e-12, 0),
            Err(ExcitationError::InvalidFrequency { field: "fc", .. })
        ));
        let mut sinus = Excitation::sinus(0.0);
        assert!(matches!(
            sinus.build(1.0e-12, 0),
            Err(ExcitationError::InvalidFrequency { field: "f0", .. })
        ));
        let mut custom = Excitation::custom("not a function(", 1.0e9, 1.0e9);
        assert!(matches!(
            custom.build(1.0e-12, 10),
            Err(ExcitationError::Parse(_))
        ));
    }

    #[test]
    fn unbuilt_signal_reads_zero() {
        let excitation = Excitation::gauss(1.0e9, 5.0e8);
        assert!(!excitation.is_built());
        assert_eq!(excitation.voltage(3), 0.0);
        assert_eq!(excitation.current(3), 0.0);
    }
}
