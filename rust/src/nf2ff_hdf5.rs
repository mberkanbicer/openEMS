//! HDF5 driver for the near-field to far-field transform (`nf2ff`).
//!
//! Mirrors the native `nf2ff` class: `AnalyseFile` reads one recorded box face
//! (frequency-domain, or time-domain with a DFT fallback), adds it to the
//! per-frequency [`Nf2ffCalc`] accumulators, and `Write2HDF5` writes the native
//! result layout (`/Mesh`, `/nf2ff/E_theta/FD`, `/nf2ff/E_phi/FD`,
//! `/nf2ff/P_rad/FD` plus the `Frequency`, `Prad` and `Dmax` attributes).

#![cfg(feature = "hdf5")]

use std::error::Error;
use std::f64::consts::PI;
use std::fmt;

use crate::hdf5::{Hdf5Error, Hdf5File};
use crate::nf2ff::{ComplexField, MirrorType, Nf2ffCalc};
use crate::processing::ComplexPoint;

#[derive(Clone, Debug, PartialEq)]
pub enum Nf2ffError {
    Hdf5(String),
    MeshMismatch(String),
    NoFrequencies,
    MissingData(String),
}

impl fmt::Display for Nf2ffError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hdf5(message) => write!(formatter, "HDF5 error: {message}"),
            Self::MeshMismatch(message) => write!(formatter, "mesh mismatch: {message}"),
            Self::NoFrequencies => formatter.write_str("no frequencies found"),
            Self::MissingData(name) => write!(formatter, "missing data: {name}"),
        }
    }
}

impl Error for Nf2ffError {}

impl From<Hdf5Error> for Nf2ffError {
    fn from(error: Hdf5Error) -> Self {
        Self::Hdf5(error.to_string())
    }
}

/// Native `nf2ff` driver: one accumulator per requested frequency.
pub struct Nf2ffRunner {
    frequencies: Vec<f64>,
    theta: Vec<f64>,
    phi: Vec<f64>,
    center: [f64; 3],
    radius: f64,
    verbose: i32,
    legacy_fmt: bool,
    calculators: Vec<Nf2ffCalc>,
}

impl Nf2ffRunner {
    pub fn new(frequencies: Vec<f64>, theta: Vec<f64>, phi: Vec<f64>, center: [f64; 3]) -> Self {
        let calculators = frequencies
            .iter()
            .map(|frequency| Nf2ffCalc::new(*frequency, theta.clone(), phi.clone(), center))
            .collect();
        Self {
            frequencies,
            theta,
            phi,
            center,
            radius: 1.0,
            verbose: 0,
            legacy_fmt: false,
            calculators,
        }
    }

    pub fn set_radius(&mut self, radius: f64) {
        self.radius = radius;
        for calculator in &mut self.calculators {
            calculator.radius = radius;
        }
    }

    pub fn set_verbose_level(&mut self, level: i32) {
        self.verbose = level;
    }

    pub fn set_legacy_format(&mut self, legacy: bool) {
        self.legacy_fmt = legacy;
    }

    /// Enable a symmetry plane (native `SetMirror`).
    pub fn set_mirror(&mut self, mirror_type: i32, axis: usize, position: f64) -> bool {
        if mirror_type <= 0 {
            return true;
        }
        if mirror_type >= 3 || axis >= 3 {
            return false;
        }
        let mirror = if mirror_type == 1 {
            MirrorType::Pec
        } else {
            MirrorType::Pmc
        };
        for calculator in &mut self.calculators {
            calculator.set_mirror(mirror, axis, position);
        }
        // the native mirror position is relative to the box centre
        let _ = position;
        true
    }

    pub fn frequencies(&self) -> &[f64] {
        &self.frequencies
    }

    pub fn radiated_power(&self, index: usize) -> f64 {
        self.calculators
            .get(index)
            .map(|calculator| calculator.radiated_power)
            .unwrap_or(0.0)
    }

    pub fn max_directivity(&self, index: usize) -> f64 {
        let Some(calculator) = self.calculators.get(index) else {
            return 0.0;
        };
        if calculator.radiated_power == 0.0 {
            return 0.0;
        }
        let mut maximum = 0.0f64;
        for row in &calculator.p_rad {
            for value in row {
                maximum = maximum.max(*value);
            }
        }
        maximum * (4.0 * PI * calculator.radius * calculator.radius / calculator.radiated_power)
    }

    /// Analyse one recorded box face and accumulate it into every frequency.
    pub fn analyse_file(&mut self, e_file: &str, h_file: &str) -> Result<bool, Nf2ffError> {
        let e_dump = Hdf5File::open(e_file, false)?;
        let h_dump = Hdf5File::open(h_file, false)?;

        let e_mesh = read_mesh(&e_dump)?;
        let h_mesh = read_mesh(&h_dump)?;
        if e_mesh.num_lines != h_mesh.num_lines || e_mesh.mesh_type != h_mesh.mesh_type {
            return Err(Nf2ffError::MeshMismatch(
                "the E- and H-field meshes do not agree".to_string(),
            ));
        }
        for axis in 0..3 {
            if e_mesh.lines[axis].len() != h_mesh.lines[axis].len()
                || e_mesh.lines[axis]
                    .iter()
                    .zip(h_mesh.lines[axis].iter())
                    .any(|(left, right)| left != right)
            {
                return Err(Nf2ffError::MeshMismatch(format!(
                    "the E- and H-field {axis}-lines do not agree"
                )));
            }
        }

        let e_frequencies = read_frequencies(&e_dump)?;
        if e_frequencies.is_empty() {
            return Err(Nf2ffError::NoFrequencies);
        }
        let h_frequencies = read_frequencies(&h_dump)?;
        let same_frequencies = e_frequencies.len() == h_frequencies.len()
            && e_frequencies
                .iter()
                .zip(h_frequencies.iter())
                .all(|(left, right)| left == right);
        let use_fd = same_frequencies
            && self
                .frequencies
                .iter()
                .all(|wanted| e_frequencies.iter().any(|value| value == wanted));

        if use_fd {
            for (index, wanted) in self.frequencies.iter().enumerate() {
                let fd_index = e_frequencies
                    .iter()
                    .position(|value| value == wanted)
                    .ok_or_else(|| Nf2ffError::MissingData(format!("frequency {wanted}")))?;
                let e_field = read_complex_field(&e_dump, fd_index, e_mesh.num_lines)?;
                let h_field = read_complex_field(&h_dump, fd_index, e_mesh.num_lines)?;
                let calculator = &mut self.calculators[index];
                calculator.add_plane(
                    [
                        &e_mesh.lines[0],
                        &e_mesh.lines[1],
                        &e_mesh.lines[2],
                    ],
                    e_mesh.num_lines,
                    &e_field,
                    &h_field,
                    e_mesh.mesh_type == 1,
                );
            }
            return Ok(true);
        }

        // time-domain fallback: transform the recorded sequence at the requested
        // frequencies (native `CalcFDVectorData`)
        let e_td = read_time_domain(&e_dump)?;
        let h_td = read_time_domain(&h_dump)?;
        if e_td.time.len() != h_td.time.len() {
            return Err(Nf2ffError::MeshMismatch(
                "the E- and H-field time sequences differ in length".to_string(),
            ));
        }
        let dt = if e_td.time.len() > 1 {
            e_td.time[1] - e_td.time[0]
        } else {
            return Err(Nf2ffError::MissingData(
                "at least two time samples are needed for the DFT".to_string(),
            ));
        };
        for (index, frequency) in self.frequencies.iter().enumerate() {
            let e_field = discrete_fourier_transform(&e_td, *frequency, dt, e_mesh.num_lines);
            let h_field = discrete_fourier_transform(&h_td, *frequency, dt, e_mesh.num_lines);
            let calculator = &mut self.calculators[index];
            calculator.add_plane(
                [&e_mesh.lines[0], &e_mesh.lines[1], &e_mesh.lines[2]],
                e_mesh.num_lines,
                &e_field,
                &h_field,
                e_mesh.mesh_type == 1,
            );
        }
        let _ = self.verbose;
        Ok(true)
    }

    /// Write the far-field result (native `Write2HDF5`).
    pub fn write_hdf5(&self, path: &str) -> Result<bool, Nf2ffError> {
        let file = Hdf5File::create(path)?;
        let num_theta = self.theta.len();
        let num_phi = self.phi.len();

        let theta: Vec<f32> = self.theta.iter().map(|value| *value as f32).collect();
        let phi: Vec<f32> = self.phi.iter().map(|value| *value as f32).collect();
        file.write_f32_dataset("/Mesh", "theta", &theta, None)?;
        file.write_f32_dataset("/Mesh", "phi", &phi, None)?;
        file.write_f32_dataset("/Mesh", "r", &[self.radius as f32], None)?;
        file.write_f32_dataset("/Mesh", "MeshType", &[2.0], None)?;
        file.write_i32_attribute("/Mesh", "mesh_type", 2)?;
        file.write_f64_attribute("/", "openEMS_HDF5_version", 0.3)?;
        if self.legacy_fmt {
            file.write_bool_attribute("/", "legacy_fmt", true)?;
        }

        for (name, values) in [
            ("E_theta", true),
            ("E_phi", false),
        ] {
            let group = format!("/nf2ff/{name}/FD");
            file.create_group(&group)?;
            for (frequency_index, calculator) in self.calculators.iter().enumerate() {
                let field = if values {
                    &calculator.e_theta
                } else {
                    &calculator.e_phi
                };
                let mut re = Vec::with_capacity(num_theta * num_phi);
                let mut im = Vec::with_capacity(num_theta * num_phi);
                for ti in 0..num_theta {
                    for pi in 0..num_phi {
                        re.push(field[ti][pi].re as f32);
                        im.push(field[ti][pi].im as f32);
                    }
                }
                if self.legacy_fmt {
                    // legacy layout: real/imaginary split, stored (phi, theta)
                    let mut legacy_re = Vec::with_capacity(re.len());
                    let mut legacy_im = Vec::with_capacity(im.len());
                    for pi in 0..num_phi {
                        for ti in 0..num_theta {
                            legacy_re.push(re[ti * num_phi + pi]);
                            legacy_im.push(im[ti * num_phi + pi]);
                        }
                    }
                    file.write_f32_dataset_nd(
                        &group,
                        &format!("f{frequency_index}_real"),
                        &[num_phi, num_theta],
                        &legacy_re,
                        None,
                    )?;
                    file.write_f32_dataset_nd(
                        &group,
                        &format!("f{frequency_index}_imag"),
                        &[num_phi, num_theta],
                        &legacy_im,
                        None,
                    )?;
                } else {
                    file.write_complex_f32_dataset(
                        &group,
                        &format!("f{frequency_index}"),
                        &[num_theta, num_phi],
                        &re,
                        &im,
                        None,
                    )?;
                }
            }
        }

        file.create_group("/nf2ff/P_rad/FD")?;
        for (frequency_index, calculator) in self.calculators.iter().enumerate() {
            let mut values = Vec::with_capacity(num_theta * num_phi);
            if self.legacy_fmt {
                for pi in 0..num_phi {
                    for ti in 0..num_theta {
                        values.push(calculator.p_rad[ti][pi] as f32);
                    }
                }
                file.write_f32_dataset_nd(
                    "/nf2ff/P_rad/FD",
                    &format!("f{frequency_index}"),
                    &[num_phi, num_theta],
                    &values,
                    None,
                )?;
            } else {
                for ti in 0..num_theta {
                    for pi in 0..num_phi {
                        values.push(calculator.p_rad[ti][pi] as f32);
                    }
                }
                file.write_f32_dataset_nd(
                    "/nf2ff/P_rad/FD",
                    &format!("f{frequency_index}"),
                    &[num_theta, num_phi],
                    &values,
                    None,
                )?;
            }
        }

        file.write_f64_attribute_vector("/nf2ff", "Frequency", &self.frequencies)?;
        let prad: Vec<f64> = (0..self.calculators.len())
            .map(|index| self.radiated_power(index))
            .collect();
        file.write_f64_attribute_vector("/nf2ff", "Prad", &prad)?;
        let dmax: Vec<f64> = (0..self.calculators.len())
            .map(|index| self.max_directivity(index))
            .collect();
        file.write_f64_attribute_vector("/nf2ff", "Dmax", &dmax)?;
        let _ = self.center;
        Ok(true)
    }
}

struct MeshData {
    lines: [Vec<f64>; 3],
    num_lines: [usize; 3],
    mesh_type: i32,
}

fn read_mesh(file: &Hdf5File) -> Result<MeshData, Nf2ffError> {
    let mut lines: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (axis, name) in ["x", "y", "z"].iter().enumerate() {
        let path = format!("/Mesh/{name}");
        lines[axis] = match file.read_f64_dataset(&path) {
            Ok((_, values)) => values,
            Err(_) => {
                let (_, values) = file.read_f32_dataset(&path)?;
                values.into_iter().map(f64::from).collect()
            }
        };
    }
    let mesh_type = file
        .read_i32_attribute("/Mesh", "mesh_type")
        .or_else(|_| file.read_i32_attribute("/Mesh", "MeshType"))
        .unwrap_or(0);
    let num_lines = [lines[0].len(), lines[1].len(), lines[2].len()];
    Ok(MeshData {
        lines,
        num_lines,
        mesh_type,
    })
}

fn read_frequencies(file: &Hdf5File) -> Result<Vec<f64>, Nf2ffError> {
    match file.read_f64_attribute_vector("/FieldData/FD", "frequency") {
        Ok(values) => Ok(values),
        Err(_) => Ok(file
            .read_f32_attribute_vector("/FieldData/FD", "frequency")?
            .into_iter()
            .map(f64::from)
            .collect()),
    }
}

fn read_complex_field(
    file: &Hdf5File,
    index: usize,
    num_lines: [usize; 3],
) -> Result<ComplexField, Nf2ffError> {
    let path = format!("/FieldData/FD/f{index}");
    let (dims, re, im) = file.read_complex_f32_dataset(&path)?;
    if dims.len() != 4 || dims[0] < 3 {
        return Err(Nf2ffError::MissingData(path));
    }
    let [nx, ny, nz] = [dims[1], dims[2], dims[3]];
    if [nx, ny, nz] != num_lines {
        return Err(Nf2ffError::MeshMismatch(format!(
            "{path}: expected {num_lines:?}, got {dims:?}"
        )));
    }
    let mut field = ComplexField::new(num_lines);
    for component in 0..3 {
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let offset = ((component * nx + i) * ny + j) * nz + k;
                    field.set(
                        component,
                        i,
                        j,
                        k,
                        ComplexPoint {
                            re: f64::from(re[offset]),
                            im: f64::from(im[offset]),
                        },
                    );
                }
            }
        }
    }
    Ok(field)
}

struct TimeDomainData {
    time: Vec<f64>,
    /// Samples as `(component, time index)` with the full grid per sample.
    samples: Vec<Vec<f32>>,
    dims: [usize; 3],
}

fn read_time_domain(file: &Hdf5File) -> Result<TimeDomainData, Nf2ffError> {
    let names = file.object_names("/FieldData/TD")?;
    let mut names: Vec<String> = names
        .into_iter()
        .filter(|name| name.chars().all(|character| character.is_ascii_digit()))
        .collect();
    names.sort();
    if names.is_empty() {
        return Err(Nf2ffError::MissingData("/FieldData/TD".to_string()));
    }
    let mut time = Vec::with_capacity(names.len());
    let mut samples = Vec::with_capacity(names.len());
    let mut dims = [0usize; 3];
    for (index, name) in names.iter().enumerate() {
        let path = format!("/FieldData/TD/{name}");
        let (shape, values) = file.read_f32_dataset(&path)?;
        if shape.len() != 4 || shape[0] < 3 {
            return Err(Nf2ffError::MissingData(path));
        }
        if index == 0 {
            dims = [shape[1], shape[2], shape[3]];
        } else if [shape[1], shape[2], shape[3]] != dims {
            return Err(Nf2ffError::MeshMismatch(format!(
                "{path}: sample shape changed"
            )));
        }
        let sample_time = file
            .read_f64_attribute(&path, "time")
            .or_else(|_| file.read_f64_attribute(&path, "Time"))
            .unwrap_or(index as f64);
        time.push(sample_time);
        samples.push(values);
    }
    Ok(TimeDomainData { time, samples, dims })
}

fn discrete_fourier_transform(
    data: &TimeDomainData,
    frequency: f64,
    dt: f64,
    num_lines: [usize; 3],
) -> ComplexField {
    let mut field = ComplexField::new(num_lines);
    let [nx, ny, nz] = data.dims;
    let mut re = vec![0.0f64; 3 * nx * ny * nz];
    let mut im = vec![0.0f64; 3 * nx * ny * nz];
    for (sample_index, sample) in data.samples.iter().enumerate() {
        let time = if data.time.len() > 1 {
            data.time[sample_index]
        } else {
            sample_index as f64 * dt
        };
        let phase = -2.0 * PI * frequency * time;
        let (sin, cos) = phase.sin_cos();
        for (offset, value) in sample.iter().enumerate() {
            let value = f64::from(*value);
            re[offset] += value * cos;
            im[offset] += value * sin;
        }
    }
    if nx == 0 || ny == 0 || nz == 0 {
        return field;
    }
    for component in 0..3 {
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let offset = ((component * nx + i) * ny + j) * nz + k;
                    field.set(
                        component,
                        i,
                        j,
                        k,
                        ComplexPoint {
                            re: re[offset],
                            im: im[offset],
                        },
                    );
                }
            }
        }
    }
    field
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write one recorded plane (`z` has a single line) as a field dump.
    fn write_plane(path: &str, amplitude_sign: f32) {
        let file = Hdf5File::create(path).unwrap();
        let x = [0.0f64, 0.5, 1.0];
        let y = [0.0f64, 0.5, 1.0];
        let z = [0.0f64];
        file.write_f64_dataset("/Mesh", "x", &x).unwrap();
        file.write_f64_dataset("/Mesh", "y", &y).unwrap();
        file.write_f64_dataset("/Mesh", "z", &z).unwrap();
        file.write_i32_attribute("/Mesh", "mesh_type", 0).unwrap();
        file.create_group("/FieldData/FD").unwrap();
        file.write_f64_attribute_vector("/FieldData/FD", "frequency", &[1.0e9])
            .unwrap();

        let dims = [3usize, 3, 3, 1];
        let count = dims.iter().product::<usize>();
        let mut re = vec![0.0f32; count];
        let mut im = vec![0.0f32; count];
        for i in 0..3 {
            for j in 0..3 {
                // tangential E (x) and H (y) with a constant Poynting vector
                re[(0 * 3 + i) * 3 + j] = amplitude_sign;
                re[((1 * 3 + i) * 3 + j)] = 1.0;
            }
        }
        file.write_complex_f32_dataset("/FieldData/FD", "f0", &dims, &re, &im, None)
            .unwrap();
    }

    #[test]
    fn analyses_planes_and_writes_the_native_result_layout() {
        let directory = std::env::temp_dir().join("openems_nf2ff_driver_test");
        std::fs::create_dir_all(&directory).unwrap();
        let e_file = directory.join("Et_0.h5");
        let h_file = directory.join("Ht_0.h5");
        let result = directory.join("nf2ff.h5");
        let e_path = e_file.to_str().unwrap();
        let h_path = h_file.to_str().unwrap();
        let result_path = result.to_str().unwrap();

        write_plane(e_path, 1.0);
        write_plane(h_path, 1.0);

        let theta = vec![0.0, PI / 4.0, PI / 2.0];
        let phi = vec![0.0, PI / 2.0];
        let mut runner = Nf2ffRunner::new(vec![1.0e9], theta.clone(), phi.clone(), [0.0; 3]);
        runner.set_radius(1.0);
        assert!(runner.analyse_file(e_path, h_path).unwrap());
        assert!(runner.write_hdf5(result_path).unwrap());

        let output = Hdf5File::open(result_path, false).unwrap();
        let (theta_dims, theta_values) = output.read_f32_dataset("/Mesh/theta").unwrap();
        assert_eq!(theta_dims, vec![theta.len()]);
        assert_eq!(theta_values.len(), theta.len());
        let (_, phi_values) = output.read_f32_dataset("/Mesh/phi").unwrap();
        assert_eq!(phi_values.len(), phi.len());

        let (dims, re, im) = output
            .read_complex_f32_dataset("/nf2ff/E_theta/FD/f0")
            .unwrap();
        assert_eq!(dims, vec![theta.len(), phi.len()]);
        assert_eq!(re.len(), theta.len() * phi.len());
        assert!(re.iter().chain(im.iter()).all(|value| value.is_finite()));
        assert!(re.iter().any(|value| value.abs() > 0.0));

        let (prad_dims, _) = output.read_f32_dataset("/nf2ff/P_rad/FD/f0").unwrap();
        assert_eq!(prad_dims, vec![theta.len(), phi.len()]);

        let frequency = output
            .read_f64_attribute_vector("/nf2ff", "Frequency")
            .unwrap();
        assert_eq!(frequency, vec![1.0e9]);
        let prad = output.read_f64_attribute_vector("/nf2ff", "Prad").unwrap();
        assert_eq!(prad.len(), 1);
        assert!(prad[0].is_finite());
        let dmax = output.read_f64_attribute_vector("/nf2ff", "Dmax").unwrap();
        assert_eq!(dmax.len(), 1);
        assert!(dmax[0].is_finite());

        let _ = std::fs::remove_file(e_path);
        let _ = std::fs::remove_file(h_path);
        let _ = std::fs::remove_file(result_path);
    }
}
