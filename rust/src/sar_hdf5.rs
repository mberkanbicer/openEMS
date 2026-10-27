//! HDF5 driver for the native SAR calculation (`SAR_Calculation`).
//!
//! Reads the raw SAR dump written by `ProcessFieldsSAR` (`/Mesh`,
//! `/CellWidth`, `/CellData/*`, `/FieldData/FD/f{n}`), runs the local or
//! mass-averaged SAR calculation of [`crate::sar`] and writes the native result
//! layout (`/Mesh`, `/FieldData/FD/f{n}` plus the native attributes).
//!
//! The calculation itself is single-threaded: the result is identical to a
//! multi-threaded run, the `numThreads` argument only selects the native worker
//! count for the reference implementation.

#![cfg(feature = "hdf5")]

use std::error::Error;
use std::fmt;

use crate::hdf5::{Hdf5Error, Hdf5File};
use crate::nf2ff::ComplexField;
use crate::processing::ComplexPoint;
use crate::sar::SarCalculation;

/// Averaging method presets, identical to the native `SARAveragingMethod`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AveragingMethod {
    Simple,
    IeeeC95_3,
    Ieee62704,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SarError {
    Hdf5(String),
    NoFrequencies,
    ShapeMismatch { path: String, detail: String },
}

impl fmt::Display for SarError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hdf5(message) => write!(formatter, "HDF5 error: {message}"),
            Self::NoFrequencies => formatter.write_str("no frequencies found in the input file"),
            Self::ShapeMismatch { path, detail } => write!(formatter, "{path}: {detail}"),
        }
    }
}

impl Error for SarError {}

impl From<Hdf5Error> for SarError {
    fn from(error: Hdf5Error) -> Self {
        Self::Hdf5(error.to_string())
    }
}

/// Native `SAR_Calculation` configuration and driver.
#[derive(Clone, Debug, PartialEq)]
pub struct SarRunner {
    /// Averaging mass in kg; 0 selects the local SAR.
    pub averaging_mass: f64,
    pub method: AveragingMethod,
    pub debug_level: i32,
    pub progress: bool,
    pub auto_range: f64,
    pub record_cube_stats: bool,
}

impl Default for SarRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl SarRunner {
    pub fn new() -> Self {
        Self {
            averaging_mass: 0.0,
            method: AveragingMethod::Simple,
            debug_level: 0,
            progress: false,
            auto_range: 0.0,
            record_cube_stats: false,
        }
    }

    /// Select the averaging method by its native name.
    pub fn set_averaging_method(&mut self, method: &str) -> bool {
        self.method = match method {
            "IEEE_C95_3" => AveragingMethod::IeeeC95_3,
            "IEEE_62704" => AveragingMethod::Ieee62704,
            "Simple" | "SIMPLE" => AveragingMethod::Simple,
            _ => return false,
        };
        true
    }

    fn apply_method(&self, calculation: &mut SarCalculation) {
        match self.method {
            AveragingMethod::Ieee62704 => {
                calculation.mass_tolerance = 0.000001;
                calculation.max_iterations = 100;
                calculation.max_background_ratio = 0.1;
                calculation.unused_relative_volume_limit = 1.05;
                calculation.ignore_face_valid = false;
                calculation.mark_partial_as_used = false;
            }
            AveragingMethod::IeeeC95_3 => {
                calculation.mass_tolerance = 0.05;
                calculation.max_iterations = 100;
                calculation.max_background_ratio = 1.0;
                calculation.unused_relative_volume_limit = 1.0;
                calculation.ignore_face_valid = false;
                calculation.mark_partial_as_used = true;
            }
            AveragingMethod::Simple => {
                calculation.mass_tolerance = 0.05;
                calculation.max_iterations = 100;
                calculation.max_background_ratio = 1.0;
                calculation.unused_relative_volume_limit = 1.0;
                calculation.ignore_face_valid = true;
                calculation.mark_partial_as_used = true;
            }
        }
    }

    /// Read `input`, calculate the SAR and write the result to `output`.
    pub fn calc_from_hdf5(&mut self, input: &str, output: &str) -> Result<bool, SarError> {
        let file = Hdf5File::open(input, false)?;

        let frequencies = match file.read_f64_attribute_vector("/FieldData/FD", "frequency") {
            Ok(values) => values,
            Err(_) => file
                .read_f32_attribute_vector("/FieldData/FD", "frequency")
                .map(|values| values.into_iter().map(f64::from).collect())?,
        };
        if frequencies.is_empty() {
            return Err(SarError::NoFrequencies);
        }

        let mesh = [
            read_axis(&file, "/Mesh", "x")?,
            read_axis(&file, "/Mesh", "y")?,
            read_axis(&file, "/Mesh", "z")?,
        ];
        let num_lines = [mesh[0].len(), mesh[1].len(), mesh[2].len()];
        let cells = num_lines[0] * num_lines[1] * num_lines[2];
        let widths = [
            read_axis(&file, "/CellWidth", "x")?,
            read_axis(&file, "/CellWidth", "y")?,
            read_axis(&file, "/CellWidth", "z")?,
        ];
        let volume = read_scalar_field(&file, "/CellData/Volume")?;
        let density = read_scalar_field(&file, "/CellData/Density")?;
        let conductivity = read_scalar_field(&file, "/CellData/Conductivity")?;
        for (path, values) in [
            ("/CellData/Volume", volume.len()),
            ("/CellData/Density", density.len()),
            ("/CellData/Conductivity", conductivity.len()),
        ] {
            if values != cells {
                return Err(SarError::ShapeMismatch {
                    path: path.to_string(),
                    detail: format!("{values} values, {cells} cells expected"),
                });
            }
        }

        let mut fields = Vec::with_capacity(frequencies.len());
        for index in 0..frequencies.len() {
            let path = format!("/FieldData/FD/f{index}");
            let (dims, re, im) = file.read_complex_f32_dataset(&path)?;
            if dims.len() != 4 || dims[0] < 3 {
                return Err(SarError::ShapeMismatch {
                    path: path.clone(),
                    detail: format!("expected (3, {num_lines:?}), got {dims:?}"),
                });
            }
            let [nx, ny, nz] = [dims[1], dims[2], dims[3]];
            if [nx, ny, nz] != num_lines {
                return Err(SarError::ShapeMismatch {
                    path: path.clone(),
                    detail: format!("expected {num_lines:?}, got {dims:?}"),
                });
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
            fields.push(field);
        }

        self.calc_from_fields(
            &mesh,
            widths,
            volume,
            density,
            Some(conductivity),
            &frequencies,
            &fields,
            None,
            output,
        )
    }

    /// Calculate the SAR from in-memory fields and write the native result.
    ///
    /// This is the shared path of the `sar_calc` tool (which reads a raw dump
    /// first) and the direct `DumpType` 20..22 result dumps, which call it
    /// while the solver still owns the DFT accumulators.
    ///
    /// `j_fields` selects the native `AddEFieldAndJField` path; `None` uses
    /// `AddEFieldAndConductivity` with the cell conductivity.
    #[allow(clippy::too_many_arguments)]
    pub fn calc_from_fields(
        &mut self,
        mesh: &[Vec<f64>; 3],
        widths: [Vec<f64>; 3],
        volume: Vec<f64>,
        density: Vec<f64>,
        conductivity: Option<Vec<f64>>,
        frequencies: &[f64],
        e_fields: &[ComplexField],
        j_fields: Option<&[ComplexField]>,
        output: &str,
    ) -> Result<bool, SarError> {
        if frequencies.is_empty() {
            return Err(SarError::NoFrequencies);
        }
        if e_fields.len() != frequencies.len() {
            return Err(SarError::ShapeMismatch {
                path: "FieldData/FD".to_string(),
                detail: format!(
                    "{} field samples for {} frequencies",
                    e_fields.len(),
                    frequencies.len()
                ),
            });
        }
        let num_lines = [mesh[0].len(), mesh[1].len(), mesh[2].len()];
        let cells = num_lines[0] * num_lines[1] * num_lines[2];
        if volume.len() != cells || density.len() != cells {
            return Err(SarError::ShapeMismatch {
                path: "CellData".to_string(),
                detail: format!(
                    "{} volumes and {} densities for {cells} cells",
                    volume.len(),
                    density.len()
                ),
            });
        }
        if let Some(values) = &conductivity {
            if values.len() != cells {
                return Err(SarError::ShapeMismatch {
                    path: "CellData/Conductivity".to_string(),
                    detail: format!("{} values for {cells} cells", values.len()),
                });
            }
        }
        if j_fields.is_some_and(|fields| fields.len() != frequencies.len()) {
            return Err(SarError::ShapeMismatch {
                path: "FieldData/FD".to_string(),
                detail: "the J-field sample count must match the frequencies".to_string(),
            });
        }

        let mut calculation = SarCalculation::new(num_lines);
        self.apply_method(&mut calculation);
        calculation.averaging_mass = self.averaging_mass;
        calculation.set_cell_width(widths);
        calculation.set_cell_volume(volume);
        calculation.set_cell_density(density);

        for (index, frequency) in frequencies.iter().enumerate() {
            match j_fields {
                Some(fields) => calculation.add_e_field_and_j_field(
                    *frequency,
                    &e_fields[index],
                    &fields[index],
                ),
                None => {
                    let conductivity =
                        conductivity
                            .as_ref()
                            .ok_or_else(|| SarError::ShapeMismatch {
                                path: "CellData/Conductivity".to_string(),
                                detail: "the conductivity is required without J fields".to_string(),
                            })?;
                    calculation.add_e_field_and_conductivity(
                        *frequency,
                        &e_fields[index],
                        conductivity,
                    );
                }
            }
        }

        calculation.calc_sar();
        self.write_result(&calculation, mesh, frequencies, output)
    }

    fn write_result(
        &self,
        calculation: &SarCalculation,
        mesh: &[Vec<f64>; 3],
        frequencies: &[f64],
        output: &str,
    ) -> Result<bool, SarError> {
        let num_lines = calculation.num_lines();
        let (start, stop) = self.auto_range_block(calculation, num_lines);
        let block = [stop[0] - start[0], stop[1] - start[1], stop[2] - start[2]];
        if block.iter().any(|size| *size == 0) {
            return Ok(false);
        }

        let file = Hdf5File::create(output)?;
        file.write_f64_attribute("/", "openEMS_HDF5_version", 0.3)?;
        for axis in 0..3 {
            let lines = &mesh[axis][start[axis]..stop[axis]];
            file.write_f64_dataset("/Mesh", ["x", "y", "z"][axis], lines)?;
        }
        file.write_i32_attribute("/Mesh", "mesh_type", 0)?;
        file.write_f64_attribute("/Mesh", "mesh_scaling", 1.0)?;

        file.create_group("/FieldData/FD")?;
        file.write_f64_attribute_vector("/FieldData/FD", "frequency", frequencies)?;
        file.write_f64_attribute("/", "mass", total_mass(calculation, start, stop))?;

        for (index, frequency) in frequencies.iter().enumerate() {
            let Some(values) = calculation.sar(index) else {
                continue;
            };
            let mut block_values = Vec::with_capacity(block[0] * block[1] * block[2]);
            let mut maximum = 0.0f64;
            let mut maximum_index = [0usize; 3];
            for i in start[0]..stop[0] {
                for j in start[1]..stop[1] {
                    for k in start[2]..stop[2] {
                        let value = values[calculation.cell_index([i, j, k])];
                        if value > maximum {
                            maximum = value;
                            maximum_index = [i - start[0], j - start[1], k - start[2]];
                        }
                        block_values.push(value as f32);
                    }
                }
            }

            let name = format!("f{index}");
            file.write_f32_dataset_nd("/FieldData/FD", &name, &block, &block_values, None)?;
            let path = format!("/FieldData/FD/{name}");
            file.write_f64_attribute(&path, "frequency", *frequency)?;
            file.write_f64_attribute(&path, "power", calculation.power(index))?;
            file.write_f64_attribute(&path, "maxSAR", maximum)?;
            file.write_i32_attribute_vector(
                &path,
                "maxSAR_idx",
                &[
                    maximum_index[0] as i32,
                    maximum_index[1] as i32,
                    maximum_index[2] as i32,
                ],
            )?;

            if self.record_cube_stats && frequencies.len() == 1 {
                let mut cube_type = Vec::with_capacity(block_values.len());
                let mut cube_mass = Vec::with_capacity(block_values.len());
                let mut cube_volume = Vec::with_capacity(block_values.len());
                for i in start[0]..stop[0] {
                    for j in start[1]..stop[1] {
                        for k in start[2]..stop[2] {
                            let cell = calculation.cell_index([i, j, k]);
                            cube_type.push(calculation.cube_type[cell]);
                            cube_mass.push(calculation.cube_mass[cell] as f32);
                            cube_volume.push(calculation.cube_volume[cell] as f32);
                        }
                    }
                }
                file.write_u8_dataset_nd(
                    "/FieldData/FD",
                    &format!("{name}_CubeType"),
                    &block,
                    &cube_type,
                    None,
                )?;
                file.write_f32_dataset_nd(
                    "/FieldData/FD",
                    &format!("{name}_CubeMass"),
                    &block,
                    &cube_mass,
                    None,
                )?;
                file.write_f32_dataset_nd(
                    "/FieldData/FD",
                    &format!("{name}_CubeVol"),
                    &block,
                    &cube_volume,
                    None,
                )?;
            }
        }

        file.write_i32_attribute(
            "/FieldData/FD",
            "valid_cubes",
            calculation.valid_cells as i32,
        )?;
        file.write_i32_attribute("/FieldData/FD", "used_cubes", calculation.used_cells as i32)?;
        file.write_i32_attribute(
            "/FieldData/FD",
            "unused_cubes",
            calculation.unused_cells as i32,
        )?;
        file.write_i32_attribute("/FieldData/FD", "air_cubes", calculation.air_cells as i32)?;
        if self.auto_range > 0.0 {
            file.write_f64_attribute("/FieldData/FD", "autorange", self.auto_range)?;
        }
        Ok(true)
    }

    /// Cell index block written to the result: everything, or the cells within
    /// `auto_range` dB of the peak local SAR padded by half an averaging cube.
    fn auto_range_block(
        &self,
        calculation: &SarCalculation,
        num_lines: [usize; 3],
    ) -> ([usize; 3], [usize; 3]) {
        let mut start = [0usize; 3];
        let mut stop = num_lines;
        if self.auto_range <= 0.0 {
            return (start, stop);
        }
        let factor = 10.0f64.powf(self.auto_range / -10.0);
        let mut minimum = [usize::MAX; 3];
        let mut maximum = [0usize; 3];
        let mut found = false;
        let frequencies = calculation.frequencies().len();
        for frequency in 0..frequencies {
            let Some(local) = calculation.local_sar(frequency) else {
                continue;
            };
            let mut peak = 0.0f64;
            for (index, value) in local.iter().enumerate() {
                let density = calculation.cell_densities()[index];
                if density > 0.0 {
                    peak = peak.max(*value);
                }
            }
            if peak <= 0.0 {
                continue;
            }
            let threshold = factor * peak;
            for i in 0..num_lines[0] {
                for j in 0..num_lines[1] {
                    for k in 0..num_lines[2] {
                        let index = calculation.cell_index([i, j, k]);
                        if local[index] > threshold {
                            found = true;
                            minimum = [minimum[0].min(i), minimum[1].min(j), minimum[2].min(k)];
                            maximum = [
                                maximum[0].max(i + 1),
                                maximum[1].max(j + 1),
                                maximum[2].max(k + 1),
                            ];
                        }
                    }
                }
            }
        }
        if !found {
            return (start, stop);
        }
        // pad by half the edge length of an air free cube of the lightest tissue
        let mut lightest = 0.0f64;
        for (index, density) in calculation.cell_densities().iter().enumerate() {
            if *density > 0.0 && (lightest == 0.0 || *density < lightest) {
                lightest = *density;
                let _ = index;
            }
        }
        let mut padding = [0usize; 3];
        if self.averaging_mass > 0.0 && lightest > 0.0 {
            let half_edge = 0.5 * (self.averaging_mass / lightest).cbrt();
            let widths = calculation.cell_width();
            for axis in 0..3 {
                let mut travelled = 0.0;
                let mut count = 0;
                let mut position = minimum[axis];
                while position > 0 && travelled < half_edge {
                    position -= 1;
                    travelled += widths[axis][position];
                    count += 1;
                }
                padding[axis] = count;
            }
        }
        for axis in 0..3 {
            start[axis] = minimum[axis].saturating_sub(padding[axis]);
            stop[axis] = (maximum[axis] + padding[axis]).min(num_lines[axis]);
            if stop[axis] <= start[axis] {
                start[axis] = 0;
                stop[axis] = num_lines[axis];
            }
        }
        (start, stop)
    }
}

fn total_mass(calculation: &SarCalculation, start: [usize; 3], stop: [usize; 3]) -> f64 {
    let volumes = calculation.cell_volumes();
    let densities = calculation.cell_densities();
    let mut mass = 0.0;
    for i in start[0]..stop[0] {
        for j in start[1]..stop[1] {
            for k in start[2]..stop[2] {
                let index = calculation.cell_index([i, j, k]);
                mass += volumes[index] * densities[index];
            }
        }
    }
    mass
}

fn read_axis(file: &Hdf5File, group: &str, axis: &str) -> Result<Vec<f64>, SarError> {
    let path = format!("{group}/{axis}");
    if let Ok((_, values)) = file.read_f64_dataset(&path) {
        return Ok(values);
    }
    let (_, values) = file.read_f32_dataset(&path)?;
    Ok(values.into_iter().map(f64::from).collect())
}

fn read_scalar_field(file: &Hdf5File, path: &str) -> Result<Vec<f64>, SarError> {
    if let Ok((_, values)) = file.read_f32_dataset(path) {
        return Ok(values.into_iter().map(f64::from).collect());
    }
    let (_, values) = file.read_f64_dataset(path)?;
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(dims: [usize; 3], value: f64) -> crate::nf2ff::ComplexField {
        let mut field = crate::nf2ff::ComplexField::new(dims);
        for i in 0..dims[0] {
            for j in 0..dims[1] {
                for k in 0..dims[2] {
                    field.set(
                        0,
                        i,
                        j,
                        k,
                        crate::processing::ComplexPoint { re: value, im: 0.0 },
                    );
                }
            }
        }
        field
    }

    #[test]
    fn direct_result_dump_matches_the_local_sar_formula() {
        let dims = [2, 2, 2];
        let mesh = [vec![0.0, 1.0], vec![0.0, 1.0], vec![0.0, 1.0]];
        let widths = [vec![1.0, 1.0], vec![1.0, 1.0], vec![1.0, 1.0]];
        let volume = vec![1.0; 8];
        let density = vec![2.0; 8];
        let e_field = field(dims, 1.0);
        let j_field = field(dims, 4.0);

        let path = std::env::temp_dir()
            .join(format!("oems_sar_result_{}.h5", std::process::id()))
            .to_string_lossy()
            .into_owned();
        let mut runner = SarRunner::new();
        runner.averaging_mass = 0.0;
        let ok = runner
            .calc_from_fields(
                &mesh,
                widths,
                volume,
                density,
                None,
                &[1.0e9],
                std::slice::from_ref(&e_field),
                Some(std::slice::from_ref(&j_field)),
                &path,
            )
            .expect("the SAR result must be written");
        assert!(ok);

        let file = Hdf5File::open(&path, false).expect("open the SAR result");
        let (shape, values) = file
            .read_f32_dataset("/FieldData/FD/f0")
            .expect("read the SAR block");
        assert_eq!(shape, vec![2, 2, 2]);
        // 0.5 * |E| * |J| / rho = 0.5 * 1 * 4 / 2 = 1 W/kg
        for value in &values {
            assert!((value - 1.0).abs() < 1e-6, "value {value}");
        }
        let max = file
            .read_f64_attribute("/FieldData/FD/f0", "maxSAR")
            .expect("maxSAR attribute");
        assert!((max - 1.0).abs() < 1e-6, "maxSAR {max}");
        drop(file);
        let _ = std::fs::remove_file(path);
    }
}
