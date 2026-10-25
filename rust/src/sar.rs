//! Specific absorption rate (SAR) calculation with mass averaging.
//!
//! Ports `tools/sar_calculation.cpp`: the local power density
//! `0.5 sigma |E|^2` is divided by the cell density for the local SAR, and the
//! 1 g / 10 g values are obtained by averaging over a cubical mass centred on
//! each cell. The cube search uses the native Newton/bisection iteration and
//! the native partial-cell weights.

/// A cubical averaging region around one cell.
#[derive(Clone, Debug)]
pub struct AveragingCube {
    pub start: [usize; 3],
    pub stop: [usize; 3],
    pub partial_start: [f64; 3],
    pub partial_stop: [f64; 3],
    pub mass: f64,
    pub volume: f64,
    pub background_ratio: f64,
}

/// Outcome of the mass search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CubeStatus {
    /// A valid cube with the requested mass was found.
    Valid,
    /// The cube mass can never reach the target (too little material).
    MassTooSmall,
    /// The mass is right but the cube is not acceptable (air/face).
    Invalid,
    /// The iteration limit was reached.
    NoConvergence,
}

/// SAR calculation state.
pub struct SarCalculation {
    num_lines: [usize; 3],
    cell_width: [Vec<f64>; 3],
    cell_volume: Vec<f64>,
    cell_density: Vec<f64>,
    frequencies: Vec<f64>,
    /// Local power density per frequency and cell (W/m^3).
    power_density: Vec<Vec<f64>>,
    /// Total dissipated power per frequency (W).
    power: Vec<f64>,
    /// Maximum local SAR per frequency (W/kg).
    local_max_sar: Vec<f64>,
    /// Result map per frequency and cell (W/kg).
    sar: Vec<Vec<f64>>,
    /// Maximum result value per frequency.
    max_sar: Vec<f64>,
    max_index: Vec<[usize; 3]>,
    /// Averaging mass in kg; 0 selects the local SAR.
    pub averaging_mass: f64,
    /// Relative mass tolerance of the cube search.
    pub mass_tolerance: f64,
    /// Maximum allowed background (air) volume fraction inside a cube.
    pub max_background_ratio: f64,
    /// Unused cells accept a cube only if its volume is close to the smallest.
    pub unused_relative_volume_limit: f64,
    /// Accept cubes whose faces do not all touch material.
    pub ignore_face_valid: bool,
    /// Iteration limit of the mass search.
    pub max_iterations: usize,
    /// Number of valid centres in the last averaged run.
    pub valid_cells: usize,
    /// Number of cells assigned by the step-2 fallback.
    pub unused_cells: usize,
    /// Background (air) cells with zero density.
    pub air_cells: usize,
    /// Per-cell averaging cube kind (native `CubeType` dataset).
    pub cube_type: Vec<u8>,
    /// Mass enclosed by the cube assigned to a cell, in kg.
    pub cube_mass: Vec<f64>,
    /// Volume of the cube assigned to a cell, in m^3.
    pub cube_volume: Vec<f64>,
}

impl SarCalculation {
    /// Create a calculation for the given grid.
    pub fn new(num_lines: [usize; 3]) -> Self {
        let cells = num_lines[0] * num_lines[1] * num_lines[2];
        Self {
            num_lines,
            cell_width: [
                vec![0.0; num_lines[0]],
                vec![0.0; num_lines[1]],
                vec![0.0; num_lines[2]],
            ],
            cell_volume: vec![0.0; cells],
            cell_density: vec![0.0; cells],
            frequencies: Vec::new(),
            power_density: Vec::new(),
            power: Vec::new(),
            local_max_sar: Vec::new(),
            sar: Vec::new(),
            max_sar: Vec::new(),
            max_index: Vec::new(),
            averaging_mass: 0.0,
            mass_tolerance: 0.05,
            max_background_ratio: 0.5,
            unused_relative_volume_limit: 1.01,
            ignore_face_valid: false,
            max_iterations: 100,
            valid_cells: 0,
            unused_cells: 0,
            air_cells: 0,
            cube_type: Vec::new(),
            cube_mass: Vec::new(),
            cube_volume: Vec::new(),
        }
    }

    /// Grid extents.
    pub fn num_lines(&self) -> [usize; 3] {
        self.num_lines
    }

    fn index(&self, pos: [usize; 3]) -> usize {
        (pos[0] * self.num_lines[1] + pos[1]) * self.num_lines[2] + pos[2]
    }

    /// Set the cell widths per direction.
    pub fn set_cell_width(&mut self, widths: [Vec<f64>; 3]) {
        self.cell_width = widths;
        let mut volume = vec![0.0; self.cell_volume.len()];
        for i in 0..self.num_lines[0] {
            for j in 0..self.num_lines[1] {
                for k in 0..self.num_lines[2] {
                    let index = self.index([i, j, k]);
                    volume[index] = self.cell_width[0][i]
                        * self.cell_width[1][j]
                        * self.cell_width[2][k];
                }
            }
        }
        self.cell_volume = volume;
    }

    /// Set the cell volumes explicitly (takes precedence over the widths).
    pub fn set_cell_volume(&mut self, volumes: Vec<f64>) {
        if volumes.len() == self.cell_volume.len() {
            self.cell_volume = volumes;
        }
    }

    /// Set the cell mass density in kg/m^3.
    pub fn set_cell_density(&mut self, density: Vec<f64>) {
        if density.len() == self.cell_density.len() {
            self.cell_density = density;
        }
    }

    /// Add a frequency result from the E-field and the cell conductivity.
    pub fn add_e_field_and_conductivity(
        &mut self,
        frequency: f64,
        e_field: &crate::nf2ff::ComplexField,
        conductivity: &[f64],
    ) {
        let mut density = vec![0.0; self.cell_volume.len()];
        let mut power = 0.0;
        let mut max_local = 0.0f64;
        for i in 0..self.num_lines[0] {
            for j in 0..self.num_lines[1] {
                for k in 0..self.num_lines[2] {
                    let index = self.index([i, j, k]);
                    let sigma = conductivity.get(index).copied().unwrap_or(0.0);
                    let mut magnitude = 0.0;
                    for component in 0..3 {
                        let value = e_field.get(component, i, j, k);
                        magnitude += value.re * value.re + value.im * value.im;
                    }
                    let local = 0.5 * sigma * magnitude;
                    density[index] = local;
                    power += local * self.cell_volume[index];
                    let cell_density = self.cell_density[index];
                    if cell_density > 0.0 {
                        max_local = max_local.max(local / cell_density);
                    }
                }
            }
        }
        self.power_density.push(density);
        self.frequencies.push(frequency);
        self.power.push(power);
        self.local_max_sar.push(max_local);
    }

    /// Add a frequency result from the E- and J-fields.
    pub fn add_e_field_and_j_field(
        &mut self,
        frequency: f64,
        e_field: &crate::nf2ff::ComplexField,
        j_field: &crate::nf2ff::ComplexField,
    ) {
        let mut density = vec![0.0; self.cell_volume.len()];
        let mut power = 0.0;
        let mut max_local = 0.0f64;
        for i in 0..self.num_lines[0] {
            for j in 0..self.num_lines[1] {
                for k in 0..self.num_lines[2] {
                    let index = self.index([i, j, k]);
                    let mut magnitude = 0.0;
                    for component in 0..3 {
                        let e = e_field.get(component, i, j, k).magnitude();
                        let j_value = j_field.get(component, i, j, k).magnitude();
                        magnitude += e * j_value;
                    }
                    let local = 0.5 * magnitude;
                    density[index] = local;
                    power += local * self.cell_volume[index];
                    let cell_density = self.cell_density[index];
                    if cell_density > 0.0 {
                        max_local = max_local.max(local / cell_density);
                    }
                }
            }
        }
        self.power_density.push(density);
        self.frequencies.push(frequency);
        self.power.push(power);
        self.local_max_sar.push(max_local);
    }

    /// Compute the local or averaged SAR for every frequency.
    pub fn calc_sar(&mut self) {
        self.valid_cells = 0;
        self.unused_cells = 0;
        self.air_cells = 0;
        let cells = self.cell_volume.len();
        self.cube_type = vec![0; cells];
        self.cube_mass = vec![0.0; cells];
        self.cube_volume = vec![0.0; cells];
        self.max_sar = vec![0.0; self.frequencies.len()];
        self.max_index = vec![[0; 3]; self.frequencies.len()];
        if self.averaging_mass <= 0.0 {
            self.calc_local_sar();
        } else {
            self.calc_averaged_sar();
        }
    }

    fn calc_local_sar(&mut self) {
        let cells = self.cell_volume.len();
        let mut sar = vec![vec![0.0; cells]; self.frequencies.len()];
        for i in 0..self.num_lines[0] {
            for j in 0..self.num_lines[1] {
                for k in 0..self.num_lines[2] {
                    let index = self.index([i, j, k]);
                    let density = self.cell_density[index];
                    if density <= 0.0 {
                        self.air_cells += 1;
                        continue;
                    }
                    for (frequency, values) in self.power_density.iter().enumerate() {
                        let value = values[index] / density;
                        sar[frequency][index] = value;
                        if value > self.max_sar[frequency] {
                            self.max_sar[frequency] = value;
                            self.max_index[frequency] = [i, j, k];
                        }
                    }
                }
            }
        }
        self.sar = sar;
    }

    fn calc_averaged_sar(&mut self) {
        let cells = self.cell_volume.len();
        let mut valid = vec![false; cells];
        let mut used = vec![false; cells];
        let mut sar = vec![vec![0.0; cells]; self.frequencies.len()];

        for i in 0..self.num_lines[0] {
            for j in 0..self.num_lines[1] {
                for k in 0..self.num_lines[2] {
                    let pos = [i, j, k];
                    let index = self.index(pos);
                    let density = self.cell_density[index];
                    if density <= 0.0 {
                        continue;
                    }
                    let guess = (self.averaging_mass / density).cbrt() / 2.0;
                    let (status, cube) =
                        self.find_fitting_cubical_mass(pos, guess, None, self.ignore_face_valid);
                    if status != CubeStatus::Valid {
                        continue;
                    }
                    let values = self.calc_cubical_sar(&cube);
                    valid[index] = true;
                    used[index] = true;
                    self.valid_cells += 1;
                    self.cube_type[index] = 7;
                    self.cube_mass[index] = cube.mass;
                    self.cube_volume[index] = cube.volume;
                    for (frequency, value) in values.iter().enumerate() {
                        if *value > sar[frequency][index] {
                            sar[frequency][index] = *value;
                        }
                        if *value > self.max_sar[frequency] {
                            self.max_sar[frequency] = *value;
                            self.max_index[frequency] = pos;
                        }
                    }
                    self.assign_used_sar(&values, &cube, &mut used, &valid, &mut sar);
                }
            }
        }

        // Step 2: cells inside another cube but without their own valid cube.
        for i in 0..self.num_lines[0] {
            for j in 0..self.num_lines[1] {
                for k in 0..self.num_lines[2] {
                    let pos = [i, j, k];
                    let index = self.index(pos);
                    let density = self.cell_density[index];
                    if density <= 0.0 || valid[index] || used[index] {
                        continue;
                    }
                    self.unused_cells += 1;
                    let guess = (self.averaging_mass / density).cbrt() / 2.0;
                    let mut best = vec![0.0f64; self.frequencies.len()];
                    let mut min_volume = f64::MAX;
                    let mut candidates = Vec::new();
                    for face in 0..6 {
                        let (status, cube) =
                            self.find_fitting_cubical_mass(pos, guess, Some(face), true);
                        if status == CubeStatus::Valid {
                            min_volume = min_volume.min(cube.volume);
                            let values = self.calc_cubical_sar(&cube);
                            candidates.push((face, cube, values));
                        }
                    }
                    let mut chosen: Option<(usize, AveragingCube)> = None;
                    for (face, cube, values) in candidates {
                        if cube.volume > self.unused_relative_volume_limit * min_volume {
                            continue;
                        }
                        for (frequency, value) in values.iter().enumerate() {
                            if *value > best[frequency] {
                                best[frequency] = *value;
                            }
                        }
                        if chosen
                            .as_ref()
                            .map(|(_, current)| cube.volume <= current.volume)
                            .unwrap_or(true)
                        {
                            chosen = Some((face, cube));
                        }
                    }
                    if let Some((face, cube)) = &chosen {
                        if self.cube_type[index] == 0 {
                            self.cube_type[index] = (*face as u8) + 1;
                            self.cube_mass[index] = cube.mass;
                            self.cube_volume[index] = cube.volume;
                        }
                    }
                    for (frequency, value) in best.iter().enumerate() {
                        if *value > sar[frequency][index] {
                            sar[frequency][index] = *value;
                            if *value > self.max_sar[frequency] {
                                self.max_sar[frequency] = *value;
                                self.max_index[frequency] = pos;
                            }
                        }
                    }
                }
            }
        }

        self.sar = sar;
    }

    fn assign_used_sar(
        &mut self,
        values: &[f64],
        cube: &AveragingCube,
        used: &mut [bool],
        valid: &[bool],
        sar: &mut [Vec<f64>],
    ) {
        for i in cube.start[0]..=cube.stop[0] {
            for j in cube.start[1]..=cube.stop[1] {
                for k in cube.start[2]..=cube.stop[2] {
                    let index = self.index([i, j, k]);
                    if valid[index] || self.cell_density[index] <= 0.0 {
                        continue;
                    }
                    used[index] = true;
                    if self.cube_type[index] == 0 {
                        self.cube_type[index] = 8;
                        self.cube_mass[index] = cube.mass;
                        self.cube_volume[index] = cube.volume;
                    }
                    // Native stores the cube value for cells covered by a cube;
                    // a cell with its own valid cube keeps its own average.
                    for (frequency, value) in values.iter().enumerate() {
                        if *value > sar[frequency][index] {
                            sar[frequency][index] = *value;
                        }
                    }
                }
            }
        }
    }
    /// The cell widths along each axis.
    pub fn cell_width(&self) -> &[Vec<f64>; 3] {
        &self.cell_width
    }

    /// Per-cell volumes in m^3.
    pub fn cell_volumes(&self) -> &[f64] {
        &self.cell_volume
    }

    /// Per-cell mass density in kg/m^3.
    pub fn cell_densities(&self) -> &[f64] {
        &self.cell_density
    }

    /// Local SAR per cell of one frequency (W/kg).
    pub fn local_sar(&self, frequency: usize) -> Option<Vec<f64>> {
        let power_density = self.power_density.get(frequency)?;
        Some(
            power_density
                .iter()
                .zip(self.cell_density.iter())
                .map(|(power, density)| {
                    if *density > 0.0 {
                        power / density
                    } else {
                        0.0
                    }
                })
                .collect(),
        )
    }

    /// Global cell index for a position (native `ArrayIJK` order).
    pub fn cell_index(&self, position: [usize; 3]) -> usize {
        self.index(position)
    }

    /// SAR map of one frequency (W/kg).
    pub fn sar(&self, frequency: usize) -> Option<&[f64]> {
        self.sar.get(frequency).map(|values| values.as_slice())
    }

    /// Maximum SAR of one frequency (W/kg).
    pub fn max_sar(&self, frequency: usize) -> f64 {
        self.max_sar.get(frequency).copied().unwrap_or(0.0)
    }

    /// Index of the maximum SAR of one frequency.
    pub fn max_index(&self, frequency: usize) -> [usize; 3] {
        self.max_index
            .get(frequency)
            .copied()
            .unwrap_or([0; 3])
    }

    /// Total dissipated power of one frequency (W).
    pub fn power(&self, frequency: usize) -> f64 {
        self.power.get(frequency).copied().unwrap_or(0.0)
    }

    /// Frequencies in insertion order.
    pub fn frequencies(&self) -> &[f64] {
        &self.frequencies
    }

    fn find_fitting_cubical_mass(
        &self,
        pos: [usize; 3],
        mut box_size: f64,
        disabled_face: Option<usize>,
        ignore_face_valid: bool,
    ) -> (CubeStatus, AveragingCube) {
        let target = self.averaging_mass;
        let tolerance = self.mass_tolerance * target;
        let mut old_mass = 0.0;
        let mut old_box_size = 0.0;
        let mut min_size = 0.0f64;
        let mut max_size = 1.0e99f64;
        let mut cube = AveragingCube {
            start: pos,
            stop: pos,
            partial_start: [0.0; 3],
            partial_stop: [0.0; 3],
            mass: 0.0,
            volume: 0.0,
            background_ratio: 0.0,
        };
        for iteration in 0..self.max_iterations {
            let face_valid = self.get_cubical_mass(pos, box_size, disabled_face, &mut cube);
            let mass_valid = (cube.mass - target).abs() <= tolerance;
            if !face_valid && cube.mass < target * (1.0 - self.mass_tolerance) && !ignore_face_valid
            {
                return (CubeStatus::MassTooSmall, cube);
            }
            if (!face_valid || cube.background_ratio >= self.max_background_ratio) && mass_valid {
                return (
                    if ignore_face_valid {
                        CubeStatus::Valid
                    } else {
                        CubeStatus::Invalid
                    },
                    cube,
                );
            }
            if mass_valid
                && face_valid
                && cube.background_ratio < self.max_background_ratio
            {
                return (CubeStatus::Valid, cube);
            }
            if iteration > 0
                && !face_valid
                && cube.mass == old_mass
                && box_size > old_box_size
                && cube.background_ratio >= self.max_background_ratio
            {
                return (CubeStatus::Invalid, cube);
            }
            if cube.mass < target {
                min_size = min_size.max(box_size);
            } else if cube.mass > target {
                max_size = max_size.min(box_size);
            }
            if iteration == 0 {
                old_box_size = box_size;
                box_size *= (target / cube.mass).cbrt();
            } else if cube.mass == old_mass {
                old_box_size = box_size;
                box_size = if cube.mass > target {
                    0.5 * (min_size + box_size)
                } else {
                    (0.5 * (max_size + box_size)).min(2.0 * box_size)
                };
            } else {
                let mut new_size = box_size
                    - (cube.mass - target) / (cube.mass - old_mass) * (box_size - old_box_size);
                if new_size < min_size {
                    new_size = 0.5 * (min_size + box_size);
                } else if new_size > max_size {
                    new_size = 0.5 * (max_size + box_size);
                }
                old_box_size = box_size;
                box_size = new_size;
            }
            old_mass = cube.mass;
            if !box_size.is_finite() || box_size <= 0.0 {
                return (CubeStatus::NoConvergence, cube);
            }
        }
        (CubeStatus::NoConvergence, cube)
    }

    fn get_cubical_mass(
        &self,
        pos: [usize; 3],
        box_size: f64,
        disabled_face: Option<usize>,
        cube: &mut AveragingCube,
    ) -> bool {
        let mut face_valid = true;
        for axis in 0..3 {
            let mut half = box_size;
            let mut start = pos[axis];
            let mut partial_start = 1.0;
            let mut distance = self.cell_width[axis][pos[axis]] / 2.0;
            if disabled_face == Some(2 * axis) || disabled_face == Some(2 * axis + 1) {
                half = 2.0 * box_size - self.cell_width[axis][pos[axis]] / 2.0;
            }
            if disabled_face != Some(2 * axis) {
                while distance < half {
                    if start == 0 {
                        partial_start = -1.0;
                        break;
                    }
                    start -= 1;
                    distance += self.cell_width[axis][start];
                }
                if partial_start != -1.0 {
                    partial_start =
                        1.0 - (distance - half) / self.cell_width[axis][start];
                }
                if partial_start != -1.0 && pos[axis] == start {
                    partial_start = 2.0 * half / self.cell_width[axis][start];
                }
            }

            let mut stop = pos[axis];
            let mut partial_stop = 1.0;
            distance = self.cell_width[axis][pos[axis]] / 2.0;
            if disabled_face != Some(2 * axis + 1) {
                while distance < half {
                    if stop == self.num_lines[axis] - 1 {
                        partial_stop = -1.0;
                        break;
                    }
                    stop += 1;
                    distance += self.cell_width[axis][stop];
                }
                if partial_stop != -1.0 {
                    partial_stop = 1.0 - (distance - half) / self.cell_width[axis][stop];
                }
                if partial_stop != -1.0 && pos[axis] == stop {
                    partial_stop = 2.0 * half / self.cell_width[axis][stop];
                }
            }
            cube.start[axis] = start;
            cube.stop[axis] = stop;
            cube.partial_start[axis] = partial_start;
            cube.partial_stop[axis] = partial_stop;
            if partial_start == -1.0 || partial_stop == -1.0 {
                face_valid = false;
            }
        }

        let mut mass = 0.0;
        let mut volume = 0.0;
        let mut background_volume = 0.0;
        let mut face_intersect = [false; 6];
        for i in cube.start[0]..=cube.stop[0] {
            let weight_i = 1.0
                * if i == cube.start[0] {
                    cube.partial_start[0].abs()
                } else {
                    1.0
                }
                * if i == cube.stop[0] {
                    cube.partial_stop[0].abs()
                } else {
                    1.0
                };
            for j in cube.start[1]..=cube.stop[1] {
                let weight_j = 1.0
                    * if j == cube.start[1] {
                        cube.partial_start[1].abs()
                    } else {
                        1.0
                    }
                    * if j == cube.stop[1] {
                        cube.partial_stop[1].abs()
                    } else {
                        1.0
                    };
                for k in cube.start[2]..=cube.stop[2] {
                    let weight_k = 1.0
                        * if k == cube.start[2] {
                            cube.partial_start[2].abs()
                        } else {
                            1.0
                        }
                        * if k == cube.stop[2] {
                            cube.partial_stop[2].abs()
                        } else {
                            1.0
                        };
                    let index = self.index([i, j, k]);
                    let cell_volume =
                        self.cell_volume[index] * weight_i * weight_j * weight_k;
                    let density = self.cell_density[index];
                    mass += density * cell_volume;
                    volume += cell_volume;
                    if density == 0.0 {
                        background_volume += cell_volume;
                    } else {
                        for axis in 0..3 {
                            let coordinate = [i, j, k][axis];
                            if cube.start[axis] == coordinate {
                                face_intersect[2 * axis] = true;
                            }
                            if cube.stop[axis] == coordinate {
                                face_intersect[2 * axis + 1] = true;
                            }
                        }
                    }
                }
            }
        }
        for value in face_intersect {
            face_valid &= value;
        }
        cube.mass = mass;
        cube.volume = volume;
        cube.background_ratio = if volume > 0.0 {
            background_volume / volume
        } else {
            0.0
        };
        face_valid
    }

    fn calc_cubical_sar(&self, cube: &AveragingCube) -> Vec<f64> {
        let mut values = vec![0.0f64; self.frequencies.len()];
        let mut mass = 0.0;
        for i in cube.start[0]..=cube.stop[0] {
            let weight_i = 1.0
                * if i == cube.start[0] {
                    cube.partial_start[0].abs()
                } else {
                    1.0
                }
                * if i == cube.stop[0] {
                    cube.partial_stop[0].abs()
                } else {
                    1.0
                };
            for j in cube.start[1]..=cube.stop[1] {
                let weight_j = 1.0
                    * if j == cube.start[1] {
                        cube.partial_start[1].abs()
                    } else {
                        1.0
                    }
                    * if j == cube.stop[1] {
                        cube.partial_stop[1].abs()
                    } else {
                        1.0
                    };
                for k in cube.start[2]..=cube.stop[2] {
                    let weight_k = 1.0
                        * if k == cube.start[2] {
                            cube.partial_start[2].abs()
                        } else {
                            1.0
                        }
                        * if k == cube.stop[2] {
                            cube.partial_stop[2].abs()
                        } else {
                            1.0
                        };
                    let index = self.index([i, j, k]);
                    let weight = weight_i * weight_j * weight_k;
                    let density = self.cell_density[index];
                    if density < 0.0 {
                        continue;
                    }
                    mass += density * self.cell_volume[index] * weight;
                    for (frequency, value) in values.iter_mut().enumerate() {
                        *value += self.power_density[frequency][index]
                            * self.cell_volume[index]
                            * weight;
                    }
                }
            }
        }
        if mass != 0.0 {
            for value in &mut values {
                *value /= mass;
            }
        }
        values
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nf2ff::ComplexField;
    use crate::processing::ComplexPoint;

    fn uniform_block(cells: usize, width: f64, density: f64) -> SarCalculation {
        let lines = cells + 1;
        let mut calc = SarCalculation::new([lines, lines, lines]);
        calc.set_cell_width([
            vec![width; lines],
            vec![width; lines],
            vec![width; lines],
        ]);
        calc.set_cell_density(vec![density; lines * lines * lines]);
        calc
    }

    #[test]
    fn local_sar_matches_the_definition() {
        let mut calc = uniform_block(5, 1.0e-3, 1000.0);
        let mut e = ComplexField::new([6, 6, 6]);
        for i in 0..6 {
            for j in 0..6 {
                for k in 0..6 {
                    e.set(0, i, j, k, ComplexPoint { re: 3.0, im: 4.0 });
                }
            }
        }
        let conductivity = vec![2.0; 6 * 6 * 6];
        calc.add_e_field_and_conductivity(1.0e9, &e, &conductivity);
        calc.calc_sar();
        // 0.5 * sigma * |E|^2 / rho = 0.5 * 2 * 25 / 1000
        let expected = 0.5 * 2.0 * 25.0 / 1000.0;
        let sar = calc.sar(0).unwrap();
        // Interior cells are away from the PEC-like domain edge.
        let index = calc.index([2, 2, 2]);
        assert!((sar[index] - expected).abs() / expected < 1e-12);
        assert!((calc.max_sar(0) - expected).abs() / expected < 1e-12);
    }

    #[test]
    fn averaged_sar_of_a_uniform_block_equals_the_local_value() {
        let mut calc = uniform_block(9, 1.0e-3, 1000.0);
        // Eight 1 mm cells at 1000 kg/m^3: 8 mg.
        let cube_mass = 8.0 * 1.0e-9 * 1000.0;
        calc.averaging_mass = cube_mass;
        let mut e = ComplexField::new([10, 10, 10]);
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    e.set(0, i, j, k, ComplexPoint { re: 1.0, im: 0.0 });
                }
            }
        }
        let conductivity = vec![1.0; 10 * 10 * 10];
        calc.add_e_field_and_conductivity(1.0e9, &e, &conductivity);
        calc.calc_sar();
        let expected = 0.5 * 1.0 / 1000.0;
        let index = calc.index([4, 4, 4]);
        let sar = calc.sar(0).unwrap();
        assert!(
            (sar[index] - expected).abs() / expected < 1e-9,
            "averaged SAR {} vs {}",
            sar[index],
            expected
        );
        assert!(calc.valid_cells > 0);
    }

    #[test]
    fn air_cells_are_skipped_and_marked_zero() {
        let mut calc = uniform_block(4, 1.0e-3, 1000.0);
        let mut density = vec![0.0; 5 * 5 * 5];
        let index = calc.index([2, 2, 2]);
        density[index] = 1000.0;
        calc.set_cell_density(density);
        let mut e = ComplexField::new([5, 5, 5]);
        e.set(0, 2, 2, 2, ComplexPoint { re: 1.0, im: 0.0 });
        calc.add_e_field_and_conductivity(1.0e9, &e, &vec![1.0; 5 * 5 * 5]);
        calc.calc_sar();
        let sar = calc.sar(0).unwrap();
        assert_eq!(sar[0], 0.0);
        assert!(sar[index] > 0.0);
    }
}
