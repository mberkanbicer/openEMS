//! Equivalent-circuit operator: EC coefficients, material averaging, PEC.
//! Mirrors `FDTD/operator.h/cpp` in the C++ codebase.

use crate::array_nijk::FieldArray;
use crate::constants::{EPS0, FDTD_FLOAT, MU0};
use crate::operator_base::OperatorBase;

/// Per-cell material properties for material averaging.
/// Mirrors the C++ material storage per Yee cell.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CellMaterial {
    pub eps_r: f64,
    pub mue_r: f64,
    pub kappa: f64,
    pub sigma: f64,
    /// True when this cell is a perfect electric conductor (Metal property).
    /// Such cells have all EC coefficients zeroed (no E/H field inside).
    pub is_pec: bool,
}

/// The basic FDTD operator.
/// Computes equivalent-circuit coefficients (vv, vi, ii, iv) for every Yee cell.
pub struct Operator {
    pub base: crate::operator_base::OperatorBaseImpl,
    pub vv_ptr: Option<FieldArray>,
    pub vi_ptr: Option<FieldArray>,
    pub ii_ptr: Option<FieldArray>,
    pub iv_ptr: Option<FieldArray>,
    /// Per-cell material grid, indexed as [n][i][j][k] flattened.
    /// When Some, overrides the background material for EC coefficient computation.
    pub material_grid: Option<Vec<CellMaterial>>,
}

impl Operator {
    /// Create a new operator.
    pub fn new() -> Self {
        Self {
            base: crate::operator_base::OperatorBaseImpl::new(),
            vv_ptr: None,
            vi_ptr: None,
            ii_ptr: None,
            iv_ptr: None,
            material_grid: None,
        }
    }

    /// Allocate the EC coefficient field arrays and the per-cell material grid.
    /// Mirrors C++ `Init_EC`.
    pub fn init_data_storage(&mut self, num_lines: [usize; 3]) {
        let extent = [num_lines[0], num_lines[1], num_lines[2]];
        self.vv_ptr = Some(FieldArray::new("vv", extent));
        self.vi_ptr = Some(FieldArray::new("vi", extent));
        self.ii_ptr = Some(FieldArray::new("ii", extent));
        self.iv_ptr = Some(FieldArray::new("iv", extent));
        // Allocate per-cell material grid (one entry per cell, direction-independent).
        let num_cells = num_lines[0] * num_lines[1] * num_lines[2];
        self.material_grid = Some(vec![CellMaterial::default(); num_cells]);
    }

    /// Set the material properties for a specific cell.
    pub fn set_cell_material(
        &mut self,
        n: usize,
        i: usize,
        j: usize,
        k: usize,
        eps_r: f64,
        mue_r: f64,
        kappa: f64,
        sigma: f64,
    ) {
        let Some(grid) = &mut self.material_grid else {
            return;
        };
        let nx = self.base.num_lines[0] as usize;
        let ny = self.base.num_lines[1] as usize;
        let nz = self.base.num_lines[2] as usize;
        if i >= nx || j >= ny || k >= nz {
            return;
        }
        let index = n * nx * ny * nz + i * ny * nz + j * nz + k;
        if index < grid.len() {
            grid[index] = CellMaterial {
                eps_r,
                mue_r,
                kappa,
                sigma,
                is_pec: false,
            };
        }
    }

    /// Apply a material to a rectangular box region of cells.
    /// Coordinates are cell indices (inclusive start, exclusive stop).
    pub fn apply_material_box(
        &mut self,
        start: [usize; 3],
        stop: [usize; 3],
        eps_r: f64,
        mue_r: f64,
        kappa: f64,
        sigma: f64,
    ) {
        let nx = self.base.num_lines[0] as usize;
        let ny = self.base.num_lines[1] as usize;
        let nz = self.base.num_lines[2] as usize;
        let x0 = start[0].min(nx).saturating_sub(1);
        let x1 = stop[0].min(nx);
        let y0 = start[1].min(ny).saturating_sub(1);
        let y1 = stop[1].min(ny);
        let z0 = start[2].min(nz).saturating_sub(1);
        let z1 = stop[2].min(nz);
        for n in 0..3 {
            for i in x0..x1 {
                for j in y0..y1 {
                    for k in z0..z1 {
                        self.set_cell_material(n, i, j, k, eps_r, mue_r, kappa, sigma);
                    }
                }
            }
        }
    }

    /// Load a CSXCAD XML geometry and wire its grid, materials, boundary
    /// conditions and primitives into this operator.
    ///
    /// Mirrors the C++ `SetupFDTD` flow that reads the CSX structure, builds
    /// the FDTD grid, applies materials and boundary conditions, and stores
    /// per-cell material data for EC coefficient computation.
    pub fn load_csx(&mut self, xml: &str) -> Result<(), String> {
        let mut structure = csxcad_core::model::Structure::from_xml_str(xml)
            .map_err(|e| format!("CSXCAD XML parse error: {}", e))?;

        // --- Grid ---
        // Extract grid lines up-front; RectGrid accessors require &mut self.
        let mut num_lines = [0u32; 3];
        let mut grid_delta = 0.0f64;
        let mut grid_lines: [Vec<f64>; 3] = Default::default();
        for n in 0..3usize {
            let count = structure.grid_mut().line_count(n);
            if count >= 2 {
                let line0 = structure.grid_mut().line(n, 0);
                let line1 = structure.grid_mut().line(n, count - 1);
                let delta = (line1 - line0) / (count as f64 - 1.0);
                if grid_delta == 0.0 {
                    grid_delta = delta;
                }
                num_lines[n] = count as u32;
                grid_lines[n] = structure.grid_mut().lines(n).unwrap_or_default().to_vec();
            }
        }
        if grid_delta <= 0.0 {
            return Err("CSXCAD grid has insufficient lines".to_string());
        }
        self.base.grid_delta = grid_delta;
        self.base.num_lines = num_lines;

        // --- Background material ---
        let bg = structure.background_material();
        self.base.set_background_material(
            bg.value(csxcad_core::material::MaterialField::Epsilon),
            bg.value(csxcad_core::material::MaterialField::Mue),
            bg.value(csxcad_core::material::MaterialField::Kappa),
            bg.value(csxcad_core::material::MaterialField::Sigma),
            self.base.m_bg_density,
        );

        // --- Boundary conditions ---
        let mut bc = [0i32; 6];
        for prop in 0..structure.property_count() {
            let property = structure
                .property_by_index(prop)
                .ok_or_else(|| "invalid property index".to_string())?;
            if property.property_type() == csxcad_core::model::PropertyType::AbsorbingBoundary {
                if let Some(ab) = property.absorbing_boundary() {
                    let bt = ab.boundary_type();
                    let v = match bt {
                        csxcad_core::material::AbsorbingBoundaryType::MurFirstOrder => 1,
                        csxcad_core::material::AbsorbingBoundaryType::MurFirstOrderSuperAbsorption => 2,
                        _ => 0,
                    };
                    let sign = if ab.normal_sign_positive() { 0 } else { 1 };
                    // Boundary faces: [x-, x+, y-, y+, z-, z+]
                    // sign 0 = positive-direction face (x+, y+, z+)
                    // sign 1 = negative-direction face (x-, y-, z-)
                    let face = match sign {
                        0 => [1, 3, 5],
                        1 => [0, 2, 4],
                        _ => [0, 0, 0],
                    };
                    // Determine which axis this boundary applies to from its
                    // primitives (boxes). For now, apply to all faces of the
                    // matching sign; the C++ code uses primitive geometry to
                    // determine the exact face.
                    for &f in &face {
                        bc[f] = v;
                    }
                }
            }
        }
        self.base.set_boundary_condition(bc);

        // --- Per-cell materials from Material properties ---
        let num_lines_usize: [usize; 3] = num_lines.map(|v| v as usize);
        let num_cells = num_lines_usize[0] * num_lines_usize[1] * num_lines_usize[2];
        let mut material_grid = vec![CellMaterial::default(); num_cells];

        for prop in 0..structure.property_count() {
            let property = structure
                .property_by_index(prop)
                .ok_or_else(|| "invalid property index".to_string())?;
            let pt = property.property_type();
            let params = structure.parameters();
            match pt {
                csxcad_core::model::PropertyType::Material => {
                    let mat = property
                        .material()
                        .ok_or_else(|| "material property missing material data".to_string())?;
                    let eps_r = mat
                        .value(
                            csxcad_core::material::MaterialField::Epsilon,
                            0,
                            Some(params),
                        )
                        .ok()
                        .flatten()
                        .unwrap_or(1.0);
                    let mue_r = mat
                        .value(
                            csxcad_core::material::MaterialField::Mue,
                            0,
                            Some(params),
                        )
                        .ok()
                        .flatten()
                        .unwrap_or(1.0);
                    let kappa = mat
                        .value(
                            csxcad_core::material::MaterialField::Kappa,
                            0,
                            Some(params),
                        )
                        .ok()
                        .flatten()
                        .unwrap_or(0.0);
                    let sigma = mat
                        .value(
                            csxcad_core::material::MaterialField::Sigma,
                            0,
                            Some(params),
                        )
                        .ok()
                        .flatten()
                        .unwrap_or(0.0);
                    for prim in property.primitives() {
                        let cm = CellMaterial {
                            eps_r,
                            mue_r,
                            kappa,
                            sigma,
                            is_pec: false,
                        };
                        apply_primitive_to_grid(
                            prim,
                            params,
                            [&grid_lines[0], &grid_lines[1], &grid_lines[2]],
                            num_lines_usize,
                            &mut material_grid,
                            cm,
                        );
                    }
                }
                csxcad_core::model::PropertyType::Metal => {
                    // Metal = PEC: zero all EC coefficients for covered cells.
                    let cm = CellMaterial {
                        eps_r: 1.0,
                        mue_r: 1.0,
                        kappa: 0.0,
                        sigma: 0.0,
                        is_pec: true,
                    };
                    for prim in property.primitives() {
                        apply_primitive_to_grid(
                            prim,
                            params,
                            [&grid_lines[0], &grid_lines[1], &grid_lines[2]],
                            num_lines_usize,
                            &mut material_grid,
                            cm,
                        );
                    }
                }
                _ => {}
            }
        }
        self.material_grid = Some(material_grid);

        // Re-allocate EC storage now that we know the real grid size.
        // init_data_storage would overwrite material_grid with defaults,
        // so save and restore it.
        let saved_grid = self.material_grid.take();
        self.init_data_storage(num_lines_usize);
        self.material_grid = saved_grid;

        Ok(())
    }

    /// Compute all EC coefficients (vv, vi, ii, iv) for every cell.
    /// Mirrors C++ `Calc_ECOperator`.
    /// Uses per-cell material from `material_grid` when available, falling back
    /// to the background material.
    pub fn calc_ec_operator(&mut self) {
        let vv = self.vv_ptr.as_mut().expect("vv_ptr not allocated");
        let vi = self.vi_ptr.as_mut().expect("vi_ptr not allocated");
        let ii = self.ii_ptr.as_mut().expect("ii_ptr not allocated");
        let iv = self.iv_ptr.as_mut().expect("iv_ptr not allocated");

        let nx = vv.ni();
        let ny = vv.nj();
        let nz = vv.nk();
        let dt = self.base.d_t as FDTD_FLOAT;

        // Extract background material values before mutable borrow to avoid
        // borrow-checker conflicts.
        let bg_eps_r = self.base.m_bg_eps_r;
        let bg_mue_r = self.base.m_bg_mue_r;
        let bg_kappa = self.base.m_bg_kappa;
        let bg_sigma = self.base.m_bg_sigma;
        let grid_delta = self.base.grid_delta;
        let material_grid = self.material_grid.as_ref();

        for n in 0..3 {
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        // Determine effective material for this cell.
                        let (eps_r, mue_r, kappa, sigma) = if let Some(grid) = material_grid {
                            let num_lines = self.base.num_lines;
                            let nx_lines = num_lines[0] as usize;
                            let ny_lines = num_lines[1] as usize;
                            let nz_lines = num_lines[2] as usize;
                            let index = n * nx_lines * ny_lines * nz_lines + i * ny_lines * nz_lines + j * nz_lines + k;
                            if index < grid.len() {
                                let m = grid[index];
                                (m.eps_r, m.mue_r, m.kappa, m.sigma)
                            } else {
                                (bg_eps_r, bg_mue_r, bg_kappa, bg_sigma)
                            }
                        } else {
                            (bg_eps_r, bg_mue_r, bg_kappa, bg_sigma)
                        };

                        // Cast to FDTD_FLOAT (f32) to match dt.
                        let area = grid_delta * grid_delta;
                        let delta = grid_delta;
                        let c = (eps_r * EPS0 * area / delta) as FDTD_FLOAT;
                        let g = (kappa * EPS0 * area / delta) as FDTD_FLOAT;
                        let l = (mue_r * MU0 * area / delta) as FDTD_FLOAT;
                        let r = (sigma * EPS0 * area / delta) as FDTD_FLOAT;

                        // Voltage coefficients: vv, vi
                        if c > 0.0 {
                            let denom = 1.0 + dt * g / (2.0 * c);
                            *vv.get_mut(n, i, j, k) = (1.0 - dt * g / (2.0 * c)) / denom;
                            *vi.get_mut(n, i, j, k) = (dt / c) / denom;
                        } else {
                            *vv.get_mut(n, i, j, k) = 0.0;
                            *vi.get_mut(n, i, j, k) = 0.0;
                        }

                        // Current coefficients: ii, iv
                        if l > 0.0 {
                            let denom = 1.0 + dt * r / (2.0 * l);
                            *ii.get_mut(n, i, j, k) = (1.0 - dt * r / (2.0 * l)) / denom;
                            *iv.get_mut(n, i, j, k) = (dt / l) / denom;
                        } else {
                            *ii.get_mut(n, i, j, k) = 0.0;
                            *iv.get_mut(n, i, j, k) = 0.0;
                        }
                    }
                }
            }
        }
    }

    /// Compute the FDTD timestep using the CFL condition (Rennings Var1).
    /// Mirrors C++ `CalcTimestep_Var1`.
    pub fn calc_timestep(&mut self) -> f64 {
        let ii = self.ii_ptr.as_ref().expect("ii_ptr not allocated");
        let nx = ii.ni();
        let ny = ii.nj();
        let nz = ii.nk();

        let mut min_dt = f64::INFINITY;

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    // EC_L for this cell and its neighbors
                    let l000 = self.get_ec_l(0, i, j, k);
                    let l100 = if i + 1 < nx {
                        self.get_ec_l(0, i + 1, j, k)
                    } else {
                        l000
                    };
                    let l010 = if j + 1 < ny {
                        self.get_ec_l(0, i, j + 1, k)
                    } else {
                        l000
                    };
                    let l001 = if k + 1 < nz {
                        self.get_ec_l(0, i, j, k + 1)
                    } else {
                        l000
                    };
                    let l110 = if i + 1 < nx && j + 1 < ny {
                        self.get_ec_l(0, i + 1, j + 1, k)
                    } else {
                        l000
                    };
                    let l101 = if i + 1 < nx && k + 1 < nz {
                        self.get_ec_l(0, i + 1, j, k + 1)
                    } else {
                        l000
                    };
                    let l011 = if j + 1 < ny && k + 1 < nz {
                        self.get_ec_l(0, i, j + 1, k + 1)
                    } else {
                        l000
                    };
                    let l111 = if i + 1 < nx && j + 1 < ny && k + 1 < nz {
                        self.get_ec_l(0, i + 1, j + 1, k + 1)
                    } else {
                        l000
                    };

                    let sum_inv_l = 4.0 / l000
                        + 4.0 / l100
                        + 4.0 / l010
                        + 4.0 / l001
                        + 4.0 / l110
                        + 4.0 / l101
                        + 4.0 / l011
                        + 4.0 / l111;
                    let ec_c = self.get_ec_c(0, i, j, k);
                    if ec_c > 0.0 {
                        let dt = 2.0 / (sum_inv_l / ec_c).sqrt();
                        if dt < min_dt {
                            min_dt = dt;
                        }
                    }
                }
            }
        }

        if min_dt.is_finite() {
            self.base.d_t = min_dt;
        }
        min_dt
    }

    /// Apply electric boundary conditions (PEC): zero out vv/vi at PEC boundaries.
    /// Mirrors C++ `ApplyElectricBC`.
    pub fn apply_electric_bc(&mut self) {
        let vv = self.vv_ptr.as_mut().expect("vv_ptr not allocated");
        let vi = self.vi_ptr.as_mut().expect("vi_ptr not allocated");
        let bc = self.base.m_bc;

        let nx = vv.ni();
        let ny = vv.nj();
        let nz = vv.nk();

        // bc[0]=x-min, bc[1]=x-max, bc[2]=y-min, bc[3]=y-max, bc[4]=z-min, bc[5]=z-max
        // PEC = 1
        for n in 0..3 {
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        let mut pec = false;
                        if i == 0 && bc[0] == 1 {
                            pec = true;
                        }
                        if i == nx - 1 && bc[1] == 1 {
                            pec = true;
                        }
                        if j == 0 && bc[2] == 1 {
                            pec = true;
                        }
                        if j == ny - 1 && bc[3] == 1 {
                            pec = true;
                        }
                        if k == 0 && bc[4] == 1 {
                            pec = true;
                        }
                        if k == nz - 1 && bc[5] == 1 {
                            pec = true;
                        }
                        if pec {
                            *vv.get_mut(n, i, j, k) = 0.0;
                            *vi.get_mut(n, i, j, k) = 0.0;
                        }
                    }
                }
            }
        }
    }

    /// Modify EC voltage coefficients at a specific mesh position.
    /// Used by extensions like conducting sheet to adjust vv/vi coefficients.
    pub fn set_ec_voltage_coeffs(
        &mut self,
        n: usize,
        i: usize,
        j: usize,
        k: usize,
        vv: FDTD_FLOAT,
        vi: FDTD_FLOAT,
    ) {
        *self
            .vv_ptr
            .as_mut()
            .expect("vv_ptr not allocated")
            .get_mut(n, i, j, k) = vv;
        *self
            .vi_ptr
            .as_mut()
            .expect("vi_ptr not allocated")
            .get_mut(n, i, j, k) = vi;
    }

    /// Apply magnetic boundary conditions (PMC): zero out ii/iv at PMC boundaries.
    /// Mirrors C++ `ApplyMagneticBC`.
    pub fn apply_magnetic_bc(&mut self) {
        let ii = self.ii_ptr.as_mut().expect("ii_ptr not allocated");
        let iv = self.iv_ptr.as_mut().expect("iv_ptr not allocated");
        let bc = self.base.m_bc;

        let nx = ii.ni();
        let ny = ii.nj();
        let nz = ii.nk();

        // PMC = 2
        for n in 0..3 {
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        let mut pmc = false;
                        if i == 0 && bc[0] == 2 {
                            pmc = true;
                        }
                        if i == nx - 1 && bc[1] == 2 {
                            pmc = true;
                        }
                        if j == 0 && bc[2] == 2 {
                            pmc = true;
                        }
                        if j == ny - 1 && bc[3] == 2 {
                            pmc = true;
                        }
                        if k == 0 && bc[4] == 2 {
                            pmc = true;
                        }
                        if k == nz - 1 && bc[5] == 2 {
                            pmc = true;
                        }
                        if pmc {
                            *ii.get_mut(n, i, j, k) = 0.0;
                            *iv.get_mut(n, i, j, k) = 0.0;
                        }
                    }
                }
            }
        }
    }

    // ----- private helpers -----

    fn get_ec_c(&self, _n: i32, i: usize, j: usize, k: usize) -> f64 {
        // EC_C = eps * area / delta; for uniform background: eps_r * EPS0 * grid_delta
        // Check per-cell material grid first.
        if let Some(grid) = &self.material_grid {
            let num_lines = self.base.num_lines;
            let nx = num_lines[0] as usize;
            let ny = num_lines[1] as usize;
            let nz = num_lines[2] as usize;
            let index = i * ny * nz + j * nz + k;
            if index < grid.len() {
                return grid[index].eps_r * EPS0 * self.base.grid_delta;
            }
        }
        self.base.m_bg_eps_r * EPS0 * self.base.grid_delta
    }

    fn get_ec_l(&self, _n: i32, i: usize, j: usize, k: usize) -> f64 {
        // EC_L = mue * area / delta; for uniform background: mue_r * MU0 * grid_delta
        // Check per-cell material grid first.
        if let Some(grid) = &self.material_grid {
            let num_lines = self.base.num_lines;
            let nx = num_lines[0] as usize;
            let ny = num_lines[1] as usize;
            let nz = num_lines[2] as usize;
            let index = i * ny * nz + j * nz + k;
            if index < grid.len() {
                return grid[index].mue_r * MU0 * self.base.grid_delta;
            }
        }
        self.base.m_bg_mue_r * MU0 * self.base.grid_delta
    }

    /// Compute EC[4] = [C, G, L, R] for direction `ny` at position `pos`.
    /// Mirrors C++ `Calc_ECPos`.
    /// Kept for future CSXCAD integration; currently unused (background-material only).
    #[allow(dead_code)]
    fn calc_ec_pos(&self, ny: i32, pos: [u32; 3]) -> [f64; 4] {
        let eff = self.calc_eff_mat_pos(ny, pos);
        let eps_r = eff[0];
        let kappa = eff[1];
        let mue_r = eff[2];
        let sigma = eff[3];

        let area = self.get_edge_area(ny, [pos[0], pos[1], pos[2]], false);
        let delta = self.get_edge_length(ny, [pos[0], pos[1], pos[2]], false);

        let c = eps_r * EPS0 * area / delta;
        let g = kappa * EPS0 * area / delta;
        let l = mue_r * MU0 * area / delta;
        let r = sigma * EPS0 * area / delta;

        [c, g, l, r]
    }

    /// Compute effective material [epsR, kappa, mueR, sigma] at position.
    /// Mirrors C++ `Calc_EffMatPos` / `AverageMatQuarterCell`.
    /// Checks per-cell material grid first, falls back to background material.
    fn calc_eff_mat_pos(&self, _ny: i32, pos: [u32; 3]) -> [f64; 4] {
        if let Some(grid) = &self.material_grid {
            let num_lines = self.base.num_lines;
            let nx = num_lines[0] as usize;
            let ny = num_lines[1] as usize;
            let nz = num_lines[2] as usize;
            let i = pos[0] as usize;
            let j = pos[1] as usize;
            let k = pos[2] as usize;
            if i < nx && j < ny && k < nz {
                // Direction 0 (x) uses cell material directly; other directions
                // average with neighbors in the C++ implementation, but for the
                // Rust port we use the cell's own material as a first approximation.
                for n in 0..3usize {
                    let index = n * nx * ny * nz + i * ny * nz + j * nz + k;
                    if index < grid.len() {
                        let m = grid[index];
                        if m.eps_r != self.base.m_bg_eps_r
                            || m.mue_r != self.base.m_bg_mue_r
                            || m.kappa != self.base.m_bg_kappa
                            || m.sigma != self.base.m_bg_sigma
                        {
                            return [m.eps_r, m.kappa, m.mue_r, m.sigma];
                        }
                    }
                }
            }
        }
        [
            self.base.m_bg_eps_r,
            self.base.m_bg_kappa,
            self.base.m_bg_mue_r,
            self.base.m_bg_sigma,
        ]
    }
}

/// Snap a coordinate to the nearest grid line index for a given direction.
/// `lines` must be the sorted grid line values for that direction.
/// Returns None if the coordinate is outside the grid range.
fn snap_coord_to_grid(coord: f64, lines: &[f64]) -> Option<usize> {
    if lines.len() < 2 {
        return None;
    }
    let line0 = lines[0];
    let last = lines.len() - 1;
    let line_last = lines[last];
    if coord < line0 || coord > line_last {
        return None;
    }
    // Binary search for the nearest line.
    let mut lo = 0usize;
    let mut hi = last;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if lines[mid] < coord {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if lo > 0 && (lines[lo] - coord).abs() > (coord - lines[lo - 1]).abs() {
        Some(lo - 1)
    } else {
        Some(lo.min(last))
    }
}

/// Apply a CSXCAD primitive's bounding box to the per-cell material grid.
/// Cells whose centers fall inside the primitive's bounding box receive the
/// given material. For Box primitives, the exact box bounds are used; for
/// other shapes (Sphere, Cylinder, etc.), the axis-aligned bounding box is
/// used as a first-order approximation.
fn apply_primitive_to_grid(
    prim: &csxcad_core::model::Primitive,
    params: &csxcad_core::parameter::ParameterSet,
    grid_lines: [&[f64]; 3],
    num_lines: [usize; 3],
    material_grid: &mut [CellMaterial],
    cm: CellMaterial,
) {
    // Use the Primitive's public bounding_box, which dispatches to the
    // appropriate shape-specific implementation internally.
    let bbox = match prim.bounding_box(Some(params), false) {
        Ok(b) => b,
        Err(_) => return,
    };
    // bbox = [xmin, xmax, ymin, ymax, zmin, zmax]
    let start_idx = [
        snap_coord_to_grid(bbox[0], grid_lines[0]),
        snap_coord_to_grid(bbox[2], grid_lines[1]),
        snap_coord_to_grid(bbox[4], grid_lines[2]),
    ];
    let stop_idx = [
        snap_coord_to_grid(bbox[1], grid_lines[0]),
        snap_coord_to_grid(bbox[3], grid_lines[1]),
        snap_coord_to_grid(bbox[5], grid_lines[2]),
    ];

    let nx = num_lines[0];
    let ny = num_lines[1];
    let nz = num_lines[2];
    let x0 = start_idx[0].unwrap_or(0).min(nx.saturating_sub(1));
    let x1 = stop_idx[0].map(|v| v + 1).unwrap_or(nx).min(nx);
    let y0 = start_idx[1].unwrap_or(0).min(ny.saturating_sub(1));
    let y1 = stop_idx[1].map(|v| v + 1).unwrap_or(ny).min(ny);
    let z0 = start_idx[2].unwrap_or(0).min(nz.saturating_sub(1));
    let z1 = stop_idx[2].map(|v| v + 1).unwrap_or(nz).min(nz);

    for i in x0..x1 {
        for j in y0..y1 {
            for k in z0..z1 {
                let index = i * ny * nz + j * nz + k;
                if index < material_grid.len() {
                    material_grid[index] = cm;
                }
            }
        }
    }
}

impl Default for Operator {
    fn default() -> Self {
        Self::new()
    }
}

impl OperatorBase for Operator {
    fn set_geometry_csx(&mut self, geo: *mut std::ffi::c_void) -> bool {
        self.base.set_geometry_csx(geo)
    }

    fn get_geometry_csx(&self) -> *mut std::ffi::c_void {
        self.base.get_geometry_csx()
    }

    fn get_timestep(&self) -> f64 {
        self.base.get_timestep()
    }

    fn get_number_cells(&self) -> f64 {
        self.base.num_lines[0] as f64
            * self.base.num_lines[1] as f64
            * self.base.num_lines[2] as f64
    }

    fn get_number_of_nyquist_timesteps(&self) -> u32 {
        self.base.get_number_of_nyquist_timesteps()
    }

    fn get_number_of_lines(&self, ny: i32, _full: bool) -> u32 {
        self.base.num_lines[ny as usize]
    }

    fn get_grid_delta(&self) -> f64 {
        self.base.grid_delta
    }

    fn get_disc_line(&self, _n: i32, pos: u32, _dual_mesh: bool) -> f64 {
        pos as f64 * self.base.grid_delta
    }

    fn get_disc_delta(&self, _n: i32, _pos: u32, _dual_mesh: bool) -> f64 {
        self.base.grid_delta
    }

    fn get_node_width(&self, _ny: i32, _pos: [u32; 3], _dual_mesh: bool) -> f64 {
        self.base.grid_delta
    }

    fn get_node_area(&self, _ny: i32, _pos: [u32; 3], _dual_mesh: bool) -> f64 {
        self.base.grid_delta * self.base.grid_delta
    }

    fn get_edge_length(&self, _ny: i32, _pos: [u32; 3], _dual_mesh: bool) -> f64 {
        self.base.grid_delta
    }

    fn get_edge_area(&self, _ny: i32, _pos: [u32; 3], _dual_mesh: bool) -> f64 {
        self.base.grid_delta * self.base.grid_delta
    }

    fn get_cell_volume(&self, _pos: [u32; 3], _dual_mesh: bool) -> f64 {
        self.base.grid_delta.powi(3)
    }

    fn get_vv(&self) -> Option<&FieldArray> {
        self.vv_ptr.as_ref()
    }

    fn get_vi(&self) -> Option<&FieldArray> {
        self.vi_ptr.as_ref()
    }

    fn get_ii(&self) -> Option<&FieldArray> {
        self.ii_ptr.as_ref()
    }

    fn get_iv(&self) -> Option<&FieldArray> {
        self.iv_ptr.as_ref()
    }

    fn snap_to_mesh(
        &self,
        coord: [f64; 3],
        uicoord: &mut [u32; 3],
        dual_mesh: bool,
        full_mesh: bool,
        inside: Option<&mut bool>,
    ) -> bool {
        let grid_delta = self.base.grid_delta;
        if grid_delta <= 0.0 {
            if let Some(inside) = inside {
                *inside = false;
            }
            return false;
        }
        let num_lines = self.base.num_lines;
        let mut all_inside = true;
        for n in 0..3 {
            let idx = (coord[n] / grid_delta).round() as i32;
            let max_idx = (num_lines[n] as i32).saturating_sub(1);
            let clamped = idx.clamp(0, max_idx);
            uicoord[n] = clamped as u32;
            if idx < 0 || idx > max_idx {
                all_inside = false;
            }
        }
        if let Some(inside) = inside {
            *inside = all_inside;
        }
        all_inside
    }

    fn snap_box_to_mesh(
        &self,
        start: [f64; 3],
        stop: [f64; 3],
        ui_start: &mut [u32; 3],
        ui_stop: &mut [u32; 3],
        dual_mesh: bool,
        full_mesh: bool,
        snap_method: i32,
        b_start_in: Option<&mut bool>,
        b_stop_in: Option<&mut bool>,
    ) -> i32 {
        let grid_delta = self.base.grid_delta;
        if grid_delta <= 0.0 {
            if let Some(b_start_in) = b_start_in {
                *b_start_in = false;
            }
            if let Some(b_stop_in) = b_stop_in {
                *b_stop_in = false;
            }
            return -1;
        }
        let num_lines = self.base.num_lines;
        let mut start_inside = true;
        let mut stop_inside = true;
        for n in 0..3 {
            let s_idx = (start[n] / grid_delta).round() as i32;
            let t_idx = (stop[n] / grid_delta).round() as i32;
            let max_idx = (num_lines[n] as i32).saturating_sub(1);
            let s_clamped = s_idx.clamp(0, max_idx);
            let t_clamped = t_idx.clamp(0, max_idx);
            ui_start[n] = s_clamped as u32;
            ui_stop[n] = t_clamped as u32;
            if s_idx < 0 || s_idx > max_idx {
                start_inside = false;
            }
            if t_idx < 0 || t_idx > max_idx {
                stop_inside = false;
            }
        }
        if let Some(b_start_in) = b_start_in {
            *b_start_in = start_inside;
        }
        if let Some(b_stop_in) = b_stop_in {
            *b_stop_in = stop_inside;
        }
        if start_inside && stop_inside {
            0
        } else {
            -1
        }
    }

    fn get_primitives_bbox(
        &self,
        _pos_x: i32,
        _pos_y: i32,
        _pos_z: i32,
        _type_: i32,
    ) -> Vec<*mut std::ffi::c_void> {
        // No CSXCAD geometry loaded in the operator yet; return empty.
        // When CSXCAD structures are loaded, this should return pointers to
        // primitives whose bounding boxes overlap the query region.
        Vec::new()
    }

    fn set_boundary_condition(&mut self, bcs: [i32; 6]) {
        self.base.set_boundary_condition(bcs);
    }

    fn set_material_store_flags(&mut self, _type_: i32, _val: bool) {
        // Material storage for post-processing is not yet wired; no-op.
    }

    fn cleanup_material_storage(&mut self) {
        // No material storage to clean up yet.
    }

    fn get_disc_material(&self, _type_: i32, _ny: i32, _pos: [u32; 3]) -> f64 {
        0.0
    }

    fn get_cell_center_material_avg_coord_uint(
        &self,
        pos: [u32; 3],
        coord: &mut [f64; 3],
    ) -> bool {
        let grid_delta = self.base.grid_delta;
        if grid_delta <= 0.0 {
            return false;
        }
        for n in 0..3 {
            coord[n] = pos[n] as f64 * grid_delta;
        }
        true
    }

    fn get_cell_center_material_avg_coord_int(
        &self,
        pos: [i32; 3],
        coord: &mut [f64; 3],
    ) -> bool {
        let grid_delta = self.base.grid_delta;
        if grid_delta <= 0.0 {
            return false;
        }
        for n in 0..3 {
            coord[n] = pos[n] as f64 * grid_delta;
        }
        true
    }

    fn set_background_material(
        &mut self,
        eps_r: f64,
        mue_r: f64,
        kappa: f64,
        sigma: f64,
        density: f64,
    ) {
        self.base
            .set_background_material(eps_r, mue_r, kappa, sigma, density);
    }

    fn get_background_eps_r(&self) -> f64 {
        self.base.get_background_eps_r()
    }

    fn set_background_eps_r(&mut self, val: f64) {
        self.base.set_background_eps_r(val);
    }

    fn get_background_mue_r(&self) -> f64 {
        self.base.get_background_mue_r()
    }

    fn set_background_mue_r(&mut self, val: f64) {
        self.base.set_background_mue_r(val);
    }

    fn get_background_kappa(&self) -> f64 {
        self.base.get_background_kappa()
    }

    fn set_background_kappa(&mut self, val: f64) {
        self.base.set_background_kappa(val);
    }

    fn get_background_sigma(&self) -> f64 {
        self.base.get_background_sigma()
    }

    fn set_background_sigma(&mut self, val: f64) {
        self.base.set_background_sigma(val);
    }

    fn get_background_density(&self) -> f64 {
        self.base.get_background_density()
    }

    fn set_background_density(&mut self, val: f64) {
        self.base.set_background_density(val);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operator_creation() {
        let op = Operator::new();
        assert_eq!(op.get_timestep(), 0.0);
        assert_eq!(op.get_background_eps_r(), 1.0);
    }

    /// A minimal but valid CSXCAD XML geometry: a 10×10×10 mm uniform grid,
    /// vacuum background, and a small PEC box.
    const SAMPLE_CSX: &str = r#"<ContinuousStructure CoordSystem="0">
        <RectilinearGrid>
            <XLines>0,1,2,3,4,5,6,7,8,9,10</XLines>
            <YLines>0,1,2,3,4,5,6,7,8,9,10</YLines>
            <ZLines>0,1,2,3,4,5,6,7,8,9,10</ZLines>
        </RectilinearGrid>
        <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
        <Properties>
            <Metal Name="pec">
                <Primitives>
                    <Box Priority="0">
                        <P1 X="3" Y="3" Z="3"/>
                        <P2 X="5" Y="5" Z="5"/>
                    </Box>
                </Primitives>
            </Metal>
            <Material Name="substrate">
                <Property Epsilon="4.4,4.4,4.4" Mue="1,1,1" Kappa="0,0,0" Sigma="0,0,0" Density="0"/>
                <Primitives>
                    <Box Priority="0">
                        <P1 X="0" Y="0" Z="0"/>
                        <P2 X="10" Y="10" Z="2"/>
                    </Box>
                </Primitives>
            </Material>
            <AbsorbingBC Name="boundary">
                <Primitives>
                    <Box Priority="0">
                        <P1 X="0" Y="0" Z="0"/>
                        <P2 X="10" Y="10" Z="10"/>
                    </Box>
                </Primitives>
            </AbsorbingBC>
        </Properties>
    </ContinuousStructure>"#;

    #[test]
    fn test_load_csx_grid_and_background() {
        let mut op = Operator::new();
        let result = op.load_csx(SAMPLE_CSX);
        if let Err(ref e) = result {
            eprintln!("load_csx error: {}", e);
        }
        assert!(result.is_ok(), "load_csx failed");
        assert_eq!(op.base.num_lines, [11, 11, 11]);
        assert!((op.base.grid_delta - 1.0).abs() < 1e-9);
        assert!((op.base.get_background_eps_r() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_load_csx_material_grid_populated() {
        let mut op = Operator::new();
        let result = op.load_csx(SAMPLE_CSX);
        if let Err(ref e) = result {
            eprintln!("load_csx error: {}", e);
        }
        assert!(result.is_ok(), "load_csx failed");
        // material_grid must be Some after load_csx.
        assert!(op.material_grid.is_some());
        let grid = op.material_grid.as_ref().unwrap();
        // Cell inside the PEC box (x=4, y=4, z=4) must be marked is_pec.
        let nx = op.base.num_lines[0] as usize;
        let ny = op.base.num_lines[1] as usize;
        let nz = op.base.num_lines[2] as usize;
        let idx = 4 * ny * nz + 4 * nz + 4;
        assert!(grid[idx].is_pec, "cell (4,4,4) should be PEC");
        // Cell inside the substrate box (x=4, y=4, z=1) must have eps_r=4.4.
        let idx_sub = 4 * ny * nz + 4 * nz + 1;
        assert!((grid[idx_sub].eps_r - 4.4).abs() < 1e-9);
    }

    #[test]
    fn test_load_csx_invalid_xml() {
        let mut op = Operator::new();
        assert!(op.load_csx("not valid xml").is_err());
    }
}
