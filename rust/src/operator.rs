//! Equivalent-circuit operator: EC coefficients, material averaging, PEC.
//! Mirrors `FDTD/operator.h/cpp` in the C++ codebase.

use crate::array_nijk::FieldArray;
use crate::constants::{C0, EPS0, FDTD_FLOAT, MU0, Z0};
use crate::excitation::Excitation;
use crate::extensions::{
    ConductingSheetCoeffs, EngineExtConductingSheet, EngineExtExcitation, EngineExtMurABC,
    EngineExtSteadyState, EngineExtUPML,
};
use crate::operator_base::OperatorBase;
use csxcad_core::material::{ConductingSheetProperties, ExcitationProperties, MaterialField};
use csxcad_core::model::{Primitive, PropertyType};
use std::collections::HashMap;
use std::sync::Arc;

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

/// Anisotropic material values used while resolving a mesh edge.
#[derive(Clone, Copy, Debug)]
struct AnisoMaterial {
    eps: [f64; 3],
    mue: [f64; 3],
    kappa: [f64; 3],
    sigma: [f64; 3],
    is_pec: bool,
}

impl AnisoMaterial {
    fn vacuum() -> Self {
        Self {
            eps: [1.0; 3],
            mue: [1.0; 3],
            kappa: [0.0; 3],
            sigma: [0.0; 3],
            is_pec: false,
        }
    }

    fn pec() -> Self {
        Self {
            eps: [1.0; 3],
            mue: [1.0; 3],
            kappa: [0.0; 3],
            sigma: [0.0; 3],
            is_pec: true,
        }
    }
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
    /// Parsed CSXCAD structure, retained for extension setup (excitation, Mur).
    pub structure: Option<csxcad_core::model::Structure>,
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
            structure: None,
        }
    }

    /// Allocate the EC coefficient field arrays and the per-cell material grid.
    /// Mirrors C++ `Init_EC`.
    pub fn init_data_storage(&mut self, num_lines: [usize; 3]) {
        let extent = [num_lines[0], num_lines[1], num_lines[2]];
        self.base.num_lines = [num_lines[0] as u32, num_lines[1] as u32, num_lines[2] as u32];
        self.vv_ptr = Some(FieldArray::new("vv", extent));
        self.vi_ptr = Some(FieldArray::new("vi", extent));
        self.ii_ptr = Some(FieldArray::new("ii", extent));
        self.iv_ptr = Some(FieldArray::new("iv", extent));
        // Allocate per-cell material grid (one entry per cell and direction,
        // initialized from the background material). Indexed as [n][i][j][k].
        let num_cells = num_lines[0] * num_lines[1] * num_lines[2];
        let background = CellMaterial {
            eps_r: self.base.m_bg_eps_r,
            mue_r: self.base.m_bg_mue_r,
            kappa: self.base.m_bg_kappa,
            sigma: self.base.m_bg_sigma,
            is_pec: false,
        };
        self.material_grid = Some(vec![background; 3 * num_cells]);
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
        // Keep the native representation: discLines hold drawing units and
        // grid_delta is the unit in metres. Non-uniform spacing is preserved.
        let mut num_lines = [0u32; 3];
        let mut grid_lines: [Vec<f64>; 3] = Default::default();
        for n in 0..3usize {
            let count = structure.grid_mut().line_count(n);
            if count < 2 {
                return Err("CSXCAD grid has insufficient lines".to_string());
            }
            num_lines[n] = count as u32;
            grid_lines[n] = structure.grid_mut().lines(n).unwrap_or_default().to_vec();
        }
        let delta_unit = structure.grid().delta_unit();
        if !delta_unit.is_finite() || delta_unit <= 0.0 {
            return Err("CSXCAD grid delta unit must be positive".to_string());
        }
        self.base.grid_delta = delta_unit;
        self.base.num_lines = num_lines;
        self.base.disc_lines = [
            Some(grid_lines[0].clone()),
            Some(grid_lines[1].clone()),
            Some(grid_lines[2].clone()),
        ];

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
                        csxcad_core::material::AbsorbingBoundaryType::MurFirstOrder
                        | csxcad_core::material::AbsorbingBoundaryType::MurFirstOrderSuperAbsorption => 2,
                        csxcad_core::material::AbsorbingBoundaryType::Other(3) => 3,
                        _ => continue,
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

        // --- Per-edge materials from Material/Metal properties ---
        // Native openEMS resolves the material at each Yee edge by coordinate
        // priority and averages it over the surrounding quarter cells
        // (`Operator::AverageMatQuarterCell`): epsilon/kappa as an area-weighted
        // arithmetic mean of four transverse quarter cells, mue/sigma as a
        // width-weighted harmonic mean along the edge. PEC is decided from the
        // single edge coordinate (`Operator::CalcPEC`).
        let num_lines_usize: [usize; 3] = num_lines.map(|v| v as usize);
        let ncells = num_lines_usize[0] * num_lines_usize[1] * num_lines_usize[2];
        let background = CellMaterial {
            eps_r: self.base.m_bg_eps_r,
            mue_r: self.base.m_bg_mue_r,
            kappa: self.base.m_bg_kappa,
            sigma: self.base.m_bg_sigma,
            is_pec: false,
        };

        let params = structure.parameters();
        struct Candidate<'a> {
            priority: i32,
            material: AnisoMaterial,
            primitive: &'a Primitive,
        }
        let mut candidates: Vec<Candidate<'_>> = Vec::new();
        for prop in 0..structure.property_count() {
            let property = structure
                .property_by_index(prop)
                .ok_or_else(|| "invalid property index".to_string())?;
            match property.property_type() {
                PropertyType::Material => {
                    let mat = property
                        .material()
                        .ok_or_else(|| "material property missing material data".to_string())?;
                    let mut material = AnisoMaterial::vacuum();
                    for component in 0..3 {
                        material.eps[component] = mat
                            .value(MaterialField::Epsilon, component, Some(params))
                            .ok()
                            .flatten()
                            .unwrap_or(1.0);
                        material.mue[component] = mat
                            .value(MaterialField::Mue, component, Some(params))
                            .ok()
                            .flatten()
                            .unwrap_or(1.0);
                        material.kappa[component] = mat
                            .value(MaterialField::Kappa, component, Some(params))
                            .ok()
                            .flatten()
                            .unwrap_or(0.0);
                        material.sigma[component] = mat
                            .value(MaterialField::Sigma, component, Some(params))
                            .ok()
                            .flatten()
                            .unwrap_or(0.0);
                    }
                    for primitive in property.primitives() {
                        candidates.push(Candidate {
                            priority: primitive.priority(),
                            material,
                            primitive,
                        });
                    }
                }
                PropertyType::Metal => {
                    for primitive in property.primitives() {
                        candidates.push(Candidate {
                            priority: primitive.priority(),
                            material: AnisoMaterial::pec(),
                            primitive,
                        });
                    }
                }
                _ => {}
            }
        }

        // Highest-priority containing primitive at `coordinate`. Metals are
        // returned as `is_pec`; native `GetMaterial` treats them as background
        // for epsilon/mue averaging, which the caller mirrors.
        let sample = |coordinate: [f64; 3]| -> Option<AnisoMaterial> {
            let mut winner: Option<&AnisoMaterial> = None;
            let mut winner_priority = i32::MIN;
            for candidate in &candidates {
                if winner.is_some() && candidate.priority <= winner_priority {
                    continue;
                }
                if candidate
                    .primitive
                    .is_inside(coordinate, Some(params))
                    .unwrap_or(false)
                {
                    winner = Some(&candidate.material);
                    winner_priority = candidate.priority;
                }
            }
            winner.copied()
        };
        let bg_eps = self.base.m_bg_eps_r;
        let bg_mue = self.base.m_bg_mue_r;
        let bg_kappa = self.base.m_bg_kappa;
        let bg_sigma = self.base.m_bg_sigma;
        let disc_lines = self.base.disc_lines.clone();
        let counts = self.base.num_lines;

        let mut material_grid = vec![background; 3 * ncells];
        for n in 0..3usize {
            let n_p = (n + 1) % 3;
            let n_pp = (n + 2) % 3;
            for i in 0..num_lines_usize[0] {
                for j in 0..num_lines_usize[1] {
                    for k in 0..num_lines_usize[2] {
                        let pos = [i, j, k];
                        let base_coordinate = [
                            disc_lines[0].as_ref().map(|l| l[i]).unwrap_or(0.0),
                            disc_lines[1].as_ref().map(|l| l[j]).unwrap_or(0.0),
                            disc_lines[2].as_ref().map(|l| l[k]).unwrap_or(0.0),
                        ];
                        let delta = raw_disc_delta(&disc_lines, counts, n, pos[n] as i64);
                        let delta_p =
                            raw_disc_delta(&disc_lines, counts, n_p, pos[n_p] as i64);
                        let delta_pp =
                            raw_disc_delta(&disc_lines, counts, n_pp, pos[n_pp] as i64);
                        let delta_m =
                            raw_disc_delta(&disc_lines, counts, n, pos[n] as i64 - 1);
                        let delta_p_m =
                            raw_disc_delta(&disc_lines, counts, n_p, pos[n_p] as i64 - 1);
                        let delta_pp_m =
                            raw_disc_delta(&disc_lines, counts, n_pp, pos[n_pp] as i64 - 1);

                        // PEC: single Yee-edge coordinate (native `CalcPEC`).
                        let edge_coordinate = yee_coordinates(
                            &disc_lines,
                            counts,
                            n,
                            pos,
                            false,
                        );
                        let is_pec = sample(edge_coordinate)
                            .map(|material| material.is_pec)
                            .unwrap_or(false);

                        // epsilon / kappa: four transverse quarter cells.
                        let mut eps_sum = 0.0;
                        let mut kappa_sum = 0.0;
                        let mut area_sum = 0.0;
                        for (positive_p, positive_pp) in
                            [(true, true), (false, true), (true, false), (false, false)]
                        {
                            let mut loc = pos;
                            if !positive_p {
                                if loc[n_p] == 0 {
                                    continue;
                                }
                                loc[n_p] -= 1;
                            }
                            if !positive_pp {
                                if loc[n_pp] == 0 {
                                    continue;
                                }
                                loc[n_pp] -= 1;
                            }
                            let mut coordinate = base_coordinate;
                            coordinate[n] = base_coordinate[n] + delta * 0.5;
                            coordinate[n_p] = base_coordinate[n_p]
                                + if positive_p {
                                    delta_p * 0.25
                                } else {
                                    -delta_p_m * 0.25
                                };
                            coordinate[n_pp] = base_coordinate[n_pp]
                                + if positive_pp {
                                    delta_pp * 0.25
                                } else {
                                    -delta_pp_m * 0.25
                                };
                            let area = edge_area_value(&disc_lines, counts, n, loc, true, 1.0);
                            if area <= 0.0 {
                                continue;
                            }
                            let material = sample(coordinate)
                                .filter(|material| !material.is_pec)
                                .unwrap_or(AnisoMaterial {
                                    eps: [bg_eps; 3],
                                    mue: [bg_mue; 3],
                                    kappa: [bg_kappa; 3],
                                    sigma: [bg_sigma; 3],
                                    is_pec: false,
                                });
                            eps_sum += material.eps[n] * area;
                            kappa_sum += material.kappa[n] * area;
                            area_sum += area;
                        }
                        // Metals contribute the background material, as in native.
                        let (eps_r, kappa) = if area_sum > 0.0 {
                            (eps_sum / area_sum, kappa_sum / area_sum)
                        } else {
                            (bg_eps, bg_kappa)
                        };

                        // mue / sigma: two longitudinal nodes, harmonic mean.
                        let mut inv_mue = 0.0;
                        let mut sigma_sum = 0.0;
                        let mut length = 0.0;
                        for lower in [true, false] {
                            let mut loc = pos;
                            if lower {
                                if loc[n] == 0 {
                                    continue;
                                }
                                loc[n] -= 1;
                            }
                            let width =
                                edge_length_value(&disc_lines, counts, n, loc, false, 1.0);
                            if width <= 0.0 {
                                continue;
                            }
                            let mut coordinate = base_coordinate;
                            coordinate[n] = base_coordinate[n]
                                + if lower { -delta_m * 0.25 } else { delta * 0.25 };
                            coordinate[n_p] = base_coordinate[n_p] + delta_p * 0.5;
                            coordinate[n_pp] = base_coordinate[n_pp] + delta_pp * 0.5;
                            let material = sample(coordinate)
                                .filter(|material| !material.is_pec)
                                .unwrap_or(AnisoMaterial {
                                    eps: [bg_eps; 3],
                                    mue: [bg_mue; 3],
                                    kappa: [bg_kappa; 3],
                                    sigma: [bg_sigma; 3],
                                    is_pec: false,
                                });
                            let mue = material.mue[n].max(f64::MIN_POSITIVE);
                            inv_mue += width / mue;
                            let sigma = material.sigma[n];
                            if sigma != 0.0 {
                                sigma_sum += width / sigma;
                            } else {
                                sigma_sum = 0.0;
                            }
                            length += width;
                        }
                        let mue_r = if inv_mue > 0.0 {
                            length / inv_mue
                        } else {
                            bg_mue
                        };
                        let sigma = if sigma_sum != 0.0 {
                            length / sigma_sum
                        } else {
                            bg_sigma
                        };

                        let index = n * ncells
                            + i * num_lines_usize[1] * num_lines_usize[2]
                            + j * num_lines_usize[2]
                            + k;
                        material_grid[index] = CellMaterial {
                            eps_r,
                            mue_r,
                            kappa,
                            sigma,
                            is_pec,
                        };
                    }
                }
            }
        }

        self.init_data_storage(num_lines_usize);
        self.material_grid = Some(material_grid);
        self.structure = Some(structure);

        Ok(())
    }

    /// Compute all EC coefficients (vv, vi, ii, iv) for every cell.
    /// Mirrors C++ `Calc_ECOperator`.
    /// Uses per-cell material from `material_grid` when available, falling back
    /// to the background material.
    pub fn calc_ec_operator(&mut self) {
        let counts = self.base.num_lines;
        let grid_delta = self.base.grid_delta;
        let disc_lines = &self.base.disc_lines;
        let material_grid = self.material_grid.as_ref();

        let nx = counts[0] as usize;
        let ny = counts[1] as usize;
        let nz = counts[2] as usize;
        let dt = self.base.d_t as FDTD_FLOAT;

        let bg_eps_r = self.base.m_bg_eps_r;
        let bg_mue_r = self.base.m_bg_mue_r;
        let bg_kappa = self.base.m_bg_kappa;
        let bg_sigma = self.base.m_bg_sigma;

        let vv = self.vv_ptr.as_mut().expect("vv_ptr not allocated");
        let vi = self.vi_ptr.as_mut().expect("vi_ptr not allocated");
        let ii = self.ii_ptr.as_mut().expect("ii_ptr not allocated");
        let iv = self.iv_ptr.as_mut().expect("iv_ptr not allocated");

        for n in 0..3 {
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        // Determine effective material for this cell.
                        let (eps_r, mue_r, kappa, sigma, is_pec) =
                            if let Some(grid) = material_grid {
                                let nx_lines = counts[0] as usize;
                                let ny_lines = counts[1] as usize;
                                let nz_lines = counts[2] as usize;
                                let index = n * nx_lines * ny_lines * nz_lines
                                    + i * ny_lines * nz_lines
                                    + j * nz_lines
                                    + k;
                                if index < grid.len() {
                                    let m = grid[index];
                                    (m.eps_r, m.mue_r, m.kappa, m.sigma, m.is_pec)
                                } else {
                                    (bg_eps_r, bg_mue_r, bg_kappa, bg_sigma, false)
                                }
                            } else {
                                (bg_eps_r, bg_mue_r, bg_kappa, bg_sigma, false)
                            };

                        // Per-direction edge geometry from the real mesh.
                        let pos = [i, j, k];
                        let area = edge_area_value(disc_lines, counts, n, pos, false, grid_delta);
                        let delta = edge_length_value(disc_lines, counts, n, pos, false, grid_delta);
                        let area_dual = edge_area_value(disc_lines, counts, n, pos, true, grid_delta);
                        let delta_dual =
                            edge_length_value(disc_lines, counts, n, pos, true, grid_delta);

                        let c = if delta > 0.0 {
                            (eps_r * EPS0 * area / delta) as FDTD_FLOAT
                        } else {
                            0.0
                        };
                        let g = if delta > 0.0 {
                            (kappa * area / delta) as FDTD_FLOAT
                        } else {
                            0.0
                        };
                        let l = if delta_dual > 0.0 {
                            (mue_r * MU0 * area_dual / delta_dual) as FDTD_FLOAT
                        } else {
                            0.0
                        };
                        let r = if delta_dual > 0.0 {
                            (sigma * area_dual / delta_dual) as FDTD_FLOAT
                        } else {
                            0.0
                        };

                        // PEC edges inside a metal primitive have no electric
                        // update, matching native `Calc_PEC` (vv/vi only).
                        if is_pec {
                            *vv.get_mut(n, i, j, k) = 0.0;
                            *vi.get_mut(n, i, j, k) = 0.0;
                        }

                        // Voltage coefficients: vv, vi
                        if c > 0.0 && !is_pec {
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

    /// Compute the FDTD timestep from the EC coefficients.
    /// Mirrors C++ `CalcTimestep`: method 3 selects `Rennings_2` (Var3, the
    /// largest stable step for rectilinear meshes), everything else Var1.
    pub fn calc_timestep(&mut self) -> f64 {
        if self.base.m_time_step_var == 3 {
            return self.calc_timestep_var3();
        }
        self.calc_timestep_var1()
    }

    /// CFL estimate `Rennings_1`, C++ `CalcTimestep_Var1`.
    pub fn calc_timestep_var1(&mut self) -> f64 {
        let counts = self.base.num_lines;
        let nx = counts[0] as usize;
        let ny = counts[1] as usize;
        let nz = counts[2] as usize;

        let mut min_dt = f64::INFINITY;

        for n in 0..3usize {
            let n_p = (n + 1) % 3;
            let n_pp = (n + 2) % 3;
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        let pos = [i, j, k];
                        let pos_pm = shift_node(pos, n_p, -1, counts);
                        let pos_ppm = shift_node(pos, n_pp, -1, counts);

                        let l_a = self.get_ec_l(n_p as i32, pos[0], pos[1], pos[2]);
                        let l_b = self.get_ec_l(n_p as i32, pos_ppm[0], pos_ppm[1], pos_ppm[2]);
                        let l_c = self.get_ec_l(n_pp as i32, pos[0], pos[1], pos[2]);
                        let l_d = self.get_ec_l(n_pp as i32, pos_pm[0], pos_pm[1], pos_pm[2]);
                        let ec_c = self.get_ec_c(n as i32, i, j, k);

                        if ec_c <= 0.0 || l_a <= 0.0 || l_b <= 0.0 || l_c <= 0.0 || l_d <= 0.0 {
                            continue;
                        }
                        let sum_inv_l = 4.0 / l_a + 4.0 / l_b + 4.0 / l_c + 4.0 / l_d;
                        let dt = 2.0 / (sum_inv_l / ec_c).sqrt();
                        if dt > 0.0 && dt < min_dt {
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

    /// CFL estimate `Rennings_2`, C++ `CalcTimestep_Var3`.
    ///
    /// Uses the equivalent-circuit C and L coefficients of the surrounding
    /// edges and their shared nodes, which gives a larger (still stable) step
    /// than Var1 on rectilinear meshes.
    pub fn calc_timestep_var3(&mut self) -> f64 {
        let counts = self.base.num_lines;
        let nx = counts[0] as usize;
        let ny = counts[1] as usize;
        let nz = counts[2] as usize;

        let mut min_dt = f64::INFINITY;
        let c_at = |operator: &Self, n: usize, pos: [usize; 3]| {
            operator.get_ec_c(n as i32, pos[0], pos[1], pos[2])
        };
        let l_at = |operator: &Self, n: usize, pos: [usize; 3]| {
            operator.get_ec_l(n as i32, pos[0], pos[1], pos[2])
        };
        let inverse_product = |l: f64, c: f64| -> Option<f64> {
            if l > 0.0 && c > 0.0 {
                Some(1.0 / (l * c))
            } else {
                None
            }
        };

        for n in 0..3usize {
            let n_p = (n + 1) % 3;
            let n_pp = (n + 2) % 3;
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        let pos = [i, j, k];
                        let pos_pm = shift_node(pos, n_p, -1, counts);
                        let pos_pp_mm = shift_node(pos_pm, n_pp, -1, counts);

                        let c_here = c_at(self, n, pos);
                        let mut terms: [Option<f64>; 8] = [None; 8];
                        terms[0] = inverse_product(
                            l_at(self, n_pp, pos),
                            c_at(self, n, shift_node(pos, n_p, 1, counts)),
                        );
                        terms[1] = inverse_product(l_at(self, n_pp, pos), c_here);
                        terms[2] = inverse_product(
                            l_at(self, n_p, pos),
                            c_at(self, n, shift_node(pos, n_pp, 1, counts)),
                        );
                        terms[3] = inverse_product(l_at(self, n_p, pos), c_here);
                        terms[4] = inverse_product(
                            l_at(self, n_pp, pos_pm),
                            c_at(self, n, shift_node(pos_pm, n_p, 1, counts)),
                        );
                        terms[5] = inverse_product(
                            l_at(self, n_pp, pos_pm),
                            c_at(self, n, pos_pm),
                        );
                        terms[6] = inverse_product(
                            l_at(self, n_p, pos_pp_mm),
                            c_at(self, n, shift_node(pos_pp_mm, n_pp, 1, counts)),
                        );
                        terms[7] = inverse_product(
                            l_at(self, n_p, pos_pp_mm),
                            c_at(self, n, pos_pp_mm),
                        );
                        let Some(wqp) = sum_terms(&terms) else {
                            continue;
                        };

                        let wt_1 = [
                            inverse_product(l_at(self, n_pp, pos), c_at(self, n_p, pos)),
                            inverse_product(
                                l_at(self, n_pp, pos_pm),
                                c_at(self, n_p, pos),
                            ),
                            inverse_product(l_at(self, n_p, pos), c_at(self, n_pp, pos)),
                            inverse_product(
                                l_at(self, n_p, shift_node(pos, n_pp, -1, counts)),
                                c_at(self, n_pp, pos),
                            ),
                        ];
                        let Some(wt1) = weighted_term(&wt_1) else {
                            continue;
                        };

                        let shifted_n = shift_node(pos, n, 1, counts);
                        let wt_2 = [
                            inverse_product(l_at(self, n_pp, pos), c_at(self, n_p, shifted_n)),
                            inverse_product(
                                l_at(self, n_pp, pos_pm),
                                c_at(self, n_p, shifted_n),
                            ),
                            inverse_product(l_at(self, n_p, pos), c_at(self, n_pp, shifted_n)),
                            inverse_product(
                                l_at(self, n_p, shift_node(pos, n_pp, -1, counts)),
                                c_at(self, n_pp, shifted_n),
                            ),
                        ];
                        let Some(wt2) = weighted_term(&wt_2) else {
                            continue;
                        };

                        let total = wqp + wt1 + wt2;
                        if total <= 0.0 || !total.is_finite() {
                            continue;
                        }
                        let dt = 2.0 / total.sqrt();
                        if dt > 0.0 && dt < min_dt {
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

    /// Apply electric boundary conditions (PEC): zero the tangential voltage
    /// coefficients on PEC faces. The normal component is untouched, matching
    /// native `ApplyElectricBC`.
    pub fn apply_electric_bc(&mut self) {
        let vv = self.vv_ptr.as_mut().expect("vv_ptr not allocated");
        let vi = self.vi_ptr.as_mut().expect("vi_ptr not allocated");
        let bc = self.base.m_bc;

        let nx = vv.ni();
        let ny = vv.nj();
        let nz = vv.nk();

        // bc[0]=x-min, bc[1]=x-max, bc[2]=y-min, bc[3]=y-max, bc[4]=z-min, bc[5]=z-max
        // PEC = 0. Only the two tangential components are forced to zero.
        for &(face, i) in &[(0usize, 0usize), (1usize, nx.saturating_sub(1))] {
            if bc[face] != 0 {
                continue;
            }
            for n in [1usize, 2] {
                for j in 0..ny {
                    for k in 0..nz {
                        *vv.get_mut(n, i, j, k) = 0.0;
                        *vi.get_mut(n, i, j, k) = 0.0;
                    }
                }
            }
        }
        for &(face, j) in &[(2usize, 0usize), (3usize, ny.saturating_sub(1))] {
            if bc[face] != 0 {
                continue;
            }
            for n in [0usize, 2] {
                for i in 0..nx {
                    for k in 0..nz {
                        *vv.get_mut(n, i, j, k) = 0.0;
                        *vi.get_mut(n, i, j, k) = 0.0;
                    }
                }
            }
        }
        for &(face, k) in &[(4usize, 0usize), (5usize, nz.saturating_sub(1))] {
            if bc[face] != 0 {
                continue;
            }
            for n in [0usize, 1] {
                for i in 0..nx {
                    for j in 0..ny {
                        *vv.get_mut(n, i, j, k) = 0.0;
                        *vi.get_mut(n, i, j, k) = 0.0;
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

    /// Apply magnetic boundary conditions (PMC): zero the tangential current
    /// coefficients on PMC faces. The normal component is untouched, matching
    /// native `ApplyMagneticBC`.
    pub fn apply_magnetic_bc(&mut self) {
        let ii = self.ii_ptr.as_mut().expect("ii_ptr not allocated");
        let iv = self.iv_ptr.as_mut().expect("iv_ptr not allocated");
        let bc = self.base.m_bc;

        let nx = ii.ni();
        let ny = ii.nj();
        let nz = ii.nk();

        // PMC = 1. Only the two tangential components are forced to zero.
        for &(face, i) in &[(0usize, 0usize), (1usize, nx.saturating_sub(1))] {
            if bc[face] != 1 {
                continue;
            }
            for n in [1usize, 2] {
                for j in 0..ny {
                    for k in 0..nz {
                        *ii.get_mut(n, i, j, k) = 0.0;
                        *iv.get_mut(n, i, j, k) = 0.0;
                    }
                }
            }
        }
        for &(face, j) in &[(2usize, 0usize), (3usize, ny.saturating_sub(1))] {
            if bc[face] != 1 {
                continue;
            }
            for n in [0usize, 2] {
                for i in 0..nx {
                    for k in 0..nz {
                        *ii.get_mut(n, i, j, k) = 0.0;
                        *iv.get_mut(n, i, j, k) = 0.0;
                    }
                }
            }
        }
        for &(face, k) in &[(4usize, 0usize), (5usize, nz.saturating_sub(1))] {
            if bc[face] != 1 {
                continue;
            }
            for n in [0usize, 1] {
                for i in 0..nx {
                    for j in 0..ny {
                        *ii.get_mut(n, i, j, k) = 0.0;
                        *iv.get_mut(n, i, j, k) = 0.0;
                    }
                }
            }
        }
    }

    // ----- private helpers -----

    fn get_ec_c(&self, n: i32, i: usize, j: usize, k: usize) -> f64 {
        let dir = n.rem_euclid(3) as usize;
        let pos = [i, j, k];
        let area = edge_area_value(
            &self.base.disc_lines,
            self.base.num_lines,
            dir,
            pos,
            false,
            self.base.grid_delta,
        );
        let delta = edge_length_value(
            &self.base.disc_lines,
            self.base.num_lines,
            dir,
            pos,
            false,
            self.base.grid_delta,
        );
        if delta <= 0.0 {
            return 0.0;
        }
        let eps_r = if let Some(grid) = &self.material_grid {
            let nx = self.base.num_lines[0] as usize;
            let ny = self.base.num_lines[1] as usize;
            let nz = self.base.num_lines[2] as usize;
            let index = dir * nx * ny * nz + i * ny * nz + j * nz + k;
            grid.get(index).map_or(self.base.m_bg_eps_r, |m| m.eps_r)
        } else {
            self.base.m_bg_eps_r
        };
        eps_r * EPS0 * area / delta
    }

    fn get_ec_l(&self, n: i32, i: usize, j: usize, k: usize) -> f64 {
        let dir = n.rem_euclid(3) as usize;
        let pos = [i, j, k];
        let area = edge_area_value(
            &self.base.disc_lines,
            self.base.num_lines,
            dir,
            pos,
            true,
            self.base.grid_delta,
        );
        let delta = edge_length_value(
            &self.base.disc_lines,
            self.base.num_lines,
            dir,
            pos,
            true,
            self.base.grid_delta,
        );
        if delta <= 0.0 {
            return 0.0;
        }
        let mue_r = if let Some(grid) = &self.material_grid {
            let nx = self.base.num_lines[0] as usize;
            let ny = self.base.num_lines[1] as usize;
            let nz = self.base.num_lines[2] as usize;
            let index = dir * nx * ny * nz + i * ny * nz + j * nz + k;
            grid.get(index).map_or(self.base.m_bg_mue_r, |m| m.mue_r)
        } else {
            self.base.m_bg_mue_r
        };
        mue_r * MU0 * area / delta
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
        let g = kappa * area / delta;
        let l = mue_r * MU0 * area / delta;
        let r = sigma * area / delta;

        [c, g, l, r]
    }

    /// Compute effective material [epsR, kappa, mueR, sigma] at position.
    /// Mirrors C++ `Calc_EffMatPos` / `AverageMatQuarterCell`.
    /// Checks per-cell material grid first, falls back to background material.
    fn calc_eff_mat_pos(&self, ny: i32, pos: [u32; 3]) -> [f64; 4] {
        if let Some(grid) = &self.material_grid {
            let num_lines = self.base.num_lines;
            let nx = num_lines[0] as usize;
            let ny_lines = num_lines[1] as usize;
            let nz = num_lines[2] as usize;
            let i = pos[0] as usize;
            let j = pos[1] as usize;
            let k = pos[2] as usize;
            if i < nx && j < ny_lines && k < nz {
                let dir = ny.rem_euclid(3) as usize;
                let index = dir * nx * ny_lines * nz + i * ny_lines * nz + j * nz + k;
                if index < grid.len() {
                    let m = grid[index];
                    return [m.eps_r, m.kappa, m.mue_r, m.sigma];
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

    /// Material (eps_r, mue_r) for a component at a mesh position, falling back
    /// to the background material.
    fn material_value(&self, component: usize, pos: [usize; 3]) -> (f64, f64) {
        if let Some(grid) = &self.material_grid {
            let counts = self.base.num_lines;
            let nx = counts[0] as usize;
            let ny = counts[1] as usize;
            let nz = counts[2] as usize;
            let index = component * nx * ny * nz + pos[0] * ny * nz + pos[1] * nz + pos[2];
            if let Some(material) = grid.get(index) {
                return (material.eps_r, material.mue_r);
            }
        }
        (self.base.m_bg_eps_r, self.base.m_bg_mue_r)
    }

    /// Build first-order Mur ABC faces from the boundary conditions.
    /// Mirrors the coefficient computation in `Operator_Ext_Mur_ABC`.
    pub fn build_mur_extension(&self, phase_velocity: [f64; 6]) -> EngineExtMurABC {
        let mut extension = EngineExtMurABC::new();
        let counts = self.base.num_lines;
        let dt = self.base.d_t;
        if dt <= 0.0 {
            return extension;
        }
        for face in 0..6usize {
            if self.base.m_bc[face] != 2 {
                continue; // BC_MUR
            }
            let dir = face / 2;
            let top = face % 2 == 1;
            let line = if top {
                counts[dir].saturating_sub(1) as usize
            } else {
                0
            };
            let shift = if top {
                counts[dir].saturating_sub(2) as usize
            } else {
                1
            };
            let n_p = (dir + 1) % 3;
            let n_pp = (dir + 2) % 3;
            let dims_p = counts[n_p] as usize;
            let dims_pp = counts[n_pp] as usize;
            let mut pos = [0usize; 3];
            pos[dir] = line;
            let delta = edge_length_value(
                &self.base.disc_lines,
                counts,
                dir,
                pos,
                false,
                self.base.grid_delta,
            )
            .abs();
            if delta <= 0.0 {
                continue;
            }

            let mut coeff_p = vec![0.0f32; dims_p * dims_pp];
            let mut coeff_pp = vec![0.0f32; dims_p * dims_pp];
            for ip in 0..dims_p {
                for ipp in 0..dims_pp {
                    let idx = ip * dims_pp + ipp;
                    let mut cell = [0usize; 3];
                    cell[dir] = shift.min(counts[dir].saturating_sub(1) as usize);
                    cell[n_p] = ip;
                    cell[n_pp] = ipp;
                    let (eps_p, mue_p) = self.material_value(n_p, cell);
                    let (eps_pp, mue_pp) = self.material_value(n_pp, cell);
                    let c_p = if phase_velocity[face] > 0.0 {
                        phase_velocity[face] * dt
                    } else {
                        C0 * dt / (eps_p * mue_p).sqrt()
                    };
                    let c_pp = if phase_velocity[face] > 0.0 {
                        phase_velocity[face] * dt
                    } else {
                        C0 * dt / (eps_pp * mue_pp).sqrt()
                    };
                    coeff_p[idx] = ((c_p - delta) / (c_p + delta)) as FDTD_FLOAT;
                    coeff_pp[idx] = ((c_pp - delta) / (c_pp + delta)) as FDTD_FLOAT;
                }
            }
            extension.add_face(
                face, dir, top, line, shift, dims_p, dims_pp, coeff_p, coeff_pp,
            );
        }
        extension
    }

    /// Build the excitation extension by mapping CSXCAD Excitation properties
    /// onto Yee edges. Mirrors `Operator_Ext_Excitation::BuildExtension`.
    pub fn build_excitation_extension(&mut self, signal: &Excitation) -> EngineExtExcitation {
        let mut extension = EngineExtExcitation::new();
        let counts = self.base.num_lines;
        let dt = self.base.d_t;
        if dt <= 0.0 {
            return extension;
        }

        let mut hard_voltage: Vec<[usize; 4]> = Vec::new();
        let mut hard_current: Vec<[usize; 4]> = Vec::new();
        {
            let Some(structure) = self.structure.as_ref() else {
                return extension;
            };
            let params = structure.parameters();
            let mut candidates: Vec<(i32, &ExcitationProperties, &Primitive)> = Vec::new();
            for index in 0..structure.property_count() {
                let Some(property) = structure.property_by_index(index) else {
                    continue;
                };
                if let Some(excitation) = property.excitation() {
                    for primitive in property.primitives() {
                        candidates.push((primitive.priority(), excitation, primitive));
                    }
                }
            }
            if candidates.is_empty() {
                return extension;
            }

            let shared = Arc::new(signal.clone());
            let disc_lines = &self.base.disc_lines;
            let grid_delta = self.base.grid_delta;

            for n in 0..3usize {
                for i in 0..counts[0] as usize {
                    for j in 0..counts[1] as usize {
                        for k in 0..counts[2] as usize {
                            for dual in [false, true] {
                                if dual
                                    && (i + 1 >= counts[0] as usize
                                        || j + 1 >= counts[1] as usize
                                        || k + 1 >= counts[2] as usize)
                                {
                                    continue;
                                }
                                let coordinate =
                                    yee_coordinates(disc_lines, counts, n, [i, j, k], dual);
                                let mut winner: Option<&ExcitationProperties> = None;
                                let mut winner_priority = i32::MIN;
                                for (priority, excitation, primitive) in &candidates {
                                    if winner.is_some() && *priority <= winner_priority {
                                        continue;
                                    }
                                    if primitive
                                        .is_inside(coordinate, Some(params))
                                        .unwrap_or(false)
                                    {
                                        winner = Some(*excitation);
                                        winner_priority = *priority;
                                    }
                                }
                                let Some(excitation) = winner else {
                                    continue;
                                };
                                if !excitation.is_enabled()
                                    || excitation.active_direction(n) != Some(true)
                                {
                                    continue;
                                }
                                let excitation_type = excitation.excitation_type();
                                let is_voltage = matches!(excitation_type, 0 | 1);
                                let is_current = matches!(excitation_type, 2 | 3);
                                if dual != is_current {
                                    continue;
                                }
                                let base_amp = excitation
                                    .excitation(n, Some(params))
                                    .ok()
                                    .flatten()
                                    .unwrap_or(0.0)
                                    * excitation
                                        .weighting(n, Some(params))
                                        .ok()
                                        .flatten()
                                        .unwrap_or(1.0);
                                if base_amp == 0.0 {
                                    continue;
                                }
                                let length = edge_length_value(
                                    disc_lines,
                                    counts,
                                    n,
                                    [i, j, k],
                                    dual,
                                    grid_delta,
                                );
                                let amp = (base_amp * length) as FDTD_FLOAT;
                                if amp == 0.0 {
                                    continue;
                                }
                                let delay = (excitation.delay(Some(params)).unwrap_or(0.0) / dt)
                                    .max(0.0) as u32;
                                if is_voltage {
                                    extension.add_voltage_excitation(
                                        n,
                                        [i, j, k],
                                        delay,
                                        amp,
                                        shared.clone(),
                                    );
                                    if excitation_type == 1 {
                                        hard_voltage.push([n, i, j, k]);
                                    }
                                } else if is_current {
                                    extension.add_current_excitation(
                                        n,
                                        [i, j, k],
                                        delay,
                                        amp,
                                        shared.clone(),
                                    );
                                    if excitation_type == 3 {
                                        hard_current.push([n, i, j, k]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if let (Some(vv), Some(vi)) = (self.vv_ptr.as_mut(), self.vi_ptr.as_mut()) {
            for [n, i, j, k] in &hard_voltage {
                *vv.get_mut(*n, *i, *j, *k) = 0.0;
                *vi.get_mut(*n, *i, *j, *k) = 0.0;
            }
        }
        if let (Some(ii), Some(iv)) = (self.ii_ptr.as_mut(), self.iv_ptr.as_mut()) {
            for [n, i, j, k] in &hard_current {
                *ii.get_mut(*n, *i, *j, *k) = 0.0;
                *iv.get_mut(*n, *i, *j, *k) = 0.0;
            }
        }

        extension
    }

    /// Build the conducting-sheet extension from the CSX `ConductingSheet`
    /// properties.
    ///
    /// Mirrors `Operator_Ext_ConductingSheet::BuildExtension`: every Yee edge
    /// inside a two-dimensional sheet primitive gets the ADE model, edges inside
    /// a boundary layer or a non-2D primitive fall back to PEC, and edges sharing
    /// their width with a non-sheet cell get the native `factor = 2`.
    pub fn build_conducting_sheet_extension(
        &mut self,
        f_max: f64,
        boundary_size: [u32; 6],
    ) -> Result<Option<EngineExtConductingSheet>, String> {
        let Some(structure) = self.structure.as_ref() else {
            return Ok(None);
        };
        let params = structure.parameters();

        struct SheetCandidate<'a> {
            priority: i32,
            properties: &'a ConductingSheetProperties,
            primitive: &'a Primitive,
        }
        let mut candidates: Vec<SheetCandidate<'_>> = Vec::new();
        for index in 0..structure.property_count() {
            let Some(property) = structure.property_by_index(index) else {
                continue;
            };
            let Some(sheet) = property.conducting_sheet() else {
                continue;
            };
            for primitive in property.primitives() {
                candidates.push(SheetCandidate {
                    priority: primitive.priority(),
                    properties: sheet,
                    primitive,
                });
            }
        }
        if candidates.is_empty() {
            return Ok(None);
        }
        let sheet_at = |coordinate: [f64; 3]| -> Option<&SheetCandidate<'_>> {
            let mut winner: Option<&SheetCandidate<'_>> = None;
            let mut winner_priority = i32::MIN;
            for candidate in &candidates {
                if winner.is_some() && candidate.priority <= winner_priority {
                    continue;
                }
                if candidate
                    .primitive
                    .is_inside(coordinate, Some(params))
                    .unwrap_or(false)
                {
                    winner = Some(candidate);
                    winner_priority = candidate.priority;
                }
            }
            winner
        };

        let counts = self.base.num_lines;
        let disc_lines = self.base.disc_lines.clone();
        let grid_delta = self.base.grid_delta;
        let dt = self.base.d_t;
        if dt <= 0.0 {
            return Err("conducting sheet requires a positive timestep".to_string());
        }
        let nx = counts[0] as usize;
        let ny = counts[1] as usize;
        let nz = counts[2] as usize;

        // Upper triangular bookkeeping: for every position, which edge
        // directions carry a sheet and along which tangential direction.
        let mut tan_dir: HashMap<[usize; 3], [i8; 3]> = HashMap::new();
        let mut active: Vec<[usize; 3]> = Vec::new();
        let mut vv = self.vv_ptr.as_mut().ok_or("vv coefficients not allocated")?;
        let mut vi = self.vi_ptr.as_mut().ok_or("vi coefficients not allocated")?;

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let pos = [i, j, k];
                    let disabled = (0..3).any(|axis| {
                        pos[axis] <= boundary_size[2 * axis] as usize
                            || pos[axis]
                                >= counts[axis] as usize
                                    - boundary_size[2 * axis + 1] as usize
                                    - 1
                    });
                    let mut local = [-1i8; 3];
                    let mut any = false;
                    for n in 0..3 {
                        let coordinate = yee_coordinates(&disc_lines, counts, n, pos, false);
                        let Some(candidate) = sheet_at(coordinate) else {
                            continue;
                        };
                        let sigma = candidate
                            .properties
                            .conductivity(Some(params))
                            .unwrap_or(0.0);
                        let thickness = candidate
                            .properties
                            .thickness(Some(params))
                            .unwrap_or(0.0);
                        let bounds = candidate
                            .primitive
                            .bounding_box(Some(params), false)
                            .unwrap_or([0.0; 6]);
                        let dimension = candidate
                            .primitive
                            .dimension(Some(params))
                            .unwrap_or(0);
                        if disabled
                            || dimension != 2
                            || sigma <= 0.0
                            || thickness <= 0.0
                        {
                            // Native fallback: this edge becomes PEC.
                            *vv.get_mut(n, i, j, k) = 0.0;
                            *vi.get_mut(n, i, j, k) = 0.0;
                            local[n] = -1;
                            continue;
                        }
                        let n_p = (n + 1) % 3;
                        let n_pp = (n + 2) % 3;
                        let mut direction = -1i8;
                        if bounds[2 * n_p] != bounds[2 * n_p + 1] {
                            direction = n_p as i8;
                        }
                        if bounds[2 * n_pp] != bounds[2 * n_pp + 1] {
                            direction = n_pp as i8;
                        }
                        if direction >= 0 {
                            local[n] = direction;
                            any = true;
                        }
                    }
                    if any {
                        tan_dir.insert(pos, local);
                        active.push(pos);
                    }
                }
            }
        }

        if active.is_empty() {
            return Ok(None);
        }

        let mut extension = EngineExtConductingSheet::new(active.len());
        for (index, &pos) in active.iter().enumerate() {
            let directions = tan_dir.get(&pos).copied().unwrap_or([-1; 3]);
            let mut coefficients = ConductingSheetCoeffs {
                v_int_ade: [[0.0; 3]; 2],
                v_ext_ade: [[0.0; 3]; 2],
                vv: [0.0; 3],
                vi: [0.0; 3],
            };
            for n in 0..3 {
                let coordinate = yee_coordinates(&disc_lines, counts, n, pos, false);
                let Some(candidate) = sheet_at(coordinate) else {
                    continue;
                };
                let direction = directions[n];
                if direction < 0 {
                    continue;
                }
                let t_dir = direction as usize;
                let sigma = candidate
                    .properties
                    .conductivity(Some(params))
                    .unwrap_or(0.0);
                let thickness = candidate
                    .properties
                    .thickness(Some(params))
                    .unwrap_or(0.0);
                let edge_length =
                    edge_length_value(&disc_lines, counts, n, pos, false, grid_delta);
                let node_width =
                    edge_length_value(&disc_lines, counts, t_dir, pos, true, grid_delta);
                if edge_length <= 0.0 || node_width <= 0.0 {
                    continue;
                }
                // Native `factor`: 2 when the neighbouring cell along the
                // tangential direction does not also carry a sheet.
                let mut factor = 1.0;
                let current = tan_dir.get(&pos).map(|map| map[t_dir]).unwrap_or(-1);
                if current < 0 {
                    factor = 2.0;
                }
                let mut neighbour = pos;
                if neighbour[t_dir] > 0 {
                    neighbour[t_dir] -= 1;
                    let previous = tan_dir
                        .get(&neighbour)
                        .map(|map| map[t_dir])
                        .unwrap_or(-1);
                    if previous < 0 {
                        factor = 2.0;
                    }
                } else {
                    factor = 2.0;
                }

                let model = crate::conducting_sheet::sheet_coefficients(
                    sigma,
                    thickness,
                    dt,
                    f_max,
                    edge_length / node_width,
                    factor,
                )?;
                coefficients.vv[n] = model.vv as f32;
                coefficients.vi[n] = model.vi as f32;
                for order in 0..2 {
                    coefficients.v_int_ade[order][n] = model.v_int_ade[order] as f32;
                    coefficients.v_ext_ade[order][n] = model.v_ext_ade[order] as f32;
                }
                *vv.get_mut(n, pos[0], pos[1], pos[2]) = model.vv as f32;
                *vi.get_mut(n, pos[0], pos[1], pos[2]) = model.vi as f32;
            }
            extension.set_position(index, pos, &coefficients);
        }
        Ok(Some(extension))
    }

    /// Build the steady-state detection extension with the native probe layout:
    /// the centre plus the quarter and three-quarter positions along each axis,
    /// all three E components.
    ///
    /// Mirrors the `Operator_Ext_SteadyState` setup in `openEMS::SetupFDTD`.
    pub fn build_steady_state_extension(&self, period_ts: u32) -> Option<EngineExtSteadyState> {
        if period_ts == 0 {
            return None;
        }
        let counts = self.base.num_lines;
        let num_lines = [
            counts[0] as usize,
            counts[1] as usize,
            counts[2] as usize,
        ];
        let center = [
            num_lines[0] / 2,
            num_lines[1] / 2,
            num_lines[2] / 2,
        ];
        let mut extension = EngineExtSteadyState::with_num_lines(period_ts as usize, num_lines);
        for component in 0..3 {
            extension.add_e_probe(center, component);
        }
        for axis in 0..3 {
            for position in [num_lines[axis] / 4, num_lines[axis] * 3 / 4] {
                let mut probe = center;
                probe[axis] = position;
                for component in 0..3 {
                    extension.add_e_probe(probe, component);
                }
            }
        }
        Some(extension)
    }

    /// Build UPML regions and their engine extensions, modifying the main
    /// operator coefficients as `Operator_Ext_UPML::BuildExtension` does.
    pub fn build_upml_extensions(
        &mut self,
        pml_size: [u32; 6],
    ) -> Result<Vec<EngineExtUPML>, String> {
        let counts = self.base.num_lines;
        let bc = self.base.m_bc;
        let sizes = pml_size;
        for dir in 0..3 {
            if sizes[2 * dir] as usize + sizes[2 * dir + 1] as usize >= counts[dir] as usize {
                return Err(format!("PML size is too large in direction {dir}"));
            }
        }
        let dt = self.base.d_t;
        if dt <= 0.0 {
            return Err("UPML requires a positive timestep".to_string());
        }

        let nx = counts[0] as usize;
        let ny = counts[1] as usize;
        let nz = counts[2] as usize;
        let lo = |face: usize| sizes[face] as usize;
        let mut regions: Vec<([usize; 3], [usize; 3])> = Vec::new();
        if bc[0] == 3 {
            regions.push(([0, 0, 0], [lo(0), ny - 1, nz - 1]));
        }
        if bc[1] == 3 {
            regions.push(([nx - 1 - lo(1), 0, 0], [nx - 1, ny - 1, nz - 1]));
        }
        let x_lo = if bc[0] == 3 { lo(0) + 1 } else { 0 };
        let x_hi = if bc[1] == 3 {
            nx.saturating_sub(2 + lo(1))
        } else {
            nx - 1
        };
        if bc[2] == 3 {
            regions.push(([x_lo, 0, 0], [x_hi, lo(2), nz - 1]));
        }
        if bc[3] == 3 {
            regions.push(([x_lo, ny - 1 - lo(3), 0], [x_hi, ny - 1, nz - 1]));
        }
        let y_lo = if bc[2] == 3 { lo(2) + 1 } else { 0 };
        let y_hi = if bc[3] == 3 {
            ny.saturating_sub(2 + lo(3))
        } else {
            ny - 1
        };
        if bc[4] == 3 {
            regions.push(([x_lo, y_lo, 0], [x_hi, y_hi, lo(4)]));
        }
        if bc[5] == 3 {
            regions.push(([x_lo, y_lo, nz - 1 - lo(5)], [x_hi, y_hi, nz - 1]));
        }
        if regions.is_empty() {
            return Err("No PML boundary is configured".to_string());
        }

        let disc_lines = &self.base.disc_lines;
        let grid_delta = self.base.grid_delta;
        let material_grid = self.material_grid.as_deref();
        let background = (
            self.base.m_bg_eps_r,
            self.base.m_bg_mue_r,
            self.base.m_bg_kappa,
        );

        let vv = self.vv_ptr.as_mut().ok_or("vv coefficients not allocated")?;
        let vi = self.vi_ptr.as_mut().ok_or("vi coefficients not allocated")?;
        let ii = self.ii_ptr.as_mut().ok_or("ii coefficients not allocated")?;
        let iv = self.iv_ptr.as_mut().ok_or("iv coefficients not allocated")?;

        let mut extensions = Vec::new();
        for (start, stop) in regions {
            if start.iter().zip(stop.iter()).any(|(low, high)| low > high) {
                continue;
            }
            let local = [
                stop[0] - start[0] + 1,
                stop[1] - start[1] + 1,
                stop[2] - start[2] + 1,
            ];
            let mut extension = EngineExtUPML::new(start, local);
            for l0 in 0..local[0] {
                for l1 in 0..local[1] {
                    for l2 in 0..local[2] {
                        let pos = [start[0] + l0, start[1] + l1, start[2] + l2];
                        for n in 0..3usize {
                            let (eps_r, mue_r, kappa) = material_full_value(
                                material_grid,
                                counts,
                                background,
                                n,
                                pos,
                            );
                            let (kappa_v, kappa_i) = upml_grading_kappa(
                                disc_lines, counts, grid_delta, bc, sizes, n, pos,
                            );
                            let n_p = (n + 1) % 3;
                            let n_pp = (n + 2) % 3;
                            let edge_len = edge_length_value(
                                disc_lines, counts, n, pos, false, grid_delta,
                            );
                            let edge_area = edge_area_value(
                                disc_lines, counts, n, pos, false, grid_delta,
                            );
                            let dual_len = edge_length_value(
                                disc_lines, counts, n, pos, true, grid_delta,
                            );
                            let dual_area = edge_area_value(
                                disc_lines, counts, n, pos, true, grid_delta,
                            );

                            let main_vv = *vv.get(n, pos[0], pos[1], pos[2]);
                            let main_vi = *vi.get(n, pos[0], pos[1], pos[2]);
                            let kappa_v_sum = kappa_v[0] + kappa_v[1] + kappa_v[2];
                            if kappa_v_sum != 0.0
                                && kappa < 1.0e3
                                && main_vv + main_vi != 0.0
                                && edge_area > 0.0
                            {
                                let denom = 2.0 * EPS0 + kappa_v[n_p] * dt;
                                let vv_new = ((2.0 * EPS0 - kappa_v[n_p] * dt) / denom) as f32;
                                let vi_new = ((2.0 * EPS0 * dt) / denom * edge_len / edge_area)
                                    as f32;
                                *vv.get_mut(n, pos[0], pos[1], pos[2]) = vv_new;
                                *vi.get_mut(n, pos[0], pos[1], pos[2]) = vi_new;
                                let denom_pp = 2.0 * EPS0 + kappa_v[n_pp] * dt;
                                let eps_scale = if eps_r > 0.0 { eps_r * EPS0 } else { EPS0 };
                                *extension.vv[n].get_mut(0, l0, l1, l2) =
                                    ((2.0 * EPS0 - kappa_v[n_pp] * dt) / denom_pp) as f32;
                                *extension.vvfn[n].get_mut(0, l0, l1, l2) =
                                    ((2.0 * EPS0 + kappa_v[n] * dt) / denom_pp / eps_scale)
                                        as f32;
                                *extension.vvfo[n].get_mut(0, l0, l1, l2) =
                                    ((2.0 * EPS0 - kappa_v[n] * dt) / denom_pp / eps_scale)
                                        as f32;
                            } else {
                                *extension.vv[n].get_mut(0, l0, l1, l2) = main_vv;
                                *vv.get_mut(n, pos[0], pos[1], pos[2]) = 0.0;
                                *extension.vvfo[n].get_mut(0, l0, l1, l2) = 0.0;
                                *extension.vvfn[n].get_mut(0, l0, l1, l2) = 1.0;
                            }

                            let main_ii = *ii.get(n, pos[0], pos[1], pos[2]);
                            let main_iv = *iv.get(n, pos[0], pos[1], pos[2]);
                            let kappa_i_sum = kappa_i[0] + kappa_i[1] + kappa_i[2];
                            if kappa_i_sum != 0.0
                                && main_ii + main_iv != 0.0
                                && dual_area > 0.0
                            {
                                let denom = 2.0 * EPS0 + kappa_i[n_p] * dt;
                                let ii_new = ((2.0 * EPS0 - kappa_i[n_p] * dt) / denom) as f32;
                                let iv_new = ((2.0 * EPS0 * dt) / denom * dual_len / dual_area)
                                    as f32;
                                *ii.get_mut(n, pos[0], pos[1], pos[2]) = ii_new;
                                *iv.get_mut(n, pos[0], pos[1], pos[2]) = iv_new;
                                let denom_pp = 2.0 * EPS0 + kappa_i[n_pp] * dt;
                                let mue_scale = if mue_r > 0.0 { mue_r * MU0 } else { MU0 };
                                *extension.ii[n].get_mut(0, l0, l1, l2) =
                                    ((2.0 * EPS0 - kappa_i[n_pp] * dt) / denom_pp) as f32;
                                *extension.iifn[n].get_mut(0, l0, l1, l2) =
                                    ((2.0 * EPS0 + kappa_i[n] * dt) / denom_pp / mue_scale)
                                        as f32;
                                *extension.iifo[n].get_mut(0, l0, l1, l2) =
                                    ((2.0 * EPS0 - kappa_i[n] * dt) / denom_pp / mue_scale)
                                        as f32;
                            } else {
                                *extension.ii[n].get_mut(0, l0, l1, l2) = main_ii;
                                *ii.get_mut(n, pos[0], pos[1], pos[2]) = 0.0;
                                *extension.iifo[n].get_mut(0, l0, l1, l2) = 0.0;
                                *extension.iifn[n].get_mut(0, l0, l1, l2) = 1.0;
                            }
                        }
                    }
                }
            }
            extensions.push(extension);
        }
        Ok(extensions)
    }
}


/// Material (eps_r, mue_r, kappa) for a component at a mesh position.
fn material_full_value(
    grid: Option<&[CellMaterial]>,
    counts: [u32; 3],
    background: (f64, f64, f64),
    component: usize,
    pos: [usize; 3],
) -> (f64, f64, f64) {
    if let Some(grid) = grid {
        let nx = counts[0] as usize;
        let ny = counts[1] as usize;
        let nz = counts[2] as usize;
        let index = component * nx * ny * nz + pos[0] * ny * nz + pos[1] * nz + pos[2];
        if let Some(material) = grid.get(index) {
            return (material.eps_r, material.mue_r, material.kappa);
        }
    }
    background
}

/// Default UPML grading profile from the native `Operator_Ext_UPML`.
fn default_upml_grading(depth: f64, dl: f64, width: f64) -> f64 {
    if depth <= 0.0 || dl <= 0.0 || width <= 0.0 {
        return 0.0;
    }
    let a = -(1.0e-6_f64).ln();
    let b = 2.5_f64.ln();
    let denominator = 2.0 * dl * Z0 * ((2.5_f64).powf(width / dl) - 1.0);
    if denominator == 0.0 {
        return 0.0;
    }
    a * b / denominator * (2.5_f64).powf(depth / dl)
}

/// Grading conductivities for one edge direction at a global position.
fn upml_grading_kappa(
    lines: &[Option<Vec<f64>>; 3],
    counts: [u32; 3],
    grid_delta: f64,
    bc: [i32; 6],
    sizes: [u32; 6],
    n_edge: usize,
    pos: [usize; 3],
) -> ([f64; 3], [f64; 3]) {
    let mut kappa_v = [0.0f64; 3];
    let mut kappa_i = [0.0f64; 3];
    let line = |dir: usize, index: usize| {
        disc_line_value(&lines[dir], counts[dir] as usize, index, false) * grid_delta
    };
    let edge = |dir: usize| edge_length_value(lines, counts, dir, pos, false, grid_delta);

    for dir in 0..3 {
        let size_lo = sizes[2 * dir] as usize;
        let size_hi = sizes[2 * dir + 1] as usize;
        let last = counts[dir] as usize - 1;
        let lower = bc[2 * dir] == 3 && size_lo > 0 && pos[dir] <= size_lo;
        let upper = bc[2 * dir + 1] == 3 && size_hi > 0 && pos[dir] + size_hi >= last;

        if lower {
            let width = (line(dir, size_lo) - line(dir, 0)).abs();
            let cells = size_lo as f64;
            let dl = width / cells;
            let mut depth = width - (line(dir, pos[dir]) - line(dir, 0)).abs();
            if dir == n_edge {
                depth -= edge(dir) / 2.0;
            }
            kappa_v[dir] = default_upml_grading(depth, dl, width);
            if dir == n_edge {
                depth += edge(dir) / 2.0;
            }
            if dir != n_edge {
                depth -= edge(dir) / 2.0;
            }
            kappa_i[dir] = default_upml_grading(depth, dl, width);
        } else if upper {
            let width = (line(dir, last) - line(dir, last - size_hi)).abs();
            let cells = size_hi as f64;
            let dl = width / cells;
            let mut depth = width - (line(dir, last) - line(dir, pos[dir])).abs();
            if dir == n_edge {
                depth += edge(dir) / 2.0;
            }
            kappa_v[dir] = default_upml_grading(depth, dl, width);
            if dir == n_edge {
                depth -= edge(dir) / 2.0;
            }
            if dir != n_edge {
                depth += edge(dir) / 2.0;
            }
            if depth > width {
                depth = 0.0;
            }
            kappa_i[dir] = default_upml_grading(depth, dl, width);
        }
    }
    (kappa_v, kappa_i)
}

/// Disc line value in drawing units. Matches native `GetDiscLine`.
/// Falls back to a uniform unit-spaced grid when no disc lines were loaded.
fn disc_line_value(lines: &Option<Vec<f64>>, count: usize, pos: usize, dual: bool) -> f64 {
    let raw = lines.as_deref().unwrap_or(&[]);
    if raw.len() >= 2 && pos < count && pos < raw.len() {
        if !dual {
            raw[pos]
        } else if pos + 1 < count && pos + 1 < raw.len() {
            0.5 * (raw[pos] + raw[pos + 1])
        } else if pos > 0 {
            raw[pos] + 0.5 * (raw[pos] - raw[pos - 1])
        } else {
            raw[pos]
        }
    } else if dual {
        pos as f64 + 0.5
    } else {
        pos as f64
    }
}

/// Disc line delta in drawing units. Matches native `GetDiscDelta`.
fn disc_delta_value(lines: &Option<Vec<f64>>, count: usize, pos: usize, dual: bool) -> f64 {
    let raw = lines.as_deref().unwrap_or(&[]);
    if raw.len() >= 2 && pos < count {
        if !dual {
            if pos + 1 < count && pos + 1 < raw.len() {
                raw[pos + 1] - raw[pos]
            } else if pos > 0 {
                raw[pos] - raw[pos - 1]
            } else {
                0.0
            }
        } else if pos > 0 {
            disc_line_value(lines, count, pos, true) - disc_line_value(lines, count, pos - 1, true)
        } else if raw.len() >= 2 {
            raw[1] - raw[0]
        } else {
            0.0
        }
    } else {
        1.0
    }
}

/// Reflect a node index across the mesh boundary, matching `AdrOp::GetPos`
/// with `SetReflection2Cell` (`uiTypeOffset = 1`).
pub fn shift_node(pos: [usize; 3], axis: usize, delta: i32, counts: [u32; 3]) -> [usize; 3] {
    let mut result = pos;
    let index = pos[axis] as i32;
    let maximum = counts[axis] as i32 - 1;
    let mut shift = delta;
    if index + shift < 0 {
        shift = -2 * index - shift - 1;
    }
    if index + shift > maximum {
        shift = 2 * (maximum - index) - shift + 1;
    }
    result[axis] = (index + shift).clamp(0, maximum) as usize;
    result
}

/// Sum `1/(L*C)` terms; `None` when any coefficient is non-positive.
fn sum_terms(terms: &[Option<f64>]) -> Option<f64> {
    let mut sum = 0.0;
    for term in terms {
        sum += (*term)?;
    }
    Some(sum)
}

/// `sum - 2*min` over four `1/(L*C)` terms, as in `CalcTimestep_Var3`.
fn weighted_term(terms: &[Option<f64>; 4]) -> Option<f64> {
    let mut sum = 0.0;
    let mut minimum = f64::INFINITY;
    for term in terms {
        let value = (*term)?;
        sum += value;
        minimum = minimum.min(value);
    }
    Some(sum - 2.0 * minimum)
}

/// Raw (non-dual) mesh delta for a node index, matching native
/// `Operator::GetRawDiscDelta`: the spacing after `index`, mirrored at the ends.
fn raw_disc_delta(
    lines: &[Option<Vec<f64>>; 3],
    counts: [u32; 3],
    axis: usize,
    index: i64,
) -> f64 {
    let Some(values) = lines.get(axis).and_then(|entry| entry.as_ref()) else {
        return 0.0;
    };
    let count = (counts.get(axis).copied().unwrap_or(0) as usize).min(values.len());
    if count < 2 {
        return 0.0;
    }
    if index < 0 {
        return values[0] - values[1];
    }
    if index >= count as i64 - 1 {
        return values[count - 2] - values[count - 1];
    }
    values[index as usize + 1] - values[index as usize]
}

/// Yee coordinate in drawing units for an edge. Matches native `GetYeeCoords`:
/// along the edge direction use the requested mesh, across it use the dual one.
fn yee_coordinates(
    lines: &[Option<Vec<f64>>; 3],
    counts: [u32; 3],
    n: usize,
    pos: [usize; 3],
    dual: bool,
) -> [f64; 3] {
    let mut coordinates = [0.0; 3];
    for d in 0..3 {
        // Native `GetYeeCoords`: transverse axes use the requested mesh
        // (`dualMesh`), the edge axis uses the opposite one.
        let use_dual = if d == n { !dual } else { dual };
        coordinates[d] = disc_line_value(&lines[d], counts[d] as usize, pos[d], use_dual);
    }
    coordinates
}

/// Edge length in metres. Matches native `GetEdgeLength`.
fn edge_length_value(
    lines: &[Option<Vec<f64>>; 3],
    counts: [u32; 3],
    n: usize,
    pos: [usize; 3],
    dual: bool,
    grid_delta: f64,
) -> f64 {
    if n >= 3 || pos[n] >= counts[n] as usize {
        return 0.0;
    }
    disc_delta_value(&lines[n], counts[n] as usize, pos[n], dual) * grid_delta
}

/// Edge area in square metres. Matches native `GetEdgeArea`.
fn edge_area_value(
    lines: &[Option<Vec<f64>>; 3],
    counts: [u32; 3],
    n: usize,
    pos: [usize; 3],
    dual: bool,
    grid_delta: f64,
) -> f64 {
    if n >= 3 {
        return 0.0;
    }
    let n_p = (n + 1) % 3;
    let n_pp = (n + 2) % 3;
    // Native `GetNodeArea` uses `GetNodeWidth(axis, pos, dual)` which is
    // `GetEdgeLength(axis, pos, !dual)`.
    edge_length_value(lines, counts, n_p, pos, !dual, grid_delta)
        * edge_length_value(lines, counts, n_pp, pos, !dual, grid_delta)
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

    fn get_disc_line(&self, n: i32, pos: u32, dual_mesh: bool) -> f64 {
        if !(0..3).contains(&n) {
            return 0.0;
        }
        let dir = n as usize;
        if pos >= self.base.num_lines[dir] {
            return 0.0;
        }
        disc_line_value(
            &self.base.disc_lines[dir],
            self.base.num_lines[dir] as usize,
            pos as usize,
            dual_mesh,
        )
    }

    fn get_disc_delta(&self, n: i32, pos: u32, dual_mesh: bool) -> f64 {
        if !(0..3).contains(&n) {
            return 0.0;
        }
        let dir = n as usize;
        if pos >= self.base.num_lines[dir] {
            return 0.0;
        }
        disc_delta_value(
            &self.base.disc_lines[dir],
            self.base.num_lines[dir] as usize,
            pos as usize,
            dual_mesh,
        )
    }

    fn get_node_width(&self, ny: i32, pos: [u32; 3], dual_mesh: bool) -> f64 {
        if !(0..3).contains(&ny) {
            return 0.0;
        }
        let dir = ny as usize;
        edge_length_value(
            &self.base.disc_lines,
            self.base.num_lines,
            dir,
            [pos[0] as usize, pos[1] as usize, pos[2] as usize],
            dual_mesh,
            self.base.grid_delta,
        )
    }

    fn get_node_area(&self, ny: i32, pos: [u32; 3], dual_mesh: bool) -> f64 {
        if !(0..3).contains(&ny) {
            return 0.0;
        }
        let dir = ny as usize;
        let ui = [pos[0] as usize, pos[1] as usize, pos[2] as usize];
        let n_p = (dir + 1) % 3;
        let n_pp = (dir + 2) % 3;
        edge_length_value(
            &self.base.disc_lines,
            self.base.num_lines,
            n_p,
            ui,
            dual_mesh,
            self.base.grid_delta,
        ) * edge_length_value(
            &self.base.disc_lines,
            self.base.num_lines,
            n_pp,
            ui,
            dual_mesh,
            self.base.grid_delta,
        )
    }

    fn get_edge_length(&self, ny: i32, pos: [u32; 3], dual_mesh: bool) -> f64 {
        if !(0..3).contains(&ny) {
            return 0.0;
        }
        edge_length_value(
            &self.base.disc_lines,
            self.base.num_lines,
            ny as usize,
            [pos[0] as usize, pos[1] as usize, pos[2] as usize],
            dual_mesh,
            self.base.grid_delta,
        )
    }

    fn get_edge_area(&self, ny: i32, pos: [u32; 3], dual_mesh: bool) -> f64 {
        if !(0..3).contains(&ny) {
            return 0.0;
        }
        edge_area_value(
            &self.base.disc_lines,
            self.base.num_lines,
            ny as usize,
            [pos[0] as usize, pos[1] as usize, pos[2] as usize],
            dual_mesh,
            self.base.grid_delta,
        )
    }

    fn get_cell_volume(&self, pos: [u32; 3], dual_mesh: bool) -> f64 {
        let ui = [pos[0] as usize, pos[1] as usize, pos[2] as usize];
        (0..3)
            .map(|n| {
                edge_length_value(
                    &self.base.disc_lines,
                    self.base.num_lines,
                    n,
                    ui,
                    dual_mesh,
                    self.base.grid_delta,
                )
            })
            .product()
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
        let nx = op.base.num_lines[0] as usize;
        let ny = op.base.num_lines[1] as usize;
        let nz = op.base.num_lines[2] as usize;
        assert_eq!(grid.len(), 3 * nx * ny * nz);
        // Cell (4,4,4) is inside the PEC box in every direction.
        let pec_cell = 4 * ny * nz + 4 * nz + 4;
        // Cell (4,4,1) is inside the substrate box in every direction.
        let sub_cell = 4 * ny * nz + 4 * nz + 1;
        for n in 0..3 {
            let pec_index = n * nx * ny * nz + pec_cell;
            let sub_index = n * nx * ny * nz + sub_cell;
            assert!(
                grid[pec_index].is_pec,
                "cell (4,4,4) should be PEC in dir {n}"
            );
            assert!(
                (grid[sub_index].eps_r - 4.4).abs() < 1e-9,
                "cell (4,4,1) should have eps_r=4.4 in dir {n}"
            );
        }
    }

    #[test]
    fn ec_coefficients_use_native_conductivity_units() {
        let mut op = Operator::new();
        op.base.grid_delta = 1e-3;
        op.base.d_t = 1.0e-12;
        op.base.set_background_material(4.0, 2.0, 0.01, 0.02, 0.0);
        op.init_data_storage([3, 3, 3]);
        op.calc_ec_operator();

        let dt = 1.0e-12;
        let delta = 1.0e-3;
        let c = 4.0 * EPS0 * delta;
        let g = 0.01 * delta;
        let l = 2.0 * MU0 * delta;
        let r = 0.02 * delta;
        let v_denom = 1.0 + dt * g / (2.0 * c);
        let i_denom = 1.0 + dt * r / (2.0 * l);
        let expected_vv = (1.0 - dt * g / (2.0 * c)) / v_denom;
        let expected_vi = (dt / c) / v_denom;
        let expected_ii = (1.0 - dt * r / (2.0 * l)) / i_denom;
        let expected_iv = (dt / l) / i_denom;

        let vv = op.vv_ptr.as_ref().unwrap();
        let vi = op.vi_ptr.as_ref().unwrap();
        let ii = op.ii_ptr.as_ref().unwrap();
        let iv = op.iv_ptr.as_ref().unwrap();
        for n in 0..3 {
            for (i, j, k) in [(0, 0, 0), (1, 1, 1), (2, 2, 2)] {
                let actual_vv = *vv.get(n, i, j, k) as f64;
                let actual_vi = *vi.get(n, i, j, k) as f64;
                let actual_ii = *ii.get(n, i, j, k) as f64;
                let actual_iv = *iv.get(n, i, j, k) as f64;
                assert!(
                    (actual_vv - expected_vv).abs() <= expected_vv.abs() * 1.0e-5,
                    "vv mismatch at ({n},{i},{j},{k}): {actual_vv:e} vs {expected_vv:e}"
                );
                assert!(
                    (actual_vi - expected_vi).abs() <= expected_vi.abs() * 1.0e-5,
                    "vi mismatch at ({n},{i},{j},{k}): {actual_vi:e} vs {expected_vi:e}"
                );
                assert!(
                    (actual_ii - expected_ii).abs() <= expected_ii.abs() * 1.0e-5,
                    "ii mismatch at ({n},{i},{j},{k}): {actual_ii:e} vs {expected_ii:e}"
                );
                assert!(
                    (actual_iv - expected_iv).abs() <= expected_iv.abs() * 1.0e-5,
                    "iv mismatch at ({n},{i},{j},{k}): {actual_iv:e} vs {expected_iv:e}"
                );
            }
        }
    }

    #[test]
    fn test_load_csx_invalid_xml() {
        let mut op = Operator::new();
        assert!(op.load_csx("not valid xml").is_err());
    }

    #[test]
    fn pec_boundary_zeroes_only_tangential_components() {
        let mut op = Operator::new();
        op.base.grid_delta = 1e-3;
        op.base.d_t = 1.0e-12;
        op.base.set_boundary_condition([0; 6]);
        op.init_data_storage([5, 5, 5]);
        op.calc_ec_operator();
        op.apply_electric_bc();

        let vv = op.vv_ptr.as_ref().unwrap();
        // x-min face at (0,2,2): Ey and Ez are tangential, Ex is normal.
        assert_eq!(*vv.get(1, 0, 2, 2), 0.0);
        assert_eq!(*vv.get(2, 0, 2, 2), 0.0);
        assert!(*vv.get(0, 0, 2, 2) > 0.0);
        // y-min face at (2,0,2): Ex and Ez are tangential, Ey is normal.
        assert_eq!(*vv.get(0, 2, 0, 2), 0.0);
        assert_eq!(*vv.get(2, 2, 0, 2), 0.0);
        assert!(*vv.get(1, 2, 0, 2) > 0.0);
        // z-min face at (2,2,0): Ex and Ey are tangential, Ez is normal.
        assert_eq!(*vv.get(0, 2, 2, 0), 0.0);
        assert_eq!(*vv.get(1, 2, 2, 0), 0.0);
        assert!(*vv.get(2, 2, 2, 0) > 0.0);
    }

    #[test]
    fn pmc_boundary_zeroes_only_tangential_components() {
        let mut op = Operator::new();
        op.base.grid_delta = 1e-3;
        op.base.d_t = 1.0e-12;
        op.base.set_boundary_condition([1; 6]);
        op.init_data_storage([5, 5, 5]);
        op.calc_ec_operator();
        op.apply_magnetic_bc();

        let ii = op.ii_ptr.as_ref().unwrap();
        // x-min face at (0,2,2): Hy and Hz are tangential, Hx is normal.
        assert_eq!(*ii.get(1, 0, 2, 2), 0.0);
        assert_eq!(*ii.get(2, 0, 2, 2), 0.0);
        assert!(*ii.get(0, 0, 2, 2) > 0.0);
        // y-min face at (2,0,2): Hx and Hz are tangential, Hy is normal.
        assert_eq!(*ii.get(0, 2, 0, 2), 0.0);
        assert_eq!(*ii.get(2, 2, 0, 2), 0.0);
        assert!(*ii.get(1, 2, 0, 2) > 0.0);
        // z-min face at (2,2,0): Hx and Hy are tangential, Hz is normal.
        assert_eq!(*ii.get(0, 2, 2, 0), 0.0);
        assert_eq!(*ii.get(1, 2, 2, 0), 0.0);
        assert!(*ii.get(2, 2, 2, 0) > 0.0);
    }

    #[test]
    fn material_priority_and_containment_select_the_right_edge() {
        // low box covers the whole domain (eps=2, priority 5 is below high box),
        // high box covers 4..6 (eps=4), sphere around (5,5,5) has priority 10.
        let xml = r#"<ContinuousStructure CoordSystem="0">
            <RectilinearGrid DeltaUnit="1">
                <XLines>0,1,2,3,4,5,6,7,8,9,10</XLines>
                <YLines>0,1,2,3,4,5,6,7,8,9,10</YLines>
                <ZLines>0,1,2,3,4,5,6,7,8,9,10</ZLines>
            </RectilinearGrid>
            <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
            <Properties>
                <Material Name="low">
                    <Property Epsilon="2" Mue="1" Kappa="0" Sigma="0"/>
                    <Primitives>
                        <Box Priority="1"><P1 X="0" Y="0" Z="0"/><P2 X="10" Y="10" Z="10"/></Box>
                    </Primitives>
                </Material>
                <Material Name="high">
                    <Property Epsilon="4" Mue="1" Kappa="0" Sigma="0"/>
                    <Primitives>
                        <Box Priority="5"><P1 X="4" Y="4" Z="4"/><P2 X="6" Y="6" Z="6"/></Box>
                    </Primitives>
                </Material>
                <Material Name="sphere">
                    <Property Epsilon="6" Mue="1" Kappa="0" Sigma="0"/>
                    <Primitives>
                        <Sphere Priority="10" Radius="1"><Center X="5" Y="5" Z="5"/></Sphere>
                    </Primitives>
                </Material>
            </Properties>
        </ContinuousStructure>"#;
        let mut op = Operator::new();
        op.load_csx(xml).expect("priority CSX should load");
        let grid = op.material_grid.as_ref().expect("material grid");
        let ny = op.base.num_lines[1] as usize;
        let nz = op.base.num_lines[2] as usize;
        let at = |i: usize, j: usize, k: usize| {
            let index = i * ny * nz + j * nz + k;
            grid[index].eps_r
        };
        // Outside the sphere and high box: low-priority box.
        assert!((at(3, 3, 3) - 2.0).abs() < 1e-12, "eps={}", at(3, 3, 3));
        // The high-priority box corner averages with the surrounding low box:
        // only one of the four quarter cells lies inside the box.
        assert!((at(4, 4, 4) - 2.5).abs() < 1e-12, "eps={}", at(4, 4, 4));
        // Inside the sphere all four quarter cells are in the sphere.
        assert!((at(5, 5, 5) - 6.0).abs() < 1e-12, "eps={}", at(5, 5, 5));
    }

    #[test]
    fn quarter_cell_averaging_matches_the_analytic_half_space() {
        // Half space eps=4 for x >= 5, background eps=1 elsewhere, uniform mesh.
        let xml = r#"<ContinuousStructure CoordSystem="0">
            <RectilinearGrid DeltaUnit="1">
                <XLines>0,1,2,3,4,5,6,7,8,9,10</XLines>
                <YLines>0,1,2,3,4,5,6,7,8,9,10</YLines>
                <ZLines>0,1,2,3,4,5,6,7,8,9,10</ZLines>
            </RectilinearGrid>
            <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
            <Properties>
                <Material Name="halfspace">
                    <Property Epsilon="4" Mue="1" Kappa="0" Sigma="0"/>
                    <Primitives>
                        <Box Priority="1"><P1 X="5" Y="-10" Z="-10"/><P2 X="10" Y="10" Z="10"/></Box>
                    </Primitives>
                </Material>
            </Properties>
        </ContinuousStructure>"#;
        let mut op = Operator::new();
        op.load_csx(xml).expect("half-space CSX should load");
        let grid = op.material_grid.as_ref().expect("material grid");
        let ny = op.base.num_lines[1] as usize;
        let nz = op.base.num_lines[2] as usize;
        // E-y edge at node (5,5,5): transverse x samples at 4.75 and 5.25,
        // so exactly two of the four quarter cells are in eps=4.
        let index = 1 * (op.base.num_lines[0] as usize) * ny * nz + 5 * ny * nz + 5 * nz + 5;
        assert!((grid[index].eps_r - 2.5).abs() < 1e-12, "eps={}", grid[index].eps_r);
        // Far inside the half space all four quarter cells are eps=4.
        let inside = 1 * (op.base.num_lines[0] as usize) * ny * nz + 7 * ny * nz + 5 * nz + 5;
        assert!((grid[inside].eps_r - 4.0).abs() < 1e-12, "eps={}", grid[inside].eps_r);
        // Far outside stays at the background.
        let outside = 1 * (op.base.num_lines[0] as usize) * ny * nz + 2 * ny * nz + 5 * nz + 5;
        assert!((grid[outside].eps_r - 1.0).abs() < 1e-12, "eps={}", grid[outside].eps_r);
    }

    #[test]
    fn metal_edges_zero_the_electric_update() {
        let mut op = Operator::new();
        op.load_csx(SAMPLE_CSX).expect("sample CSX should load");
        op.base.d_t = 1.0e-12;
        op.calc_ec_operator();
        let vv = op.vv_ptr.as_ref().unwrap();
        // Cell (4,4,4) is inside the PEC box in every direction.
        for n in 0..3 {
            assert_eq!(*vv.get(n, 4, 4, 4), 0.0, "PEC vv should be zero in dir {n}");
        }
        // A vacuum edge outside the metal keeps its update coefficient.
        assert!(*vv.get(0, 8, 8, 8) > 0.0);
    }

    #[test]
    fn mur_extension_builds_six_faces() {
        let mut op = Operator::new();
        op.base.grid_delta = 1.0e-3;
        op.base.d_t = 1.9e-12;
        op.base.set_boundary_condition([2; 6]);
        op.init_data_storage([6, 6, 6]);
        let extension = op.build_mur_extension([0.0; 6]);
        assert_eq!(extension.faces.len(), 6);
        for face in &extension.faces {
            assert_eq!(face.coeff_p.len(), face.dims_p * face.dims_pp);
            assert_eq!(face.coeff_pp.len(), face.dims_p * face.dims_pp);
            assert!(face.coeff_p.iter().all(|coefficient| coefficient.abs() < 1.0));
            assert!(face
                .coeff_pp
                .iter()
                .all(|coefficient| coefficient.abs() < 1.0));
        }
    }

    #[test]
    fn upml_builds_six_regions_with_finite_coefficients() {
        let mut op = Operator::new();
        op.base.grid_delta = 1.0e-3;
        op.base.d_t = 1.0e-12;
        op.base.set_boundary_condition([3; 6]);
        op.init_data_storage([10, 10, 10]);
        op.calc_ec_operator();
        let extensions = op
            .build_upml_extensions([2; 6])
            .expect("UPML should build on a 10^3 grid");
        assert_eq!(extensions.len(), 6, "one region per PML face");
        let mut non_trivial = false;
        for extension in &extensions {
            assert_eq!(extension.vv.len(), 3);
            let [nx, ny, nz] = extension.num_lines;
            for comp in 0..3 {
                for i in 0..nx {
                    for j in 0..ny {
                        for k in 0..nz {
                            let vv = *extension.vv[comp].get(0, i, j, k);
                            let vvfn = *extension.vvfn[comp].get(0, i, j, k);
                            let ii = *extension.ii[comp].get(0, i, j, k);
                            assert!(vv.is_finite() && vvfn.is_finite() && ii.is_finite());
                            if (vvfn - 1.0).abs() > 1e-12 {
                                non_trivial = true;
                            }
                        }
                    }
                }
            }
        }
        assert!(non_trivial, "UPML coefficients were never activated");
    }

    #[test]
    fn nonuniform_mesh_edge_lengths_and_ec_coefficients() {
        // DeltaUnit 1 mm; graded X (1,2,3), Y (2,2), Z (5,5) in drawing units.
        let xml = r#"<ContinuousStructure CoordSystem="0">
            <RectilinearGrid DeltaUnit="0.001">
                <XLines>0,1,3,6</XLines>
                <YLines>0,2,4</YLines>
                <ZLines>0,5,10</ZLines>
            </RectilinearGrid>
            <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
            <Properties></Properties>
        </ContinuousStructure>"#;
        let mut op = Operator::new();
        op.load_csx(xml).expect("non-uniform CSX should load");
        assert_eq!(op.base.num_lines, [4, 3, 3]);
        assert!((op.base.grid_delta - 1.0e-3).abs() < 1e-15);

        // Edge lengths in metres follow the real line spacing.
        assert!((op.get_edge_length(0, [0, 0, 0], false) - 1.0e-3).abs() < 1e-15);
        assert!((op.get_edge_length(0, [1, 0, 0], false) - 2.0e-3).abs() < 1e-15);
        assert!((op.get_edge_length(0, [2, 0, 0], false) - 3.0e-3).abs() < 1e-15);
        // The last node repeats the previous delta (native GetDiscDelta).
        assert!((op.get_edge_length(0, [3, 0, 0], false) - 3.0e-3).abs() < 1e-15);
        assert!((op.get_edge_length(1, [0, 0, 0], false) - 2.0e-3).abs() < 1e-15);
        assert!((op.get_edge_length(2, [0, 0, 0], false) - 5.0e-3).abs() < 1e-15);
        assert!(
            (op.get_edge_area(0, [0, 0, 0], false) - 2.0e-3 * 5.0e-3).abs() < 1e-18
        );

        // EC_C uses area/length, so vi_y / vi_x equals c_x / c_y for vacuum.
        op.base.d_t = 1.0e-15;
        op.calc_ec_operator();
        let vi = op.vi_ptr.as_ref().unwrap();
        let vi_x = *vi.get(0, 0, 0, 0) as f64;
        let vi_y = *vi.get(1, 0, 0, 0) as f64;
        assert!(vi_x > 0.0 && vi_y > 0.0);
        let ratio = vi_y / vi_x;
        assert!(
            (ratio - 4.0).abs() < 1.0e-4,
            "vi_y/vi_x should be c_x/c_y = 4.0, got {ratio}"
        );

        // A non-uniform mesh must still produce a positive CFL timestep.
        let dt = op.calc_timestep();
        assert!(dt > 0.0 && dt.is_finite(), "CFL timestep = {dt}");
    }
}


#[cfg(test)]
mod timestep_tests {
    use super::*;

    fn uniform_operator(delta_unit: f64, lines: usize) -> Operator {
        let values: Vec<String> = (0..lines).map(|index| index.to_string()).collect();
        let joined = values.join(",");
        let xml = format!(
            r#"<ContinuousStructure CoordSystem="0">
                <RectilinearGrid DeltaUnit="{delta_unit}">
                    <XLines>{joined}</XLines>
                    <YLines>{joined}</YLines>
                    <ZLines>{joined}</ZLines>
                </RectilinearGrid>
                <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
                <Properties></Properties>
            </ContinuousStructure>"#
        );
        let mut op = Operator::new();
        op.load_csx(&xml).expect("uniform CSX should load");
        op
    }

    #[test]
    fn var1_and_var3_match_the_uniform_mesh_analytic_values() {
        let delta = 1.0e-3;
        let mut op = uniform_operator(delta, 5);
        let c = 1.0 / (EPS0 * MU0).sqrt();

        let var1 = op.calc_timestep_var1();
        let expected_var1 = delta / (2.0 * c);
        assert!(
            (var1 - expected_var1).abs() / expected_var1 < 1.0e-6,
            "Var1 {var1:e} vs analytic {expected_var1:e}"
        );

        op.base.m_time_step_var = 3;
        let var3 = op.calc_timestep();
        let expected_var3 = delta / (3.0f64.sqrt() * c);
        assert!(
            (var3 - expected_var3).abs() / expected_var3 < 1.0e-6,
            "Var3 {var3:e} vs analytic {expected_var3:e}"
        );
        assert!(var3 > var1, "Var3 {var3:e} should exceed Var1 {var1:e}");
    }

    #[test]
    fn var3_is_larger_than_var1_on_a_nonuniform_mesh() {
        let xml = r#"<ContinuousStructure CoordSystem="0">
            <RectilinearGrid DeltaUnit="0.001">
                <XLines>0,1,1.5,3,6</XLines>
                <YLines>0,0.6,1.2,2.5,5</YLines>
                <ZLines>0,2,2.4,2.8,4</ZLines>
            </RectilinearGrid>
            <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
            <Properties></Properties>
        </ContinuousStructure>"#;
        let mut op = Operator::new();
        op.load_csx(xml).expect("non-uniform CSX should load");
        let var1 = op.calc_timestep_var1();
        op.base.m_time_step_var = 3;
        let var3 = op.calc_timestep();
        assert!(var1 > 0.0 && var3 > 0.0);
        assert!(
            var3 >= var1,
            "Var3 {var3:e} should not be smaller than Var1 {var1:e}"
        );
    }
}

#[cfg(test)]
mod anisotropy_tests {
    use super::*;

    #[test]
    fn anisotropic_epsilon_scales_each_edge_direction() {
        let xml = r#"<ContinuousStructure CoordSystem="0">
            <RectilinearGrid DeltaUnit="0.001">
                <XLines>0,1,2,3,4</XLines>
                <YLines>0,1,2,3,4</YLines>
                <ZLines>0,1,2,3,4</ZLines>
            </RectilinearGrid>
            <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
            <Properties>
                <Material Name="aniso">
                    <Property Epsilon="2,4,6" Mue="1" Kappa="0" Sigma="0"/>
                    <Primitives>
                        <Box Priority="1"><P1 X="-1" Y="-1" Z="-1"/><P2 X="10" Y="10" Z="10"/></Box>
                    </Primitives>
                </Material>
            </Properties>
        </ContinuousStructure>"#;
        let mut op = Operator::new();
        op.load_csx(xml).expect("anisotropic CSX should load");
        let grid = op.material_grid.as_ref().expect("material grid");
        let ny = op.base.num_lines[1] as usize;
        let nz = op.base.num_lines[2] as usize;
        for n in 0..3 {
            let index = n * (op.base.num_lines[0] as usize) * ny * nz + 2 * ny * nz + 2 * nz + 2;
            let expected = [2.0, 4.0, 6.0][n];
            assert!(
                (grid[index].eps_r - expected).abs() < 1e-12,
                "direction {n}: eps {} vs {expected}",
                grid[index].eps_r
            );
        }

        op.base.d_t = 1.0e-12;
        op.calc_ec_operator();
        let vi = op.vi_ptr.as_ref().unwrap();
        let vx = *vi.get(0, 2, 2, 2);
        let vy = *vi.get(1, 2, 2, 2);
        let vz = *vi.get(2, 2, 2, 2);
        // vi = dt / C and C is proportional to eps, so vi_x:vi_y:vi_z = 1/2:1/4:1/6.
        assert!((vx / vy - 2.0).abs() < 1e-5, "vi_x/vi_y = {}", vx / vy);
        assert!((vy / vz - 1.5).abs() < 1e-5, "vi_y/vi_z = {}", vy / vz);
    }
}

#[cfg(test)]
mod steady_state_layout_tests {
    use super::*;

    #[test]
    fn steady_state_extension_uses_the_native_probe_layout() {
        let lines: Vec<String> = (0..9).map(|index| index.to_string()).collect();
        let joined = lines.join(",");
        let xml = format!(
            r#"<ContinuousStructure CoordSystem="0">
                <RectilinearGrid DeltaUnit="0.001">
                    <XLines>{joined}</XLines>
                    <YLines>{joined}</YLines>
                    <ZLines>{joined}</ZLines>
                </RectilinearGrid>
                <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
                <Properties></Properties>
            </ContinuousStructure>"#
        );
        let mut op = Operator::new();
        op.load_csx(&xml).expect("mesh should load");
        assert!(op.build_steady_state_extension(0).is_none());

        let extension = op
            .build_steady_state_extension(10)
            .expect("periodic signals get detection");
        // Centre plus quarter and three-quarter positions on each axis,
        // all three components: 3 + 3 * 2 * 3 = 21 probes.
        assert_eq!(extension.probe_positions().len(), 21);
        assert_eq!(extension.probe_components().len(), 21);
        // Centre of a 9-line mesh is index 4.
        assert_eq!(extension.probe_positions()[0], [4, 4, 4]);
        assert_eq!(extension.probe_components()[0], 0);
        // First offset probe is x at 2 (quarter) for all components.
        assert_eq!(extension.probe_positions()[3], [2, 4, 4]);
        // The next group is x at 6 (three quarters).
        assert_eq!(extension.probe_positions()[6], [6, 4, 4]);
        // y quarter probe.
        assert_eq!(extension.probe_positions()[9], [4, 2, 4]);
        // z three-quarter probe (last group, first component).
        assert_eq!(extension.probe_positions()[18], [4, 4, 6]);
    }
}

#[cfg(test)]
mod conducting_sheet_tests {
    use super::*;

    fn mesh_with_sheet(box_bounds: &str) -> String {
        let lines: Vec<String> = (0..11).map(|index| index.to_string()).collect();
        let joined = lines.join(",");
        format!(
            r#"<ContinuousStructure CoordSystem="0">
                <RectilinearGrid DeltaUnit="0.001">
                    <XLines>{joined}</XLines>
                    <YLines>{joined}</YLines>
                    <ZLines>{joined}</ZLines>
                </RectilinearGrid>
                <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
                <Properties>
                    <ConductingSheet Name="sheet" Conductivity="1e5" Thickness="35e-6">
                        <Primitives>{box_bounds}</Primitives>
                    </ConductingSheet>
                </Properties>
            </ContinuousStructure>"#
        )
    }

    #[test]
    fn two_dimensional_sheet_gets_the_ade_model() {
        let xml = mesh_with_sheet(
            r#"<Box Priority="1"><P1 X="5" Y="-1" Z="-1"/><P2 X="5" Y="12" Z="12"/></Box>"#,
        );
        let mut op = Operator::new();
        op.load_csx(&xml).expect("sheet CSX should load");
        op.base.d_t = 1.0e-12;
        op.calc_ec_operator();

        let extension = op
            .build_conducting_sheet_extension(10.0e9, [0; 6])
            .expect("sheet model should build")
            .expect("the sheet must be active");
        assert!(extension.num_positions > 0);
        // Tangential E edges on the sheet plane are modified; the normal
        // component keeps its vacuum update coefficient.
        let vv = op.vv_ptr.as_ref().unwrap();
        let tangential = *vv.get(2, 5, 5, 5);
        let normal = *vv.get(0, 5, 5, 5);
        assert!(
            tangential < 0.999_9,
            "tangential vv should include the sheet loss, got {tangential}"
        );
        assert!(
            normal > 0.999,
            "the normal edge is not on the sheet, got {normal}"
        );
        // The extension carries finite ADE coefficients.
        for order in 0..2 {
            let coefficient = extension.v_int_ade[order][2][0];
            assert!(coefficient.is_finite() && coefficient.abs() < 1.0);
        }
    }

    #[test]
    fn volume_primitive_falls_back_to_pec_and_is_not_registered() {
        let xml = mesh_with_sheet(
            r#"<Box Priority="1"><P1 X="4" Y="0" Z="0"/><P2 X="6" Y="10" Z="10"/></Box>"#,
        );
        let mut op = Operator::new();
        op.load_csx(&xml).expect("sheet CSX should load");
        op.base.d_t = 1.0e-12;
        op.calc_ec_operator();
        let extension = op
            .build_conducting_sheet_extension(10.0e9, [0; 6])
            .expect("the model must not error");
        eprintln!(
            "3d extension positions: {:?}",
            extension.as_ref().map(|e| e.num_positions)
        );
        assert!(extension.is_none(), "3-D primitives are not sheets");
        let vv = op.vv_ptr.as_ref().unwrap();
        assert_eq!(*vv.get(2, 5, 5, 5), 0.0, "fallback edges become PEC");
    }

    #[test]
    fn boundary_layer_edges_fall_back_to_pec() {
        let xml = mesh_with_sheet(
            r#"<Box Priority="1"><P1 X="5" Y="-1" Z="-1"/><P2 X="5" Y="12" Z="12"/></Box>"#,
        );
        let mut op = Operator::new();
        op.load_csx(&xml).expect("sheet CSX should load");
        op.base.d_t = 1.0e-12;
        op.calc_ec_operator();
        // A 5-cell boundary layer covers x = 5 (0..5 and 5..10).
        let extension = op
            .build_conducting_sheet_extension(10.0e9, [5, 5, 0, 0, 0, 0])
            .expect("the model must not error");
        eprintln!(
            "layer extension positions: {:?}",
            extension.as_ref().map(|e| e.num_positions)
        );
        assert!(extension.is_none(), "edges inside the layer are disabled");
        let vv = op.vv_ptr.as_ref().unwrap();
        assert_eq!(*vv.get(2, 5, 5, 5), 0.0);
    }
}

#[cfg(test)]
mod yee_convention_tests {
    use super::*;

    fn lines(values: [&[f64]; 3]) -> ([Option<Vec<f64>>; 3], [u32; 3]) {
        (
            [
                Some(values[0].to_vec()),
                Some(values[1].to_vec()),
                Some(values[2].to_vec()),
            ],
            [
                values[0].len() as u32,
                values[1].len() as u32,
                values[2].len() as u32,
            ],
        )
    }

    #[test]
    fn yee_coordinates_use_nodes_transverse_and_dual_along_the_edge() {
        // Non-uniform mesh so node and dual positions are distinguishable.
        let x: &[f64] = &[0.0, 1.0, 3.0];
        let y: &[f64] = &[0.0, 3.0, 7.0];
        let z: &[f64] = &[0.0, 5.0, 11.0];
        let (disc, counts) = lines([x, y, z]);

        // E_y at (1,1,1): x and z are nodes, y is the dual midpoint.
        let coordinate = yee_coordinates(&disc, counts, 1, [1, 1, 1], false);
        assert_eq!(coordinate, [1.0, 5.0, 5.0]);

        // The dual request swaps along/transverse.
        let dual = yee_coordinates(&disc, counts, 1, [1, 1, 1], true);
        assert_eq!(dual, [2.0, 3.0, 8.0]);
    }

    #[test]
    fn capacitance_uses_the_dual_area_on_a_nonuniform_mesh() {
        let x: &[f64] = &[0.0, 1.0, 3.0];
        let y: &[f64] = &[0.0, 3.0, 7.0];
        let z: &[f64] = &[0.0, 5.0, 11.0];
        let (disc, counts) = lines([x, y, z]);
        let mut op = Operator::new();
        op.base.num_lines = counts;
        op.base.disc_lines = disc;
        op.base.grid_delta = 1.0e-3;

        // Native `Calc_ECPos` at the interior position (1,1,1):
        // C = eps * dual_y * dual_z / primary_x.
        let dual_y = 3.5e-3;
        let dual_z = 5.5e-3;
        let primary_x = 2.0e-3;
        let expected = EPS0 * dual_y * dual_z / primary_x;
        let measured = op.get_ec_c(0, 1, 1, 1);
        assert!(
            (measured - expected).abs() / expected < 1e-12,
            "C {measured:e} vs {expected:e}"
        );

        // L = mue * primary_y * primary_z / dual_x.
        let primary_y = 4.0e-3;
        let primary_z = 6.0e-3;
        let dual_x = 1.5e-3;
        let expected_l = MU0 * primary_y * primary_z / dual_x;
        let measured_l = op.get_ec_l(0, 1, 1, 1);
        assert!(
            (measured_l - expected_l).abs() / expected_l < 1e-12,
            "L {measured_l:e} vs {expected_l:e}"
        );
    }
}
