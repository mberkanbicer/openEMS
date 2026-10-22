//! Equivalent-circuit operator: EC coefficients, material averaging, PEC.
//! Mirrors `FDTD/operator.h/cpp` in the C++ codebase.

use crate::array_nijk::FieldArray;
use crate::constants::{EPS0, FDTD_FLOAT, MU0};
use crate::operator_base::OperatorBase;

/// The basic FDTD operator.
/// Computes equivalent-circuit coefficients (vv, vi, ii, iv) for every Yee cell.
pub struct Operator {
    pub base: crate::operator_base::OperatorBaseImpl,
    pub vv_ptr: Option<FieldArray>,
    pub vi_ptr: Option<FieldArray>,
    pub ii_ptr: Option<FieldArray>,
    pub iv_ptr: Option<FieldArray>,
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
        }
    }

    /// Allocate the EC coefficient field arrays.
    /// Mirrors C++ `Init_EC`.
    pub fn init_data_storage(&mut self, num_lines: [usize; 3]) {
        let extent = [num_lines[0], num_lines[1], num_lines[2]];
        self.vv_ptr = Some(FieldArray::new("vv", extent));
        self.vi_ptr = Some(FieldArray::new("vi", extent));
        self.ii_ptr = Some(FieldArray::new("ii", extent));
        self.iv_ptr = Some(FieldArray::new("iv", extent));
    }

    /// Compute all EC coefficients (vv, vi, ii, iv) for every cell.
    /// Mirrors C++ `Calc_ECOperator`.
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
        // borrow-checker conflicts (calc_ec_pos borrows self immutably).
        let bg_eps_r = self.base.m_bg_eps_r;
        let bg_mue_r = self.base.m_bg_mue_r;
        let bg_kappa = self.base.m_bg_kappa;
        let bg_sigma = self.base.m_bg_sigma;
        let grid_delta = self.base.grid_delta;

        for n in 0..3 {
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        // Background-material EC values (no self borrow needed).
                        // Cast to FDTD_FLOAT (f32) to match dt.
                        let area = grid_delta * grid_delta;
                        let delta = grid_delta;
                        let c = (bg_eps_r * EPS0 * area / delta) as FDTD_FLOAT;
                        let g = (bg_kappa * EPS0 * area / delta) as FDTD_FLOAT;
                        let l = (bg_mue_r * MU0 * area / delta) as FDTD_FLOAT;
                        let r = (bg_sigma * EPS0 * area / delta) as FDTD_FLOAT;

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

    fn get_ec_c(&self, _n: i32, _i: usize, _j: usize, _k: usize) -> f64 {
        // EC_C = eps * area / delta; for uniform background: eps_r * EPS0 * grid_delta
        self.base.m_bg_eps_r * EPS0 * self.base.grid_delta
    }

    fn get_ec_l(&self, _n: i32, _i: usize, _j: usize, _k: usize) -> f64 {
        // EC_L = mue * area / delta; for uniform background: mue_r * MU0 * grid_delta
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
    /// Background-material-only implementation (no CSXCAD integration yet).
    /// Kept for future CSXCAD integration; currently unused.
    #[allow(dead_code)]
    fn calc_eff_mat_pos(&self, _ny: i32, _pos: [u32; 3]) -> [f64; 4] {
        [
            self.base.m_bg_eps_r,
            self.base.m_bg_kappa,
            self.base.m_bg_mue_r,
            self.base.m_bg_sigma,
        ]
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
        self.base
            .snap_to_mesh(coord, uicoord, dual_mesh, full_mesh, inside)
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
        self.base.snap_box_to_mesh(
            start,
            stop,
            ui_start,
            ui_stop,
            dual_mesh,
            full_mesh,
            snap_method,
            b_start_in,
            b_stop_in,
        )
    }

    fn set_boundary_condition(&mut self, bcs: [i32; 6]) {
        self.base.set_boundary_condition(bcs);
    }

    fn set_material_store_flags(&mut self, type_: i32, val: bool) {
        self.base.set_material_store_flags(type_, val);
    }

    fn cleanup_material_storage(&mut self) {
        self.base.cleanup_material_storage();
    }

    fn get_disc_material(&self, type_: i32, ny: i32, pos: [u32; 3]) -> f64 {
        self.base.get_disc_material(type_, ny, pos)
    }

    fn get_cell_center_material_avg_coord_uint(&self, pos: [u32; 3], coord: &mut [f64; 3]) -> bool {
        self.base
            .get_cell_center_material_avg_coord_uint(pos, coord)
    }

    fn get_cell_center_material_avg_coord_int(&self, pos: [i32; 3], coord: &mut [f64; 3]) -> bool {
        self.base.get_cell_center_material_avg_coord_int(pos, coord)
    }

    fn get_primitives_bbox(
        &self,
        pos_x: i32,
        pos_y: i32,
        pos_z: i32,
        type_: i32,
    ) -> Vec<*mut std::ffi::c_void> {
        self.base.get_primitives_bbox(pos_x, pos_y, pos_z, type_)
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
}
