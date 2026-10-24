//! Engine: time-stepping loop, voltage/current updates.
//! Mirrors `FDTD/engine.h/cpp` in the C++ codebase.

use crate::array_nijk::FieldArray;
use crate::constants::FDTD_FLOAT;
use crate::extensions::EngineExtension;
use crate::operator_base::OperatorBase;

/// The basic FDTD engine.
/// Time-steps the equivalent-circuit coefficients over the field arrays.
pub struct Engine {
    pub op: Box<dyn OperatorBase>,
    pub volt_ptr: FieldArray,
    pub curr_ptr: FieldArray,
    pub num_lines: [usize; 3],
    pub num_ts: u32,
    pub extensions: Vec<Box<dyn EngineExtension>>,
}

impl Engine {
    /// Create a new engine for the given operator.
    pub fn new(op: Box<dyn OperatorBase>) -> Self {
        let num_lines = [
            op.get_number_of_lines(0, true) as usize,
            op.get_number_of_lines(1, true) as usize,
            op.get_number_of_lines(2, true) as usize,
        ];
        Self {
            op,
            volt_ptr: FieldArray::new("volt", [num_lines[0], num_lines[1], num_lines[2]]),
            curr_ptr: FieldArray::new("curr", [num_lines[0], num_lines[1], num_lines[2]]),
            num_lines,
            num_ts: 0,
            extensions: Vec::new(),
        }
    }

    /// Add a physics extension to the engine.
    pub fn add_extension(&mut self, ext: Box<dyn EngineExtension>) {
        self.extensions.push(ext);
    }

    /// Get the number of timesteps executed.
    pub fn get_number_of_timesteps(&self) -> u32 {
        self.num_ts
    }

    /// Query the steady-state convergence diff from the steady-state extension,
    /// if one is registered. Returns `None` if no steady-state extension exists.
    pub fn get_steady_state_diff(&self) -> Option<FDTD_FLOAT> {
        for ext in self.extensions.iter() {
            if let Some(diff) = ext.get_last_diff() {
                return Some(diff);
            }
        }
        None
    }

    /// Sum of the squared voltage and current components over the interior
    /// cells, mirroring `Engine_Interface_FDTD::CalcFastEnergy`.
    pub fn fast_energy(&self) -> f64 {
        let mut energy = 0.0;
        for n in 0..3 {
            for i in 0..self.num_lines[0].saturating_sub(1) {
                for j in 0..self.num_lines[1].saturating_sub(1) {
                    for k in 0..self.num_lines[2].saturating_sub(1) {
                        let v = *self.volt_ptr.get(n, i, j, k) as f64;
                        let c = *self.curr_ptr.get(n, i, j, k) as f64;
                        energy += v * v + c * c;
                    }
                }
            }
        }
        energy
    }

    /// Execute pre-voltage extension updates (highest priority last).
    pub fn do_pre_voltage_updates(&mut self) {
        self.extensions
            .sort_by_key(|a| std::cmp::Reverse(a.get_priority()));
        for ext in self.extensions.iter_mut().rev() {
            ext.do_pre_voltage_updates(&mut self.volt_ptr, &self.curr_ptr);
        }
    }

    /// Main FDTD engine voltage updates (Yee E-field update).
    pub fn update_voltages(&mut self) {
        let vv = self.op.get_vv().expect("EC coefficients not initialized");
        let vi = self.op.get_vi().expect("EC coefficients not initialized");

        let nx = self.num_lines[0];
        let ny = self.num_lines[1];
        let nz = self.num_lines[2];

        for i in 0..nx {
            let shift_i = if i > 0 { 1usize } else { 0usize };
            for j in 0..ny {
                let shift_j = if j > 0 { 1usize } else { 0usize };
                for k in 0..nz {
                    let shift_k = if k > 0 { 1usize } else { 0usize };

                    // Read all current values before taking mutable borrow
                    let c2_ijk = curr_get(self, 2, i, j, k);
                    let c2_ijm1k = curr_get(self, 2, i, j - shift_j, k);
                    let c1_ijk = curr_get(self, 1, i, j, k);
                    let c1_ijkm1 = curr_get(self, 1, i, j, k - shift_k);
                    let c0_ijk = curr_get(self, 0, i, j, k);
                    let c0_ijkm1 = curr_get(self, 0, i, j, k - shift_k);
                    let c2_im1jk = curr_get(self, 2, i - shift_i, j, k);
                    let c1_im1jk = curr_get(self, 1, i - shift_i, j, k);
                    let c0_ijm1k = curr_get(self, 0, i, j - shift_j, k);

                    // Ex update: curl of H
                    let ex_ref = self.volt_ptr.get_mut(0, i, j, k);
                    let vv0 = *vv.get(0, i, j, k);
                    let vi0 = *vi.get(0, i, j, k);
                    let curl_h = c2_ijk - c2_ijm1k - c1_ijk + c1_ijkm1;
                    *ex_ref = *ex_ref * vv0 + vi0 * curl_h;

                    // Ey update
                    let ey_ref = self.volt_ptr.get_mut(1, i, j, k);
                    let vv1 = *vv.get(1, i, j, k);
                    let vi1 = *vi.get(1, i, j, k);
                    let curl_h = c0_ijk - c0_ijkm1 - c2_ijk + c2_im1jk;
                    *ey_ref = *ey_ref * vv1 + vi1 * curl_h;

                    // Ez update
                    let ez_ref = self.volt_ptr.get_mut(2, i, j, k);
                    let vv2 = *vv.get(2, i, j, k);
                    let vi2 = *vi.get(2, i, j, k);
                    let curl_h = c1_ijk - c1_im1jk - c0_ijk + c0_ijm1k;
                    *ez_ref = *ez_ref * vv2 + vi2 * curl_h;
                }
            }
        }
    }

    /// Execute post-voltage extension updates (highest priority first).
    pub fn do_post_voltage_updates(&mut self) {
        self.extensions
            .sort_by_key(|a| std::cmp::Reverse(a.get_priority()));
        for ext in self.extensions.iter_mut() {
            ext.do_post_voltage_updates(&mut self.volt_ptr, &self.curr_ptr);
        }
    }

    /// Apply extension voltage changes.
    pub fn apply2_voltages(&mut self) {
        self.extensions
            .sort_by_key(|a| std::cmp::Reverse(a.get_priority()));
        for ext in self.extensions.iter_mut() {
            ext.apply2_voltages(&mut self.volt_ptr, &self.curr_ptr);
        }
    }

    /// Execute pre-current extension updates (highest priority last).
    pub fn do_pre_current_updates(&mut self) {
        self.extensions
            .sort_by_key(|a| std::cmp::Reverse(a.get_priority()));
        for ext in self.extensions.iter_mut().rev() {
            ext.do_pre_current_updates(&self.volt_ptr, &mut self.curr_ptr);
        }
    }

    /// Main FDTD engine current updates (Yee H-field update).
    pub fn update_currents(&mut self) {
        let ii = self.op.get_ii().expect("EC coefficients not initialized");
        let iv = self.op.get_iv().expect("EC coefficients not initialized");

        let nx = self.num_lines[0];
        let ny = self.num_lines[1];
        let nz = self.num_lines[2];

        for i in 0..nx.saturating_sub(1) {
            for j in 0..ny.saturating_sub(1) {
                for k in 0..nz.saturating_sub(1) {
                    // Read all voltage values before taking mutable borrow
                    let v2_ijk = volt_get(self, 2, i, j, k);
                    let v2_ijp1k = volt_get(self, 2, i, j + 1, k);
                    let v1_ijk = volt_get(self, 1, i, j, k);
                    let v1_ijkp1 = volt_get(self, 1, i, j, k + 1);
                    let v0_ijk = volt_get(self, 0, i, j, k);
                    let v0_ijkp1 = volt_get(self, 0, i, j, k + 1);
                    let v2_ip1jk = volt_get(self, 2, i + 1, j, k);
                    let v1_ip1jk = volt_get(self, 1, i + 1, j, k);
                    let v0_ijp1k = volt_get(self, 0, i, j + 1, k);

                    // Hx update: curl of E
                    let hx_ref = self.curr_ptr.get_mut(0, i, j, k);
                    let ii0 = *ii.get(0, i, j, k);
                    let iv0 = *iv.get(0, i, j, k);
                    let curl_e = v2_ijk - v2_ijp1k - v1_ijk + v1_ijkp1;
                    *hx_ref = *hx_ref * ii0 + iv0 * curl_e;

                    // Hy update
                    let hy_ref = self.curr_ptr.get_mut(1, i, j, k);
                    let ii1 = *ii.get(1, i, j, k);
                    let iv1 = *iv.get(1, i, j, k);
                    let curl_e = v0_ijk - v0_ijkp1 - v2_ijk + v2_ip1jk;
                    *hy_ref = *hy_ref * ii1 + iv1 * curl_e;

                    // Hz update
                    let hz_ref = self.curr_ptr.get_mut(2, i, j, k);
                    let ii2 = *ii.get(2, i, j, k);
                    let iv2 = *iv.get(2, i, j, k);
                    let curl_e = v1_ijk - v1_ip1jk - v0_ijk + v0_ijp1k;
                    *hz_ref = *hz_ref * ii2 + iv2 * curl_e;
                }
            }
        }
    }

    /// Execute post-current extension updates (highest priority first).
    pub fn do_post_current_updates(&mut self) {
        self.extensions
            .sort_by_key(|a| std::cmp::Reverse(a.get_priority()));
        for ext in self.extensions.iter_mut() {
            ext.do_post_current_updates(&self.volt_ptr, &mut self.curr_ptr);
        }
    }

    /// Apply extension current changes.
    pub fn apply2_current(&mut self) {
        self.extensions
            .sort_by_key(|a| std::cmp::Reverse(a.get_priority()));
        for ext in self.extensions.iter_mut() {
            ext.apply2_current(&self.volt_ptr, &mut self.curr_ptr);
        }
    }

    /// Iterate a number of timesteps.
    pub fn iterate_ts(&mut self, iter_ts: u32) -> bool {
        for _ in 0..iter_ts {
            // Voltage updates with extensions
            self.do_pre_voltage_updates();
            self.update_voltages();
            self.do_post_voltage_updates();
            self.apply2_voltages();

            // Current updates with extensions
            self.do_pre_current_updates();
            self.update_currents();
            self.do_post_current_updates();
            self.apply2_current();

            self.num_ts += 1;
        }
        true
    }
}

/// Helper to read current field value with boundary shift (0 at lower boundary).
fn curr_get(eng: &Engine, n: usize, i: usize, j: usize, k: usize) -> FDTD_FLOAT {
    *eng.curr_ptr.get(n, i, j, k)
}

/// Helper to read voltage field value.
fn volt_get(eng: &Engine, n: usize, i: usize, j: usize, k: usize) -> FDTD_FLOAT {
    *eng.volt_ptr.get(n, i, j, k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let op = Box::new(crate::operator::Operator::new());
        let eng = Engine::new(op);
        assert_eq!(eng.get_number_of_timesteps(), 0);
    }
}
