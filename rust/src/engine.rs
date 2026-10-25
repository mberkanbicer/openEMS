//! Engine: time-stepping loop, voltage/current updates.
//! Mirrors `FDTD/engine.h/cpp` in the C++ codebase.

use crate::array_nijk::FieldArray;
use crate::constants::FDTD_FLOAT;
use crate::extensions::EngineExtension;
use crate::operator_base::OperatorBase;

/// A raw field pointer that may be handed to a worker thread.
///
/// The update kernels partition the `i` range, so every worker touches a
/// disjoint slice of the written array while the other arrays stay read-only.
/// The pointer is only dereferenced inside the scoped-thread block.
#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);

// SAFETY: the engine upholds disjoint write ranges and read-only sharing;
// see the safety comments in `update_voltages` / `update_currents`.
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    /// Offset the pointer by `offset` elements.
    ///
    /// # Safety
    /// The resulting address must stay inside the allocation.
    unsafe fn add(&self, offset: usize) -> *mut T {
        self.0.add(offset)
    }
}

/// The basic FDTD engine.
/// Time-steps the equivalent-circuit coefficients over the field arrays.
pub struct Engine {
    pub op: Box<dyn OperatorBase>,
    pub volt_ptr: FieldArray,
    pub curr_ptr: FieldArray,
    pub num_lines: [usize; 3],
    pub num_ts: u32,
    pub extensions: Vec<Box<dyn EngineExtension>>,
    /// Number of worker threads used by the update kernels (1 = serial).
    pub num_threads: usize,
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
            num_threads: 1,
        }
    }

    /// Set the number of worker threads (clamped to at least one).
    pub fn set_number_of_threads(&mut self, threads: i32) {
        self.num_threads = threads.max(1) as usize;
    }

    fn worker_count(&self, max_workers: usize) -> usize {
        self.num_threads.max(1).min(max_workers.max(1))
    }

    /// Add a physics extension to the engine.
    pub fn add_extension(&mut self, ext: Box<dyn EngineExtension>) {
        self.extensions.push(ext);
    }

    /// Names of the registered extensions, in update order.
    pub fn extension_names(&self) -> Vec<&'static str> {
        self.extensions
            .iter()
            .map(|extension| extension.get_extension_name())
            .collect()
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
    ///
    /// The update is embarrassingly parallel over the `i` index: it reads
    /// only `curr` and writes only `volt`, so the field arrays are never in a
    /// read/write race. With more than one worker the `i` range is partitioned;
    /// each cell performs exactly the same arithmetic as the serial kernel,
    /// which keeps the results bit-identical.
    pub fn update_voltages(&mut self) {
        let vv = self.op.get_vv().expect("EC coefficients not initialized");
        let vi = self.op.get_vi().expect("EC coefficients not initialized");

        let nx = self.num_lines[0];
        let ny = self.num_lines[1];
        let nz = self.num_lines[2];

        let workers = self.worker_count(nx);
        if workers > 1 {
            let volt_ptr = SendPtr(self.volt_ptr.as_mut_ptr());
            let curr_ptr = SendPtr(self.curr_ptr.as_ptr() as *mut FDTD_FLOAT);
            let vv_ptr = SendPtr(vv.as_ptr() as *mut FDTD_FLOAT);
            let vi_ptr = SendPtr(vi.as_ptr() as *mut FDTD_FLOAT);
            let vs = self.volt_ptr.strides();
            let cs = self.curr_ptr.strides();
            let chunk = nx.div_ceil(workers);
            std::thread::scope(|scope| {
                for worker in 0..workers {
                    let start = (worker * chunk).min(nx);
                    let end = ((worker + 1) * chunk).min(nx);
                    if start >= end {
                        continue;
                    }
                    scope.spawn(move || {
                        // SAFETY: the workers own disjoint `i` ranges. `volt`
                        // is written only at the current cell, while `curr`,
                        // `vv` and `vi` are read-only for the whole phase. All
                        // indices are bounded by the loop limits, which mirror
                        // the serial kernel.
                        unsafe {
                            let cget = |n: usize, i: usize, j: usize, k: usize| {
                                *curr_ptr.add(n * cs[0] + i * cs[1] + j * cs[2] + k * cs[3])
                            };
                            for i in start..end {
                                let shift_i = if i > 0 { 1usize } else { 0usize };
                                for j in 0..ny {
                                    let shift_j = if j > 0 { 1usize } else { 0usize };
                                    for k in 0..nz {
                                        let shift_k = if k > 0 { 1usize } else { 0usize };
                                        let c2_ijk = cget(2, i, j, k);
                                        let c2_ijm1k = cget(2, i, j - shift_j, k);
                                        let c1_ijk = cget(1, i, j, k);
                                        let c1_ijkm1 = cget(1, i, j, k - shift_k);
                                        let c0_ijk = cget(0, i, j, k);
                                        let c0_ijkm1 = cget(0, i, j, k - shift_k);
                                        let c2_im1jk = cget(2, i - shift_i, j, k);
                                        let c1_im1jk = cget(1, i - shift_i, j, k);
                                        let c0_ijm1k = cget(0, i, j - shift_j, k);
                                        let base = i * vs[1] + j * vs[2] + k * vs[3];

                                        let ex = volt_ptr.add(base);
                                        let vv0 = *vv_ptr.add(base);
                                        let vi0 = *vi_ptr.add(base);
                                        let curl_h = c2_ijk - c2_ijm1k - c1_ijk + c1_ijkm1;
                                        *ex = *ex * vv0 + vi0 * curl_h;

                                        let ey = volt_ptr.add(vs[0] + base);
                                        let vv1 = *vv_ptr.add(vs[0] + base);
                                        let vi1 = *vi_ptr.add(vs[0] + base);
                                        let curl_h = c0_ijk - c0_ijkm1 - c2_ijk + c2_im1jk;
                                        *ey = *ey * vv1 + vi1 * curl_h;

                                        let ez = volt_ptr.add(2 * vs[0] + base);
                                        let vv2 = *vv_ptr.add(2 * vs[0] + base);
                                        let vi2 = *vi_ptr.add(2 * vs[0] + base);
                                        let curl_h = c1_ijk - c1_im1jk - c0_ijk + c0_ijm1k;
                                        *ez = *ez * vv2 + vi2 * curl_h;
                                    }
                                }
                            }
                        }
                    });
                }
            });
            return;
        }

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
    ///
    /// Like the voltage update, this is partitioned over `i` for the threaded
    /// path; it reads only `volt` and writes only `curr`.
    pub fn update_currents(&mut self) {
        let ii = self.op.get_ii().expect("EC coefficients not initialized");
        let iv = self.op.get_iv().expect("EC coefficients not initialized");

        let nx = self.num_lines[0];
        let ny = self.num_lines[1];
        let nz = self.num_lines[2];
        let cells_x = nx.saturating_sub(1);

        let workers = self.worker_count(cells_x);
        if workers > 1 {
            let curr_ptr = SendPtr(self.curr_ptr.as_mut_ptr());
            let volt_ptr = SendPtr(self.volt_ptr.as_ptr() as *mut FDTD_FLOAT);
            let ii_ptr = SendPtr(ii.as_ptr() as *mut FDTD_FLOAT);
            let iv_ptr = SendPtr(iv.as_ptr() as *mut FDTD_FLOAT);
            let cs = self.curr_ptr.strides();
            let vs = self.volt_ptr.strides();
            let chunk = cells_x.div_ceil(workers);
            std::thread::scope(|scope| {
                for worker in 0..workers {
                    let start = (worker * chunk).min(cells_x);
                    let end = ((worker + 1) * chunk).min(cells_x);
                    if start >= end {
                        continue;
                    }
                    scope.spawn(move || {
                        // SAFETY: the workers own disjoint `i` ranges. `curr`
                        // is written only at the current cell, while `volt`,
                        // `ii` and `iv` are read-only for the whole phase.
                        unsafe {
                            let vget = |n: usize, i: usize, j: usize, k: usize| {
                                *volt_ptr.add(n * vs[0] + i * vs[1] + j * vs[2] + k * vs[3])
                            };
                            for i in start..end {
                                for j in 0..ny.saturating_sub(1) {
                                    for k in 0..nz.saturating_sub(1) {
                                        let v2_ijk = vget(2, i, j, k);
                                        let v2_ijp1k = vget(2, i, j + 1, k);
                                        let v1_ijk = vget(1, i, j, k);
                                        let v1_ijkp1 = vget(1, i, j, k + 1);
                                        let v0_ijk = vget(0, i, j, k);
                                        let v0_ijkp1 = vget(0, i, j, k + 1);
                                        let v2_ip1jk = vget(2, i + 1, j, k);
                                        let v1_ip1jk = vget(1, i + 1, j, k);
                                        let v0_ijp1k = vget(0, i, j + 1, k);
                                        let base = i * cs[1] + j * cs[2] + k * cs[3];

                                        let hx = curr_ptr.add(base);
                                        let ii0 = *ii_ptr.add(base);
                                        let iv0 = *iv_ptr.add(base);
                                        let curl_e =
                                            v2_ijk - v2_ijp1k - v1_ijk + v1_ijkp1;
                                        *hx = *hx * ii0 + iv0 * curl_e;

                                        let hy = curr_ptr.add(cs[0] + base);
                                        let ii1 = *ii_ptr.add(cs[0] + base);
                                        let iv1 = *iv_ptr.add(cs[0] + base);
                                        let curl_e =
                                            v0_ijk - v0_ijkp1 - v2_ijk + v2_ip1jk;
                                        *hy = *hy * ii1 + iv1 * curl_e;

                                        let hz = curr_ptr.add(2 * cs[0] + base);
                                        let ii2 = *ii_ptr.add(2 * cs[0] + base);
                                        let iv2 = *iv_ptr.add(2 * cs[0] + base);
                                        let curl_e =
                                            v1_ijk - v1_ip1jk - v0_ijk + v0_ijp1k;
                                        *hz = *hz * ii2 + iv2 * curl_e;
                                    }
                                }
                            }
                        }
                    });
                }
            });
            return;
        }

        for i in 0..cells_x {
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
            // Native `GetNumberOfTimesteps()` is the step index during the
            // update phases; the counter is incremented only at the end.
            let number = self.num_ts;
            for extension in self.extensions.iter_mut() {
                extension.set_timestep(number);
            }

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
