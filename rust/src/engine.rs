//! Engine: time-stepping loop, voltage/current updates.
//! Mirrors `FDTD/engine.h/cpp` in the C++ codebase.

use crate::array_nijk::FieldArray;
use crate::constants::{EPS0, FDTD_FLOAT, MU0};
use crate::extensions::EngineExtension;
use crate::operator_base::OperatorBase;

/// Native `Engine_Interface_FDTD::CalcFastEnergy`: the rough field energy
/// `eps0*sum(E^2) + mue0*sum(H^2)` over the interior cells, accumulated in
/// the native order (per cell: E0,E1,E2 then H0,H1,H2).
///
/// `counts` are the *non-full* line counts (`GetNumberOfLines(n, false)`):
/// like native, a closed cylindrical azimuth contributes `num_lines - 2`.
pub fn calc_fast_energy(volt: &FieldArray, curr: &FieldArray, counts: [usize; 3]) -> f64 {
    let mut e_energy = 0.0f64;
    let mut h_energy = 0.0f64;
    for i in 0..counts[0].saturating_sub(1) {
        for j in 0..counts[1].saturating_sub(1) {
            for k in 0..counts[2].saturating_sub(1) {
                for component in 0..3 {
                    let value = *volt.get(component, i, j, k) as f64;
                    e_energy += value * value;
                }
                for component in 0..3 {
                    let value = *curr.get(component, i, j, k) as f64;
                    h_energy += value * value;
                }
            }
        }
    }
    EPS0 * e_energy + MU0 * h_energy
}

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

/// Interface data of a cylindrical multi-grid link
/// (native `Operator_CylinderMultiGrid` tables plus the split position).
#[derive(Clone, Debug)]
pub struct MultiGridLink {
    pub split_pos: usize,
    /// Physical base azimuthal lines (the closed-mesh padding plane is excluded).
    pub alpha_count: usize,
    pub interpolation: crate::multigrid::AlphaInterpolation,
}

/// The basic FDTD engine.
/// Time-steps the equivalent-circuit coefficients over the field arrays.
pub struct Engine {
    pub op: Box<dyn OperatorBase>,
    pub volt_ptr: FieldArray,
    pub curr_ptr: FieldArray,
    pub num_lines: [usize; 3],
    /// Line counts used by the native energy estimate (non-full counts).
    pub energy_counts: [usize; 3],
    pub num_ts: u32,
    pub extensions: Vec<Box<dyn EngineExtension>>,
    /// Number of worker threads used by the update kernels (1 = serial).
    pub num_threads: usize,
    /// Inner (child) engine of a cylindrical multi-grid.
    pub inner: Option<Box<Engine>>,
    /// Split/interpolation data shared by the base and inner engines.
    pub multi_grid: Option<MultiGridLink>,
    /// Persistent GPU field/coefficient buffers (ADR 0012).
    #[cfg(feature = "gpu")]
    pub gpu_fields: Option<crate::gpu::GpuFields>,
    /// The host copy of `volt` has changes the device has not seen yet.
    #[cfg(feature = "gpu")]
    host_volt_dirty: bool,
    /// The host copy of `curr` has changes the device has not seen yet.
    #[cfg(feature = "gpu")]
    host_curr_dirty: bool,
    /// A GPU batch advanced the device fields past the host copies.
    #[cfg(feature = "gpu")]
    gpu_pending: bool,
}

#[cfg(feature = "gpu")]
fn source_slot(
    slots: &mut Vec<(usize, usize)>,
    values: &mut Vec<Vec<f32>>,
    signal: &std::sync::Arc<crate::excitation::Excitation>,
    voltage: bool,
) -> usize {
    let key = std::sync::Arc::as_ptr(signal) as usize;
    if let Some((_, slot)) = slots.iter().find(|(entry, _)| *entry == key) {
        return *slot;
    }
    let length = signal.signal_length() as usize;
    let mut samples = Vec::with_capacity(length);
    for index in 0..length {
        samples.push(if voltage {
            signal.voltage(index)
        } else {
            signal.current(index)
        });
    }
    let slot = slots.len();
    slots.push((key, slot));
    values.push(samples);
    slot
}

impl Engine {
    /// Create a new engine for the given operator.
    pub fn new(op: Box<dyn OperatorBase>) -> Self {
        let num_lines = [
            op.get_number_of_lines(0, true) as usize,
            op.get_number_of_lines(1, true) as usize,
            op.get_number_of_lines(2, true) as usize,
        ];
        let energy_counts = [
            op.get_number_of_lines(0, false) as usize,
            op.get_number_of_lines(1, false) as usize,
            op.get_number_of_lines(2, false) as usize,
        ];
        Self {
            op,
            volt_ptr: FieldArray::new("volt", [num_lines[0], num_lines[1], num_lines[2]]),
            curr_ptr: FieldArray::new("curr", [num_lines[0], num_lines[1], num_lines[2]]),
            num_lines,
            energy_counts,
            num_ts: 0,
            extensions: Vec::new(),
            num_threads: 1,
            inner: None,
            multi_grid: None,
            #[cfg(feature = "gpu")]
            gpu_fields: None,
            #[cfg(feature = "gpu")]
            host_volt_dirty: false,
            #[cfg(feature = "gpu")]
            host_curr_dirty: false,
            #[cfg(feature = "gpu")]
            gpu_pending: false,
        }
    }

    /// Enable the GPU update path (ADR 0012).
    ///
    /// Extension-free models keep the fields resident on the GPU. Models with
    /// engine extensions use the native host-sync path: the E/H kernels still
    /// run on the GPU, and the fields are downloaded/uploaded around each
    /// extension phase so the extension sees the authoritative CPU state.
    #[cfg(feature = "gpu")]
    pub fn enable_gpu(&mut self) -> Result<(), String> {
        let context =
            crate::gpu::GpuContext::new().ok_or_else(|| "no GPU adapter available".to_string())?;
        let vv = self.op.get_vv().ok_or("vv coefficients not initialized")?;
        let vi = self.op.get_vi().ok_or("vi coefficients not initialized")?;
        let ii = self.op.get_ii().ok_or("ii coefficients not initialized")?;
        let iv = self.op.get_iv().ok_or("iv coefficients not initialized")?;
        // SAFETY: the slices only live for the duration of this call, which
        // copies the data into GPU buffers.
        let volt = unsafe {
            std::slice::from_raw_parts(self.volt_ptr.as_ptr(), self.volt_ptr.len())
        };
        let curr = unsafe {
            std::slice::from_raw_parts(self.curr_ptr.as_ptr(), self.curr_ptr.len())
        };
        // SAFETY: the coefficient arrays live for the call and are copied
        // into GPU buffers.
        let (vv, vi, ii, iv) = unsafe {
            (
                std::slice::from_raw_parts(vv.as_ptr(), vv.len()),
                std::slice::from_raw_parts(vi.as_ptr(), vi.len()),
                std::slice::from_raw_parts(ii.as_ptr(), ii.len()),
                std::slice::from_raw_parts(iv.as_ptr(), iv.len()),
            )
        };
        self.gpu_fields = crate::gpu::GpuFields::new(
            &context, volt, curr, vv, vi, ii, iv, self.num_lines,
        )
        .ok_or_else(|| "GPU buffer size mismatch".to_string())?
        .into();
        self.setup_gpu_extensions();
        Ok(())
    }

    /// Upload every extension that has a device implementation.
    ///
    /// Taken only when **all** registered extensions are device-native
    /// (excitation or UPML). A mixed model must apply the device extension
    /// phases between host extension phases, where the host array is
    /// authoritative, so it keeps the host path.
    #[cfg(feature = "gpu")]
    fn setup_gpu_extensions(&mut self) {
        if self.gpu_fields.is_none() || self.extensions.is_empty() {
            return;
        }
        let all_native = self.extensions.iter().all(|extension| {
            extension.excitation_terms().is_some() || extension.upml_terms().is_some()
        });
        if !all_native {
            if std::env::var_os("OPENEMS_DEBUG_GPU").is_some() {
                let names: Vec<&str> = self
                    .extensions
                    .iter()
                    .map(|extension| extension.get_extension_name())
                    .collect();
                eprintln!("[openEMS] GPU host path, extensions {names:?}");
            }
            return;
        }
        let dims = self.num_lines;
        let i_stride = dims[1] * dims[2];
        let j_stride = dims[2];
        let plane = dims[0] * dims[1] * dims[2];
        let mut terms = crate::gpu::SourceTerms::default();
        let mut volt_slots: Vec<(usize, usize)> = Vec::new();
        let mut curr_slots: Vec<(usize, usize)> = Vec::new();
        for extension in &self.extensions {
            let Some(source) = extension.excitation_terms() else {
                continue;
            };
            for excitation in &source.voltage_excitations {
                let slot = source_slot(&mut volt_slots, &mut terms.volt_values, &excitation.signal, true);
                terms.volt.push(crate::gpu::SourceTerm {
                    index: (excitation.component * plane
                        + excitation.pos[0] * i_stride
                        + excitation.pos[1] * j_stride
                        + excitation.pos[2]) as u32,
                    coeff: excitation.amp,
                    delay: excitation.delay,
                    period: excitation.signal.signal_period(),
                    slot: slot as u32,
                    length: excitation.signal.signal_length(),
                });
            }
            for excitation in &source.current_excitations {
                let slot = source_slot(&mut curr_slots, &mut terms.curr_values, &excitation.signal, false);
                terms.curr.push(crate::gpu::SourceTerm {
                    index: (excitation.component * plane
                        + excitation.pos[0] * i_stride
                        + excitation.pos[1] * j_stride
                        + excitation.pos[2]) as u32,
                    coeff: excitation.amp,
                    delay: excitation.delay,
                    period: excitation.signal.signal_period(),
                    slot: slot as u32,
                    length: excitation.signal.signal_length(),
                });
            }
        }
        let upmls: Vec<&crate::extensions::EngineExtUPML> = self
            .extensions
            .iter()
            .filter_map(|extension| extension.upml_terms())
            .collect();
        let Some(gpu) = self.gpu_fields.as_mut() else {
            return;
        };
        let mut prepared = false;
        if !terms.is_empty() {
            prepared |= gpu.setup_source(&terms);
        }
        if !upmls.is_empty() {
            prepared |= gpu.setup_upml(&upmls);
        }
        if !prepared {
            return;
        }
        if std::env::var_os("OPENEMS_DEBUG_GPU").is_some() {
            eprintln!(
                "[openEMS] GPU batched path: source {}, UPML slabs {}",
                if gpu.has_source() { "on device" } else { "none" },
                upmls.len()
            );
        }
        for extension in self.extensions.iter_mut() {
            extension.mark_gpu_handled();
        }
    }

    /// True when at least one extension must run on the host.
    #[cfg(feature = "gpu")]
    fn needs_host_extension(&self) -> bool {
        self.extensions
            .iter()
            .any(|extension| !extension.is_gpu_native())
    }

    /// Copy the device `volt` result back into the CPU array.
    #[cfg(feature = "gpu")]
    fn download_volt(&mut self) {
        let Some(gpu) = self.gpu_fields.as_ref() else {
            return;
        };
        let len = self.volt_ptr.len();
        // SAFETY: the GPU buffer has the same length; `download_volt` only
        // writes those elements and the pointer is not aliased by `gpu`.
        let volt = unsafe { std::slice::from_raw_parts_mut(self.volt_ptr.as_mut_ptr(), len) };
        gpu.download_volt(volt);
        self.host_volt_dirty = false;
    }

    /// Copy the device `curr` result back into the CPU array.
    #[cfg(feature = "gpu")]
    fn download_curr(&mut self) {
        let Some(gpu) = self.gpu_fields.as_ref() else {
            return;
        };
        let len = self.curr_ptr.len();
        // SAFETY: as in `download_volt`.
        let curr = unsafe { std::slice::from_raw_parts_mut(self.curr_ptr.as_mut_ptr(), len) };
        gpu.download_curr(curr);
        self.host_curr_dirty = false;
    }

    /// Fetch both device fields with one round trip when a batch advanced
    /// them; a no-op otherwise.
    #[cfg(feature = "gpu")]
    pub fn sync_fields(&mut self) {
        if !self.gpu_pending {
            return;
        }
        let Some(gpu) = self.gpu_fields.as_ref() else {
            self.gpu_pending = false;
            return;
        };
        let len = self.volt_ptr.len();
        // SAFETY: both GPU buffers have the same length; `download_pair` only
        // writes those elements and the pointers are not aliased by `gpu`.
        let (volt, curr) = unsafe {
            (
                std::slice::from_raw_parts_mut(self.volt_ptr.as_mut_ptr(), len),
                std::slice::from_raw_parts_mut(self.curr_ptr.as_mut_ptr(), len),
            )
        };
        gpu.download_pair(volt, curr);
        self.gpu_pending = false;
        self.host_volt_dirty = false;
        self.host_curr_dirty = false;
    }

    /// Upload the CPU voltage field into the GPU buffer when it changed on the
    /// host (host-extension path); a no-op otherwise.
    #[cfg(feature = "gpu")]
    fn upload_volt(&mut self) {
        if !self.host_volt_dirty {
            return;
        }
        let Some(gpu) = self.gpu_fields.as_ref() else {
            return;
        };
        // SAFETY: the slice only lives for the upload call, which copies it
        // into the GPU buffer.
        let volt =
            unsafe { std::slice::from_raw_parts(self.volt_ptr.as_ptr(), self.volt_ptr.len()) };
        gpu.upload_volt(volt);
        self.host_volt_dirty = false;
    }

    /// Upload the CPU current field into the GPU buffer when it changed on the
    /// host (host-extension path); a no-op otherwise.
    #[cfg(feature = "gpu")]
    fn upload_curr(&mut self) {
        if !self.host_curr_dirty {
            return;
        }
        let Some(gpu) = self.gpu_fields.as_ref() else {
            return;
        };
        // SAFETY: as in `upload_volt`.
        let curr =
            unsafe { std::slice::from_raw_parts(self.curr_ptr.as_ptr(), self.curr_ptr.len()) };
        gpu.upload_curr(curr);
        self.host_curr_dirty = false;
    }

    /// Set the number of worker threads (clamped to at least one).
    pub fn set_number_of_threads(&mut self, threads: i32) {
        self.num_threads = threads.max(1) as usize;
    }

    /// Cells per worker below which an update stays on the calling thread.
    ///
    /// The update kernels start and join the worker set per time step, so for a
    /// small grid that cost exceeds the arithmetic: the 9152-cell cavity case
    /// measured 12.4 s threaded against 2.6 s serial, while the 2.1M-cell Coax
    /// case profits from the threads (25 s against 54 s serial). 16384 cells
    /// per worker keeps both regimes on their faster path.
    const MIN_CELLS_PER_WORKER: usize = 16_384;

    fn worker_count(&self, max_workers: usize, cells: usize) -> usize {
        let workers = self.num_threads.max(1).min(max_workers.max(1));
        if cells / workers < Self::MIN_CELLS_PER_WORKER {
            1
        } else {
            workers
        }
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

    /// Rough field energy, mirroring `Engine_Interface_FDTD::CalcFastEnergy`.
    pub fn fast_energy(&self) -> f64 {
        calc_fast_energy(&self.volt_ptr, &self.curr_ptr, self.energy_counts)
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
        #[cfg(feature = "gpu")]
        if let Some(gpu) = self.gpu_fields.as_ref() {
            gpu.dispatch_e();
            return;
        }
        let vv = self.op.get_vv().expect("EC coefficients not initialized");
        let vi = self.op.get_vi().expect("EC coefficients not initialized");

        let nx = self.num_lines[0];
        let ny = self.num_lines[1];
        let nz = self.num_lines[2];

        let workers = self.worker_count(nx, nx * ny * nz);
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
        #[cfg(feature = "gpu")]
        if let Some(gpu) = self.gpu_fields.as_ref() {
            gpu.dispatch_h();
            return;
        }
        let ii = self.op.get_ii().expect("EC coefficients not initialized");
        let iv = self.op.get_iv().expect("EC coefficients not initialized");

        let nx = self.num_lines[0];
        let ny = self.num_lines[1];
        let nz = self.num_lines[2];
        let cells_x = nx.saturating_sub(1);

        let workers = self.worker_count(
            cells_x,
            cells_x * ny.saturating_sub(1) * nz.saturating_sub(1),
        );
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
                                        let curl_e = v2_ijk - v2_ijp1k - v1_ijk + v1_ijkp1;
                                        *hx = *hx * ii0 + iv0 * curl_e;

                                        let hy = curr_ptr.add(cs[0] + base);
                                        let ii1 = *ii_ptr.add(cs[0] + base);
                                        let iv1 = *iv_ptr.add(cs[0] + base);
                                        let curl_e = v0_ijk - v0_ijkp1 - v2_ijk + v2_ip1jk;
                                        *hy = *hy * ii1 + iv1 * curl_e;

                                        let hz = curr_ptr.add(2 * cs[0] + base);
                                        let ii2 = *ii_ptr.add(2 * cs[0] + base);
                                        let iv2 = *iv_ptr.add(2 * cs[0] + base);
                                        let curl_e = v1_ijk - v1_ip1jk - v0_ijk + v0_ijp1k;
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

    /// Attach the inner (child) engine and the interface tables.
    ///
    /// The base engine keeps the full mesh, so probes and dumps can sample it
    /// unchanged; the overlap rows are filled from the child every step.
    pub fn enable_multi_grid(&mut self, mut inner: Engine, link: MultiGridLink) {
        inner.set_number_of_threads(self.num_threads as i32);
        self.inner = Some(Box::new(inner));
        self.multi_grid = Some(link);
    }

    /// True for a base engine that drives an inner multi-grid engine.
    pub fn is_multi_grid(&self) -> bool {
        self.multi_grid.is_some()
    }

    /// Iterate a number of timesteps.
    pub fn iterate_ts(&mut self, iter_ts: u32) -> bool {
        if self.multi_grid.is_some() {
            return self.iterate_multi_grid(iter_ts);
        }
        #[cfg(feature = "gpu")]
        if self.gpu_fields.is_some() && self.needs_host_extension() {
            return self.iterate_ts_gpu_host(iter_ts);
        }
        #[cfg(feature = "gpu")]
        if self.gpu_fields.is_some() {
            // Extension-free or all-device batch: record every E/H step and
            // the excitation source into one submission and keep the fields
            // resident. The batch ends with one download, so the CPU arrays
            // are current for the caller as before.
            if let Some(gpu) = self.gpu_fields.as_ref() {
                gpu.dispatch_steps(iter_ts, self.num_ts);
            }
            self.gpu_pending = true;
            self.num_ts = self.num_ts.saturating_add(iter_ts);
            self.sync_fields();
            return true;
        }
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
        // The CPU arrays are only needed by observables/extensions; sync once
        // per batch instead of every step (extensions force the CPU path).
        #[cfg(feature = "gpu")]
        self.sync_fields();
        true
    }

    /// GPU update path with host engine extensions.
    ///
    /// Native `Engine_GPU` runs non-GPU-safe extensions on the host copy and
    /// synchronizes around the phase. The E/H kernels stay on the device; only
    /// the extension phases force a transfer.
    #[cfg(feature = "gpu")]
    fn iterate_ts_gpu_host(&mut self, iter_ts: u32) -> bool {
        for _ in 0..iter_ts {
            let number = self.num_ts;
            for extension in self.extensions.iter_mut() {
                extension.set_timestep(number);
            }

            // The CPU arrays hold the completed previous step, including the
            // edits the extensions made after the last download.
            self.do_pre_voltage_updates();
            self.host_volt_dirty = true;
            // The E kernel reads `curr` as its peer and updates `volt`.
            self.upload_curr();
            self.upload_volt();
            self.update_voltages();
            self.download_volt();
            self.do_post_voltage_updates();
            self.apply2_voltages();
            self.host_volt_dirty = true;

            self.do_pre_current_updates();
            self.host_curr_dirty = true;
            // The H kernel reads `volt` as its peer and updates `curr`.
            self.upload_volt();
            self.upload_curr();
            self.update_currents();
            self.download_curr();
            self.do_post_current_updates();
            self.apply2_current();
            self.host_curr_dirty = true;

            self.num_ts += 1;
        }
        true
    }

    /// Native `Engine_CylinderMultiGrid` + `Engine_Ext_CylinderMultiGrid`
    /// barrier order, run sequentially (the phases never read the other
    /// engine's fields, so the ordering is equivalent).
    fn iterate_multi_grid(&mut self, iter_ts: u32) -> bool {
        let Some(link) = self.multi_grid.clone() else {
            return false;
        };
        if self.inner.is_none() {
            return false;
        }
        let mut ok = true;
        for _ in 0..iter_ts {
            let number = self.num_ts;
            for extension in self.extensions.iter_mut() {
                extension.set_timestep(number);
            }
            if let Some(inner) = self.inner.as_mut() {
                for extension in inner.extensions.iter_mut() {
                    extension.set_timestep(number);
                }
            }

            // Voltage half-step: both engines, then the interface sync.
            self.do_pre_voltage_updates();
            self.update_voltages();
            self.do_post_voltage_updates();
            self.apply2_voltages();
            if let Some(inner) = self.inner.as_mut() {
                inner.do_pre_voltage_updates();
                inner.update_voltages();
                inner.do_post_voltage_updates();
                inner.apply2_voltages();
            }
            self.multi_grid_sync_voltages(&link);

            // Current half-step.
            self.do_pre_current_updates();
            self.update_currents();
            self.do_post_current_updates();
            self.apply2_current();
            if let Some(inner) = self.inner.as_mut() {
                inner.do_pre_current_updates();
                inner.update_currents();
                inner.do_post_current_updates();
                inner.apply2_current();
            }
            if self.multi_grid_sync_currents(&link).is_err() {
                ok = false;
            }

            // End of step: fill the base overlap from the child.
            for rho in 0..link.split_pos.saturating_sub(1) {
                if self.multi_grid_interpolate_row(&link, rho, true).is_err() {
                    ok = false;
                }
            }
            for rho in 0..link.split_pos.saturating_sub(2) {
                if self.multi_grid_interpolate_row(&link, rho, false).is_err() {
                    ok = false;
                }
            }

            self.num_ts += 1;
            if let Some(inner) = self.inner.as_mut() {
                inner.num_ts += 1;
            }
        }
        ok
    }

    /// Native `Engine_Ext_CylinderMultiGrid::SyncVoltages`.
    pub fn multi_grid_sync_voltages(&mut self, link: &MultiGridLink) {
        if link.split_pos == 0 {
            return;
        }
        let rho = link.split_pos - 1;
        let Some(inner) = self.inner.as_mut() else {
            return;
        };
        let nz = self.num_lines[2].min(inner.num_lines[2]);
        let inner_ny = inner.num_lines[1];
        for a in (0..link.alpha_count.saturating_sub(1)).step_by(2) {
            let h = a / 2;
            if h >= inner_ny {
                break;
            }
            let a_next = (a + 1).min(link.alpha_count - 1);
            for z in 0..nz {
                // Native zeroes the normal (rho) component and copies the
                // tangential components from the base.
                *inner.volt_ptr.get_mut(0, rho, h, z) = 0.0;
                *inner.volt_ptr.get_mut(2, rho, h, z) =
                    *self.volt_ptr.get(2, rho, a, z);
                *inner.volt_ptr.get_mut(1, rho, h, z) =
                    *self.volt_ptr.get(1, rho, a, z)
                        + *self.volt_ptr.get(1, rho, a_next, z);
            }
        }
    }

    /// Native `Engine_Ext_CylinderMultiGrid::SyncCurrents`.
    pub fn multi_grid_sync_currents(&mut self, link: &MultiGridLink) -> Result<(), String> {
        if link.split_pos < 2 {
            return Ok(());
        }
        self.multi_grid_interpolate_row(link, link.split_pos - 2, false)
    }

    /// Interpolate one child rho row into the base overlap.
    pub fn multi_grid_interpolate_row(
        &mut self,
        link: &MultiGridLink,
        rho: usize,
        voltage: bool,
    ) -> Result<(), String> {
        let Some(inner) = self.inner.as_mut() else {
            return Err("multi-grid base has no inner engine".to_string());
        };
        let nz = self.num_lines[2].min(inner.num_lines[2]);
        for z in 0..nz {
            if voltage {
                crate::multigrid::interpolate_child_to_base(
                    &mut self.volt_ptr,
                    &inner.volt_ptr,
                    rho,
                    z,
                    link.alpha_count,
                    &link.interpolation,
                    true,
                )?;
            } else {
                crate::multigrid::interpolate_child_to_base(
                    &mut self.curr_ptr,
                    &inner.curr_ptr,
                    rho,
                    z,
                    link.alpha_count,
                    &link.interpolation,
                    false,
                )?;
            }
        }
        Ok(())
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

    #[test]
    fn fast_energy_matches_the_native_formula() {
        let counts = [4usize, 3, 5];
        let mut volt = FieldArray::new("volt", counts);
        let mut curr = FieldArray::new("curr", counts);
        for i in 0..counts[0] {
            for j in 0..counts[1] {
                for k in 0..counts[2] {
                    for component in 0..3 {
                        *volt.get_mut(component, i, j, k) = (2 + component) as FDTD_FLOAT;
                        *curr.get_mut(component, i, j, k) = (5 + component) as FDTD_FLOAT;
                    }
                }
            }
        }
        let cells = (counts[0] - 1) * (counts[1] - 1) * (counts[2] - 1);
        let e_sum = (4 + 9 + 16) as f64 * cells as f64;
        let h_sum = (25 + 36 + 49) as f64 * cells as f64;
        let expected = EPS0 * e_sum + MU0 * h_sum;
        let energy = calc_fast_energy(&volt, &curr, counts);
        assert!(
            (energy - expected).abs() / expected < 1e-12,
            "energy {energy} vs native {expected}"
        );
    }

    #[cfg(feature = "gpu")]
    fn cavity_operator(n: usize) -> crate::operator::Operator {
        let lines = (0..n)
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let xml = format!(
            r#"<ContinuousStructure CoordSystem="0">
                <RectilinearGrid DeltaUnit="1">
                    <XLines>{lines}</XLines>
                    <YLines>{lines}</YLines>
                    <ZLines>{lines}</ZLines>
                </RectilinearGrid>
                <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
                <Properties/>
            </ContinuousStructure>"#
        );
        let mut op = crate::operator::Operator::new();
        op.load_csx(&xml).expect("cavity CSX");
        op
    }

    #[cfg(feature = "gpu")]
    fn seed_fields(engine: &mut Engine) {
        let [nx, ny, nz] = engine.num_lines;
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let phase = 0.11 * (i as f64 + 2.0 * j as f64 + 0.5 * k as f64);
                    for component in 0..3 {
                        *engine.volt_ptr.get_mut(component, i, j, k) =
                            (phase + component as f64).sin() as FDTD_FLOAT;
                        *engine.curr_ptr.get_mut(component, i, j, k) =
                            (phase * 0.7 + component as f64).cos() as FDTD_FLOAT;
                    }
                }
            }
        }
    }

    #[cfg(feature = "gpu")]
    fn field_diff(a: &Engine, b: &Engine) -> f32 {
        let left = unsafe { std::slice::from_raw_parts(a.volt_ptr.as_ptr(), a.volt_ptr.len()) };
        let right = unsafe { std::slice::from_raw_parts(b.volt_ptr.as_ptr(), b.volt_ptr.len()) };
        left.iter()
            .zip(right)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_device_source_matches_the_cpu_engine() {
        let n = 24;
        let steps = 120;
        // The source waveform only needs a fixed sample interval; both
        // engines are fed the same samples, so the comparison is exact.
        let mut signal = crate::excitation::Excitation::gauss(2.0e9, 1.0e9);
        signal.build(1.0e-12, 4096).expect("signal must build");
        let build = || {
            let mut engine = Engine::new(Box::new(cavity_operator(n)));
            let mut excitation = crate::extensions::EngineExtExcitation::new();
            excitation.add_voltage_excitation(
                0,
                [3, 3, 3],
                0,
                1.5,
                std::sync::Arc::new(signal.clone()),
            );
            excitation.add_current_excitation(
                1,
                [4, 4, 4],
                0,
                0.75,
                std::sync::Arc::new(signal.clone()),
            );
            engine.add_extension(Box::new(excitation));
            seed_fields(&mut engine);
            engine
        };

        let mut cpu = build();
        assert!(cpu.iterate_ts(steps));

        let mut gpu = build();
        if let Err(reason) = gpu.enable_gpu() {
            eprintln!("skipping the GPU device-source test: {reason}");
            return;
        }
        assert!(
            gpu.gpu_fields
                .as_ref()
                .map(|fields| fields.has_source())
                .unwrap_or(false),
            "the excitation must be uploaded to the device"
        );
        assert!(
            !gpu.needs_host_extension(),
            "a source-only model must use the batched device path"
        );

        let start = std::time::Instant::now();
        assert!(gpu.iterate_ts(steps));
        let gpu_time = start.elapsed();
        let diff = field_diff(&cpu, &gpu);
        assert!(diff < 1.0e-5, "CPU/GPU device-source field diff {diff}");

        // One batch and one step at a time must select the same source
        // samples, i.e. the per-step uniform slots are indexed correctly.
        let mut stepped = build();
        if stepped.enable_gpu().is_err() {
            return;
        }
        for _ in 0..steps {
            assert!(stepped.iterate_ts(1));
        }
        let batch_diff = field_diff(&gpu, &stepped);
        assert!(
            batch_diff < 1.0e-6,
            "batch vs single-step scheduling diff {batch_diff}"
        );
        eprintln!(
            "GPU device source: {n}^3, {steps} steps, CPU diff {diff:.2e}, batch/step diff {batch_diff:.2e}, GPU {gpu_time:.2?}"
        );
    }

    #[cfg(feature = "gpu")]
    fn upml_extension(
        start: [usize; 3],
        slab: [usize; 3],
    ) -> crate::extensions::EngineExtUPML {
        let mut extension = crate::extensions::EngineExtUPML::new(start, slab);
        for comp in 0..3 {
            for i in 0..slab[0] {
                for j in 0..slab[1] {
                    for k in 0..slab[2] {
                        *extension.vv[comp].get_mut(0, i, j, k) = 1.0 + 0.01 * i as f32;
                        *extension.vvfo[comp].get_mut(0, i, j, k) = 0.25;
                        *extension.vvfn[comp].get_mut(0, i, j, k) = 0.5 + 0.02 * j as f32;
                        *extension.ii[comp].get_mut(0, i, j, k) = 1.0 + 0.01 * k as f32;
                        *extension.iifo[comp].get_mut(0, i, j, k) = 0.25;
                        *extension.iifn[comp].get_mut(0, i, j, k) = 0.5 + 0.02 * i as f32;
                    }
                }
            }
        }
        extension
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_device_upml_matches_the_cpu_engine() {
        let n = 12;
        let steps = 120;
        let slab = [4, 4, 4];
        let start = [2, 2, 2];
        let build = || {
            let mut engine = Engine::new(Box::new(cavity_operator(n)));
            engine.add_extension(Box::new(upml_extension(start, slab)));
            seed_fields(&mut engine);
            engine
        };

        let mut cpu = build();
        assert!(cpu.iterate_ts(steps));

        let mut gpu = build();
        if let Err(reason) = gpu.enable_gpu() {
            eprintln!("skipping the GPU UPML test: {reason}");
            return;
        }
        assert!(
            gpu.gpu_fields
                .as_ref()
                .map(|fields| fields.has_upml())
                .unwrap_or(false),
            "the PML slab must be uploaded to the device"
        );
        assert!(
            !gpu.needs_host_extension(),
            "a PML-only model must use the batched device path"
        );

        let start_time = std::time::Instant::now();
        assert!(gpu.iterate_ts(steps));
        let gpu_time = start_time.elapsed();
        let diff = field_diff(&cpu, &gpu);
        assert!(diff < 1.0e-5, "CPU/GPU device-UPML field diff {diff}");

        let mut stepped = build();
        if stepped.enable_gpu().is_err() {
            return;
        }
        for _ in 0..steps {
            assert!(stepped.iterate_ts(1));
        }
        let batch_diff = field_diff(&gpu, &stepped);
        assert!(
            batch_diff < 1.0e-6,
            "batch vs single-step UPML scheduling diff {batch_diff}"
        );
        eprintln!(
            "GPU device UPML: {n}^3, {steps} steps, CPU diff {diff:.2e}, batch/step diff {batch_diff:.2e}, GPU {gpu_time:.2?}"
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_engine_matches_cpu_on_a_cavity() {
        let n = 48;
        let steps = 120;
        let mut cpu = Engine::new(Box::new(cavity_operator(n)));
        cpu.set_number_of_threads(1);
        seed_fields(&mut cpu);

        let mut gpu = Engine::new(Box::new(cavity_operator(n)));
        seed_fields(&mut gpu);
        if let Err(reason) = gpu.enable_gpu() {
            eprintln!("skipping the GPU engine test: {reason}");
            return;
        }

        let start = std::time::Instant::now();
        assert!(cpu.iterate_ts(steps));
        let cpu_time = start.elapsed();

        let start = std::time::Instant::now();
        assert!(gpu.iterate_ts(steps));
        let gpu_time = start.elapsed();

        let diff = field_diff(&cpu, &gpu);
        assert!(diff < 1.0e-5, "CPU/GPU field diff {diff}");
        eprintln!(
            "GPU engine vs CPU: {n}^3, {steps} steps, E diff {diff:.2e}, CPU {:.2?}, GPU {:.2?}",
            cpu_time, gpu_time
        );
    }

    #[cfg(feature = "gpu")]
    struct BumpExtension;

    #[cfg(feature = "gpu")]
    impl EngineExtension for BumpExtension {
        fn get_priority(&self) -> i32 {
            0
        }
        fn get_extension_name(&self) -> &'static str {
            "bump"
        }
        fn do_post_voltage_updates(
            &mut self,
            volt: &mut FieldArray,
            _curr: &FieldArray,
        ) {
            *volt.get_mut(0, 1, 1, 1) += 0.5;
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_host_extension_path_matches_the_cpu_engine() {
        let n = 48;
        let steps = 120;
        let mut cpu = Engine::new(Box::new(cavity_operator(n)));
        cpu.add_extension(Box::new(BumpExtension));
        seed_fields(&mut cpu);

        let mut gpu = Engine::new(Box::new(cavity_operator(n)));
        gpu.add_extension(Box::new(BumpExtension));
        seed_fields(&mut gpu);
        if let Err(reason) = gpu.enable_gpu() {
            eprintln!("skipping the GPU host-extension test: {reason}");
            return;
        }

        let start = std::time::Instant::now();
        assert!(cpu.iterate_ts(steps));
        let cpu_time = start.elapsed();

        let start = std::time::Instant::now();
        assert!(gpu.iterate_ts(steps));
        let gpu_time = start.elapsed();

        let diff = field_diff(&cpu, &gpu);
        assert!(diff < 1.0e-5, "CPU/GPU host-extension field diff {diff}");
        eprintln!(
            "GPU host extension vs CPU: {n}^3, {steps} steps, E diff {diff:.2e}, CPU {:.2?}, GPU {:.2?}",
            cpu_time, gpu_time
        );
    }
}
