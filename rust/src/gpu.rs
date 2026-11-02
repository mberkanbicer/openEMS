//! Optional `wgpu` compute backend (ADR 0012).
//!
//! Stage 1: device acquisition and buffer round-trip.
//! Stage 2: the Yee E/H update kernels, validated against the CPU update in
//! the unit test. The engine still runs the CPU path; wiring these buffers into
//! `Engine` with the extension hooks is the next stage.

use crate::constants::FDTD_FLOAT;

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_name: String,
    pub backend: String,
}

const SCALE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> factor: vec4<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < arrayLength(&input)) {
        output[i] = input[i] * factor.x;
    }
}
"#;

const UPDATE_SHADER: &str = r#"
struct Dims {
    nx: u32,
    ny: u32,
    nz: u32,
    pad: u32,
};
@group(0) @binding(0) var<storage, read_write> field: array<f32>;
@group(0) @binding(1) var<storage, read> peer: array<f32>;
@group(0) @binding(2) var<storage, read> vv: array<f32>;
@group(0) @binding(3) var<storage, read> vi: array<f32>;
@group(0) @binding(4) var<storage, read> ii: array<f32>;
@group(0) @binding(5) var<storage, read> iv: array<f32>;
@group(0) @binding(6) var<uniform> dims: Dims;

fn idx(n: u32, i: u32, j: u32, k: u32) -> u32 {
    return n * dims.nx * dims.ny * dims.nz + i * dims.ny * dims.nz + j * dims.nz + k;
}

@compute @workgroup_size(64)
fn update_e(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = dims.nx * dims.ny * dims.nz;
    let t = gid.x;
    if (t >= total) { return; }
    let i = t / (dims.ny * dims.nz);
    let rem = t % (dims.ny * dims.nz);
    let j = rem / dims.nz;
    let k = rem % dims.nz;
    let si = select(0u, 1u, i > 0u);
    let sj = select(0u, 1u, j > 0u);
    let sk = select(0u, 1u, k > 0u);

    let c2_ijk = peer[idx(2u, i, j, k)];
    let c2_ijm1k = peer[idx(2u, i, j - sj, k)];
    let c1_ijk = peer[idx(1u, i, j, k)];
    let c1_ijkm1 = peer[idx(1u, i, j, k - sk)];
    let c0_ijk = peer[idx(0u, i, j, k)];
    let c0_ijkm1 = peer[idx(0u, i, j, k - sk)];
    let c2_im1jk = peer[idx(2u, i - si, j, k)];
    let c1_im1jk = peer[idx(1u, i - si, j, k)];
    let c0_ijm1k = peer[idx(0u, i, j - sj, k)];

    let b = idx(0u, i, j, k);
    let comp = dims.nx * dims.ny * dims.nz;
    let ex = b;
    let ey = b + comp;
    let ez = ey + comp;
    field[ex] = field[ex] * vv[ex] + vi[ex] * (c2_ijk - c2_ijm1k - c1_ijk + c1_ijkm1);
    field[ey] = field[ey] * vv[ey] + vi[ey] * (c0_ijk - c0_ijkm1 - c2_ijk + c2_im1jk);
    field[ez] = field[ez] * vv[ez] + vi[ez] * (c1_ijk - c1_im1jk - c0_ijk + c0_ijm1k);
}

@compute @workgroup_size(64)
fn update_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cxs = dims.nx - 1u;
    let cys = dims.ny - 1u;
    let czs = dims.nz - 1u;
    let total = cxs * cys * czs;
    let t = gid.x;
    if (t >= total) { return; }
    let i = t / (cys * czs);
    let rem = t % (cys * czs);
    let j = rem / czs;
    let k = rem % czs;

    let v2_ijk = peer[idx(2u, i, j, k)];
    let v2_ijp1k = peer[idx(2u, i, j + 1u, k)];
    let v1_ijk = peer[idx(1u, i, j, k)];
    let v1_ijkp1 = peer[idx(1u, i, j, k + 1u)];
    let v0_ijk = peer[idx(0u, i, j, k)];
    let v0_ijkp1 = peer[idx(0u, i, j, k + 1u)];
    let v2_ip1jk = peer[idx(2u, i + 1u, j, k)];
    let v1_ip1jk = peer[idx(1u, i + 1u, j, k)];
    let v0_ijp1k = peer[idx(0u, i, j + 1u, k)];

    let b = idx(0u, i, j, k);
    let comp = dims.nx * dims.ny * dims.nz;
    let hx = b;
    let hy = b + comp;
    let hz = hy + comp;
    field[hx] = field[hx] * ii[hx] + iv[hx] * (v2_ijk - v2_ijp1k - v1_ijk + v1_ijkp1);
    field[hy] = field[hy] * ii[hy] + iv[hy] * (v0_ijk - v0_ijkp1 - v2_ijk + v2_ip1jk);
    field[hz] = field[hz] * ii[hz] + iv[hz] * (v1_ijk - v1_ip1jk - v0_ijk + v0_ijp1k);
}

// Device-side excitation source: `target[index] += coeff * value(ts)`.
//
// Mirrors `EngineExtExcitation::apply2_voltages` / `apply2_current`: the
// sample index is `(ts - delay)` modulo the signal period when the signal is
// periodic, and the term is skipped once the index runs past the built
// signal. The waveform samples live in one concatenated buffer, one slot per
// distinct signal, so a batch only needs the per-step `ts` in a uniform.

struct SourceParams {
    ts: u32,
    stride: u32,
    count: u32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read> src_index: array<u32>;
@group(0) @binding(1) var<storage, read> src_coeff: array<f32>;
@group(0) @binding(2) var<storage, read> src_delay: array<u32>;
@group(0) @binding(3) var<storage, read> src_period: array<u32>;
@group(0) @binding(4) var<storage, read> src_slot: array<u32>;
@group(0) @binding(5) var<storage, read> src_length: array<u32>;
@group(0) @binding(6) var<storage, read> src_values: array<f32>;
@group(0) @binding(7) var<storage, read_write> src_target: array<f32>;
@group(0) @binding(8) var<uniform> src_params: SourceParams;

@compute @workgroup_size(64)
fn apply_source(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e = gid.x;
    if (e >= src_params.count) { return; }
    let ts = src_params.ts;
    let delay = src_delay[e];
    if (ts < delay) { return; }
    var pos = ts - delay;
    let period = src_period[e];
    if (period > 0u) { pos = pos % period; }
    if (pos >= src_length[e]) { return; }
    let value = src_values[src_slot[e] * src_params.stride + pos];
    let index = src_index[e];
    src_target[index] = src_target[index] + src_coeff[e] * value;
}

// Device-side UPML: the four PML phases over one slab, mirroring
// `EngineExtUPML`'s host loops. The slab arrays are concatenated over the
// three components, so the invocation index is also the slab-array index and
// the global field index is decoded from the uniform.

struct UpmlDims {
    global: vec4<u32>,
    /// `x` = slab count, `y` = total element count over all slabs.
    counts: vec4<u32>,
};

@group(0) @binding(0) var<storage, read_write> up_volt: array<f32>;
@group(0) @binding(1) var<storage, read_write> up_curr: array<f32>;
@group(0) @binding(2) var<storage, read> up_vv: array<f32>;
@group(0) @binding(3) var<storage, read> up_vvfo: array<f32>;
@group(0) @binding(4) var<storage, read> up_vvfn: array<f32>;
@group(0) @binding(5) var<storage, read> up_ii: array<f32>;
@group(0) @binding(6) var<storage, read> up_iifo: array<f32>;
@group(0) @binding(7) var<storage, read> up_iifn: array<f32>;
@group(0) @binding(8) var<storage, read_write> up_volt_flux: array<f32>;
@group(0) @binding(9) var<storage, read_write> up_curr_flux: array<f32>;
// Slab table: even entries are `start.xyz`, odd entries are `extent.xyz`.
@group(0) @binding(10) var<storage, read> up_slabs: array<vec4<u32>>;
@group(0) @binding(11) var<uniform> up_dims: UpmlDims;

// Locate the slab containing concatenated index `t`; returns
// `(local index, slab index, 0, 0)`. The table is a prefix scan, which is
// cheap because a model has at most a handful of PML slabs.
fn upml_locate(t: u32) -> vec4<u32> {
    var base = 0u;
    for (var s = 0u; s < up_dims.counts.x; s = s + 1u) {
        let ext = up_slabs[s * 2u + 1u].xyz;
        let n = 3u * ext.x * ext.y * ext.z;
        if (t < base + n) {
            return vec4<u32>(t - base, s, 0u, 0u);
        }
        base = base + n;
    }
    return vec4<u32>(0u, 0u, 0u, 0u);
}

fn upml_global_index(t: u32) -> u32 {
    let loc = upml_locate(t);
    let s = loc.y;
    let local = loc.x;
    let ext = up_slabs[s * 2u + 1u].xyz;
    let start = up_slabs[s * 2u].xyz;
    let slab_n = ext.x * ext.y * ext.z;
    let comp = local / slab_n;
    let rem = local % slab_n;
    let nj = ext.y;
    let nk = ext.z;
    let i = rem / (nj * nk);
    let rem2 = rem % (nj * nk);
    let j = rem2 / nk;
    let k = rem2 % nk;
    let plane = up_dims.global.x * up_dims.global.y * up_dims.global.z;
    return comp * plane
        + (i + start.x) * up_dims.global.y * up_dims.global.z
        + (j + start.y) * up_dims.global.z
        + (k + start.z);
}

@compute @workgroup_size(64)
fn upml_pre_volt(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= up_dims.counts.y) { return; }
    let g = upml_global_index(t);
    let help = up_vv[t] * up_volt[g] - up_vvfo[t] * up_volt_flux[t];
    let flux_old = up_volt_flux[t];
    up_volt[g] = flux_old;
    up_volt_flux[t] = help;
}

@compute @workgroup_size(64)
fn upml_post_volt(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= up_dims.counts.y) { return; }
    let g = upml_global_index(t);
    let help = up_volt_flux[t];
    up_volt_flux[t] = up_volt[g];
    up_volt[g] = help + up_vvfn[t] * up_volt_flux[t];
}

@compute @workgroup_size(64)
fn upml_pre_curr(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= up_dims.counts.y) { return; }
    let g = upml_global_index(t);
    let help = up_ii[t] * up_curr[g] - up_iifo[t] * up_curr_flux[t];
    let flux_old = up_curr_flux[t];
    up_curr[g] = flux_old;
    up_curr_flux[t] = help;
}

@compute @workgroup_size(64)
fn upml_post_curr(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= up_dims.counts.y) { return; }
    let g = upml_global_index(t);
    let help = up_curr_flux[t];
    up_curr_flux[t] = up_curr[g];
    up_curr[g] = help + up_iifn[t] * up_curr_flux[t];
}

// Fused PML: the post phases run inside the E/H kernels, which removes two
// dispatches per step. The pre phases stay separate because they read and
// write the field *between* the two updates (a global barrier is required, and
// WGSL has none).
//
// The post phase only touches its own cell, so running it at the end of the
// E/H thread that owns the cell is exact. `update_h_pml` iterates the full
// grid and guards the H update, because PML slabs can include the last line,
// which the H update itself does not visit but the post phase must.

struct PmlDims {
    global: vec4<u32>,
    counts: vec4<u32>,
};

@group(0) @binding(12) var<storage, read> pml_coeff: array<f32>;
@group(0) @binding(13) var<storage, read_write> pml_flux: array<f32>;
@group(0) @binding(14) var<storage, read> pml_slabs: array<vec4<u32>>;
@group(0) @binding(15) var<uniform> pml_dims: PmlDims;
// Pre-phase coefficients (`vv`/`vvfo` for E, `ii`/`iifo` for H).
@group(0) @binding(16) var<storage, read> pml_pre_a: array<f32>;
@group(0) @binding(17) var<storage, read> pml_pre_b: array<f32>;

// Returns (slab index, absolute element index, 0, 0), or `0xffffffff` in `x`
// when the cell is outside every slab. The absolute index includes the
// preceding slabs' element ranges, because all slabs share one array.
fn pml_hit(i: u32, j: u32, k: u32) -> vec4<u32> {
    var base = 0u;
    for (var s = 0u; s < pml_dims.counts.x; s = s + 1u) {
        let start = pml_slabs[s * 2u].xyz;
        let ext = pml_slabs[s * 2u + 1u].xyz;
        let n = 3u * ext.x * ext.y * ext.z;
        if (i >= start.x && i < start.x + ext.x
            && j >= start.y && j < start.y + ext.y
            && k >= start.z && k < start.z + ext.z) {
            let li = i - start.x;
            let lj = j - start.y;
            let lk = k - start.z;
            return vec4<u32>(s, base + li * ext.y * ext.z + lj * ext.z + lk, 0u, 0u);
        }
        base = base + n;
    }
    return vec4<u32>(0xffffffffu, 0u, 0u, 0u);
}

fn pml_pre_phase(i: u32, j: u32, k: u32) {
    let hit = pml_hit(i, j, k);
    if (hit.x == 0xffffffffu) { return; }
    let ext = pml_slabs[hit.x * 2u + 1u].xyz;
    let slab_n = ext.x * ext.y * ext.z;
    let plane = pml_dims.global.x * pml_dims.global.y * pml_dims.global.z;
    let base = idx(0u, i, j, k);
    for (var c = 0u; c < 3u; c = c + 1u) {
        let t = hit.y + c * slab_n;
        let g = base + c * plane;
        let help = pml_pre_a[t] * field[g] - pml_pre_b[t] * pml_flux[t];
        let flux_old = pml_flux[t];
        field[g] = flux_old;
        pml_flux[t] = help;
    }
}

fn pml_post_phase(i: u32, j: u32, k: u32) {
    let hit = pml_hit(i, j, k);
    if (hit.x == 0xffffffffu) { return; }
    let ext = pml_slabs[hit.x * 2u + 1u].xyz;
    let slab_n = ext.x * ext.y * ext.z;
    let plane = pml_dims.global.x * pml_dims.global.y * pml_dims.global.z;
    let base = idx(0u, i, j, k);
    for (var c = 0u; c < 3u; c = c + 1u) {
        let t = hit.y + c * slab_n;
        let g = base + c * plane;
        let help = pml_flux[t];
        pml_flux[t] = field[g];
        field[g] = help + pml_coeff[t] * pml_flux[t];
    }
}

@compute @workgroup_size(64)
fn update_e_pml(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = dims.nx * dims.ny * dims.nz;
    let t = gid.x;
    if (t >= total) { return; }
    let i = t / (dims.ny * dims.nz);
    let rem = t % (dims.ny * dims.nz);
    let j = rem / dims.nz;
    let k = rem % dims.nz;
    // The E update reads only `peer` (curr), so the pre phase can run on the
    // same cell before it without a global barrier.
    pml_pre_phase(i, j, k);
    let si = select(0u, 1u, i > 0u);
    let sj = select(0u, 1u, j > 0u);
    let sk = select(0u, 1u, k > 0u);

    let c2_ijk = peer[idx(2u, i, j, k)];
    let c2_ijm1k = peer[idx(2u, i, j - sj, k)];
    let c1_ijk = peer[idx(1u, i, j, k)];
    let c1_ijkm1 = peer[idx(1u, i, j, k - sk)];
    let c0_ijk = peer[idx(0u, i, j, k)];
    let c0_ijkm1 = peer[idx(0u, i, j, k - sk)];
    let c2_im1jk = peer[idx(2u, i - si, j, k)];
    let c1_im1jk = peer[idx(1u, i - si, j, k)];
    let c0_ijm1k = peer[idx(0u, i, j - sj, k)];

    let b = idx(0u, i, j, k);
    let comp = dims.nx * dims.ny * dims.nz;
    let ex = b;
    let ey = b + comp;
    let ez = ey + comp;
    field[ex] = field[ex] * vv[ex] + vi[ex] * (c2_ijk - c2_ijm1k - c1_ijk + c1_ijkm1);
    field[ey] = field[ey] * vv[ey] + vi[ey] * (c0_ijk - c0_ijkm1 - c2_ijk + c2_im1jk);
    field[ez] = field[ez] * vv[ez] + vi[ez] * (c1_ijk - c1_im1jk - c0_ijk + c0_ijm1k);
    pml_post_phase(i, j, k);
}

@compute @workgroup_size(64)
fn update_h_pml(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = dims.nx * dims.ny * dims.nz;
    let t = gid.x;
    if (t >= total) { return; }
    let i = t / (dims.ny * dims.nz);
    let rem = t % (dims.ny * dims.nz);
    let j = rem / dims.nz;
    let k = rem % dims.nz;
    // The H update reads only `peer` (volt), so the pre phase is same-cell.
    pml_pre_phase(i, j, k);
    if (i + 1u < dims.nx && j + 1u < dims.ny && k + 1u < dims.nz) {
        let v2_ijk = peer[idx(2u, i, j, k)];
        let v2_ijp1k = peer[idx(2u, i, j + 1u, k)];
        let v1_ijk = peer[idx(1u, i, j, k)];
        let v1_ijkp1 = peer[idx(1u, i, j, k + 1u)];
        let v0_ijk = peer[idx(0u, i, j, k)];
        let v0_ijkp1 = peer[idx(0u, i, j, k + 1u)];
        let v2_ip1jk = peer[idx(2u, i + 1u, j, k)];
        let v1_ip1jk = peer[idx(1u, i + 1u, j, k)];
        let v0_ijp1k = peer[idx(0u, i, j + 1u, k)];

        let b = idx(0u, i, j, k);
        let comp = dims.nx * dims.ny * dims.nz;
        let hx = b;
        let hy = b + comp;
        let hz = hy + comp;
        field[hx] = field[hx] * ii[hx] + iv[hx] * (v2_ijk - v2_ijp1k - v1_ijk + v1_ijkp1);
        field[hy] = field[hy] * ii[hy] + iv[hy] * (v0_ijk - v0_ijkp1 - v2_ijk + v2_ip1jk);
        field[hz] = field[hz] * ii[hz] + iv[hz] * (v1_ijk - v1_ip1jk - v0_ijk + v0_ijp1k);
    }
    pml_post_phase(i, j, k);
}

// Device-side energy reduction: one workgroup sums 64 cells and writes two
// partials (E and H). The host sums the partials in f64 and applies the
// constants, so the value stays within ~1e-7 of the CPU accumulation.
struct EnergyDims {
    field: vec4<u32>,
    counts: vec4<u32>,
};

@group(0) @binding(0) var<storage, read> energy_volt: array<f32>;
@group(0) @binding(1) var<storage, read> energy_curr: array<f32>;
@group(0) @binding(2) var<storage, read_write> energy_parts: array<f32>;
@group(0) @binding(3) var<uniform> energy_dims: EnergyDims;

var<workgroup> energy_e: array<f32, 64>;
var<workgroup> energy_h: array<f32, 64>;

@compute @workgroup_size(64)
fn energy_sum(@builtin(global_invocation_id) gid: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>) {
    let cx = energy_dims.counts.x - 1u;
    let cy = energy_dims.counts.y - 1u;
    let cz = energy_dims.counts.z - 1u;
    let total = cx * cy * cz;
    let t = gid.x;
    var e_sum = 0.0;
    var h_sum = 0.0;
    if (t < total) {
        let i = t / (cy * cz);
        let rem = t % (cy * cz);
        let j = rem / cz;
        let k = rem % cz;
        let nx = energy_dims.field.x;
        let ny = energy_dims.field.y;
        let nz = energy_dims.field.z;
        let plane = nx * ny * nz;
        let base = i * ny * nz + j * nz + k;
        for (var c = 0u; c < 3u; c = c + 1u) {
            let ev = energy_volt[base + c * plane];
            let hv = energy_curr[base + c * plane];
            e_sum = e_sum + ev * ev;
            h_sum = h_sum + hv * hv;
        }
    }
    energy_e[lid.x] = e_sum;
    energy_h[lid.x] = h_sum;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (lid.x < stride) {
            energy_e[lid.x] = energy_e[lid.x] + energy_e[lid.x + stride];
            energy_h[lid.x] = energy_h[lid.x] + energy_h[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (lid.x == 0u) {
        let group = gid.x / 64u;
        energy_parts[group * 2u] = energy_e[0];
        energy_parts[group * 2u + 1u] = energy_h[0];
    }
}

// Device-side probe gather: one invocation per probe writes the sampled value
// into a compact per-step matrix, so a probe-only model never downloads the
// field.
struct ProbeParams {
    ts: u32,
    step: u32,
    count: u32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read> probe_volt: array<f32>;
@group(0) @binding(1) var<storage, read> probe_curr: array<f32>;
@group(0) @binding(2) var<storage, read> probe_index: array<u32>;
@group(0) @binding(3) var<storage, read> probe_buffer: array<u32>;
@group(0) @binding(4) var<storage, read> probe_interval: array<u32>;
@group(0) @binding(5) var<storage, read_write> probe_values: array<f32>;
@group(0) @binding(6) var<uniform> probe_params: ProbeParams;

@compute @workgroup_size(64)
fn probe_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= probe_params.count) { return; }
    let interval = probe_interval[p];
    if (interval == 0u || probe_params.ts % interval != 0u) { return; }
    let index = probe_index[p];
    var value = probe_volt[index];
    if (probe_buffer[p] == 1u) {
        value = probe_curr[index];
    }
    probe_values[probe_params.step * probe_params.count + p] = value;
}

// Device-side Mur ABC: the three host phases (pre, post, apply2). Every face
// of a model shares one set of concatenated coefficient/history buffers and a
// face table; the uniform carries the element range of the current pass.
//
// Faces that share a corner edge must run in the host's registration order,
// so the host colours the faces: a face's colour is one more than the highest
// colour of an earlier face it overlaps. Faces of one colour never overlap and
// run in a single dispatch, and a later colour always runs after an earlier
// one, which reproduces the host order exactly.

struct MurDims {
    global: vec4<u32>,
    /// `x` = first element, `y` = one past the last element, `z` = face count.
    range: vec4<u32>,
};

@group(0) @binding(0) var<storage, read_write> mur_volt: array<f32>;
@group(0) @binding(1) var<storage, read> mur_coeff_p: array<f32>;
@group(0) @binding(2) var<storage, read> mur_coeff_pp: array<f32>;
@group(0) @binding(3) var<storage, read_write> mur_hist_p: array<f32>;
@group(0) @binding(4) var<storage, read_write> mur_hist_pp: array<f32>;
// 12 u32 per face: offset, dims_p, dims_pp, line, shift, dir, n_p, n_pp, group, pad x3
@group(0) @binding(5) var<storage, read> mur_table: array<u32>;
@group(0) @binding(6) var<uniform> mur_dims: MurDims;

struct MurFace {
    /// offset, dims_p, dims_pp, line
    a: vec4<u32>,
    /// shift, dir, n_p, n_pp
    b: vec4<u32>,
};

fn mur_find(t: u32) -> MurFace {
    for (var f = 0u; f < mur_dims.range.z; f = f + 1u) {
        let base = f * 12u;
        let off = mur_table[base];
        let dp = mur_table[base + 1u];
        let dpp = mur_table[base + 2u];
        if (t >= off && t < off + dp * dpp) {
            return MurFace(
                vec4<u32>(off, dp, dpp, mur_table[base + 3u]),
                vec4<u32>(
                    mur_table[base + 4u],
                    mur_table[base + 5u],
                    mur_table[base + 6u],
                    mur_table[base + 7u],
                ),
            );
        }
    }
    return MurFace(vec4<u32>(0u), vec4<u32>(0u));
}

fn mur_pos_index(face: MurFace, ip: u32, ipp: u32, shifted: bool, comp: u32) -> u32 {
    var pos = vec3<u32>(0u, 0u, 0u);
    pos[face.b.y] = select(face.a.w, face.b.x, shifted);
    pos[face.b.z] = ip;
    pos[face.b.w] = ipp;
    let gy = mur_dims.global.y;
    let gz = mur_dims.global.z;
    let plane = mur_dims.global.x * gy * gz;
    return comp * plane + pos.x * gy * gz + pos.y * gz + pos.z;
}

@compute @workgroup_size(64)
fn mur_pre(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x + mur_dims.range.x;
    if (t < mur_dims.range.x || t >= mur_dims.range.y) { return; }
    let face = mur_find(t);
    let ip = (t - face.a.x) / face.a.z;
    let ipp = (t - face.a.x) % face.a.z;
    let np = face.b.z;
    let npp = face.b.w;
    let p_shift = mur_volt[mur_pos_index(face, ip, ipp, true, np)];
    let p_bound = mur_volt[mur_pos_index(face, ip, ipp, false, np)];
    mur_hist_p[t] = p_shift - mur_coeff_p[t] * p_bound;
    let pp_shift = mur_volt[mur_pos_index(face, ip, ipp, true, npp)];
    let pp_bound = mur_volt[mur_pos_index(face, ip, ipp, false, npp)];
    mur_hist_pp[t] = pp_shift - mur_coeff_pp[t] * pp_bound;
}

@compute @workgroup_size(64)
fn mur_post(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x + mur_dims.range.x;
    if (t < mur_dims.range.x || t >= mur_dims.range.y) { return; }
    let face = mur_find(t);
    let ip = (t - face.a.x) / face.a.z;
    let ipp = (t - face.a.x) % face.a.z;
    mur_hist_p[t] = mur_hist_p[t]
        + mur_coeff_p[t] * mur_volt[mur_pos_index(face, ip, ipp, true, face.b.z)];
    mur_hist_pp[t] = mur_hist_pp[t]
        + mur_coeff_pp[t] * mur_volt[mur_pos_index(face, ip, ipp, true, face.b.w)];
}

@compute @workgroup_size(64)
fn mur_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x + mur_dims.range.x;
    if (t < mur_dims.range.x || t >= mur_dims.range.y) { return; }
    let face = mur_find(t);
    let ip = (t - face.a.x) / face.a.z;
    let ipp = (t - face.a.x) % face.a.z;
    mur_volt[mur_pos_index(face, ip, ipp, false, face.b.z)] = mur_hist_p[t];
    mur_volt[mur_pos_index(face, ip, ipp, false, face.b.w)] = mur_hist_pp[t];
}

// Device-side lumped RLC: the two host phases (pre voltage rotation and the
// apply2 series update) over independent edges. Coefficients are packed as
// [vvd, vv2, vj1, vj2, ib0, b1, b2, ilv, i2v] per edge.

struct LumpedDims {
    count: vec4<u32>,
};

@group(0) @binding(0) var<storage, read_write> lump_volt: array<f32>;
@group(0) @binding(1) var<storage, read> lump_index: array<u32>;
@group(0) @binding(2) var<storage, read> lump_coeff: array<f32>;
@group(0) @binding(3) var<storage, read_write> lump_hist: array<f32>;
@group(0) @binding(4) var<storage, read_write> lump_jn: array<f32>;
@group(0) @binding(5) var<storage, read_write> lump_il: array<f32>;
@group(0) @binding(6) var<uniform> lump_dims: LumpedDims;

@compute @workgroup_size(64)
fn lumped_pre(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= lump_dims.count.x) { return; }
    let b = t * 3u;
    let h2 = lump_hist[b + 2u];
    lump_hist[b + 2u] = lump_hist[b + 1u];
    lump_hist[b + 1u] = lump_hist[b];
    lump_hist[b] = h2;
    let c = t * 9u;
    lump_il[t] = lump_il[t] + lump_coeff[c + 8u] * lump_coeff[c + 7u] * lump_hist[b + 1u];
}

@compute @workgroup_size(64)
fn lumped_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= lump_dims.count.x) { return; }
    let b = t * 3u;
    let j2 = lump_jn[b + 2u];
    lump_jn[b + 2u] = lump_jn[b + 1u];
    lump_jn[b + 1u] = lump_jn[b];
    lump_jn[b] = j2;
    let c = t * 9u;
    let index = lump_index[t];
    let raw = lump_volt[index];
    lump_hist[b] = raw;
    let updated = lump_coeff[c] * (raw - lump_il[t]
        + lump_coeff[c + 1u] * lump_hist[b + 2u]
        + lump_coeff[c + 2u] * lump_jn[b + 1u]
        + lump_coeff[c + 3u] * lump_jn[b + 2u]);
    lump_hist[b] = updated;
    lump_jn[b] = lump_coeff[c + 4u] * (lump_hist[b] - lump_hist[b + 2u])
        - (lump_coeff[c + 5u] * lump_coeff[c + 4u]) * lump_jn[b + 1u]
        - (lump_coeff[c + 6u] * lump_coeff[c + 4u]) * lump_jn[b + 2u];
    lump_volt[index] = lump_hist[b];
}

// Device-side conducting sheet: two ADE orders over the sheet cells. The
// per-cell state and coefficients are packed as (order * 3 + component) * n.

struct SheetDims {
    count: vec4<u32>,
    plane: vec4<u32>,
};

@group(0) @binding(0) var<storage, read_write> sheet_volt: array<f32>;
@group(0) @binding(1) var<storage, read> sheet_pos: array<u32>;
@group(0) @binding(2) var<storage, read> sheet_int: array<f32>;
@group(0) @binding(3) var<storage, read> sheet_ext: array<f32>;
@group(0) @binding(4) var<storage, read_write> sheet_state: array<f32>;
@group(0) @binding(5) var<uniform> sheet_dims: SheetDims;

@compute @workgroup_size(64)
fn sheet_pre(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= sheet_dims.count.x) { return; }
    let base = sheet_pos[t];
    let plane = sheet_dims.plane.x;
    let count = sheet_dims.count.x;
    for (var n = 0u; n < 3u; n = n + 1u) {
        let v = sheet_volt[base + n * plane];
        for (var o = 0u; o < 2u; o = o + 1u) {
            let s = (o * 3u + n) * count + t;
            sheet_state[s] = sheet_int[s] * sheet_state[s] + sheet_ext[s] * v;
        }
    }
}

@compute @workgroup_size(64)
fn sheet_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= sheet_dims.count.x) { return; }
    let base = sheet_pos[t];
    let plane = sheet_dims.plane.x;
    let count = sheet_dims.count.x;
    for (var n = 0u; n < 3u; n = n + 1u) {
        let correction = sheet_state[(0u * 3u + n) * count + t]
            + sheet_state[(1u * 3u + n) * count + t];
        let index = base + n * plane;
        sheet_volt[index] = sheet_volt[index] - correction;
    }
}

// Device-side TFSF: the two correction phases. Per term the table stores the
// flat field index, the amplitude, the delay and the interpolation factor;
// the per-step timestamp arrives in a uniform slot and the signal samples are
// uploaded once.

struct TfsfParams {
    ts: u32,
    length: u32,
    period: u32,
    max_delay: u32,
};

@group(0) @binding(0) var<storage, read_write> tfsf_target: array<f32>;
@group(0) @binding(1) var<storage, read> tfsf_index: array<u32>;
@group(0) @binding(2) var<storage, read> tfsf_amp: array<f32>;
@group(0) @binding(3) var<storage, read> tfsf_delay: array<u32>;
@group(0) @binding(4) var<storage, read> tfsf_delta: array<f32>;
@group(0) @binding(5) var<storage, read> tfsf_samples: array<f32>;
@group(0) @binding(6) var<uniform> tfsf_params: TfsfParams;

// `EngineExtTFSF::update_delay_lookup`: negative becomes 0, a run past the
// signal (for non-periodic signals) becomes 0, then the period folds it.
// `delay + 1` past `max_delay` falls back to index 0, like the host table.
fn tfsf_lookup(ts: u32, delay: u32) -> u32 {
    if (delay > tfsf_params.max_delay) { return 0u; }
    var value = 0u;
    if (ts >= delay) {
        let diff = ts - delay;
        if (tfsf_params.period == 0u && diff >= tfsf_params.length) {
            value = 0u;
        } else {
            value = diff;
        }
    }
    if (tfsf_params.period > 0u) { value = value % tfsf_params.period; }
    return value;
}

fn tfsf_sample(index: u32) -> f32 {
    if (index < tfsf_params.length) { return tfsf_samples[index]; }
    return 0.0;
}

@compute @workgroup_size(64)
fn tfsf_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= arrayLength(&tfsf_index)) { return; }
    let ts = tfsf_params.ts;
    let delay = tfsf_delay[t];
    let first = tfsf_lookup(ts, delay);
    let second = tfsf_lookup(ts, delay + 1u);
    let delta = tfsf_delta[t];
    let amp = tfsf_amp[t];
    let correction = (1.0 - delta) * amp * tfsf_sample(first)
        + delta * amp * tfsf_sample(second);
    let index = tfsf_index[t];
    tfsf_target[index] = tfsf_target[index] + correction;
}

// Device-side Debye: three voltage phases over the dispersive cells. The
// pole state and the per-pole coefficients are packed as
// (pole * 3 + component) * count.

struct DebyeDims {
    count: vec4<u32>,
    poles: vec4<u32>,
    plane: vec4<u32>,
};

@group(0) @binding(0) var<storage, read_write> debye_volt: array<f32>;
@group(0) @binding(1) var<storage, read> debye_pos: array<u32>;
@group(0) @binding(2) var<storage, read> debye_relax: array<f32>;
@group(0) @binding(3) var<storage, read> debye_drive: array<f32>;
@group(0) @binding(4) var<storage, read> debye_solve: array<f32>;
@group(0) @binding(5) var<storage, read_write> debye_pole: array<f32>;
@group(0) @binding(6) var<storage, read_write> debye_pre_ade: array<f32>;
@group(0) @binding(7) var<storage, read_write> debye_ade_state: array<f32>;
@group(0) @binding(8) var<uniform> debye_dims: DebyeDims;

@compute @workgroup_size(64)
fn debye_pre(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= debye_dims.count.x) { return; }
    let base = debye_pos[t];
    let count = debye_dims.count.x;
    let poles = debye_dims.poles.x;
    let plane = debye_dims.plane.x;
    for (var comp = 0u; comp < 3u; comp = comp + 1u) {
        let v = debye_volt[base + comp * plane];
        var pre = 0.0;
        for (var o = 0u; o < poles; o = o + 1u) {
            let s = (o * 3u + comp) * count + t;
            let w = debye_relax[s] * debye_pole[s] + debye_drive[s] * v;
            pre = pre + (w - debye_pole[s]);
            debye_pole[s] = w;
        }
        debye_pre_ade[comp * count + t] = pre;
    }
}

@compute @workgroup_size(64)
fn debye_post(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= debye_dims.count.x) { return; }
    let base = debye_pos[t];
    let count = debye_dims.count.x;
    let poles = debye_dims.poles.x;
    let plane = debye_dims.plane.x;
    for (var comp = 0u; comp < 3u; comp = comp + 1u) {
        let v_raw = debye_volt[base + comp * plane];
        let key = comp * count + t;
        let v_new = (v_raw - debye_pre_ade[key]) * debye_solve[key];
        debye_ade_state[key] = v_raw - v_new;
        for (var o = 0u; o < poles; o = o + 1u) {
            let s = (o * 3u + comp) * count + t;
            debye_pole[s] = debye_pole[s] + debye_drive[s] * v_new;
        }
    }
}

@compute @workgroup_size(64)
fn debye_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= debye_dims.count.x) { return; }
    let base = debye_pos[t];
    let count = debye_dims.count.x;
    let plane = debye_dims.plane.x;
    for (var comp = 0u; comp < 3u; comp = comp + 1u) {
        let index = base + comp * plane;
        debye_volt[index] = debye_volt[index] - debye_ade_state[comp * count + t];
    }
}

// Device-side Lorentz: four phases over the dispersive cells. The state and
// coefficients are packed as (order * 3 + component) * count; `active`
// carries one flag per order and side.

struct LorentzDims {
    count: vec4<u32>,
    orders: vec4<u32>,
    plane: vec4<u32>,
};

@group(0) @binding(0) var<storage, read_write> lor_volt: array<f32>;
@group(0) @binding(1) var<storage, read_write> lor_curr: array<f32>;
@group(0) @binding(2) var<storage, read> lor_pos: array<u32>;
@group(0) @binding(3) var<storage, read> lor_active: array<u32>;
@group(0) @binding(4) var<storage, read> lor_v_int: array<f32>;
@group(0) @binding(5) var<storage, read> lor_v_ext: array<f32>;
@group(0) @binding(6) var<storage, read> lor_v_lor: array<f32>;
@group(0) @binding(7) var<storage, read> lor_i_int: array<f32>;
@group(0) @binding(8) var<storage, read> lor_i_ext: array<f32>;
@group(0) @binding(9) var<storage, read> lor_i_lor: array<f32>;
@group(0) @binding(10) var<storage, read_write> lor_volt_ade: array<f32>;
@group(0) @binding(11) var<storage, read_write> lor_volt_lor: array<f32>;
@group(0) @binding(12) var<storage, read_write> lor_curr_ade: array<f32>;
@group(0) @binding(13) var<storage, read_write> lor_curr_lor: array<f32>;
@group(0) @binding(14) var<uniform> lor_dims: LorentzDims;

@compute @workgroup_size(64)
fn lorentz_pre_volt(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= lor_dims.count.x) { return; }
    let base = lor_pos[t];
    let count = lor_dims.count.x;
    let orders = lor_dims.orders.x;
    let plane = lor_dims.plane.x;
    for (var o = 0u; o < orders; o = o + 1u) {
        if (lor_active[o * 2u] == 0u) { continue; }
        for (var comp = 0u; comp < 3u; comp = comp + 1u) {
            let key = (o * 3u + comp) * count + t;
            let v = lor_volt[base + comp * plane];
            let lor = lor_v_lor[key];
            if (lor != 0.0) {
                let lor_new = lor_volt_lor[key] + lor * lor_volt_ade[key];
                lor_volt_lor[key] = lor_new;
                lor_volt_ade[key] = lor_volt_ade[key] * lor_v_int[key]
                    + lor_v_ext[key] * (v - lor_new);
            } else {
                lor_volt_ade[key] = lor_volt_ade[key] * lor_v_int[key] + lor_v_ext[key] * v;
            }
        }
    }
}

@compute @workgroup_size(64)
fn lorentz_pre_curr(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= lor_dims.count.x) { return; }
    let base = lor_pos[t];
    let count = lor_dims.count.x;
    let orders = lor_dims.orders.x;
    let plane = lor_dims.plane.x;
    for (var o = 0u; o < orders; o = o + 1u) {
        if (lor_active[o * 2u + 1u] == 0u) { continue; }
        for (var comp = 0u; comp < 3u; comp = comp + 1u) {
            let key = (o * 3u + comp) * count + t;
            let v = lor_curr[base + comp * plane];
            let lor = lor_i_lor[key];
            if (lor != 0.0) {
                let lor_new = lor_curr_lor[key] + lor * lor_curr_ade[key];
                lor_curr_lor[key] = lor_new;
                lor_curr_ade[key] = lor_curr_ade[key] * lor_i_int[key]
                    + lor_i_ext[key] * (v - lor_new);
            } else {
                lor_curr_ade[key] = lor_curr_ade[key] * lor_i_int[key] + lor_i_ext[key] * v;
            }
        }
    }
}

@compute @workgroup_size(64)
fn lorentz_apply_volt(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= lor_dims.count.x) { return; }
    let base = lor_pos[t];
    let count = lor_dims.count.x;
    let orders = lor_dims.orders.x;
    let plane = lor_dims.plane.x;
    for (var comp = 0u; comp < 3u; comp = comp + 1u) {
        let index = base + comp * plane;
        var v = lor_volt[index];
        for (var o = 0u; o < orders; o = o + 1u) {
            if (lor_active[o * 2u] == 0u) { continue; }
            v = v - lor_volt_ade[(o * 3u + comp) * count + t];
        }
        lor_volt[index] = v;
    }
}

@compute @workgroup_size(64)
fn lorentz_apply_curr(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= lor_dims.count.x) { return; }
    let base = lor_pos[t];
    let count = lor_dims.count.x;
    let orders = lor_dims.orders.x;
    let plane = lor_dims.plane.x;
    for (var comp = 0u; comp < 3u; comp = comp + 1u) {
        let index = base + comp * plane;
        var v = lor_curr[index];
        for (var o = 0u; o < orders; o = o + 1u) {
            if (lor_active[o * 2u + 1u] == 0u) { continue; }
            v = v - lor_curr_ade[(o * 3u + comp) * count + t];
        }
        lor_curr[index] = v;
    }
}
"#;

fn f32_bytes(values: &[FDTD_FLOAT]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Host-side TFSF table flattening for one direction: one (index, amplitude,
/// delay, delta) term per corrected edge, in the same order as
/// `EngineExtTFSF`'s loops. Faces of one direction sit on opposite planes, so
/// they never write the same edge; different directions overlap at the box
/// corners, which is why the engine dispatches one pass per direction.
fn tfsf_terms(
    extension: &crate::extensions::EngineExtTFSF,
    volt: bool,
    dims: [usize; 3],
    n: usize,
) -> (Vec<u32>, Vec<f32>, Vec<u32>, Vec<f32>) {
    let (i_stride, j_stride, plane) = (
        dims[1] * dims[2],
        dims[2],
        dims[0] * dims[1] * dims[2],
    );
    let mut indices = Vec::new();
    let mut amps = Vec::new();
    let mut delays = Vec::new();
    let mut deltas = Vec::new();
    let n_p = (n + 1) % 3;
    let n_pp = (n + 2) % 3;
    for face in 0..2 {
        if !extension.active_dir[n][face] {
            continue;
        }
        let mut pos = [0usize; 3];
        pos[n_p] = extension.start[n_p];
        let mut ui_pos = 0usize;
        for _i in 0..extension.num_lines[n_p] {
            pos[n_pp] = extension.start[n_pp];
            for _j in 0..extension.num_lines[n_pp] {
                pos[n] = if volt {
                    if face == 0 {
                        extension.start[n]
                    } else {
                        extension.stop[n]
                    }
                } else if face == 0 {
                    extension.start[n].saturating_sub(1)
                } else {
                    extension.stop[n]
                };
                for component in 0..2 {
                    let (amp, delay, delta) = if volt {
                        (
                            extension.volt_amp[n][face][component].get(ui_pos).copied(),
                            extension.volt_delay[n][face][component].get(ui_pos).copied(),
                            extension.volt_delay_delta[n][face][component]
                                .get(ui_pos)
                                .copied(),
                        )
                    } else {
                        (
                            extension.curr_amp[n][face][component].get(ui_pos).copied(),
                            extension.curr_delay[n][face][component].get(ui_pos).copied(),
                            extension.curr_delay_delta[n][face][component]
                                .get(ui_pos)
                                .copied(),
                        )
                    };
                    if let (Some(amp), Some(delay), Some(delta)) = (amp, delay, delta) {
                        if amp != 0.0 {
                            let edge = if component == 0 { n_p } else { n_pp };
                            indices.push(
                                (edge * plane
                                    + pos[0] * i_stride
                                    + pos[1] * j_stride
                                    + pos[2]) as u32,
                            );
                            amps.push(amp);
                            delays.push(delay);
                            deltas.push(delta);
                        }
                    }
                }
                pos[n_pp] += 1;
                ui_pos += 1;
            }
            pos[n_p] += 1;
        }
    }
    (indices, amps, delays, deltas)
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn readback(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    bytes: u64,
) -> Option<Vec<FDTD_FLOAT>> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("openems gpu readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("openems gpu copy"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes);
    queue.submit(std::iter::once(encoder.finish()));
    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
    receiver.recv().ok()?.ok()?;
    let data = slice.get_mapped_range().ok()?;
    let values = data
        .chunks_exact(4)
        .map(|chunk| FDTD_FLOAT::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    drop(data);
    staging.unmap();
    Some(values)
}

fn readback_pair(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    first: &wgpu::Buffer,
    second: &wgpu::Buffer,
    bytes: u64,
) -> Option<(Vec<FDTD_FLOAT>, Vec<FDTD_FLOAT>)> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("openems gpu pair readback"),
        size: 2 * bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("openems gpu pair copy"),
    });
    encoder.copy_buffer_to_buffer(first, 0, &staging, 0, bytes);
    encoder.copy_buffer_to_buffer(second, 0, &staging, bytes, bytes);
    queue.submit(std::iter::once(encoder.finish()));
    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
    receiver.recv().ok()?.ok()?;
    let data = slice.get_mapped_range().ok()?;
    let parse = |chunk: &[u8]| {
        FDTD_FLOAT::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
    };
    let first_values = data[..bytes as usize]
        .chunks_exact(4)
        .map(parse)
        .collect();
    let second_values = data[bytes as usize..]
        .chunks_exact(4)
        .map(parse)
        .collect();
    drop(data);
    staging.unmap();
    Some((first_values, second_values))
}

/// Copy one buffer into a persistent staging buffer and decode it into `out`.
///
/// The staging buffer is allocated once per [`GpuFields`] instance; the earlier
/// path created a fresh map-read buffer for every download, which added a
/// multi-megabyte allocation to every host sync.
fn readback_into(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    staging: &wgpu::Buffer,
    out: &mut [FDTD_FLOAT],
) -> Option<()> {
    let bytes = (out.len() * 4) as u64;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("openems gpu copy"),
    });
    encoder.copy_buffer_to_buffer(source, 0, staging, 0, bytes);
    queue.submit(std::iter::once(encoder.finish()));
    let slice = staging.slice(..bytes);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
    receiver.recv().ok()?.ok()?;
    let data = slice.get_mapped_range().ok()?;
    for (value, chunk) in out.iter_mut().zip(data.chunks_exact(4)) {
        *value = FDTD_FLOAT::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    drop(data);
    staging.unmap();
    Some(())
}

impl GpuContext {
    /// Create a context on the default high-performance adapter, or `None`
    /// when no adapter is available (the caller must fall back to the CPU).
    pub fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            flags: wgpu::InstanceFlags::default(),
            backend_options: Default::default(),
            display: Default::default(),
            memory_budget_thresholds: Default::default(),
        });
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            },
        ))
        .ok()?;
        let info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                // The UPML kernels bind the two field buffers, six coefficient
                // arrays and two flux arrays (10 storage buffers), above the
                // WebGPU default of 8. Every desktop adapter offers more; ask
                // for 16 when the adapter allows it.
                required_limits: {
                    let mut limits = wgpu::Limits::default();
                    limits.max_storage_buffers_per_shader_stage = adapter
                        .limits()
                        .max_storage_buffers_per_shader_stage
                        .clamp(8, 16);
                    limits
                },
                label: Some("openems-core gpu context"),
                trace: wgpu::Trace::default(),
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: wgpu::ExperimentalFeatures::default(),
            },
        ))
        .ok()?;
        Some(Self {
            device,
            queue,
            adapter_name: info.name,
            backend: format!("{:?}", info.backend),
        })
    }

    /// Stage-1 kernel: multiply every value by `factor` on the GPU.
    pub fn scale(&self, values: &[FDTD_FLOAT], factor: f32) -> Option<Vec<FDTD_FLOAT>> {
        if values.is_empty() {
            return Some(Vec::new());
        }
        use wgpu::util::DeviceExt;
        let bytes = (values.len() * 4) as u64;
        let input = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("openems gpu input"),
                contents: &f32_bytes(values),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("openems gpu output"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let uniform = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("openems gpu factor"),
                contents: &f32_bytes(&[factor, 0.0, 0.0, 0.0]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("openems gpu scale"),
                source: wgpu::ShaderSource::Wgsl(SCALE_SHADER.into()),
            });
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("openems gpu scale layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("openems gpu scale"),
                layout: Some(&self.device.create_pipeline_layout(
                    &wgpu::PipelineLayoutDescriptor {
                        label: Some("openems gpu scale pipeline layout"),
                        bind_group_layouts: &[Some(&layout)],
                        immediate_size: 0,
                    },
                )),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("openems gpu scale bind group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: output.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: uniform.as_entire_binding() },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("openems gpu scale encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("openems gpu scale pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((values.len() as u32).div_ceil(64), 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        readback(&self.device, &self.queue, &output, bytes)
    }

    /// Run one E update followed by one H update on the GPU.
    ///
    /// All arrays are component-major `[3*nx*ny*nz]`, exactly like
    /// `ArrayNIJK<f32>`. The CPU engine uses the same layout and formulas; the
    /// unit test compares both.
    pub fn update_fields(
        &self,
        volt: &mut [FDTD_FLOAT],
        curr: &mut [FDTD_FLOAT],
        vv: &[FDTD_FLOAT],
        vi: &[FDTD_FLOAT],
        ii: &[FDTD_FLOAT],
        iv: &[FDTD_FLOAT],
        dims: [usize; 3],
    ) -> Option<()> {
        let [nx, ny, nz] = dims;
        let total = nx * ny * nz;
        if volt.len() != 3 * total
            || curr.len() != 3 * total
            || vv.len() != 3 * total
            || vi.len() != 3 * total
            || ii.len() != 3 * total
            || iv.len() != 3 * total
        {
            return None;
        }
        let uniform: [u32; 4] = [nx as u32, ny as u32, nz as u32, 0];
        self.dispatch_update(
            "update_e",
            total as u32,
            volt,
            curr,
            [vv, vi, ii, iv],
            uniform,
        )?;
        self.dispatch_update(
            "update_h",
            ((nx - 1) * (ny - 1) * (nz - 1)) as u32,
            curr,
            volt,
            [vv, vi, ii, iv],
            uniform,
        )?;
        Some(())
    }

    fn dispatch_update(
        &self,
        entry: &str,
        groups: u32,
        target: &mut [FDTD_FLOAT],
        peer: &[FDTD_FLOAT],
        coeffs: [&[FDTD_FLOAT]; 4],
        uniform: [u32; 4],
    ) -> Option<()> {
        use wgpu::util::DeviceExt;
        let bytes = (target.len() * 4) as u64;
        let target_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("openems gpu target"),
                contents: &f32_bytes(target),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        let peer_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("openems gpu peer"),
                contents: &f32_bytes(peer),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let coeff_bufs: Vec<wgpu::Buffer> = coeffs
            .iter()
            .map(|values| {
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("openems gpu coefficient"),
                        contents: &f32_bytes(values),
                        usage: wgpu::BufferUsages::STORAGE,
                    })
            })
            .collect();
        let uniform_bytes: Vec<u8> = uniform.iter().flat_map(|v| v.to_le_bytes()).collect();
        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("openems gpu dims"),
                contents: &uniform_bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("openems gpu update"),
                source: wgpu::ShaderSource::Wgsl(UPDATE_SHADER.into()),
            });
        let mut entries = Vec::with_capacity(7);
        for binding in 0..7u32 {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 6 {
                        wgpu::BufferBindingType::Uniform
                    } else if binding <= 1 {
                        wgpu::BufferBindingType::Storage { read_only: binding == 1 }
                    } else {
                        wgpu::BufferBindingType::Storage { read_only: true }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("openems gpu update layout"),
                entries: &entries,
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("openems gpu update"),
                layout: Some(&self.device.create_pipeline_layout(
                    &wgpu::PipelineLayoutDescriptor {
                        label: Some("openems gpu update pipeline layout"),
                        bind_group_layouts: &[Some(&layout)],
                        immediate_size: 0,
                    },
                )),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("openems gpu update bind group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: target_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: peer_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: coeff_bufs[0].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: coeff_bufs[1].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: coeff_bufs[2].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: coeff_bufs[3].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: uniform_buf.as_entire_binding() },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("openems gpu update encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("openems gpu update pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups.div_ceil(64), 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        let values = readback(&self.device, &self.queue, &target_buf, bytes)?;
        target.copy_from_slice(&values[..target.len()]);
        Some(())
    }
}


/// One device-side excitation term.
///
/// `index` is the flat N-I-J-K field index, `coeff` the weighted amplitude,
/// and `slot`/`delay`/`period`/`length` reproduce
/// `EngineExtExcitation::sample_index` on the device.
#[derive(Clone, Copy, Debug, Default)]
pub struct SourceTerm {
    pub index: u32,
    pub coeff: f32,
    pub delay: u32,
    pub period: u32,
    pub slot: u32,
    pub length: u32,
}

/// Host description of the excitation source handed to the GPU kernels.
#[derive(Default)]
pub struct SourceTerms {
    pub volt: Vec<SourceTerm>,
    pub curr: Vec<SourceTerm>,
    /// `volt_values[slot][sample]` and `curr_values[slot][sample]`.
    pub volt_values: Vec<Vec<f32>>,
    pub curr_values: Vec<Vec<f32>>,
}

impl SourceTerms {
    pub fn is_empty(&self) -> bool {
        self.volt.is_empty() && self.curr.is_empty()
    }

    fn stride(&self) -> u32 {
        self.volt_values
            .iter()
            .chain(self.curr_values.iter())
            .map(|values| values.len())
            .max()
            .unwrap_or(0) as u32
    }
}

/// Maximum number of steps recorded into one submission; the per-step uniform
/// slots are sized for it.
const SOURCE_BATCH: u32 = 256;

/// Largest batch a caller may request when device-side sampling is active.
pub const GPU_MAX_BATCH: u32 = SOURCE_BATCH;

struct SourceSide {
    pipeline: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    count: u32,
    params: wgpu::Buffer,
    scratch: std::cell::RefCell<Vec<u8>>,
}

/// Device-side excitation kernels and their per-step parameters.
struct SourceBinds {
    volt: Option<SourceSide>,
    curr: Option<SourceSide>,
    slot_size: u32,
    stride: u32,
}

impl SourceBinds {
    fn write_params(
        queue: &wgpu::Queue,
        side: &SourceSide,
        slot_size: u32,
        stride: u32,
        start_ts: u32,
        steps: u32,
    ) {
        let mut scratch = side.scratch.borrow_mut();
        scratch.clear();
        scratch.resize(steps as usize * slot_size as usize, 0);
        for step in 0..steps {
            let base = step as usize * slot_size as usize;
            scratch[base..base + 4].copy_from_slice(&(start_ts + step).to_le_bytes());
            scratch[base + 4..base + 8].copy_from_slice(&stride.to_le_bytes());
            scratch[base + 8..base + 12].copy_from_slice(&side.count.to_le_bytes());
        }
        queue.write_buffer(&side.params, 0, &scratch);
    }

    fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        side: Option<&SourceSide>,
        step: u32,
    ) -> u32 {
        let Some(side) = side else {
            return 0;
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("openems source pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&side.pipeline);
        pass.set_bind_group(0, &side.bind, &[step * self.slot_size]);
        pass.dispatch_workgroups(side.count.div_ceil(64).max(1), 1, 1);
        1
    }
}

/// Device-side UPML kernels shared by every PML slab of a model.
///
/// All slabs share one bind group: their arrays are concatenated and a slab
/// table tells the kernel which slab an element belongs to, so a phase costs
/// one dispatch regardless of the slab count.
struct UpmlBinds {
    pre_volt: wgpu::ComputePipeline,
    post_volt: wgpu::ComputePipeline,
    pre_curr: wgpu::ComputePipeline,
    post_curr: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    groups: u32,
    /// Buffers the fused E/H kernels reuse for the post phases.
    vvfn: wgpu::Buffer,
    volt_flux: wgpu::Buffer,
    iifn: wgpu::Buffer,
    curr_flux: wgpu::Buffer,
    table: wgpu::Buffer,
    dims: wgpu::Buffer,
    /// Pre-phase coefficients (`vv`/`vvfo` for E, `ii`/`iifo` for H).
    vv: wgpu::Buffer,
    vvfo: wgpu::Buffer,
    ii: wgpu::Buffer,
    iifo: wgpu::Buffer,
}

impl UpmlBinds {
    fn record(&self, encoder: &mut wgpu::CommandEncoder, phase: usize) -> u32 {
        let pipeline = match phase {
            0 => &self.pre_volt,
            1 => &self.post_volt,
            2 => &self.pre_curr,
            _ => &self.post_curr,
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("openems upml pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.bind, &[]);
        pass.dispatch_workgroups(self.groups, 1, 1);
        1
    }
}

/// Device-side Mur kernels shared by every Mur face of a model.
///
/// Faces are coloured so that one colour never overlaps; each colour is one
/// dispatch over its contiguous element range, and colours run in increasing
/// order, which reproduces the host's registration order on shared edges.
struct MurBinds {
    pre: wgpu::ComputePipeline,
    post: wgpu::ComputePipeline,
    apply: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    passes: Vec<MurPass>,
}

struct MurPass {
    /// Dynamic uniform offset of this colour's range.
    offset: u32,
    groups: u32,
}

impl MurBinds {
    fn record(&self, encoder: &mut wgpu::CommandEncoder, phase: usize) -> u32 {
        let pipeline = match phase {
            0 => &self.pre,
            1 => &self.post,
            _ => &self.apply,
        };
        for pass_info in &self.passes {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("openems mur pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.bind, &[pass_info.offset]);
            pass.dispatch_workgroups(pass_info.groups, 1, 1);
        }
        self.passes.len() as u32
    }
}

/// Device-side lumped RLC kernels.
struct LumpedBinds {
    pre: wgpu::ComputePipeline,
    apply: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    count: u32,
    groups: u32,
}

impl LumpedBinds {
    fn record(&self, encoder: &mut wgpu::CommandEncoder, pre: bool) -> u32 {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("openems lumped pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(if pre { &self.pre } else { &self.apply });
        pass.set_bind_group(0, &self.bind, &[]);
        pass.dispatch_workgroups(self.groups, 1, 1);
        1
    }
}

/// Device-side conducting-sheet kernels.
struct SheetBinds {
    pre: wgpu::ComputePipeline,
    apply: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    groups: u32,
}

impl SheetBinds {
    fn record(&self, encoder: &mut wgpu::CommandEncoder, pre: bool) -> u32 {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("openems sheet pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(if pre { &self.pre } else { &self.apply });
        pass.set_bind_group(0, &self.bind, &[]);
        pass.dispatch_workgroups(self.groups, 1, 1);
        1
    }
}

/// Device-side TFSF kernels, one bind group per extension and side.
struct TfsfBinds {
    pipeline: wgpu::ComputePipeline,
    volt: Vec<TfsfSide>,
    curr: Vec<TfsfSide>,
    slot_size: u32,
}

struct TfsfSide {
    bind: wgpu::BindGroup,
    groups: u32,
    params: wgpu::Buffer,
    scratch: std::cell::RefCell<Vec<u8>>,
    length: u32,
    period: u32,
    max_delay: u32,
}

impl TfsfBinds {
    fn write_params(&self, queue: &wgpu::Queue, side: &TfsfSide, start_ts: u32, steps: u32) {
        let mut scratch = side.scratch.borrow_mut();
        scratch.clear();
        scratch.resize(steps as usize * self.slot_size as usize, 0);
        for step in 0..steps {
            let base = step as usize * self.slot_size as usize;
            scratch[base..base + 4].copy_from_slice(&(start_ts + step).to_le_bytes());
            scratch[base + 4..base + 8].copy_from_slice(&side.length.to_le_bytes());
            scratch[base + 8..base + 12].copy_from_slice(&side.period.to_le_bytes());
            scratch[base + 12..base + 16].copy_from_slice(&side.max_delay.to_le_bytes());
        }
        queue.write_buffer(&side.params, 0, &scratch);
    }

    fn record(&self, encoder: &mut wgpu::CommandEncoder, side: &TfsfSide, step: u32) -> u32 {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("openems tfsf pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &side.bind, &[step * self.slot_size]);
        pass.dispatch_workgroups(side.groups, 1, 1);
        1
    }
}

/// One device-side Debye region.
struct DebyeBinds {
    pre: wgpu::ComputePipeline,
    post: wgpu::ComputePipeline,
    apply: wgpu::ComputePipeline,
    instances: Vec<DebyeInstance>,
}

struct DebyeInstance {
    bind: wgpu::BindGroup,
    groups: u32,
}

impl DebyeBinds {
    fn record(&self, encoder: &mut wgpu::CommandEncoder, phase: usize) -> u32 {
        let pipeline = match phase {
            0 => &self.pre,
            1 => &self.post,
            _ => &self.apply,
        };
        for instance in &self.instances {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("openems debye pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &instance.bind, &[]);
            pass.dispatch_workgroups(instance.groups, 1, 1);
        }
        self.instances.len() as u32
    }
}

/// Device-side Lorentz kernels.
struct LorentzBinds {
    pre_volt: wgpu::ComputePipeline,
    pre_curr: wgpu::ComputePipeline,
    apply_volt: wgpu::ComputePipeline,
    apply_curr: wgpu::ComputePipeline,
    instances: Vec<LorentzInstance>,
}

struct LorentzInstance {
    bind: wgpu::BindGroup,
    groups: u32,
}

impl LorentzBinds {
    fn record(&self, encoder: &mut wgpu::CommandEncoder, phase: usize) -> u32 {
        let pipeline = match phase {
            0 => &self.pre_volt,
            1 => &self.pre_curr,
            2 => &self.apply_volt,
            _ => &self.apply_curr,
        };
        for instance in &self.instances {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("openems lorentz pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &instance.bind, &[]);
            pass.dispatch_workgroups(instance.groups, 1, 1);
        }
        self.instances.len() as u32
    }
}

/// Device-side probe gather: a compact per-step value matrix.
struct ProbeBinds {
    pipe: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    params: wgpu::Buffer,
    values: wgpu::Buffer,
    staging: wgpu::Buffer,
    count: u32,
    slot_size: u32,
    scratch: std::cell::RefCell<Vec<u8>>,
}

/// Fused PML E/H kernels: the post phases run at the end of the update thread,
/// which removes the two post dispatches per step.
struct PmlEngine {
    pipe_e: wgpu::ComputePipeline,
    pipe_h: wgpu::ComputePipeline,
    bind_e: wgpu::BindGroup,
    bind_h: wgpu::BindGroup,
}

/// Persistent GPU field/coefficient buffers for `Engine`.
///
/// The engine uploads once, dispatches the E and H kernels per step, and
/// downloads the fields before the extension phase or an observable read.
pub struct GpuFields {
    device: wgpu::Device,
    queue: wgpu::Queue,
    dims: [usize; 3],
    volt: wgpu::Buffer,
    curr: wgpu::Buffer,
    bind_e: wgpu::BindGroup,
    bind_h: wgpu::BindGroup,
    pipe_e: wgpu::ComputePipeline,
    pipe_h: wgpu::ComputePipeline,
    /// Persistent map-read staging buffer for `volt` downloads.
    staging_volt: wgpu::Buffer,
    /// Persistent map-read staging buffer for `curr` downloads.
    staging_curr: wgpu::Buffer,
    /// Persistent map-read staging buffer for a `volt`+`curr` pair download.
    staging_pair: wgpu::Buffer,
    /// Device-side excitation kernels, when the model has a source.
    source: Option<SourceBinds>,
    /// Device-side UPML kernels, when the model has PML regions.
    upml: Option<UpmlBinds>,
    /// Fused PML E/H kernels (post phases folded into the updates).
    pml_engine: Option<PmlEngine>,
    /// EC coefficient buffers, kept so the fused PML bind groups can rebind
    /// them next to the PML buffers.
    ec_vv: wgpu::Buffer,
    ec_vi: wgpu::Buffer,
    ec_ii: wgpu::Buffer,
    ec_iv: wgpu::Buffer,
    /// The `Dims` uniform bound by the E/H kernels.
    dims_uniform: wgpu::Buffer,
    /// Device-side energy reduction (partial sums per workgroup).
    energy_pipe: wgpu::ComputePipeline,
    energy_bind: wgpu::BindGroup,
    energy_uniform: wgpu::Buffer,
    energy_parts: wgpu::Buffer,
    energy_staging: wgpu::Buffer,
    energy_groups: u32,
    /// Device-side probe gather, when the model only has point probes.
    probes: Option<ProbeBinds>,
    /// Device-side Mur ABC kernels, when the model has Mur faces.
    mur: Option<MurBinds>,
    /// Device-side lumped RLC kernels, when the model has lumped elements.
    lumped: Option<LumpedBinds>,
    /// Device-side conducting-sheet kernels.
    sheet: Option<SheetBinds>,
    /// Device-side TFSF kernels.
    tfsf: Option<TfsfBinds>,
    /// Device-side Debye dispersive kernels.
    debye: Option<DebyeBinds>,
    /// Device-side Lorentz dispersive kernels.
    lorentz: Option<LorentzBinds>,
    /// Compute passes recorded by the last `dispatch_steps` call.
    passes: std::cell::Cell<u64>,
    /// Reused little-endian upload scratch, so a host-phase upload does not
    /// allocate a new byte vector per field.
    upload_scratch: std::cell::RefCell<Vec<u8>>,
}

fn f32_buffer(device: &wgpu::Device, label: &str, values: &[FDTD_FLOAT],
              extra: wgpu::BufferUsages) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: &f32_bytes(values),
        usage: wgpu::BufferUsages::STORAGE | extra,
    })
}

impl GpuFields {
    pub fn new(
        context: &GpuContext,
        volt: &[FDTD_FLOAT],
        curr: &[FDTD_FLOAT],
        vv: &[FDTD_FLOAT],
        vi: &[FDTD_FLOAT],
        ii: &[FDTD_FLOAT],
        iv: &[FDTD_FLOAT],
        dims: [usize; 3],
    ) -> Option<Self> {
        let total = dims[0] * dims[1] * dims[2];
        if volt.len() != 3 * total
            || curr.len() != 3 * total
            || vv.len() != 3 * total
            || vi.len() != 3 * total
            || ii.len() != 3 * total
            || iv.len() != 3 * total
        {
            return None;
        }
        let device = context.device.clone();
        let queue = context.queue.clone();
        let volt_buf = f32_buffer(
            &device,
            "openems engine volt",
            volt,
            wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        );
        let curr_buf = f32_buffer(
            &device,
            "openems engine curr",
            curr,
            wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        );
        let vv_buf = f32_buffer(&device, "openems engine vv", vv, wgpu::BufferUsages::empty());
        let vi_buf = f32_buffer(&device, "openems engine vi", vi, wgpu::BufferUsages::empty());
        let ii_buf = f32_buffer(&device, "openems engine ii", ii, wgpu::BufferUsages::empty());
        let iv_buf = f32_buffer(&device, "openems engine iv", iv, wgpu::BufferUsages::empty());
        let uniform: [u32; 4] = [dims[0] as u32, dims[1] as u32, dims[2] as u32, 0];
        use wgpu::util::DeviceExt;
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("openems engine dims"),
            contents: &uniform.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("openems engine kernels"),
            source: wgpu::ShaderSource::Wgsl(UPDATE_SHADER.into()),
        });
        let entries = (0..7u32)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 6 {
                        wgpu::BufferBindingType::Uniform
                    } else if binding <= 1 {
                        wgpu::BufferBindingType::Storage { read_only: binding == 1 }
                    } else {
                        wgpu::BufferBindingType::Storage { read_only: true }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect::<Vec<_>>();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("openems engine layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("openems engine pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipe_e = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("openems engine update_e"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("update_e"),
            compilation_options: Default::default(),
            cache: None,
        });
        let pipe_h = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("openems engine update_h"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("update_h"),
            compilation_options: Default::default(),
            cache: None,
        });
        let make_bind = |target: &wgpu::Buffer, peer: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("openems engine bind group"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: target.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: peer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: vv_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: vi_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: ii_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: iv_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: uniform_buf.as_entire_binding() },
                ],
            })
        };
        let bind_e = make_bind(&volt_buf, &curr_buf);
        let bind_h = make_bind(&curr_buf, &volt_buf);
        let staging = |label: &str, len: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (len * 4) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        };
        let staging_volt = staging("openems engine volt readback", volt.len());
        let staging_curr = staging("openems engine curr readback", curr.len());
        let staging_pair = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("openems engine pair readback"),
            size: (2 * volt.len() * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        // Energy reduction: one partial (E and H) per 64-cell workgroup.
        let energy_groups = (volt.len() as u32).div_ceil(64).max(1);
        let energy_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("openems energy layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let energy_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("openems energy pipeline layout"),
                bind_group_layouts: &[Some(&energy_layout)],
                immediate_size: 0,
            });
        let energy_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("openems energy sum"),
            layout: Some(&energy_pipeline_layout),
            module: &shader,
            entry_point: Some("energy_sum"),
            compilation_options: Default::default(),
            cache: None,
        });
        let energy_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("openems energy dims"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let energy_parts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("openems energy parts"),
            size: (energy_groups as u64) * 2 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let energy_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("openems energy readback"),
            size: (energy_groups as u64) * 2 * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let energy_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("openems energy bind group"),
            layout: &energy_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: volt_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: curr_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: energy_parts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: energy_uniform.as_entire_binding(),
                },
            ],
        });
        Some(Self {
            device,
            queue,
            dims,
            volt: volt_buf,
            curr: curr_buf,
            bind_e,
            bind_h,
            pipe_e,
            pipe_h,
            staging_volt,
            staging_curr,
            staging_pair,
            source: None,
            upml: None,
            pml_engine: None,
            ec_vv: vv_buf,
            ec_vi: vi_buf,
            ec_ii: ii_buf,
            ec_iv: iv_buf,
            dims_uniform: uniform_buf,
            energy_pipe,
            energy_bind,
            energy_uniform,
            energy_parts,
            energy_staging,
            energy_groups,
            probes: None,
            mur: None,
            lumped: None,
            sheet: None,
            tfsf: None,
            debye: None,
            lorentz: None,
            passes: std::cell::Cell::new(0),
            upload_scratch: std::cell::RefCell::new(Vec::new()),
        })
    }

    /// Build the device-side excitation kernels from the host terms.
    ///
    /// Returns `true` when at least one side was created; the caller marks the
    /// extension as GPU-handled so its host phase becomes a no-op.
    pub fn setup_source(&mut self, terms: &SourceTerms) -> bool {
        if terms.is_empty() {
            return false;
        }
        let stride = terms.stride();
        let alignment = self
            .device
            .limits()
            .min_uniform_buffer_offset_alignment
            .max(16);
        let slot_size = 16u32.div_ceil(alignment) * alignment;
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("openems source kernels"),
            source: wgpu::ShaderSource::Wgsl(UPDATE_SHADER.into()),
        });
        let entries = (0..9u32)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 8 {
                        wgpu::BufferBindingType::Uniform
                    } else if binding == 7 {
                        wgpu::BufferBindingType::Storage { read_only: false }
                    } else {
                        wgpu::BufferBindingType::Storage { read_only: true }
                    },
                    has_dynamic_offset: binding == 8,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect::<Vec<_>>();
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("openems source layout"),
                entries: &entries,
            });
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("openems source pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("openems source apply"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("apply_source"),
            compilation_options: Default::default(),
            cache: None,
        });
        let mut make_side = |terms: &[SourceTerm],
                             values: &[Vec<f32>],
                             target: &wgpu::Buffer|
         -> Option<SourceSide> {
            if terms.is_empty() || values.is_empty() {
                return None;
            }
            let count = terms.len() as u32;
            let indices: Vec<u32> = terms.iter().map(|term| term.index).collect();
            let coeffs: Vec<f32> = terms.iter().map(|term| term.coeff).collect();
            let delays: Vec<u32> = terms.iter().map(|term| term.delay).collect();
            let periods: Vec<u32> = terms.iter().map(|term| term.period).collect();
            let slots: Vec<u32> = terms.iter().map(|term| term.slot).collect();
            let lengths: Vec<u32> = terms.iter().map(|term| term.length).collect();
            let mut samples = Vec::with_capacity(values.len() * stride as usize);
            for slot in values {
                samples.extend_from_slice(slot);
                samples.resize(samples.len() + (stride as usize - slot.len()), 0.0);
            }
            let buffer_u32 = |label: &str, data: &[u32]| {
                use wgpu::util::DeviceExt;
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(label),
                        contents: &u32_bytes(data),
                        usage: wgpu::BufferUsages::STORAGE,
                    })
            };
            let buffer_f32 = |label: &str, data: &[f32]| {
                use wgpu::util::DeviceExt;
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(label),
                        contents: &f32_bytes(data),
                        usage: wgpu::BufferUsages::STORAGE,
                    })
            };
            let index_buf = buffer_u32("openems source index", &indices);
            let coeff_buf = buffer_f32("openems source coeff", &coeffs);
            let delay_buf = buffer_u32("openems source delay", &delays);
            let period_buf = buffer_u32("openems source period", &periods);
            let slot_buf = buffer_u32("openems source slot", &slots);
            let length_buf = buffer_u32("openems source length", &lengths);
            let value_buf = buffer_f32("openems source values", &samples);
            let params = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("openems source params"),
                size: (SOURCE_BATCH * slot_size) as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("openems source bind group"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: index_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: coeff_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: delay_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: period_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: slot_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: length_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: value_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: target.as_entire_binding() },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &params,
                            offset: 0,
                            size: wgpu::BufferSize::new(16),
                        }),
                    },
                ],
            });
            Some(SourceSide {
                pipeline: pipeline.clone(),
                bind,
                count,
                params,
                scratch: std::cell::RefCell::new(Vec::new()),
            })
        };
        let volt = make_side(&terms.volt, &terms.volt_values, &self.volt);
        let curr = make_side(&terms.curr, &terms.curr_values, &self.curr);
        if volt.is_none() && curr.is_none() {
            return false;
        }
        self.source = Some(SourceBinds {
            volt,
            curr,
            slot_size,
            stride,
        });
        true
    }

    /// Whether a device-side excitation source is active.
    pub fn has_source(&self) -> bool {
        self.source.is_some()
    }

    /// Build the device-side UPML kernels from the host slabs.
    pub fn setup_upml(&mut self, upmls: &[&crate::extensions::EngineExtUPML]) -> bool {
        if upmls.is_empty() {
            return false;
        }
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("openems upml kernels"),
            source: wgpu::ShaderSource::Wgsl(UPDATE_SHADER.into()),
        });
        let entries = (0..12u32)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: if binding == 11 {
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    }
                } else {
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: (binding >= 2 && binding <= 7) || binding == 10,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    }
                },
                count: None,
            })
            .collect::<Vec<_>>();
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("openems upml layout"),
                entries: &entries,
            });
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("openems upml pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry: &str| {
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };
        let pre_volt = pipeline("upml_pre_volt");
        let post_volt = pipeline("upml_post_volt");
        let pre_curr = pipeline("upml_pre_curr");
        let post_curr = pipeline("upml_post_curr");
        // Concatenate every slab's arrays in slab order, and record where each
        // slab starts in the global field plus its own extent.
        let mut vv = Vec::new();
        let mut vvfo = Vec::new();
        let mut vvfn = Vec::new();
        let mut ii = Vec::new();
        let mut iifo = Vec::new();
        let mut iifn = Vec::new();
        let mut volt_flux = Vec::new();
        let mut curr_flux = Vec::new();
        let mut table = Vec::with_capacity(upmls.len() * 8);
        let mut total = 0usize;
        for upml in upmls {
            let slab = upml.num_lines;
            let mut append = |dst: &mut Vec<f32>, arrays: &[crate::array_nijk::FieldArray]| {
                for comp in 0..3 {
                    for i in 0..slab[0] {
                        for j in 0..slab[1] {
                            for k in 0..slab[2] {
                                dst.push(*arrays[comp].get(0, i, j, k));
                            }
                        }
                    }
                }
            };
            append(&mut vv, &upml.vv);
            append(&mut vvfo, &upml.vvfo);
            append(&mut vvfn, &upml.vvfn);
            append(&mut ii, &upml.ii);
            append(&mut iifo, &upml.iifo);
            append(&mut iifn, &upml.iifn);
            append(&mut volt_flux, &upml.volt_flux);
            append(&mut curr_flux, &upml.curr_flux);
            table.extend_from_slice(&[
                upml.start_pos[0] as u32,
                upml.start_pos[1] as u32,
                upml.start_pos[2] as u32,
                0,
                slab[0] as u32,
                slab[1] as u32,
                slab[2] as u32,
                0,
            ]);
            total += 3 * slab[0] * slab[1] * slab[2];
        }
        let buffer = |label: &str, data: &[f32]| {
            use wgpu::util::DeviceExt;
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: &f32_bytes(data),
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };
        let vv_buf = buffer("openems upml vv", &vv);
        let vvfo_buf = buffer("openems upml vvfo", &vvfo);
        let vvfn_buf = buffer("openems upml vvfn", &vvfn);
        let ii_buf = buffer("openems upml ii", &ii);
        let iifo_buf = buffer("openems upml iifo", &iifo);
        let iifn_buf = buffer("openems upml iifn", &iifn);
        let volt_flux_buf = buffer("openems upml volt flux", &volt_flux);
        let curr_flux_buf = buffer("openems upml curr flux", &curr_flux);
        let table_buf = {
            use wgpu::util::DeviceExt;
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("openems upml slabs"),
                    contents: &u32_bytes(&table),
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };
        let mut dims = Vec::<u8>::with_capacity(32);
        for value in [
            self.dims[0] as u32,
            self.dims[1] as u32,
            self.dims[2] as u32,
            0,
            upmls.len() as u32,
            total as u32,
            0,
            0,
        ] {
            dims.extend_from_slice(&value.to_le_bytes());
        }
        let uniform = {
            use wgpu::util::DeviceExt;
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("openems upml dims"),
                    contents: &dims,
                    usage: wgpu::BufferUsages::UNIFORM,
                })
        };
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("openems upml bind group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.volt.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.curr.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: vv_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: vvfo_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: vvfn_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: ii_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: iifo_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: iifn_buf.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: volt_flux_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: curr_flux_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry { binding: 10, resource: table_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: uniform.as_entire_binding() },
            ],
        });
        self.upml = Some(UpmlBinds {
            pre_volt,
            post_volt,
            pre_curr,
            post_curr,
            bind,
            groups: (total as u32).div_ceil(64).max(1),
            vvfn: vvfn_buf,
            volt_flux: volt_flux_buf,
            iifn: iifn_buf,
            curr_flux: curr_flux_buf,
            table: table_buf,
            dims: uniform,
            vv: vv_buf,
            vvfo: vvfo_buf,
            ii: ii_buf,
            iifo: iifo_buf,
        });
        true
    }

    /// Whether device-side UPML kernels are active.
    pub fn has_upml(&self) -> bool {
        self.upml.is_some()
    }

    /// Build the fused PML E/H kernels.
    ///
    /// The post phases only read and write their own cell, so they run at the
    /// end of the E/H thread that owns it; the pre phases stay separate
    /// dispatches because they need a global barrier before the update.
    pub fn setup_pml_engine(&mut self) -> bool {
        let Some(upml) = self.upml.as_ref() else {
            return false;
        };
        let vvfn = upml.vvfn.clone();
        let volt_flux = upml.volt_flux.clone();
        let iifn = upml.iifn.clone();
        let curr_flux = upml.curr_flux.clone();
        let table = upml.table.clone();
        let pml_dims = upml.dims.clone();
        let vv = upml.vv.clone();
        let vvfo = upml.vvfo.clone();
        let ii = upml.ii.clone();
        let iifo = upml.iifo.clone();
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("openems pml engine kernels"),
            source: wgpu::ShaderSource::Wgsl(UPDATE_SHADER.into()),
        });
        let mut entries: Vec<wgpu::BindGroupLayoutEntry> = (0..7u32)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 6 {
                        wgpu::BufferBindingType::Uniform
                    } else if binding <= 1 {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding == 1,
                        }
                    } else {
                        wgpu::BufferBindingType::Storage { read_only: true }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        for binding in 12..15u32 {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: binding != 13,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        for binding in 16..18u32 {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 15,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("openems pml engine layout"),
                entries: &entries,
            });
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("openems pml engine pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipe = |entry: &str| {
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };
        let pipe_e = pipe("update_e_pml");
        let pipe_h = pipe("update_h_pml");
        let (bind_e, bind_h) = {
            let make_bind = |field: &wgpu::Buffer,
                             peer: &wgpu::Buffer,
                             coeff: &wgpu::Buffer,
                             flux: &wgpu::Buffer,
                             pre_a: &wgpu::Buffer,
                             pre_b: &wgpu::Buffer| {
                self.device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("openems pml engine bind group"),
                        layout: &layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: field.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: peer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: self.ec_vv.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: self.ec_vi.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: self.ec_ii.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: self.ec_iv.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: self.dims_uniform.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 12,
                                resource: coeff.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 13,
                                resource: flux.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 14,
                                resource: table.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 15,
                                resource: pml_dims.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 16,
                                resource: pre_a.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 17,
                                resource: pre_b.as_entire_binding(),
                            },
                        ],
                    })
            };
            (
                make_bind(&self.volt, &self.curr, &vvfn, &volt_flux, &vv, &vvfo),
                make_bind(&self.curr, &self.volt, &iifn, &curr_flux, &ii, &iifo),
            )
        };
        self.pml_engine = Some(PmlEngine {
            pipe_e,
            pipe_h,
            bind_e,
            bind_h,
        });
        true
    }

    /// Whether the fused PML E/H kernels are active.
    pub fn has_pml_engine(&self) -> bool {
        self.pml_engine.is_some()
    }

    /// Build the device-side probe gather from the host probe plan.
    pub fn setup_probes(&mut self, points: &[crate::processing::GpuProbePoint]) -> bool {
        if points.is_empty() {
            return false;
        }
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("openems probe kernels"),
            source: wgpu::ShaderSource::Wgsl(UPDATE_SHADER.into()),
        });
        let entries = (0..7u32)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: if binding == 6 {
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(16),
                    }
                } else {
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: binding != 5,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    }
                },
                count: None,
            })
            .collect::<Vec<_>>();
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("openems probe layout"),
                entries: &entries,
            });
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("openems probe pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipe = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("openems probe gather"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("probe_gather"),
                compilation_options: Default::default(),
                cache: None,
            });
        let alignment = self
            .device
            .limits()
            .min_uniform_buffer_offset_alignment
            .max(16);
        let slot_size = 16u32.div_ceil(alignment) * alignment;
        let count = points.len() as u32;
        let plane = self.dims[0] * self.dims[1] * self.dims[2];
        let i_stride = self.dims[1] * self.dims[2];
        let j_stride = self.dims[2];
        let mut indices = Vec::with_capacity(points.len());
        let mut intervals = Vec::with_capacity(points.len());
        let mut buffers = Vec::with_capacity(points.len());
        for point in points {
            indices.push(
                (point.component * plane
                    + point.index[0] * i_stride
                    + point.index[1] * j_stride
                    + point.index[2]) as u32,
            );
            intervals.push(point.interval);
            buffers.push(point.buffer as u32);
        }
        let index_buf = {
            use wgpu::util::DeviceExt;
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("openems probe index"),
                    contents: &u32_bytes(&indices),
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };
        let interval_buf = {
            use wgpu::util::DeviceExt;
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("openems probe interval"),
                    contents: &u32_bytes(&intervals),
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };
        let buffer_buf = {
            use wgpu::util::DeviceExt;
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("openems probe buffer"),
                    contents: &u32_bytes(&buffers),
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };
        let params = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("openems probe params"),
            size: (SOURCE_BATCH * slot_size) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let values = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("openems probe values"),
            size: (count as u64) * (SOURCE_BATCH as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("openems probe readback"),
            size: (count as u64) * (SOURCE_BATCH as u64) * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("openems probe bind group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.volt.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.curr.as_entire_binding(),
                },
                wgpu::BindGroupEntry { binding: 2, resource: index_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: buffer_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: interval_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: values.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params,
                        offset: 0,
                        size: wgpu::BufferSize::new(16),
                    }),
                },
            ],
        });
        self.probes = Some(ProbeBinds {
            pipe,
            bind,
            params,
            values,
            staging,
            count,
            slot_size,
            scratch: std::cell::RefCell::new(Vec::new()),
        });
        true
    }

    /// Whether device-side probe gathering is active.
    pub fn has_probes(&self) -> bool {
        self.probes.is_some()
    }

    /// Download the per-step probe matrix for `steps` steps.
    pub fn download_probe_values(&self, steps: u32) -> Option<Vec<f32>> {
        let probes = self.probes.as_ref()?;
        let mut values = vec![0.0f32; probes.count as usize * steps as usize];
        readback_into(
            &self.device,
            &self.queue,
            &probes.values,
            &probes.staging,
            &mut values,
        )?;
        Some(values)
    }

    /// Build the device-side Mur kernels from the host faces.
    pub fn setup_mur(&mut self, mur: &[&crate::extensions::EngineExtMurABC]) -> bool {
        if mur.is_empty() {
            return false;
        }
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("openems mur kernels"),
            source: wgpu::ShaderSource::Wgsl(UPDATE_SHADER.into()),
        });
        let entries = (0..7u32)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: if binding == 6 {
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(32),
                    }
                } else {
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: binding == 1 || binding == 2 || binding == 5,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    }
                },
                count: None,
            })
            .collect::<Vec<_>>();
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("openems mur layout"),
                entries: &entries,
            });
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("openems mur pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry: &str| {
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };
        let pre = pipeline("mur_pre");
        let post = pipeline("mur_post");
        let apply = pipeline("mur_apply");
        let faces: Vec<&crate::extensions::MurFace> = mur
            .iter()
            .flat_map(|extension| extension.faces.iter())
            .collect();
        if faces.is_empty() {
            return false;
        }
        // Extent of each face along the absolute axes (its own axis is fixed).
        let extents: Vec<[usize; 3]> = faces
            .iter()
            .map(|face| {
                let mut extent = [0usize; 3];
                extent[(face.dir + 1) % 3] = face.dims_p;
                extent[(face.dir + 2) % 3] = face.dims_pp;
                extent
            })
            .collect();
        // Two faces conflict when their planes intersect inside both extents;
        // they then touch the same corner edges and must keep the host order.
        let conflicts = |a: usize, b: usize| -> bool {
            let (fa, fb) = (faces[a], faces[b]);
            if fa.dir == fb.dir {
                return fa.line == fb.line;
            }
            fb.line < extents[a][fb.dir] && fa.line < extents[b][fa.dir]
        };
        // Colour = 1 + the highest colour of an earlier conflicting face, so a
        // later colour never runs before an earlier one on a shared edge and
        // faces of one colour never overlap.
        let mut colour = vec![0usize; faces.len()];
        for i in 0..faces.len() {
            let mut value = 0usize;
            for j in 0..i {
                if conflicts(j, i) {
                    value = value.max(colour[j] + 1);
                }
            }
            colour[i] = value;
        }
        let colour_count = colour.iter().copied().max().map_or(0, |c| c + 1);
        let mut order: Vec<usize> = (0..faces.len()).collect();
        order.sort_by_key(|&index| colour[index]);
        // Concatenate in colour order, so each colour owns a contiguous range.
        let mut coeff_p = Vec::new();
        let mut coeff_pp = Vec::new();
        let mut hist_p = Vec::new();
        let mut hist_pp = Vec::new();
        let mut table = Vec::with_capacity(faces.len() * 12);
        let mut offsets = vec![0usize; faces.len()];
        for &index in &order {
            let face = faces[index];
            offsets[index] = coeff_p.len();
            coeff_p.extend_from_slice(&face.coeff_p);
            coeff_pp.extend_from_slice(&face.coeff_pp);
            hist_p.extend_from_slice(&face.hist_p);
            hist_pp.extend_from_slice(&face.hist_pp);
            table.extend_from_slice(&[
                offsets[index] as u32,
                face.dims_p as u32,
                face.dims_pp as u32,
                face.line as u32,
                face.shift as u32,
                face.dir as u32,
                ((face.dir + 1) % 3) as u32,
                ((face.dir + 2) % 3) as u32,
                colour[index] as u32,
                0,
                0,
                0,
            ]);
        }
        // One uniform slot per colour: global dims + element range + face count.
        let alignment = self
            .device
            .limits()
            .min_uniform_buffer_offset_alignment
            .max(32);
        let slot_size = 32u32.div_ceil(alignment) * alignment;
        let mut uniforms = vec![0u8; colour_count.max(1) * slot_size as usize];
        let mut passes = Vec::with_capacity(colour_count);
        for c in 0..colour_count {
            let members: Vec<usize> = order
                .iter()
                .copied()
                .filter(|&index| colour[index] == c)
                .collect();
            let Some(&first) = members.first() else {
                continue;
            };
            let last = *members.last().expect("members is not empty");
            let start = offsets[first];
            let end = offsets[last] + faces[last].dims_p * faces[last].dims_pp;
            let base = c * slot_size as usize;
            let words = [
                self.dims[0] as u32,
                self.dims[1] as u32,
                self.dims[2] as u32,
                0,
                start as u32,
                end as u32,
                faces.len() as u32,
                0,
            ];
            for (n, word) in words.iter().enumerate() {
                uniforms[base + n * 4..base + n * 4 + 4].copy_from_slice(&word.to_le_bytes());
            }
            passes.push(MurPass {
                offset: (c as u32) * slot_size,
                groups: ((end - start) as u32).div_ceil(64).max(1),
            });
        }
        let buffer = |label: &str, data: &[FDTD_FLOAT]| {
            use wgpu::util::DeviceExt;
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: &f32_bytes(data),
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };
        let coeff_p_buf = buffer("openems mur coeff p", &coeff_p);
        let coeff_pp_buf = buffer("openems mur coeff pp", &coeff_pp);
        let hist_p_buf = buffer("openems mur hist p", &hist_p);
        let hist_pp_buf = buffer("openems mur hist pp", &hist_pp);
        let table_buf = {
            use wgpu::util::DeviceExt;
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("openems mur table"),
                    contents: &u32_bytes(&table),
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };
        let uniform_buf = {
            use wgpu::util::DeviceExt;
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("openems mur dims"),
                    contents: &uniforms,
                    usage: wgpu::BufferUsages::UNIFORM,
                })
        };
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("openems mur bind group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.volt.as_entire_binding(),
                },
                wgpu::BindGroupEntry { binding: 1, resource: coeff_p_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: coeff_pp_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: hist_p_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: hist_pp_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: table_buf.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &uniform_buf,
                        offset: 0,
                        size: wgpu::BufferSize::new(32),
                    }),
                },
            ],
        });
        self.mur = Some(MurBinds {
            pre,
            post,
            apply,
            bind,
            passes,
        });
        true
    }

    /// Whether device-side Mur kernels are active.
    pub fn has_mur(&self) -> bool {
        self.mur.is_some()
    }

    /// Build the device-side lumped RLC kernels from the host edges.
    pub fn setup_lumped(&mut self, lumped: &[&crate::extensions::EngineExtLumpedRLC]) -> bool {
        let count: usize = lumped.iter().map(|extension| extension.num_positions).sum();
        if count == 0 {
            return false;
        }
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("openems lumped kernels"),
            source: wgpu::ShaderSource::Wgsl(UPDATE_SHADER.into()),
        });
        let entries = (0..7u32)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: if binding == 6 {
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    }
                } else {
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: binding == 1 || binding == 2,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    }
                },
                count: None,
            })
            .collect::<Vec<_>>();
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("openems lumped layout"),
                entries: &entries,
            });
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("openems lumped pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry: &str| {
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };
        let pre = pipeline("lumped_pre");
        let apply = pipeline("lumped_apply");
        let dims = self.dims;
        let (i_stride, j_stride, plane) = (
            dims[1] * dims[2],
            dims[2],
            dims[0] * dims[1] * dims[2],
        );
        let mut indices = Vec::with_capacity(count * 4);
        let mut coefficients = Vec::with_capacity(count * 9);
        let mut hist = Vec::with_capacity(count * 3);
        let mut jn = Vec::with_capacity(count * 3);
        let mut il = Vec::with_capacity(count);
        for extension in lumped {
            for pos in 0..extension.num_positions {
                let [i, j, k] = extension.positions[pos];
                let coeff = &extension.coeffs[pos];
                let index = (coeff.direction.min(2) * plane + i * i_stride + j * j_stride + k)
                    as u32;
                indices.push(index);
                coefficients.extend_from_slice(&[
                    coeff.vvd,
                    coeff.vv2,
                    coeff.vj1,
                    coeff.vj2,
                    coeff.ib0,
                    coeff.b1,
                    coeff.b2,
                    coeff.ilv,
                    coeff.i2v,
                ]);
                hist.extend_from_slice(&extension.v_vdn[pos]);
                jn.extend_from_slice(&extension.v_jn[pos]);
                il.push(extension.v_il[pos]);
            }
        }
        let buffer_u32 = |label: &str, data: &[u32]| {
            use wgpu::util::DeviceExt;
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: &u32_bytes(data),
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };
        let buffer_f32 = |label: &str, data: &[f32]| {
            use wgpu::util::DeviceExt;
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: &f32_bytes(data),
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };
        let index_buf = buffer_u32("openems lumped index", &indices);
        let coeff_buf = buffer_f32("openems lumped coeff", &coefficients);
        let hist_buf = buffer_f32("openems lumped hist", &hist);
        let jn_buf = buffer_f32("openems lumped jn", &jn);
        let il_buf = buffer_f32("openems lumped il", &il);
        let uniform = {
            use wgpu::util::DeviceExt;
            let mut data = Vec::with_capacity(16);
            for value in [count as u32, 0, 0, 0] {
                data.extend_from_slice(&value.to_le_bytes());
            }
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("openems lumped dims"),
                    contents: &data,
                    usage: wgpu::BufferUsages::UNIFORM,
                })
        };
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("openems lumped bind group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.volt.as_entire_binding(),
                },
                wgpu::BindGroupEntry { binding: 1, resource: index_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: coeff_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: hist_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: jn_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: il_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: uniform.as_entire_binding() },
            ],
        });
        self.lumped = Some(LumpedBinds {
            pre,
            apply,
            bind,
            count: count as u32,
            groups: (count as u32).div_ceil(64).max(1),
        });
        true
    }

    /// Whether device-side lumped RLC kernels are active.
    pub fn has_lumped(&self) -> bool {
        self.lumped.is_some()
    }

    /// Build the device-side conducting-sheet kernels from the host cells.
    pub fn setup_sheet(
        &mut self,
        sheets: &[&crate::extensions::EngineExtConductingSheet],
    ) -> bool {
        let count: usize = sheets.iter().map(|sheet| sheet.num_positions).sum();
        if count == 0 {
            return false;
        }
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("openems sheet kernels"),
            source: wgpu::ShaderSource::Wgsl(UPDATE_SHADER.into()),
        });
        let entries = (0..6u32)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: if binding == 5 {
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    }
                } else {
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: binding == 1 || binding == 2 || binding == 3,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    }
                },
                count: None,
            })
            .collect::<Vec<_>>();
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("openems sheet layout"),
                entries: &entries,
            });
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("openems sheet pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry: &str| {
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };
        let pre = pipeline("sheet_pre");
        let apply = pipeline("sheet_apply");
        let dims = self.dims;
        let (i_stride, j_stride, plane) = (
            dims[1] * dims[2],
            dims[2],
            dims[0] * dims[1] * dims[2],
        );
        let mut positions = Vec::with_capacity(count);
        let mut int_coeffs = Vec::with_capacity(6 * count);
        let mut ext_coeffs = Vec::with_capacity(6 * count);
        let mut state = Vec::with_capacity(6 * count);
        for sheet in sheets {
            // The kernels work on the component-0 base index; `n * plane` is
            // added inside the shader.
            for pos in 0..sheet.num_positions {
                let [i, j, k] = sheet.positions[pos];
                positions.push((i * i_stride + j * j_stride + k) as u32);
            }
        }
        // Global (order, component, position) layout, so several sheet
        // extensions concatenate into one matrix.
        for order in 0..2 {
            for component in 0..3 {
                for sheet in sheets {
                    int_coeffs.extend_from_slice(&sheet.v_int_ade[order][component]);
                }
            }
        }
        for order in 0..2 {
            for component in 0..3 {
                for sheet in sheets {
                    ext_coeffs.extend_from_slice(&sheet.v_ext_ade[order][component]);
                }
            }
        }
        for order in 0..2 {
            for component in 0..3 {
                for sheet in sheets {
                    state.extend_from_slice(&sheet.volt_ade[order][component]);
                }
            }
        }
        let buffer_u32 = |label: &str, data: &[u32]| {
            use wgpu::util::DeviceExt;
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: &u32_bytes(data),
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };
        let buffer_f32 = |label: &str, data: &[f32]| {
            use wgpu::util::DeviceExt;
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: &f32_bytes(data),
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };
        let pos_buf = buffer_u32("openems sheet pos", &positions);
        let int_buf = buffer_f32("openems sheet int", &int_coeffs);
        let ext_buf = buffer_f32("openems sheet ext", &ext_coeffs);
        let state_buf = buffer_f32("openems sheet state", &state);
        let uniform = {
            use wgpu::util::DeviceExt;
            let mut data = Vec::with_capacity(32);
            for value in [count as u32, 0, 0, 0] {
                data.extend_from_slice(&value.to_le_bytes());
            }
            for value in [plane as u32, 0, 0, 0] {
                data.extend_from_slice(&value.to_le_bytes());
            }
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("openems sheet dims"),
                    contents: &data,
                    usage: wgpu::BufferUsages::UNIFORM,
                })
        };
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("openems sheet bind group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.volt.as_entire_binding(),
                },
                wgpu::BindGroupEntry { binding: 1, resource: pos_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: int_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: ext_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: state_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: uniform.as_entire_binding() },
            ],
        });
        self.sheet = Some(SheetBinds {
            pre,
            apply,
            bind,
            groups: (count as u32).div_ceil(64).max(1),
        });
        true
    }

    /// Whether device-side conducting-sheet kernels are active.
    pub fn has_sheet(&self) -> bool {
        self.sheet.is_some()
    }

    /// Build the device-side TFSF kernels from the host tables.
    pub fn setup_tfsf(&mut self, tfsfs: &[&crate::extensions::EngineExtTFSF]) -> bool {
        if tfsfs.is_empty() {
            return false;
        }
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("openems tfsf kernels"),
            source: wgpu::ShaderSource::Wgsl(UPDATE_SHADER.into()),
        });
        let alignment = self
            .device
            .limits()
            .min_uniform_buffer_offset_alignment
            .max(16);
        let slot_size = 16u32.div_ceil(alignment) * alignment;
        let entries = (0..7u32)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: if binding == 6 {
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    }
                } else {
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: binding >= 1 && binding != 0,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    }
                },
                count: None,
            })
            .collect::<Vec<_>>();
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("openems tfsf layout"),
                entries: &entries,
            });
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("openems tfsf pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("openems tfsf apply"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("tfsf_apply"),
            compilation_options: Default::default(),
            cache: None,
        });
        let dims = self.dims;
        let mut make_side = |extension: &crate::extensions::EngineExtTFSF,
                             volt: bool,
                             direction: usize|
         -> Option<TfsfSide> {
            let (indices, amps, delays, deltas) = tfsf_terms(extension, volt, dims, direction);
            if indices.is_empty() {
                return None;
            }
            let count = indices.len() as u32;
            let length = extension.signal.signal_length();
            let samples: Vec<f32> = (0..length as usize)
                .map(|index| {
                    if volt {
                        extension.signal.current(index)
                    } else {
                        extension.signal.voltage(index)
                    }
                })
                .collect();
            let buffer_u32 = |label: &str, data: &[u32]| {
                use wgpu::util::DeviceExt;
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(label),
                        contents: &u32_bytes(data),
                        usage: wgpu::BufferUsages::STORAGE,
                    })
            };
            let buffer_f32 = |label: &str, data: &[f32]| {
                use wgpu::util::DeviceExt;
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(label),
                        contents: &f32_bytes(data),
                        usage: wgpu::BufferUsages::STORAGE,
                    })
            };
            let index_buf = buffer_u32("openems tfsf index", &indices);
            let amp_buf = buffer_f32("openems tfsf amp", &amps);
            let delay_buf = buffer_u32("openems tfsf delay", &delays);
            let delta_buf = buffer_f32("openems tfsf delta", &deltas);
            let sample_buf = buffer_f32("openems tfsf samples", &samples);
            let params = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("openems tfsf params"),
                size: (SOURCE_BATCH * slot_size) as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("openems tfsf bind group"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: if volt {
                            self.volt.as_entire_binding()
                        } else {
                            self.curr.as_entire_binding()
                        },
                    },
                    wgpu::BindGroupEntry { binding: 1, resource: index_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: amp_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: delay_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: delta_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: sample_buf.as_entire_binding() },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &params,
                            offset: 0,
                            size: wgpu::BufferSize::new(16),
                        }),
                    },
                ],
            });
            Some(TfsfSide {
                bind,
                groups: count.div_ceil(64).max(1),
                params,
                scratch: std::cell::RefCell::new(Vec::new()),
                length,
                period: extension.signal.signal_period(),
                max_delay: extension.max_delay,
            })
        };
        let mut volt = Vec::new();
        let mut curr = Vec::new();
        for extension in tfsfs {
            // One pass per direction, in the host's n-outer order: corners are
            // shared between directions and a single dispatch would race.
            for direction in 0..3 {
                if let Some(side) = make_side(extension, true, direction) {
                    volt.push(side);
                }
                if let Some(side) = make_side(extension, false, direction) {
                    curr.push(side);
                }
            }
        }
        if volt.is_empty() && curr.is_empty() {
            return false;
        }
        self.tfsf = Some(TfsfBinds {
            pipeline,
            volt,
            curr,
            slot_size,
        });
        true
    }

    /// Whether device-side TFSF kernels are active.
    pub fn has_tfsf(&self) -> bool {
        self.tfsf.is_some()
    }

    /// Build the device-side Debye kernels from the host state.
    pub fn setup_debye(
        &mut self,
        debye: &[&crate::extensions::EngineExtDebyeMaterial],
    ) -> bool {
        let mut instances = Vec::new();
        if debye.is_empty() {
            return false;
        }
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("openems debye kernels"),
            source: wgpu::ShaderSource::Wgsl(UPDATE_SHADER.into()),
        });
        let entries = (0..9u32)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: if binding == 8 {
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    }
                } else {
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: binding >= 1 && binding <= 4,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    }
                },
                count: None,
            })
            .collect::<Vec<_>>();
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("openems debye layout"),
                entries: &entries,
            });
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("openems debye pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry: &str| {
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };
        let pre = pipeline("debye_pre");
        let post = pipeline("debye_post");
        let apply = pipeline("debye_apply");
        let dims = self.dims;
        let (i_stride, j_stride, plane) = (
            dims[1] * dims[2],
            dims[2],
            dims[0] * dims[1] * dims[2],
        );
        for extension in debye {
            let count = extension.num_positions;
            if count == 0 {
                continue;
            }
            let poles = extension.num_poles;
            let mut positions = Vec::with_capacity(count);
            for pos in 0..count {
                let [i, j, k] = extension.positions[pos];
                positions.push((i * i_stride + j * j_stride + k) as u32);
            }
            let mut relax = Vec::with_capacity(poles * 3 * count);
            let mut drive = Vec::with_capacity(poles * 3 * count);
            let mut poles_state = Vec::with_capacity(poles * 3 * count);
            for pole in 0..poles {
                for component in 0..3 {
                    relax.extend_from_slice(&extension.v_relax_ade[pole][component]);
                    drive.extend_from_slice(&extension.v_drive_ade[pole][component]);
                    poles_state.extend_from_slice(&extension.volt_pole_ade[pole][component]);
                }
            }
            let mut solve = Vec::with_capacity(3 * count);
            let mut pre_state = Vec::with_capacity(3 * count);
            let mut ade = Vec::with_capacity(3 * count);
            for component in 0..3 {
                solve.extend_from_slice(&extension.v_solve_ade[component]);
                pre_state.extend_from_slice(&extension.volt_pre_ade[component]);
                ade.extend_from_slice(&extension.volt_ade[component]);
            }
            let buffer_u32 = |label: &str, data: &[u32]| {
                use wgpu::util::DeviceExt;
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(label),
                        contents: &u32_bytes(data),
                        usage: wgpu::BufferUsages::STORAGE,
                    })
            };
            let buffer_f32 = |label: &str, data: &[f32]| {
                use wgpu::util::DeviceExt;
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(label),
                        contents: &f32_bytes(data),
                        usage: wgpu::BufferUsages::STORAGE,
                    })
            };
            let pos_buf = buffer_u32("openems debye pos", &positions);
            let relax_buf = buffer_f32("openems debye relax", &relax);
            let drive_buf = buffer_f32("openems debye drive", &drive);
            let solve_buf = buffer_f32("openems debye solve", &solve);
            let pole_buf = buffer_f32("openems debye pole", &poles_state);
            let pre_buf = buffer_f32("openems debye pre", &pre_state);
            let ade_buf = buffer_f32("openems debye ade", &ade);
            let uniform = {
                use wgpu::util::DeviceExt;
                let mut data = Vec::with_capacity(48);
                for value in [count as u32, 0, 0, 0] {
                    data.extend_from_slice(&value.to_le_bytes());
                }
                for value in [poles as u32, 0, 0, 0] {
                    data.extend_from_slice(&value.to_le_bytes());
                }
                for value in [plane as u32, 0, 0, 0] {
                    data.extend_from_slice(&value.to_le_bytes());
                }
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("openems debye dims"),
                        contents: &data,
                        usage: wgpu::BufferUsages::UNIFORM,
                    })
            };
            let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("openems debye bind group"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.volt.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry { binding: 1, resource: pos_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: relax_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: drive_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: solve_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: pole_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: pre_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: ade_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: uniform.as_entire_binding() },
                ],
            });
            instances.push(DebyeInstance {
                bind,
                groups: (count as u32).div_ceil(64).max(1),
            });
        }
        if instances.is_empty() {
            return false;
        }
        self.debye = Some(DebyeBinds {
            pre,
            post,
            apply,
            instances,
        });
        true
    }

    /// Whether device-side Debye kernels are active.
    pub fn has_debye(&self) -> bool {
        self.debye.is_some()
    }

    /// Build the device-side Lorentz kernels from the host state.
    pub fn setup_lorentz(
        &mut self,
        lorentz: &[&crate::extensions::EngineExtLorentzMaterial],
    ) -> bool {
        if lorentz.is_empty() {
            return false;
        }
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("openems lorentz kernels"),
            source: wgpu::ShaderSource::Wgsl(UPDATE_SHADER.into()),
        });
        let entries = (0..15u32)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: if binding == 14 {
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    }
                } else {
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: binding >= 2 && binding <= 9,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    }
                },
                count: None,
            })
            .collect::<Vec<_>>();
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("openems lorentz layout"),
                entries: &entries,
            });
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("openems lorentz pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry: &str| {
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };
        let pre_volt = pipeline("lorentz_pre_volt");
        let pre_curr = pipeline("lorentz_pre_curr");
        let apply_volt = pipeline("lorentz_apply_volt");
        let apply_curr = pipeline("lorentz_apply_curr");
        let dims = self.dims;
        let (i_stride, j_stride, plane) = (
            dims[1] * dims[2],
            dims[2],
            dims[0] * dims[1] * dims[2],
        );
        let mut instances = Vec::new();
        for extension in lorentz {
            let count = extension.num_positions;
            if count == 0 {
                continue;
            }
            let orders = extension.num_orders;
            let mut positions = Vec::with_capacity(count);
            for pos in 0..count {
                let [i, j, k] = extension.positions[pos];
                positions.push((i * i_stride + j * j_stride + k) as u32);
            }
            let mut active = Vec::with_capacity(orders * 2);
            for order in 0..orders {
                active.push(u32::from(extension.volt_ade_on[order]));
                active.push(u32::from(extension.curr_ade_on[order]));
            }
            let mut v_int = Vec::with_capacity(orders * 3 * count);
            let mut v_ext = Vec::with_capacity(orders * 3 * count);
            let mut v_lor = Vec::with_capacity(orders * 3 * count);
            let mut i_int = Vec::with_capacity(orders * 3 * count);
            let mut i_ext = Vec::with_capacity(orders * 3 * count);
            let mut i_lor = Vec::with_capacity(orders * 3 * count);
            let mut volt_ade = Vec::with_capacity(orders * 3 * count);
            let mut volt_lor = Vec::with_capacity(orders * 3 * count);
            let mut curr_ade = Vec::with_capacity(orders * 3 * count);
            let mut curr_lor = Vec::with_capacity(orders * 3 * count);
            for order in 0..orders {
                for component in 0..3 {
                    v_int.extend_from_slice(&extension.v_int_ade[order][component]);
                    v_ext.extend_from_slice(&extension.v_ext_ade[order][component]);
                    v_lor.extend_from_slice(&extension.v_lor_ade[order][component]);
                    i_int.extend_from_slice(&extension.i_int_ade[order][component]);
                    i_ext.extend_from_slice(&extension.i_ext_ade[order][component]);
                    i_lor.extend_from_slice(&extension.i_lor_ade[order][component]);
                    volt_ade.extend_from_slice(&extension.volt_ade[order][component]);
                    volt_lor.extend_from_slice(&extension.volt_lor_ade[order][component]);
                    curr_ade.extend_from_slice(&extension.curr_ade[order][component]);
                    curr_lor.extend_from_slice(&extension.curr_lor_ade[order][component]);
                }
            }
            let buffer_u32 = |label: &str, data: &[u32]| {
                use wgpu::util::DeviceExt;
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(label),
                        contents: &u32_bytes(data),
                        usage: wgpu::BufferUsages::STORAGE,
                    })
            };
            let buffer_f32 = |label: &str, data: &[f32]| {
                use wgpu::util::DeviceExt;
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(label),
                        contents: &f32_bytes(data),
                        usage: wgpu::BufferUsages::STORAGE,
                    })
            };
            let pos_buf = buffer_u32("openems lorentz pos", &positions);
            let active_buf = buffer_u32("openems lorentz active", &active);
            let v_int_buf = buffer_f32("openems lorentz v int", &v_int);
            let v_ext_buf = buffer_f32("openems lorentz v ext", &v_ext);
            let v_lor_buf = buffer_f32("openems lorentz v lor", &v_lor);
            let i_int_buf = buffer_f32("openems lorentz i int", &i_int);
            let i_ext_buf = buffer_f32("openems lorentz i ext", &i_ext);
            let i_lor_buf = buffer_f32("openems lorentz i lor", &i_lor);
            let volt_ade_buf = buffer_f32("openems lorentz volt ade", &volt_ade);
            let volt_lor_buf = buffer_f32("openems lorentz volt lor", &volt_lor);
            let curr_ade_buf = buffer_f32("openems lorentz curr ade", &curr_ade);
            let curr_lor_buf = buffer_f32("openems lorentz curr lor", &curr_lor);
            let uniform = {
                use wgpu::util::DeviceExt;
                let mut data = Vec::with_capacity(48);
                for value in [count as u32, 0, 0, 0] {
                    data.extend_from_slice(&value.to_le_bytes());
                }
                for value in [orders as u32, 0, 0, 0] {
                    data.extend_from_slice(&value.to_le_bytes());
                }
                for value in [plane as u32, 0, 0, 0] {
                    data.extend_from_slice(&value.to_le_bytes());
                }
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("openems lorentz dims"),
                        contents: &data,
                        usage: wgpu::BufferUsages::UNIFORM,
                    })
            };
            let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("openems lorentz bind group"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.volt.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.curr.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry { binding: 2, resource: pos_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: active_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: v_int_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: v_ext_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: v_lor_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: i_int_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: i_ext_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 9, resource: i_lor_buf.as_entire_binding() },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: volt_ade_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: volt_lor_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 12,
                        resource: curr_ade_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 13,
                        resource: curr_lor_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry { binding: 14, resource: uniform.as_entire_binding() },
                ],
            });
            instances.push(LorentzInstance {
                bind,
                groups: (count as u32).div_ceil(64).max(1),
            });
        }
        if instances.is_empty() {
            return false;
        }
        self.lorentz = Some(LorentzBinds {
            pre_volt,
            pre_curr,
            apply_volt,
            apply_curr,
            instances,
        });
        true
    }

    /// Whether device-side Lorentz kernels are active.
    pub fn has_lorentz(&self) -> bool {
        self.lorentz.is_some()
    }

    /// Apply the device-side voltage source for one step (host path).
    pub fn dispatch_source_volt(&self, ts: u32) {
        self.dispatch_source(true, ts);
    }

    /// Apply the device-side current source for one step (host path).
    pub fn dispatch_source_curr(&self, ts: u32) {
        self.dispatch_source(false, ts);
    }

    fn dispatch_source(&self, volt: bool, ts: u32) {
        let Some(source) = self.source.as_ref() else {
            return;
        };
        let side = if volt {
            source.volt.as_ref()
        } else {
            source.curr.as_ref()
        };
        let Some(side) = side else {
            return;
        };
        SourceBinds::write_params(&self.queue, side, source.slot_size, source.stride, ts, 1);
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("openems source step"),
        });
        if volt {
            source.record(&mut encoder, source.volt.as_ref(), 0);
        } else {
            source.record(&mut encoder, source.curr.as_ref(), 0);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    pub fn dispatch_e(&self) {
        let total = (self.dims[0] * self.dims[1] * self.dims[2]) as u32;
        self.dispatch(&self.pipe_e, &self.bind_e, total);
    }

    pub fn dispatch_h(&self) {
        let cells = ((self.dims[0] - 1) * (self.dims[1] - 1) * (self.dims[2] - 1)) as u32;
        self.dispatch(&self.pipe_h, &self.bind_h, cells);
    }

    /// Run `steps` complete E/H steps in one submission.
    ///
    /// The earlier path submitted one command buffer per half step, so a batch
    /// of N steps paid 2N queue submissions plus the per-step host sync. Here
    /// the passes are recorded back to back and submitted once per chunk, and
    /// the fields stay on the device between steps. A device-side source is
    /// interleaved after the E and H kernels, matching the host order
    /// (`UpdateVoltages` -> `Apply2Voltages` -> `UpdateCurrents` ->
    /// `Apply2Current`).
    pub fn dispatch_steps(&self, steps: u32, start_ts: u32) {
        // Bound the encoder size; a chunk is still thousands of kernel
        // launches, far fewer submissions than before. The per-step uniforms
        // are sized for `SOURCE_BATCH`, so the chunk never exceeds it.
        let chunk_limit = std::env::var("OPENEMS_GPU_CHUNK")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .filter(|value| *value > 0 && *value <= SOURCE_BATCH)
            .unwrap_or(SOURCE_BATCH);
        let volt_cells = (self.dims[0] * self.dims[1] * self.dims[2]) as u32;
        let curr_cells = (self.dims[0].saturating_sub(1)
            * self.dims[1].saturating_sub(1)
            * self.dims[2].saturating_sub(1)) as u32;
        let mut done = 0u32;
        let mut passes = 0u32;
        while done < steps {
            let chunk = (steps - done).min(chunk_limit);
            if let Some(source) = self.source.as_ref() {
                if let Some(side) = source.volt.as_ref() {
                    SourceBinds::write_params(
                        &self.queue,
                        side,
                        source.slot_size,
                        source.stride,
                        start_ts + done,
                        chunk,
                    );
                }
                if let Some(side) = source.curr.as_ref() {
                    SourceBinds::write_params(
                        &self.queue,
                        side,
                        source.slot_size,
                        source.stride,
                        start_ts + done,
                        chunk,
                    );
                }
            }
            if let Some(tfsf) = self.tfsf.as_ref() {
                for side in tfsf.volt.iter().chain(tfsf.curr.iter()) {
                    tfsf.write_params(&self.queue, side, start_ts + done, chunk);
                }
            }
            self.write_probe_params(start_ts + done, chunk);
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("openems engine batch"),
            });
            for step in 0..chunk {
                if let Some(mur) = self.mur.as_ref() {
                    passes += mur.record(&mut encoder, 0);
                }
                if let Some(lumped) = self.lumped.as_ref() {
                    passes += lumped.record(&mut encoder, true);
                }
                if let Some(sheet) = self.sheet.as_ref() {
                    passes += sheet.record(&mut encoder, true);
                }
                if let Some(debye) = self.debye.as_ref() {
                    passes += debye.record(&mut encoder, 0);
                }
                if let Some(lorentz) = self.lorentz.as_ref() {
                    passes += lorentz.record(&mut encoder, 0);
                }
                if self.pml_engine.is_none() {
                    if let Some(upml) = self.upml.as_ref() {
                        passes += upml.record(&mut encoder, 0);
                    }
                }
                if let Some(pml) = self.pml_engine.as_ref() {
                    record_dispatch(&mut encoder, &pml.pipe_e, &pml.bind_e, volt_cells);
                } else {
                    record_dispatch(&mut encoder, &self.pipe_e, &self.bind_e, volt_cells);
                }
                passes += 1;
                if self.pml_engine.is_none() {
                    if let Some(upml) = self.upml.as_ref() {
                        passes += upml.record(&mut encoder, 1);
                    }
                }
                if let Some(debye) = self.debye.as_ref() {
                    passes += debye.record(&mut encoder, 1);
                }
                if let Some(tfsf) = self.tfsf.as_ref() {
                    for side in &tfsf.volt {
                        passes += tfsf.record(&mut encoder, side, step);
                    }
                }
                if let Some(mur) = self.mur.as_ref() {
                    passes += mur.record(&mut encoder, 1);
                }
                if let Some(sheet) = self.sheet.as_ref() {
                    passes += sheet.record(&mut encoder, false);
                }
                if let Some(debye) = self.debye.as_ref() {
                    passes += debye.record(&mut encoder, 2);
                }
                if let Some(lorentz) = self.lorentz.as_ref() {
                    passes += lorentz.record(&mut encoder, 2);
                }
                if let Some(lumped) = self.lumped.as_ref() {
                    passes += lumped.record(&mut encoder, false);
                }
                if let Some(mur) = self.mur.as_ref() {
                    passes += mur.record(&mut encoder, 2);
                }
                if let Some(source) = self.source.as_ref() {
                    passes += source.record(&mut encoder, source.volt.as_ref(), step);
                }
                if self.pml_engine.is_none() {
                    if let Some(upml) = self.upml.as_ref() {
                        passes += upml.record(&mut encoder, 2);
                    }
                }
                if let Some(lorentz) = self.lorentz.as_ref() {
                    passes += lorentz.record(&mut encoder, 1);
                }
                if let Some(pml) = self.pml_engine.as_ref() {
                    record_dispatch(&mut encoder, &pml.pipe_h, &pml.bind_h, volt_cells);
                } else {
                    record_dispatch(&mut encoder, &self.pipe_h, &self.bind_h, curr_cells);
                }
                passes += 1;
                if self.pml_engine.is_none() {
                    if let Some(upml) = self.upml.as_ref() {
                        passes += upml.record(&mut encoder, 3);
                    }
                }
                if let Some(tfsf) = self.tfsf.as_ref() {
                    for side in &tfsf.curr {
                        passes += tfsf.record(&mut encoder, side, step);
                    }
                }
                if let Some(lorentz) = self.lorentz.as_ref() {
                    passes += lorentz.record(&mut encoder, 3);
                }
                if let Some(source) = self.source.as_ref() {
                    passes += source.record(&mut encoder, source.curr.as_ref(), step);
                }
                passes += self.record_probe_gather(&mut encoder, step);
            }
            self.queue.submit(std::iter::once(encoder.finish()));
            done += chunk;
        }
        self.passes.set(passes as u64);
    }

    /// Write the per-step probe parameters for a batch.
    fn write_probe_params(&self, start_ts: u32, steps: u32) {
        let Some(probes) = self.probes.as_ref() else {
            return;
        };
        let mut scratch = probes.scratch.borrow_mut();
        scratch.clear();
        scratch.resize(steps as usize * probes.slot_size as usize, 0);
        for step in 0..steps {
            let base = step as usize * probes.slot_size as usize;
            scratch[base..base + 4].copy_from_slice(&(start_ts + step).to_le_bytes());
            scratch[base + 4..base + 8].copy_from_slice(&step.to_le_bytes());
            scratch[base + 8..base + 12].copy_from_slice(&probes.count.to_le_bytes());
        }
        self.queue.write_buffer(&probes.params, 0, &scratch);
    }

    /// Record the probe gather for one step.
    fn record_probe_gather(&self, encoder: &mut wgpu::CommandEncoder, step: u32) -> u32 {
        let Some(probes) = self.probes.as_ref() else {
            return 0;
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("openems probe pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&probes.pipe);
        pass.set_bind_group(0, &probes.bind, &[step * probes.slot_size]);
        pass.dispatch_workgroups(probes.count.div_ceil(64).max(1), 1, 1);
        1
    }

    /// Number of compute passes recorded by the last [`dispatch_steps`] call.
    ///
    /// The performance gate asserts a budget per model shape, so a duplicated
    /// phase is caught even when the numerical result is unchanged.
    pub fn last_batch_passes(&self) -> u64 {
        self.passes.get()
    }

    /// Storage buffers per shader stage this device offers.
    pub fn storage_buffer_limit(&self) -> u32 {
        self.device
            .limits()
            .max_storage_buffers_per_shader_stage
    }

    /// Device-side field energy (`CalcFastEnergy`).
    ///
    /// One workgroup sums 64 cells into an E and an H partial; the host sums
    /// the partials in f64 and applies the constants, so the value tracks the
    /// CPU accumulation to ~1e-7 relative without downloading the fields.
    pub fn compute_energy(&self, counts: [usize; 3]) -> Option<f64> {
        let total = counts[0].saturating_sub(1)
            * counts[1].saturating_sub(1)
            * counts[2].saturating_sub(1);
        if total == 0 {
            return Some(0.0);
        }
        let groups = (total as u32).div_ceil(64);
        if groups > self.energy_groups {
            return None;
        }
        let mut data = Vec::with_capacity(32);
        for value in [
            self.dims[0] as u32,
            self.dims[1] as u32,
            self.dims[2] as u32,
            0,
            counts[0] as u32,
            counts[1] as u32,
            counts[2] as u32,
            0,
        ] {
            data.extend_from_slice(&value.to_le_bytes());
        }
        self.queue.write_buffer(&self.energy_uniform, 0, &data);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("openems energy"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("openems energy pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.energy_pipe);
            pass.set_bind_group(0, &self.energy_bind, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        let mut parts = vec![0.0f32; groups as usize * 2];
        readback_into(
            &self.device,
            &self.queue,
            &self.energy_parts,
            &self.energy_staging,
            &mut parts,
        )?;
        let mut e_energy = 0.0f64;
        let mut h_energy = 0.0f64;
        for pair in parts.chunks_exact(2) {
            e_energy += pair[0] as f64;
            h_energy += pair[1] as f64;
        }
        Some(crate::constants::EPS0 * e_energy + crate::constants::MU0 * h_energy)
    }

    fn dispatch(&self, pipeline: &wgpu::ComputePipeline, bind: &wgpu::BindGroup, groups: u32) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("openems engine step"),
        });
        record_dispatch(&mut encoder, pipeline, bind, groups);
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Download the voltage field only (host-extension E phase).
    pub fn download_volt(&self, volt: &mut [FDTD_FLOAT]) -> Option<()> {
        readback_into(
            &self.device,
            &self.queue,
            &self.volt,
            &self.staging_volt,
            volt,
        )
    }

    /// Download the current field only (host-extension H phase).
    pub fn download_curr(&self, curr: &mut [FDTD_FLOAT]) -> Option<()> {
        readback_into(
            &self.device,
            &self.queue,
            &self.curr,
            &self.staging_curr,
            curr,
        )
    }

    pub fn download(&self, volt: &mut [FDTD_FLOAT], curr: &mut [FDTD_FLOAT]) -> Option<()> {
        self.download_pair(volt, curr)
    }

    /// Download both fields with a single copy submission and one map, which
    /// is the round trip a batch of GPU steps ends with.
    pub fn download_pair(
        &self,
        volt: &mut [FDTD_FLOAT],
        curr: &mut [FDTD_FLOAT],
    ) -> Option<()> {
        let bytes = (volt.len() * 4) as u64;
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("openems gpu pair copy"),
        });
        encoder.copy_buffer_to_buffer(&self.volt, 0, &self.staging_pair, 0, bytes);
        encoder.copy_buffer_to_buffer(&self.curr, 0, &self.staging_pair, bytes, bytes);
        self.queue.submit(std::iter::once(encoder.finish()));
        let slice = self.staging_pair.slice(..2 * bytes);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .ok()?;
        receiver.recv().ok()?.ok()?;
        let data = slice.get_mapped_range().ok()?;
        for (value, chunk) in volt.iter_mut().zip(data.chunks_exact(4)) {
            *value = FDTD_FLOAT::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        for (value, chunk) in curr
            .iter_mut()
            .zip(data[bytes as usize..].chunks_exact(4))
        {
            *value = FDTD_FLOAT::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        drop(data);
        self.staging_pair.unmap();
        Some(())
    }

    /// Upload the voltage field only.
    pub fn upload_volt(&self, volt: &[FDTD_FLOAT]) {
        self.write_field(&self.volt, volt);
    }

    /// Upload the current field only.
    pub fn upload_curr(&self, curr: &[FDTD_FLOAT]) {
        self.write_field(&self.curr, curr);
    }

    /// Upload host fields into the persistent GPU buffers (host-extension path).
    pub fn upload(&self, volt: &[FDTD_FLOAT], curr: &[FDTD_FLOAT]) {
        self.upload_volt(volt);
        self.upload_curr(curr);
    }

    fn write_field(&self, target: &wgpu::Buffer, values: &[FDTD_FLOAT]) {
        let mut scratch = self.upload_scratch.borrow_mut();
        scratch.clear();
        scratch.reserve(values.len() * 4);
        for value in values {
            scratch.extend_from_slice(&value.to_le_bytes());
        }
        self.queue.write_buffer(target, 0, &scratch);
    }
}

fn record_dispatch(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bind: &wgpu::BindGroup,
    cells: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("openems engine pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind, &[]);
    pass.dispatch_workgroups(cells.div_ceil(64).max(1), 1, 1);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_update(
        volt: &mut [f32],
        curr: &mut [f32],
        vv: &[f32],
        vi: &[f32],
        ii: &[f32],
        iv: &[f32],
        dims: [usize; 3],
    ) {
        let [nx, ny, nz] = dims;
        let comp = nx * ny * nz;
        let idx = |n: usize, i: usize, j: usize, k: usize| n * comp + i * ny * nz + j * nz + k;
        for i in 0..nx {
            let si = usize::from(i > 0);
            for j in 0..ny {
                let sj = usize::from(j > 0);
                for k in 0..nz {
                    let sk = usize::from(k > 0);
                    let c2 = curr[idx(2, i, j, k)];
                    let c2j = curr[idx(2, i, j - sj, k)];
                    let c1 = curr[idx(1, i, j, k)];
                    let c1k = curr[idx(1, i, j, k - sk)];
                    let c0 = curr[idx(0, i, j, k)];
                    let c0k = curr[idx(0, i, j, k - sk)];
                    let c2i = curr[idx(2, i - si, j, k)];
                    let c1i = curr[idx(1, i - si, j, k)];
                    let c0j = curr[idx(0, i, j - sj, k)];
                    let ex = idx(0, i, j, k);
                    let ey = ex + comp;
                    let ez = ey + comp;
                    volt[ex] = volt[ex] * vv[ex] + vi[ex] * (c2 - c2j - c1 + c1k);
                    volt[ey] = volt[ey] * vv[ey] + vi[ey] * (c0 - c0k - c2 + c2i);
                    volt[ez] = volt[ez] * vv[ez] + vi[ez] * (c1 - c1i - c0 + c0j);
                }
            }
        }
        for i in 0..nx - 1 {
            for j in 0..ny - 1 {
                for k in 0..nz - 1 {
                    let v2 = volt[idx(2, i, j, k)];
                    let v2j = volt[idx(2, i, j + 1, k)];
                    let v1 = volt[idx(1, i, j, k)];
                    let v1k = volt[idx(1, i, j, k + 1)];
                    let v0 = volt[idx(0, i, j, k)];
                    let v0k = volt[idx(0, i, j, k + 1)];
                    let v2i = volt[idx(2, i + 1, j, k)];
                    let v1i = volt[idx(1, i + 1, j, k)];
                    let v0j = volt[idx(0, i, j + 1, k)];
                    let hx = idx(0, i, j, k);
                    let hy = hx + comp;
                    let hz = hy + comp;
                    curr[hx] = curr[hx] * ii[hx] + iv[hx] * (v2 - v2j - v1 + v1k);
                    curr[hy] = curr[hy] * ii[hy] + iv[hy] * (v0 - v0k - v2 + v2i);
                    curr[hz] = curr[hz] * ii[hz] + iv[hz] * (v1 - v1i - v0 + v0j);
                }
            }
        }
    }

    #[test]
    fn gpu_context_and_scale_round_trip() {
        let Some(context) = GpuContext::new() else {
            eprintln!("skipping the GPU round-trip: no adapter available");
            return;
        };
        let values = vec![1.0f32, -2.0, 0.5, 4.0];
        let scaled = context.scale(&values, 2.0).expect("gpu scale");
        assert_eq!(scaled, vec![2.0, -4.0, 1.0, 8.0]);
        eprintln!(
            "GPU round-trip on {} ({})",
            context.adapter_name, context.backend
        );
    }

    #[test]
    fn gpu_update_matches_the_cpu_formula() {
        let Some(context) = GpuContext::new() else {
            eprintln!("skipping the GPU update test: no adapter available");
            return;
        };
        let dims = [5usize, 4usize, 3usize];
        let total = dims[0] * dims[1] * dims[2];
        let mut volt: Vec<f32> = (0..3 * total).map(|i| (i as f32 * 0.013).sin()).collect();
        let mut curr: Vec<f32> = (0..3 * total).map(|i| (i as f32 * 0.017).cos()).collect();
        let vv: Vec<f32> = (0..3 * total).map(|i| 0.9 + (i % 7) as f32 * 0.01).collect();
        let vi: Vec<f32> = (0..3 * total).map(|i| 0.1 + (i % 5) as f32 * 0.02).collect();
        let ii: Vec<f32> = (0..3 * total).map(|i| 0.8 + (i % 3) as f32 * 0.03).collect();
        let iv: Vec<f32> = (0..3 * total).map(|i| 0.2 + (i % 4) as f32 * 0.01).collect();

        let mut cpu_volt = volt.clone();
        let mut cpu_curr = curr.clone();
        cpu_update(&mut cpu_volt, &mut cpu_curr, &vv, &vi, &ii, &iv, dims);

        context
            .update_fields(&mut volt, &mut curr, &vv, &vi, &ii, &iv, dims)
            .expect("gpu update");

        let mut max_volt = 0.0f32;
        let mut max_curr = 0.0f32;
        for (a, b) in volt.iter().zip(&cpu_volt) {
            max_volt = max_volt.max((a - b).abs());
        }
        for (a, b) in curr.iter().zip(&cpu_curr) {
            max_curr = max_curr.max((a - b).abs());
        }
        assert!(max_volt < 1.0e-5, "E max diff {max_volt}");
        assert!(max_curr < 1.0e-5, "H max diff {max_curr}");
        eprintln!(
            "GPU update vs CPU on {} ({}): E {:.2e}, H {:.2e}",
            context.adapter_name, context.backend, max_volt, max_curr
        );
    }
}
