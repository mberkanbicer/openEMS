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
"#;

fn f32_bytes(values: &[FDTD_FLOAT]) -> Vec<u8> {
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
                required_limits: wgpu::Limits::default(),
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
            upload_scratch: std::cell::RefCell::new(Vec::new()),
        })
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
    /// the fields stay on the device between steps.
    pub fn dispatch_steps(&self, steps: u32) {
        // Bound the encoder size; a chunk is still thousands of kernel
        // launches, far fewer submissions than before.
        const CHUNK: u32 = 256;
        let volt_cells = (self.dims[0] * self.dims[1] * self.dims[2]) as u32;
        let curr_cells = (self.dims[0].saturating_sub(1)
            * self.dims[1].saturating_sub(1)
            * self.dims[2].saturating_sub(1)) as u32;
        let mut remaining = steps;
        while remaining > 0 {
            let chunk = remaining.min(CHUNK);
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("openems engine batch"),
            });
            for _ in 0..chunk {
                record_dispatch(&mut encoder, &self.pipe_e, &self.bind_e, volt_cells);
                record_dispatch(&mut encoder, &self.pipe_h, &self.bind_h, curr_cells);
            }
            self.queue.submit(std::iter::once(encoder.finish()));
            remaining -= chunk;
        }
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
