//! Optional `wgpu` compute device (ADR 0012, stage 1).
//!
//! This module is the device/buffer layer only: it proves that an adapter can
//! be acquired, that a storage-buffer compute dispatch runs, and that results
//! round-trip. The FDTD kernels are not wired to it yet; the CPU engine remains
//! the reference. `gpu` is an additive feature, off by default.

use crate::constants::FDTD_FLOAT;

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_name: String,
    pub backend: String,
}

const SHADER: &str = r#"
struct Scale {
    factor: vec4<f32>,
};
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> scale: Scale;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < arrayLength(&input)) {
        output[i] = input[i] * scale.factor.x;
    }
}
"#;

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

    /// Stage-1 kernel: multiply every value by `factor` on the GPU and read it
    /// back. This exercises the same buffer layout the field updates will use
    /// (component-major, contiguous f32).
    pub fn scale(&self, values: &[FDTD_FLOAT], factor: f32) -> Option<Vec<FDTD_FLOAT>> {
        if values.is_empty() {
            return Some(Vec::new());
        }
        use wgpu::util::DeviceExt;
        let bytes = std::mem::size_of_val(values) as u64;
        let input_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let factor_bytes: Vec<u8> = [factor, 0.0, 0.0, 0.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let input = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("openems gpu input"),
                contents: &input_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });
        let output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("openems gpu output"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("openems gpu readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let uniform = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("openems gpu scale"),
                contents: &factor_bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("openems gpu scale shader"),
                source: wgpu::ShaderSource::Wgsl(SHADER.into()),
            });
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("openems gpu layout"),
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
                        label: Some("openems gpu pipeline layout"),
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
            label: Some("openems gpu bind group"),
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
                label: Some("openems gpu encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("openems gpu pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (values.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, bytes);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
        receiver.recv().ok()?.ok()?;
        let data = slice.get_mapped_range().ok()?;
        let result: Vec<FDTD_FLOAT> = data
            .chunks_exact(4)
            .map(|chunk| FDTD_FLOAT::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        drop(data);
        readback.unmap();
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_context_and_scale_round_trip() {
        let Some(context) = GpuContext::new() else {
            eprintln!("skipping the GPU round-trip: no adapter available");
            return;
        };
        let values = vec![1.0f32, -2.0, 0.5, 4.0];
        let scaled = context
            .scale(&values, 2.0)
            .expect("gpu scale round-trip");
        assert_eq!(scaled, vec![2.0, -4.0, 1.0, 8.0]);
        eprintln!(
            "GPU round-trip on {} ({})",
            context.adapter_name, context.backend
        );
    }
}
