use std::sync::OnceLock;

use bytemuck::{Pod, Zeroable};
use pollster::block_on;
use wgpu::util::DeviceExt;

use crate::RenderError;
use crate::model::{Derived, Plane};
use crate::params::{Params, RadiusDist};

const PIXEL_SHADER: &str = include_str!("shaders/pixel_wise.wgsl");
const GRAIN_SHADER: &str = include_str!("shaders/grain_wise.wgsl");

static GPU_CONTEXT: OnceLock<GpuContext> = OnceLock::new();

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    in_w: u32,
    in_h: u32,
    out_w: u32,
    out_h: u32,
    n_samples: u32,
    lanes: u32,
    seed: u32,
    _pad0: u32,
    s: f32,
    delta: f32,
    rm: f32,
    inv_e_pi_r2: f32,
    dist_kind: u32,
    _pad1: u32,
    radius_mean: f32,
    radius_log_mu: f32,
    radius_log_sigma: f32,
    _pad2: f32,
}

pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pixel_bind_group_layout: wgpu::BindGroupLayout,
    grain_splat_bind_group_layout: wgpu::BindGroupLayout,
    grain_reduce_bind_group_layout: wgpu::BindGroupLayout,
    pixel_pipeline: wgpu::ComputePipeline,
    grain_splat_pipeline: wgpu::ComputePipeline,
    grain_reduce_pipeline: wgpu::ComputePipeline,
}

pub fn context() -> Result<&'static GpuContext, RenderError> {
    if let Some(ctx) = GPU_CONTEXT.get() {
        return Ok(ctx);
    }
    let ctx = init()?;
    GPU_CONTEXT
        .set(ctx)
        .map_err(|_| RenderError::Gpu("GPU context already initialized".into()))?;
    Ok(GPU_CONTEXT.get().expect("gpu context set"))
}

fn init() -> Result<GpuContext, RenderError> {
    let instance = wgpu::Instance::default();
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or_else(|| RenderError::Gpu("no compatible GPU adapter found".into()))?;

    let (device, queue) = block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("filmgrain-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        },
        None,
    ))
    .map_err(|err| RenderError::Gpu(format!("request_device failed: {err}")))?;

    let pixel_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("pixel-wise shader"),
        source: wgpu::ShaderSource::Wgsl(PIXEL_SHADER.into()),
    });
    let grain_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("grain-wise shader"),
        source: wgpu::ShaderSource::Wgsl(GRAIN_SHADER.into()),
    });

    let pixel_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pixel-bind-group-layout"),
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
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    let grain_splat_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("grain-splat-bind-group-layout"),
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
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    let grain_reduce_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("grain-reduce-bind-group-layout"),
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
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    let pixel_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pixel-pipeline-layout"),
        bind_group_layouts: &[&pixel_bind_group_layout],
        push_constant_ranges: &[],
    });

    let grain_splat_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("grain-splat-pipeline-layout"),
            bind_group_layouts: &[&grain_splat_bind_group_layout],
            push_constant_ranges: &[],
        });

    let grain_reduce_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("grain-reduce-pipeline-layout"),
            bind_group_layouts: &[&grain_reduce_bind_group_layout],
            push_constant_ranges: &[],
        });

    let pixel_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pixel-pipeline"),
        layout: Some(&pixel_pipeline_layout),
        module: &pixel_shader,
        entry_point: "main",
    });

    let grain_splat_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("grain-splat-pipeline"),
        layout: Some(&grain_splat_pipeline_layout),
        module: &grain_shader,
        entry_point: "splat",
    });

    let grain_reduce_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("grain-reduce-pipeline"),
        layout: Some(&grain_reduce_pipeline_layout),
        module: &grain_shader,
        entry_point: "reduce",
    });

    Ok(GpuContext {
        device,
        queue,
        pixel_bind_group_layout,
        grain_splat_bind_group_layout,
        grain_reduce_bind_group_layout,
        pixel_pipeline,
        grain_splat_pipeline,
        grain_reduce_pipeline,
    })
}

pub fn render_pixelwise_gpu(
    ctx: &GpuContext,
    lambda: &Plane,
    params: &Params,
    d: &Derived,
) -> Result<Plane, RenderError> {
    let offsets = &d.offsets_input;
    if offsets.len() != params.n_samples as usize {
        return Err(RenderError::Gpu(
            "offset count does not match sample count".into(),
        ));
    }

    let uniforms = build_uniforms(params, d, 0);
    let lambda_bytes = bytemuck::cast_slice(lambda.pixels());
    let offsets_bytes = bytemuck::cast_slice(offsets);
    let uniforms_bytes = bytemuck::bytes_of(&uniforms);
    let out_len = (d.output_width * d.output_height) as usize;
    let out_bytes = out_len * std::mem::size_of::<f32>();

    let lambda_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lambda-buffer"),
            contents: lambda_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });

    let offsets_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("offsets-input-buffer"),
            contents: offsets_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });

    let uniforms_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms-buffer"),
            contents: uniforms_bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let out_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pixel-output-buffer"),
        size: out_bytes as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let zero_out = vec![0u8; out_bytes];
    ctx.queue.write_buffer(&out_buffer, 0, &zero_out);

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pixel-bind-group"),
        layout: &ctx.pixel_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: lambda_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: offsets_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniforms_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pixel-encoder"),
        });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pixel-pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&ctx.pixel_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let wg_x = div_round_up(d.output_width, 16);
        let wg_y = div_round_up(d.output_height, 16);
        cpass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    let staging_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pixel-staging"),
        size: out_bytes as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&out_buffer, 0, &staging_buffer, 0, out_bytes as u64);
    ctx.queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
        sender.send(res).ok();
    });
    ctx.device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .unwrap()
        .map_err(|_| RenderError::Gpu("map_async failed for pixel output".into()))?;
    let data = buffer_slice.get_mapped_range();
    let mut output_plane = Plane::new(d.output_width, d.output_height);
    let floats: &[f32] = bytemuck::cast_slice(&data);
    output_plane.pixels_mut().copy_from_slice(floats);
    drop(data);
    staging_buffer.unmap();

    Ok(output_plane)
}

pub fn render_grainwise_gpu(
    ctx: &GpuContext,
    lambda: &Plane,
    params: &Params,
    d: &Derived,
) -> Result<Plane, RenderError> {
    let offsets = &d.offsets;
    if offsets.len() != params.n_samples as usize {
        return Err(RenderError::Gpu(
            "offset count does not match sample count".into(),
        ));
    }

    let lanes = lanes_for_samples(params.n_samples);
    let uniforms = build_uniforms(params, d, lanes);
    let lambda_bytes = bytemuck::cast_slice(lambda.pixels());
    let offsets_bytes = bytemuck::cast_slice(offsets);
    let uniforms_bytes = bytemuck::bytes_of(&uniforms);

    let out_len = (d.output_width * d.output_height) as usize;
    let bitset_words = out_len * lanes as usize;
    let bitset_bytes = bitset_words * std::mem::size_of::<u32>();
    let out_bytes = out_len * std::mem::size_of::<f32>();

    let lambda_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lambda-buffer"),
            contents: lambda_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });

    let offsets_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("offsets-output-buffer"),
            contents: offsets_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });

    let uniforms_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms-buffer"),
            contents: uniforms_bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bitset_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bitset-buffer"),
        size: bitset_bytes as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let zero_bitset = vec![0u8; bitset_bytes];
    ctx.queue.write_buffer(&bitset_buffer, 0, &zero_bitset);

    let out_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("grain-output-buffer"),
        size: out_bytes as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let zero_out = vec![0u8; out_bytes];
    ctx.queue.write_buffer(&out_buffer, 0, &zero_out);

    let splat_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("grain-splat-bind-group"),
        layout: &ctx.grain_splat_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: lambda_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: offsets_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniforms_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: bitset_buffer.as_entire_binding(),
            },
        ],
    });

    let reduce_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("grain-reduce-bind-group"),
        layout: &ctx.grain_reduce_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: lambda_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: offsets_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniforms_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: bitset_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: out_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("grain-encoder"),
        });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("grain-splat-pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&ctx.grain_splat_pipeline);
        cpass.set_bind_group(0, &splat_bind_group, &[]);
        let wg_x = div_round_up(d.input_width, 8);
        let wg_y = div_round_up(d.input_height, 8);
        cpass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("grain-reduce-pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&ctx.grain_reduce_pipeline);
        cpass.set_bind_group(0, &reduce_bind_group, &[]);
        let wg_x = div_round_up(d.output_width, 16);
        let wg_y = div_round_up(d.output_height, 16);
        cpass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    let staging_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("grain-staging"),
        size: out_bytes as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&out_buffer, 0, &staging_buffer, 0, out_bytes as u64);
    ctx.queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
        sender.send(res).ok();
    });
    ctx.device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .unwrap()
        .map_err(|_| RenderError::Gpu("map_async failed for grain output".into()))?;
    let data = buffer_slice.get_mapped_range();
    let mut output_plane = Plane::new(d.output_width, d.output_height);
    let floats: &[f32] = bytemuck::cast_slice(&data);
    output_plane.pixels_mut().copy_from_slice(floats);
    drop(data);
    staging_buffer.unmap();

    Ok(output_plane)
}

fn build_uniforms(params: &Params, derived: &Derived, lanes: u32) -> Uniforms {
    let (log_mu, log_sigma) = match params.radius_dist {
        RadiusDist::Const => (0.0, 0.0),
        RadiusDist::Lognorm => (
            params.radius_log_mu.unwrap_or(params.radius_mean.ln()),
            params.radius_log_sigma.unwrap_or(0.0),
        ),
    };
    Uniforms {
        in_w: derived.input_width as u32,
        in_h: derived.input_height as u32,
        out_w: derived.output_width as u32,
        out_h: derived.output_height as u32,
        n_samples: params.n_samples,
        lanes,
        seed: params.seed as u32,
        _pad0: 0,
        s: params.zoom,
        delta: derived.delta,
        rm: derived.rm,
        inv_e_pi_r2: derived.inv_e_pi_r2,
        dist_kind: match params.radius_dist {
            RadiusDist::Const => 0,
            RadiusDist::Lognorm => 1,
        },
        _pad1: 0,
        radius_mean: params.radius_mean,
        radius_log_mu: log_mu,
        radius_log_sigma: log_sigma,
        _pad2: 0.0,
    }
}

fn div_round_up(a: usize, b: u32) -> u32 {
    ((a as u32) + b - 1) / b
}

fn lanes_for_samples(samples: u32) -> u32 {
    ((samples + 31) / 32).max(1)
}
