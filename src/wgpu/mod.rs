#[cfg(target_arch = "wasm32")]
use std::mem;
use std::sync::{Arc, Mutex, OnceLock};
use std::task::Poll;

use bytemuck::{Pod, Zeroable};
use futures::{FutureExt, channel::oneshot, executor::block_on, future::poll_fn};
use log::{error, warn};
use wgpu::util::DeviceExt;

#[cfg(target_arch = "wasm32")]
pub const WEBGPU_MAX_BUFFER_BYTES: u64 = 256 * 1024 * 1024;
#[cfg(target_arch = "wasm32")]
pub const WEBGPU_MAX_STORAGE_BUFFER_BINDING_BYTES: u64 = 128 * 1024 * 1024;
#[cfg(target_arch = "wasm32")]
pub const WEBGPU_MAX_UNIFORM_BUFFER_BINDING_BYTES: u64 = 64 * 1024;
#[cfg(target_arch = "wasm32")]
pub const WEBGPU_MAX_OUTPUT_PIXELS: usize =
    (WEBGPU_MAX_STORAGE_BUFFER_BINDING_BYTES / mem::size_of::<f32>() as u64) as usize;

use crate::RenderError;
use crate::model::{Derived, Plane};
use crate::params::{Params, RadiusDist};

const PIXEL_SHADER: &str = include_str!("shaders/pixel_wise.wgsl");
const GRAIN_SHADER: &str = include_str!("shaders/grain_wise.wgsl");

static GPU_CONTEXT: OnceLock<GpuContextCache> = OnceLock::new();

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

struct GpuContextCache {
    ctx: Mutex<Option<Arc<GpuContext>>>,
}

impl Default for GpuContextCache {
    fn default() -> Self {
        Self {
            ctx: Mutex::new(None),
        }
    }
}

impl GpuContextCache {
    fn get(&self) -> Result<Arc<GpuContext>, RenderError> {
        if let Some(ctx) = self.cached() {
            return Ok(ctx);
        }
        let ctx = Arc::new(init_blocking()?);
        Ok(self.store(ctx))
    }

    async fn get_async(&self) -> Result<Arc<GpuContext>, RenderError> {
        if let Some(ctx) = self.cached() {
            return Ok(ctx);
        }
        let ctx = Arc::new(init_async().await?);
        Ok(self.store(ctx))
    }

    fn invalidate(&self) {
        let mut guard = self.ctx.lock().unwrap_or_else(|err| err.into_inner());
        guard.take();
    }

    fn cached(&self) -> Option<Arc<GpuContext>> {
        let guard = self.ctx.lock().unwrap_or_else(|err| err.into_inner());
        guard.as_ref().cloned()
    }

    fn store(&self, ctx: Arc<GpuContext>) -> Arc<GpuContext> {
        let mut guard = self.ctx.lock().unwrap_or_else(|err| err.into_inner());
        if let Some(existing) = guard.as_ref() {
            return existing.clone();
        }
        *guard = Some(ctx.clone());
        ctx
    }
}

fn cache() -> &'static GpuContextCache {
    GPU_CONTEXT.get_or_init(GpuContextCache::default)
}

pub fn context() -> Result<Arc<GpuContext>, RenderError> {
    cache().get()
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
pub async fn context_async() -> Result<Arc<GpuContext>, RenderError> {
    cache().get_async().await
}

fn invalidate_context() {
    if let Some(cache) = GPU_CONTEXT.get() {
        cache.invalidate();
    }
}

fn create_instance() -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: preferred_backends(),
        dx12_shader_compiler: Default::default(),
        flags: wgpu::InstanceFlags::empty(),
        gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
    })
}

fn preferred_backends() -> wgpu::Backends {
    #[cfg(target_arch = "wasm32")]
    {
        wgpu::Backends::BROWSER_WEBGPU
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        wgpu::Backends::PRIMARY | wgpu::Backends::GL
    }
}

fn negotiated_limits(adapter: &wgpu::Adapter) -> wgpu::Limits {
    let adapter_limits = adapter.limits();
    #[cfg(target_arch = "wasm32")]
    {
        let mut limits = wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter_limits);
        limits.max_buffer_size = limits.max_buffer_size.min(WEBGPU_MAX_BUFFER_BYTES);
        limits.max_storage_buffer_binding_size = limits
            .max_storage_buffer_binding_size
            .min(WEBGPU_MAX_STORAGE_BUFFER_BINDING_BYTES);
        limits.max_uniform_buffer_binding_size = limits
            .max_uniform_buffer_binding_size
            .min(WEBGPU_MAX_UNIFORM_BUFFER_BINDING_BYTES);
        limits
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        adapter_limits
    }
}

fn init_blocking() -> Result<GpuContext, RenderError> {
    block_on(init_async())
}

async fn init_async() -> Result<GpuContext, RenderError> {
    let instance = create_instance();
    let adapter_options = wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    };
    let adapter = instance
        .request_adapter(&adapter_options)
        .await
        .ok_or_else(|| RenderError::Gpu("no compatible GPU adapter found".into()))?;

    let adapter_limits = negotiated_limits(&adapter);
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("filmgrain-device"),
                required_features: wgpu::Features::empty(),
                // Request the full limits the adapter reports so large outputs aren't capped by the
                // conservative WebGPU defaults (e.g. 256 MiB max buffer size).
                required_limits: adapter_limits,
            },
            None,
        )
        .await
        .map_err(|err| RenderError::Gpu(format!("request_device failed: {err}")))?;

    device.on_uncaptured_error(Box::new(|err| {
        log_uncaptured_error(err);
    }));

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
        compilation_options: Default::default(),
    });

    let grain_splat_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("grain-splat-pipeline"),
        layout: Some(&grain_splat_pipeline_layout),
        module: &grain_shader,
        entry_point: "splat",
        compilation_options: Default::default(),
    });

    let grain_reduce_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("grain-reduce-pipeline"),
        layout: Some(&grain_reduce_pipeline_layout),
        module: &grain_shader,
        entry_point: "reduce",
        compilation_options: Default::default(),
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
    run_with_gpu_error_scope(ctx, "Pixel renderer", || {
        render_pixelwise_gpu_inner(ctx, lambda, params, d)
    })
}

fn render_pixelwise_gpu_inner(
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
    let out_len = d.output_width * d.output_height;
    let out_bytes = out_len * std::mem::size_of::<f32>();

    let limits = ctx.device.limits();
    ensure_storage_buffer_size(&limits, lambda_bytes.len(), "Pixel lambda buffer")?;
    ensure_storage_buffer_size(&limits, offsets_bytes.len(), "Pixel offsets buffer")?;
    ensure_uniform_buffer_size(&limits, uniforms_bytes.len(), "Pixel uniforms")?;
    ensure_storage_buffer_size(&limits, out_bytes, "Pixel output buffer")?;

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

    wait_for_buffer_map_sync(&ctx.device, staging_buffer.slice(..), "pixel output")?;
    let data = staging_buffer.slice(..).get_mapped_range();
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
    run_with_gpu_error_scope(ctx, "Grain renderer", || {
        render_grainwise_gpu_inner(ctx, lambda, params, d)
    })
}

fn render_grainwise_gpu_inner(
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

    let out_len = d.output_width * d.output_height;
    let bitset_words = out_len * lanes as usize;
    let bitset_bytes = bitset_words * std::mem::size_of::<u32>();
    let out_bytes = out_len * std::mem::size_of::<f32>();

    let limits = ctx.device.limits();
    ensure_storage_buffer_size(&limits, lambda_bytes.len(), "Grain lambda buffer")?;
    ensure_storage_buffer_size(&limits, offsets_bytes.len(), "Grain offsets buffer")?;
    ensure_uniform_buffer_size(&limits, uniforms_bytes.len(), "Grain uniforms")?;
    ensure_storage_buffer_size(&limits, bitset_bytes, "Grain accumulator buffer")?;
    ensure_storage_buffer_size(&limits, out_bytes, "Grain output buffer")?;

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

    wait_for_buffer_map_sync(&ctx.device, staging_buffer.slice(..), "grain output")?;
    let data = staging_buffer.slice(..).get_mapped_range();
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
    (a as u32).div_ceil(b)
}

fn lanes_for_samples(samples: u32) -> u32 {
    samples.div_ceil(32)
}

fn ensure_storage_buffer_size(
    limits: &wgpu::Limits,
    bytes: usize,
    label: &str,
) -> Result<(), RenderError> {
    let requirement = bytes as u64;
    if requirement > limits.max_buffer_size {
        return Err(RenderError::Gpu(format!(
            "{label} requires {} but the adapter only supports buffers up to {}.",
            format_bytes(requirement),
            format_bytes(limits.max_buffer_size)
        )));
    }
    if requirement > limits.max_storage_buffer_binding_size as u64 {
        return Err(RenderError::Gpu(format!(
            "{label} exceeds the storage binding limit of {}.",
            format_bytes(limits.max_storage_buffer_binding_size as u64)
        )));
    }
    Ok(())
}

fn ensure_uniform_buffer_size(
    limits: &wgpu::Limits,
    bytes: usize,
    label: &str,
) -> Result<(), RenderError> {
    let requirement = bytes as u64;
    if requirement > limits.max_uniform_buffer_binding_size as u64 {
        return Err(RenderError::Gpu(format!(
            "{label} exceeds the uniform buffer limit of {}.",
            format_bytes(limits.max_uniform_buffer_binding_size as u64)
        )));
    }
    Ok(())
}

fn format_bytes(value: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    if value as f64 >= MIB {
        format!("{:.1} MiB", value as f64 / MIB)
    } else if value as f64 >= KIB {
        format!("{:.1} KiB", value as f64 / KIB)
    } else {
        format!("{value} B")
    }
}

fn run_with_gpu_error_scope<T, F>(
    ctx: &GpuContext,
    label: &'static str,
    work: F,
) -> Result<T, RenderError>
where
    F: FnOnce() -> Result<T, RenderError>,
{
    ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    ctx.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    ctx.device.push_error_scope(wgpu::ErrorFilter::Internal);

    let work_result = work();

    let internal = block_on(ctx.device.pop_error_scope());
    let out_of_memory = block_on(ctx.device.pop_error_scope());
    let validation = block_on(ctx.device.pop_error_scope());

    if let Some(err) = validation.or(out_of_memory).or(internal) {
        return Err(handle_gpu_error(label, err));
    }

    work_result
}

fn handle_gpu_error(label: &str, err: wgpu::Error) -> RenderError {
    let (message, fatal) = describe_gpu_error(label, &err);
    if fatal {
        error!("{msg}", msg = &message);
        invalidate_context();
    } else {
        warn!("{msg}", msg = &message);
    }
    RenderError::Gpu(message)
}

fn describe_gpu_error(label: &str, err: &wgpu::Error) -> (String, bool) {
    match err {
        wgpu::Error::OutOfMemory { .. } => (
            format!("{label} ran out of GPU memory. Try lowering the output size or sample count."),
            true,
        ),
        wgpu::Error::Validation { description, .. } => {
            (format!("{label} validation error: {description}"), false)
        }
        wgpu::Error::Internal { description, .. } => (
            format!("{label} hit an internal GPU error: {description}"),
            true,
        ),
    }
}

fn log_uncaptured_error(err: wgpu::Error) {
    let (message, fatal) = describe_gpu_error("wgpu", &err);
    if fatal {
        error!("{msg}", msg = &message);
        invalidate_context();
    } else {
        warn!("{msg}", msg = &message);
    }
}

async fn wait_for_buffer_map(
    device: &wgpu::Device,
    slice: wgpu::BufferSlice<'_>,
    label: &'static str,
) -> Result<(), RenderError> {
    let (sender, mut receiver) = oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    poll_fn(|cx| match receiver.poll_unpin(cx) {
        Poll::Ready(Ok(Ok(()))) => Poll::Ready(Ok(())),
        Poll::Ready(Ok(Err(err))) => Poll::Ready(Err(RenderError::Gpu(format!(
            "{label} map_async failed: {err}"
        )))),
        Poll::Ready(Err(_)) => Poll::Ready(Err(RenderError::Gpu(format!(
            "{label} map_async cancelled"
        )))),
        Poll::Pending => {
            device.poll(wgpu::Maintain::Poll);
            Poll::Pending
        }
    })
    .await
}

fn wait_for_buffer_map_sync(
    device: &wgpu::Device,
    slice: wgpu::BufferSlice<'_>,
    label: &'static str,
) -> Result<(), RenderError> {
    block_on(wait_for_buffer_map(device, slice, label))
}
