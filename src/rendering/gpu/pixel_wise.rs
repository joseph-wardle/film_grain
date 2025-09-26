use crate::prng::Prng;
use crate::rendering::FilmGrainOptions;
use ndarray::Array2;
use tracing::{debug, info, instrument, trace};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuOptions {
    m_in: u32,
    n_in: u32,
    m_out: u32,
    n_out: u32,
    n_monte_carlo: u32,
    grain_seed: u32,
    _pad: [u32; 2],
    grain_radius: f32,
    sigma_r: f32,
    sigma_filter: f32,
    _pad2: f32,
    x_a: f32,
    y_a: f32,
    x_b: f32,
    y_b: f32,
}

#[instrument(level = "info", skip(img_in, opts))]
pub fn render_pixel_wise(img_in: &Array2<f32>, opts: &FilmGrainOptions) -> Array2<f32> {
    info!(?opts, "Starting GPU pixel-wise rendering");
    pollster::block_on(render_pixel_wise_async(img_in, opts))
}

#[instrument(level = "info", skip(img_in, opts))]
async fn render_pixel_wise_async(img_in: &Array2<f32>, opts: &FilmGrainOptions) -> Array2<f32> {
    /* ---------- WGPU setup ---------- */
    trace!("Creating WGPU instance");
    let instance = wgpu::Instance::default();
    trace!("Requesting GPU adapter");
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("No suitable GPU adapters found");
    debug!("Adapter acquired");
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: Default::default(),
            required_limits: Default::default(),
            memory_hints: Default::default(),
            trace: Default::default(),
        })
        .await
        .expect("Failed to create device");
    debug!("Device and queue created");

    /* ---------- Pre-compute tables ---------- */
    debug!("Preparing Monte Carlo tables for GPU execution");
    let mut prng = Prng::new(opts.grain_seed);
    let n_mc = opts.n_monte_carlo;
    let (mut x_offsets, mut y_offsets) = (Vec::with_capacity(n_mc), Vec::with_capacity(n_mc));
    for _ in 0..n_mc {
        x_offsets.push(prng.next_standard_normal());
        y_offsets.push(prng.next_standard_normal());
    }

    const MAX_GREY_LEVEL: usize = 255;
    const EPSILON: f32 = 0.1;
    let cell_size = 1.0 / ((1.0 / opts.grain_radius).ceil());
    let mut lambda_lookup = vec![0.0f32; MAX_GREY_LEVEL + 1];
    trace!("Computing lambda lookup for GPU shader");
    lambda_lookup
        .iter_mut()
        .enumerate()
        .for_each(|(i, lambda)| {
            let u = i as f32 / (MAX_GREY_LEVEL as f32 + EPSILON);
            let denom = std::f32::consts::PI
                * (opts.grain_radius * opts.grain_radius
                    + if opts.sigma_r > 0.0 {
                        opts.sigma_r * opts.sigma_r
                    } else {
                        0.0
                    });
            *lambda = -((cell_size * cell_size) / denom) * (1.0 - u).ln();
        });

    /* ---------- Buffers ---------- */
    let img_in_flat = img_in.as_slice().unwrap();
    trace!("Creating GPU buffers");
    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("img_in"),
        contents: bytemuck::cast_slice(img_in_flat),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_size = (opts.m_out * opts.n_out) as u64 * 4;

    // GPU-only buffer written by the shader
    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("img_out"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Staging buffer we’ll copy into and then map on the CPU
    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read_back"),
        size: output_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let x_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("x_offsets"),
        contents: bytemuck::cast_slice(&x_offsets),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let y_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("y_offsets"),
        contents: bytemuck::cast_slice(&y_offsets),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let lambda_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("lambda"),
        contents: bytemuck::cast_slice(&lambda_lookup),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let opts_gpu = GpuOptions {
        m_in: img_in.shape()[0] as u32,
        n_in: img_in.shape()[1] as u32,
        m_out: opts.m_out as u32,
        n_out: opts.n_out as u32,
        n_monte_carlo: opts.n_monte_carlo as u32,
        grain_seed: opts.grain_seed,
        _pad: [0, 0],
        grain_radius: opts.grain_radius,
        sigma_r: opts.sigma_r,
        sigma_filter: opts.sigma_filter,
        _pad2: 0.0,
        x_a: opts.x_a,
        y_a: opts.y_a,
        x_b: opts.x_b,
        y_b: opts.y_b,
    };
    let opts_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("opts"),
        contents: bytemuck::bytes_of(&opts_gpu),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    /* ---------- Bind group & pipeline ---------- */
    trace!("Configuring bind group layout");
    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("layout"),
        entries: &[
            // 0 - input image
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
            // 1 - output image
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
            // 2,3,4 - tables
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 5 - uniform options
            wgpu::BindGroupLayoutEntry {
                binding: 5,
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

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind_group"),
        layout: &bind_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: x_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: y_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: lambda_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: opts_buf.as_entire_binding(),
            },
        ],
    });

    trace!("Creating compute shader module");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("pixel_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/pixel.wgsl").into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pipeline_layout"),
        bind_group_layouts: &[&bind_layout],
        push_constant_ranges: &[],
    });

    trace!("Creating compute pipeline");
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    /* ---------- Dispatch ---------- */
    trace!("Encoding compute pass");
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("encoder"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cpass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let wg_size = 16u32;
        let dispatch_x = (opts.n_out as u32 + wg_size - 1) / wg_size;
        let dispatch_y = (opts.m_out as u32 + wg_size - 1) / wg_size;
        debug!(dispatch_x, dispatch_y, "Dispatching compute workload");
        cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }
    // Copy GPU results into the CPU-visible staging buffer
    encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, output_size);
    queue.submit(Some(encoder.finish()));

    /* ---------- Read-back ---------- */
    trace!("Mapping staging buffer for read-back");
    let buffer_slice = staging_buf.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    device.poll(wgpu::PollType::Wait).expect("poll failed");
    receiver.receive().await.unwrap().expect("map failed");

    let data = buffer_slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buf.unmap();

    Array2::from_shape_vec((opts.m_out, opts.n_out), result).expect("shape error")
}
