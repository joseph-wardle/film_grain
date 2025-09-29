use crate::prng::Prng;
use crate::rendering::FilmGrainOptions;
use ndarray::Array2;
use tracing::{debug, info, instrument, trace};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuOptions {
    input_height: u32,
    input_width: u32,
    output_height: u32,
    output_width: u32,
    monte_carlo_sample_count: u32,
    grain_seed_offset: u32,
    padding: [u32; 2],
    grain_radius: f32,
    grain_radius_stddev_factor: f32,
    gaussian_filter_stddev_factor: f32,
    padding_f32: f32,
    input_region_min_x: f32,
    input_region_min_y: f32,
    input_region_max_x: f32,
    input_region_max_y: f32,
}

#[instrument(level = "info", skip(input_image, options))]
pub fn render_grain_wise(input_image: &Array2<f32>, options: &FilmGrainOptions) -> Array2<f32> {
    info!(?options, "Starting GPU grain-wise rendering");
    pollster::block_on(render_grain_wise_async(input_image, options))
}

#[instrument(level = "info", skip(input_image, options))]
async fn render_grain_wise_async(
    input_image: &Array2<f32>,
    options: &FilmGrainOptions,
) -> Array2<f32> {
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

    /* ---------- Pre-compute offsets ---------- */
    debug!("Preparing Monte Carlo offsets for GPU execution");
    let mut prng = Prng::new(options.grain_seed_offset);
    let sample_count = options.monte_carlo_sample_count;
    let (mut x_offsets, mut y_offsets) = (
        Vec::with_capacity(sample_count),
        Vec::with_capacity(sample_count),
    );
    for _ in 0..sample_count {
        x_offsets.push(prng.next_standard_normal());
        y_offsets.push(prng.next_standard_normal());
    }

    /* ---------- Buffers ---------- */
    trace!("Creating GPU buffers");
    let input_flat = input_image.as_slice().unwrap();
    let input_image_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("img_in"),
        contents: bytemuck::cast_slice(input_flat),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let num_pixels = (options.output_height * options.output_width) as usize;
    let flags_size = num_pixels * 4;
    // Zero-initialized buffer for Monte Carlo flags
    let zero_flags = vec![0u8; flags_size];
    let monte_carlo_flags_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("flags"),
        contents: &zero_flags,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read_back"),
        size: flags_size as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let x_offset_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("x_offsets"),
        contents: bytemuck::cast_slice(&x_offsets),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let y_offset_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("y_offsets"),
        contents: bytemuck::cast_slice(&y_offsets),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let opts_gpu = GpuOptions {
        input_height: input_image.shape()[0] as u32,
        input_width: input_image.shape()[1] as u32,
        output_height: options.output_height as u32,
        output_width: options.output_width as u32,
        monte_carlo_sample_count: options.monte_carlo_sample_count as u32,
        grain_seed_offset: options.grain_seed_offset,
        padding: [0, 0],
        grain_radius: options.grain_radius,
        grain_radius_stddev_factor: options.grain_radius_stddev_factor,
        gaussian_filter_stddev_factor: options.gaussian_filter_stddev,
        padding_f32: 0.0,
        input_region_min_x: options.input_region_min_x,
        input_region_min_y: options.input_region_min_y,
        input_region_max_x: options.input_region_max_x,
        input_region_max_y: options.input_region_max_y,
    };
    let options_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
            // 1 - flags
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
            // 2,3 - offsets
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
            // 4 - opts
            wgpu::BindGroupLayoutEntry {
                binding: 4,
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
                resource: input_image_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: monte_carlo_flags_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: x_offset_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: y_offset_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: options_buffer.as_entire_binding(),
            },
        ],
    });

    trace!("Creating compute shader module");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("grain_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/grain.wgsl").into()),
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
    let mut command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("encoder"),
    });
    {
        let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cpass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroup_size = 16u32;
        let dispatch_width = (input_image.shape()[1] as u32 + workgroup_size - 1) / workgroup_size;
        let dispatch_height = (input_image.shape()[0] as u32 + workgroup_size - 1) / workgroup_size;
        debug!(
            dispatch_width,
            dispatch_height, "Dispatching grain-wise workload"
        );
        compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
    }
    command_encoder.copy_buffer_to_buffer(
        &monte_carlo_flags_buffer,
        0,
        &readback_buffer,
        0,
        flags_size as u64,
    );
    queue.submit(Some(command_encoder.finish()));

    /* ---------- Read-back and aggregate ---------- */
    trace!("Mapping results for read-back");
    let readback_slice = readback_buffer.slice(..);
    let (mapping_sender, mapping_receiver) = futures_intrusive::channel::shared::oneshot_channel();
    readback_slice.map_async(wgpu::MapMode::Read, move |v| {
        mapping_sender.send(v).unwrap()
    });
    device.poll(wgpu::PollType::Wait).expect("poll failed");
    mapping_receiver
        .receive()
        .await
        .unwrap()
        .expect("map failed");

    let mapped_flags = readback_slice.get_mapped_range();
    let flags: Vec<u32> = bytemuck::cast_slice(&mapped_flags).to_vec();
    drop(mapped_flags);
    readback_buffer.unmap();

    let mut rendered_pixels = vec![0f32; num_pixels];
    trace!("Aggregating Monte Carlo flags");
    for i in 0..num_pixels {
        rendered_pixels[i] = flags[i] as f32 / options.monte_carlo_sample_count as f32;
    }

    Array2::from_shape_vec(
        (options.output_height, options.output_width),
        rendered_pixels,
    )
    .expect("shape error")
}
