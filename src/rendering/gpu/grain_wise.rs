use ndarray::Array2;
use wgpu::util::DeviceExt;
use crate::prng::Prng;
use crate::rendering::FilmGrainOptions;

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

pub fn render_grain_wise(img_in: &Array2<f32>, opts: &FilmGrainOptions) -> Array2<f32> {
    pollster::block_on(render_grain_wise_async(img_in, opts))
}

async fn render_grain_wise_async(img_in: &Array2<f32>, opts: &FilmGrainOptions) -> Array2<f32> {
    /* ---------- WGPU setup ---------- */
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("No suitable GPU adapters found");
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: Default::default(),
                required_limits: Default::default(),
                memory_hints: Default::default(),
                trace: Default::default(),
            },
        )
        .await
        .expect("Failed to create device");

    /* ---------- Pre-compute offsets ---------- */
    let mut prng = Prng::new(opts.grain_seed);
    let n_mc = opts.n_monte_carlo;
    let (mut x_offsets, mut y_offsets) = (Vec::with_capacity(n_mc), Vec::with_capacity(n_mc));
    for _ in 0..n_mc {
        x_offsets.push(prng.next_standard_normal());
        y_offsets.push(prng.next_standard_normal());
    }

    /* ---------- Buffers ---------- */
    let img_in_flat = img_in.as_slice().unwrap();
    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("img_in"),
        contents: bytemuck::cast_slice(img_in_flat),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let num_pixels = (opts.m_out * opts.n_out) as usize;
    let words_per_pixel = (opts.n_monte_carlo + 31) / 32;
    let flags_size = num_pixels * words_per_pixel * 4;
    // Zero-initialized buffer for Monte Carlo flags
    let zero_flags = vec![0u8; flags_size];
    let flags_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("flags"),
        contents: &zero_flags,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let read_back = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read_back"),
        size: flags_size as u64,
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
            wgpu::BindGroupEntry { binding: 0, resource: input_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: flags_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: x_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: y_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: opts_buf.as_entire_binding() },
        ],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("grain_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/grain.wgsl").into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pipeline_layout"),
        bind_group_layouts: &[&bind_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    /* ---------- Dispatch ---------- */
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("encoder") });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("cpass"), timestamp_writes: None });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let wg_size = 16u32;
        let dispatch_x = (img_in.shape()[1] as u32 + wg_size - 1) / wg_size;
        let dispatch_y = (img_in.shape()[0] as u32 + wg_size - 1) / wg_size;
        cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }
    encoder.copy_buffer_to_buffer(&flags_buf, 0, &read_back, 0, flags_size as u64);
    queue.submit(Some(encoder.finish()));

    /* ---------- Read-back and aggregate ---------- */
    let slice = read_back.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    device.poll(wgpu::PollType::Wait).expect("poll failed");
    receiver.receive().await.unwrap().expect("map failed");

    let data = slice.get_mapped_range();
    let flags: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    read_back.unmap();

    let mut result = vec![0f32; num_pixels];
    for i in 0..num_pixels {
        let mut count = 0u32;
        for w in 0..words_per_pixel {
            let word = flags[i * words_per_pixel + w];
            count += word.count_ones();
        }
        result[i] = count as f32 / opts.n_monte_carlo as f32;
    }

    Array2::from_shape_vec((opts.m_out, opts.n_out), result).expect("shape error")
}