use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use film_grain::rendering::{self, FilmGrainOptions};
use image::{imageops::FilterType, GrayImage};
use ndarray::Array2;
use once_cell::sync::Lazy;
use rayon::ThreadPool;

const SQUARE_SIDES: &[u32] = &[64, 128, 256, 512, 1024, 2048, 4096];
const SIZE_SWEEP_RADII: &[f32] = &[0.01, 0.1, 1.0];
const RADIUS_SWEEP_SIZES: &[u32] = &[256, 1024, 4096];
const RADIUS_SAMPLE_COUNT: usize = 5;
const MONTE_CARLO_SAMPLES: usize = 128;
const GRAIN_STDDEV_FACTOR: f32 = 0.35;
const GAUSSIAN_STDDEV_FLOOR: f32 = 0.7;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Algorithm {
    GpuGrainWise,
    GpuPixelWise,
    CpuGrainWiseMultithreaded,
    CpuPixelWiseMultithreaded,
    CpuGrainWiseSingleThreaded,
    CpuPixelWiseSingleThreaded,
}

impl Algorithm {
    const ALL: &'static [Algorithm] = &[
        Algorithm::GpuGrainWise,
        Algorithm::GpuPixelWise,
        Algorithm::CpuGrainWiseMultithreaded,
        Algorithm::CpuPixelWiseMultithreaded,
        Algorithm::CpuGrainWiseSingleThreaded,
        Algorithm::CpuPixelWiseSingleThreaded,
    ];

    fn slug(self) -> &'static str {
        match self {
            Algorithm::GpuGrainWise => "gpu-grain-wise",
            Algorithm::GpuPixelWise => "gpu-pixel-wise",
            Algorithm::CpuGrainWiseMultithreaded => "cpu-grain-wise-mt",
            Algorithm::CpuPixelWiseMultithreaded => "cpu-pixel-wise-mt",
            Algorithm::CpuGrainWiseSingleThreaded => "cpu-grain-wise-st",
            Algorithm::CpuPixelWiseSingleThreaded => "cpu-pixel-wise-st",
        }
    }

    fn render(self, input: &Array2<f32>, options: &FilmGrainOptions) -> Array2<f32> {
        match self {
            Algorithm::GpuGrainWise => {
                rendering::gpu::grain_wise::render_grain_wise(input, options)
            }
            Algorithm::GpuPixelWise => {
                rendering::gpu::pixel_wise::render_pixel_wise(input, options)
            }
            Algorithm::CpuGrainWiseMultithreaded => {
                rendering::cpu::grain_wise::render_grain_wise(input, options)
            }
            Algorithm::CpuPixelWiseMultithreaded => {
                rendering::cpu::pixel_wise::render_pixel_wise(input, options)
            }
            Algorithm::CpuGrainWiseSingleThreaded => SINGLE_THREAD_POOL
                .install(|| rendering::cpu::grain_wise::render_grain_wise(input, options)),
            Algorithm::CpuPixelWiseSingleThreaded => SINGLE_THREAD_POOL
                .install(|| rendering::cpu::pixel_wise::render_pixel_wise(input, options)),
        }
    }
}

static BASE_IMAGES: Lazy<Vec<(PathBuf, GrayImage)>> = Lazy::new(load_base_images);
static RESIZE_CACHE: Lazy<Mutex<HashMap<(usize, u32), Arc<Array2<f32>>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
static SINGLE_THREAD_POOL: Lazy<ThreadPool> = Lazy::new(|| {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("failed to build single-threaded rayon pool")
});

fn load_base_images() -> Vec<(PathBuf, GrayImage)> {
    let data_dir = Path::new("benches/data");
    let mut bases = Vec::new();
    for entry in fs::read_dir(data_dir).expect("failed to read benches/data") {
        let entry = entry.expect("failed to read benches/data entry");
        if !entry
            .file_type()
            .expect("failed to obtain benches/data entry type")
            .is_file()
        {
            continue;
        }
        let path = entry.path();
        let image = image::open(&path)
            .unwrap_or_else(|err| panic!("failed to open {}: {err}", path.display()))
            .to_luma8();
        if image.width() == 4096 && image.height() == 4096 {
            bases.push((path, image));
        }
    }
    assert!(
        !bases.is_empty(),
        "no 4096×4096 base images found in benches/data"
    );
    bases
}

fn scaled_image(base_idx: usize, side: u32) -> Arc<Array2<f32>> {
    {
        let cache = RESIZE_CACHE.lock().expect("resize cache poisoned");
        if let Some(image) = cache.get(&(base_idx, side)) {
            return Arc::clone(image);
        }
    }

    let (_, base_image) = &BASE_IMAGES[base_idx];
    let resized = image::imageops::resize(base_image, side, side, FilterType::CatmullRom);
    let pixels: Vec<f32> = resized.pixels().map(|p| p.0[0] as f32 / 255.0).collect();
    let array = Array2::from_shape_vec((side as usize, side as usize), pixels)
        .expect("failed to reshape resized pixels");
    let array = Arc::new(array);

    let mut cache = RESIZE_CACHE.lock().expect("resize cache poisoned");
    let entry = cache
        .entry((base_idx, side))
        .or_insert_with(|| Arc::clone(&array));
    Arc::clone(entry)
}

fn make_options(side: u32, radius: f32, base_idx: usize) -> FilmGrainOptions {
    let side_usize = side as usize;
    FilmGrainOptions {
        grain_radius: radius,
        grain_radius_stddev_factor: GRAIN_STDDEV_FACTOR,
        gaussian_filter_stddev: radius.max(GAUSSIAN_STDDEV_FLOOR),
        monte_carlo_sample_count: MONTE_CARLO_SAMPLES,
        grain_seed_offset: (base_idx as u32) ^ radius.to_bits() ^ side,
        input_region_min_x: 0.0,
        input_region_min_y: 0.0,
        input_region_max_x: side as f32,
        input_region_max_y: side as f32,
        output_height: side_usize,
        output_width: side_usize,
    }
}

fn logarithmic_radii(count: usize, min: f32, max: f32) -> Vec<f32> {
    assert!(count >= 2, "logarithmic_radii expects at least two samples");
    let min_log = min.ln();
    let max_log = max.ln();
    (0..count)
        .map(|idx| {
            let t = idx as f32 / (count as f32 - 1.0);
            (min_log + t * (max_log - min_log)).exp()
        })
        .collect()
}

fn size_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("size_sweep");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(4));

    for &algorithm in Algorithm::ALL {
        for (base_idx, (path, _)) in BASE_IMAGES.iter().enumerate() {
            let base_label = path
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.display().to_string());
            for &radius in SIZE_SWEEP_RADII {
                for &side in SQUARE_SIDES {
                    group.throughput(Throughput::Elements((side as u64).pow(2)));
                    let bench_id = BenchmarkId::new(
                        format!("{}-{}-r{:.2}", algorithm.slug(), base_label, radius),
                        side,
                    );
                    let image = scaled_image(base_idx, side);
                    let options = make_options(side, radius, base_idx);
                    group.bench_with_input(bench_id, &options, |b, options| {
                        let input = Arc::clone(&image);
                        let opts = options.clone();
                        b.iter(|| {
                            let result = algorithm.render(input.as_ref(), &opts);
                            black_box(result);
                        });
                    });
                }
            }
        }
    }

    group.finish();
}

fn radius_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("radius_sweep");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(4));

    let radii = logarithmic_radii(RADIUS_SAMPLE_COUNT, 0.01, 10.0);

    for &algorithm in Algorithm::ALL {
        for (base_idx, (path, _)) in BASE_IMAGES.iter().enumerate() {
            let base_label = path
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.display().to_string());
            for &side in RADIUS_SWEEP_SIZES {
                group.throughput(Throughput::Elements((side as u64).pow(2)));
                for &radius in &radii {
                    let bench_id = BenchmarkId::new(
                        format!("{}-{}-n{}", algorithm.slug(), base_label, side),
                        format!("{radius:.4}"),
                    );
                    let image = scaled_image(base_idx, side);
                    let options = make_options(side, radius, base_idx);
                    group.bench_with_input(bench_id, &options, |b, options| {
                        let input = Arc::clone(&image);
                        let opts = options.clone();
                        b.iter(|| {
                            let result = algorithm.render(input.as_ref(), &opts);
                            black_box(result);
                        });
                    });
                }
            }
        }
    }

    group.finish();
}

fn film_grain_benches(c: &mut Criterion) {
    size_sweep(c);
    radius_sweep(c);
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = film_grain_benches
);
criterion_main!(benches);