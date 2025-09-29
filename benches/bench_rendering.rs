use criterion::{black_box, criterion_group, criterion_main, Criterion};
use film_grain::io::read_image;
use film_grain::rendering::pixel_wise::render_pixel_wise;
use film_grain::rendering::FilmGrainOptions;
use rand::Rng;

fn benchmark_rendering(c: &mut Criterion) {
    // List of three test input images (adjust the paths as needed)
    let test_images = vec![
        "benches/data/test1.png",
        "benches/data/test2.png",
        "benches/data/test3.png",
        "benches/data/test4.png",
    ];

    let mut group = c.benchmark_group("film_grain_rendering");

    for image_path in test_images {
        // Load the image from disk.
        let image_in = read_image(image_path);
        // Create a default set of rendering options.
        let opts = FilmGrainOptions {
            grain_radius: 0.1,
            grain_radius_stddev_factor: 0.0,
            gaussian_filter_stddev: 0.8,
            monte_carlo_sample_count: 800,
            grain_seed_offset: rand::rng().random::<u32>(),
            input_region_min_x: 0.0,
            input_region_min_y: 0.0,
            input_region_max_x: image_in.shape()[1] as f32,
            input_region_max_y: image_in.shape()[0] as f32,
            output_height: image_in.shape()[0],
            output_width: image_in.shape()[1],
        };

        // Benchmark the pixel-wise rendering function for this image.
        group.bench_function(image_path, |b| {
            b.iter(|| {
                // Use black_box to avoid unwanted optimizations.
                let result = render_pixel_wise(black_box(&image_in), black_box(&opts));
                black_box(result);
            });
        });
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = benchmark_rendering);
criterion_main!(benches);
