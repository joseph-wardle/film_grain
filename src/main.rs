use clap::Parser;
use film_grain::cli::Cli;
use film_grain::io::{read_image, write_image};
use film_grain::rendering::cpu::grain_wise::render_grain_wise;
use film_grain::rendering::cpu::pixel_wise::render_pixel_wise;
use film_grain::rendering::{FilmGrainOptions, RenderingAlgorithm};
use film_grain::rendering::gpu;

fn main() {
    // Parse command line arguments.
    let args = Cli::parse();

    // Read input image.
    let image_in = read_image(&args.input);

    // Determine default region and output dimensions if not provided.
    let (img_rows, img_cols) = (image_in.shape()[0] as f32, image_in.shape()[1] as f32);
    let x_b = args.x_b.unwrap_or(img_cols);
    let y_b = args.y_b.unwrap_or(img_rows);
    let m_out = args.height.unwrap_or((img_rows * args.zoom) as u32) as usize;
    let n_out = args.width.unwrap_or((img_cols * args.zoom) as u32) as usize;

    // Set up rendering options.
    let opts = FilmGrainOptions {
        grain_radius: args.mu_r,
        sigma_r: args.sigma_r,
        sigma_filter: args.sigma_filter,
        n_monte_carlo: args.n_monte_carlo as usize,
        grain_seed: args.seed,
        x_a: args.x_a,
        y_a: args.y_a,
        x_b,
        y_b,
        m_out,
        n_out,
    };

    let start_time = std::time::Instant::now();
    // Choose and run the rendering algorithm.
    let image_out = match args.algorithm {
        RenderingAlgorithm::GrainWise => render_grain_wise(&image_in, &opts),
        RenderingAlgorithm::PixelWise => render_pixel_wise(&image_in, &opts),
        RenderingAlgorithm::GpuPixelWise => gpu::render_pixel_wise(&image_in, &opts),
        RenderingAlgorithm::GpuGrainWise => gpu::render_grain_wise(&image_in, &opts),
        RenderingAlgorithm::Automatic => {
            // Use grain-wise if grain radius is large; otherwise, pixel-wise.
            if opts.grain_radius > 1.0 {
                render_grain_wise(&image_in, &opts)
            } else {
                render_pixel_wise(&image_in, &opts)
            }
        }
    };
    let elapsed_time = start_time.elapsed();
    println!("Rendering took {} ms", elapsed_time.as_millis());

    // Write the rendered output image.
    write_image(&args.output, &image_out).expect("Failed to write output image");
}
