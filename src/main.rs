use clap::Parser;
use film_grain::cli::Cli;
use film_grain::io::{read_image, write_image};
use film_grain::rendering::cpu::grain_wise::render_grain_wise;
use film_grain::rendering::cpu::pixel_wise::render_pixel_wise;
use film_grain::rendering::gpu;
use film_grain::rendering::{FilmGrainOptions, RenderingAlgorithm};
use tracing::{debug, error, info, trace, warn};

fn main() {
    film_grain::init_logging();
    trace!("Logging initialized, starting application");

    // Parse command line arguments.
    let args = Cli::parse();
    debug!(?args, "Parsed CLI arguments");

    // Read input image.
    info!(input = ?args.input, "Reading input image");
    let image_in = read_image(&args.input);
    info!(
        rows = image_in.shape()[0],
        cols = image_in.shape()[1],
        "Loaded input image"
    );

    // Determine default region and output dimensions if not provided.
    let (img_rows, img_cols) = (image_in.shape()[0] as f32, image_in.shape()[1] as f32);
    let x_b = args.region_max_x.unwrap_or(img_cols);
    let y_b = args.region_max_y.unwrap_or(img_rows);
    let m_out = args.output_height.unwrap_or((img_rows * args.zoom) as u32) as usize;
    let n_out = args.output_width.unwrap_or((img_cols * args.zoom) as u32) as usize;
    if m_out == 0 || n_out == 0 {
        warn!(
            output_rows = m_out,
            output_cols = n_out,
            "Output dimensions evaluate to zero; rendering will produce an empty image"
        );
    }
    debug!(
        input_rows = img_rows,
        input_cols = img_cols,
        region_x_a = args.region_min_x,
        region_y_a = args.region_min_y,
        region_x_b = x_b,
        region_y_b = y_b,
        output_rows = m_out,
        output_cols = n_out,
        "Computed rendering region and output dimensions"
    );

    // Set up rendering options.
    let opts = FilmGrainOptions {
        grain_radius: args.average_grain_radius,
        sigma_r: args.grain_radius_stddev_factor,
        sigma_filter: args.gaussian_filter_stddev,
        n_monte_carlo: args.monte_carlo_sample_count as usize,
        grain_seed: args.rng_seed,
        x_a: args.region_min_x,
        y_a: args.region_min_y,
        x_b,
        y_b,
        m_out,
        n_out,
    };
    debug!(?opts, "Constructed rendering options");

    let start_time = std::time::Instant::now();
    info!(algorithm = ?args.algorithm, "Starting rendering");
    // Choose and run the rendering algorithm.
    let image_out = match args.algorithm {
        RenderingAlgorithm::GrainWise => {
            trace!("Using CPU grain-wise renderer");
            render_grain_wise(&image_in, &opts)
        }
        RenderingAlgorithm::PixelWise => {
            trace!("Using CPU pixel-wise renderer");
            render_pixel_wise(&image_in, &opts)
        }
        RenderingAlgorithm::GpuPixelWise => {
            trace!("Using GPU pixel-wise renderer");
            gpu::render_pixel_wise(&image_in, &opts)
        }
        RenderingAlgorithm::GpuGrainWise => {
            trace!("Using GPU grain-wise renderer");
            gpu::render_grain_wise(&image_in, &opts)
        }
        RenderingAlgorithm::Automatic => {
            // Use grain-wise if grain radius is large; otherwise, pixel-wise.
            if opts.grain_radius > 1.0 {
                info!("Automatic mode selected CPU grain-wise renderer");
                render_grain_wise(&image_in, &opts)
            } else {
                info!("Automatic mode selected CPU pixel-wise renderer");
                render_pixel_wise(&image_in, &opts)
            }
        }
    };
    let elapsed_time = start_time.elapsed();
    info!(duration_ms = elapsed_time.as_millis(), "Rendering finished");

    // Write the rendered output image.
    info!(output = ?args.output, "Writing rendered image to disk");
    if let Err(err) = write_image(&args.output, &image_out) {
        error!(?err, path = ?args.output, "Failed to write output image");
        std::process::exit(1);
    }
    info!("Film grain rendering completed successfully");
}
