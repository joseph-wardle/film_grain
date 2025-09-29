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
    let input_image = read_image(&args.input);
    info!(
        rows = input_image.shape()[0],
        cols = input_image.shape()[1],
        "Loaded input image"
    );

    // Determine default region and output dimensions if not provided.
    let (img_rows, img_cols) = (input_image.shape()[0] as f32, input_image.shape()[1] as f32);
    let region_max_x = args.region_max_x.unwrap_or(img_cols);
    let region_max_y = args.region_max_y.unwrap_or(img_rows);
    let output_height = args.output_height.unwrap_or((img_rows * args.zoom) as u32) as usize;
    let output_width = args.output_width.unwrap_or((img_cols * args.zoom) as u32) as usize;
    if output_height == 0 || output_width == 0 {
        warn!(
            output_rows = output_height,
            output_cols = output_width,
            "Output dimensions evaluate to zero; rendering will produce an empty image"
        );
    }
    debug!(
        input_rows = img_rows,
        input_cols = img_cols,
        region_x_a = args.region_min_x,
        region_y_a = args.region_min_y,
        region_x_b = region_max_x,
        region_y_b = region_max_y,
        output_rows = output_height,
        output_cols = output_width,
        "Computed rendering region and output dimensions"
    );

    // Set up rendering options.
    let rendering_options = FilmGrainOptions {
        grain_radius: args.average_grain_radius,
        grain_radius_stddev_factor: args.grain_radius_stddev_factor,
        gaussian_filter_stddev: args.gaussian_filter_stddev,
        monte_carlo_sample_count: args.monte_carlo_sample_count as usize,
        grain_seed_offset: args.rng_seed,
        input_region_min_x: args.region_min_x,
        input_region_min_y: args.region_min_y,
        input_region_max_x: region_max_x,
        input_region_max_y: region_max_y,
        output_height,
        output_width,
    };
    debug!(?rendering_options, "Constructed rendering options");

    let start_time = std::time::Instant::now();
    info!(algorithm = ?args.algorithm, "Starting rendering");
    // Choose and run the rendering algorithm.
    let image_out = match args.algorithm {
        RenderingAlgorithm::GrainWise => {
            trace!("Using CPU grain-wise renderer");
            render_grain_wise(&input_image, &rendering_options)
        }
        RenderingAlgorithm::PixelWise => {
            trace!("Using CPU pixel-wise renderer");
            render_pixel_wise(&input_image, &rendering_options)
        }
        RenderingAlgorithm::GpuPixelWise => {
            trace!("Using GPU pixel-wise renderer");
            gpu::render_pixel_wise(&input_image, &rendering_options)
        }
        RenderingAlgorithm::GpuGrainWise => {
            trace!("Using GPU grain-wise renderer");
            gpu::render_grain_wise(&input_image, &rendering_options)
        }
        RenderingAlgorithm::Automatic => {
            // Use grain-wise if grain radius is large; otherwise, pixel-wise.
            if rendering_options.grain_radius > 1.0 {
                info!("Automatic mode selected CPU grain-wise renderer");
                render_grain_wise(&input_image, &rendering_options)
            } else {
                info!("Automatic mode selected CPU pixel-wise renderer");
                render_pixel_wise(&input_image, &rendering_options)
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
