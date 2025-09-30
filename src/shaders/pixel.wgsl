struct Options {
    input_height: u32,
    input_width: u32,
    output_height: u32,
    output_width: u32,
    monte_carlo_sample_count: u32,
    grain_seed_offset: u32,
    padding: vec2<u32>,
    grain_radius: f32,
    grain_radius_stddev_factor: f32,
    gaussian_filter_stddev: f32,
    padding_f32: f32,
    input_region_min_x: f32,
    input_region_min_y: f32,
    input_region_max_x: f32,
    input_region_max_y: f32,
};

@group(0) @binding(0) var<storage, read>        input_image: array<f32>;
@group(0) @binding(1) var<storage, read_write>  output_image: array<f32>;
@group(0) @binding(2) var<storage, read>        sample_x_offsets: array<f32>;
@group(0) @binding(3) var<storage, read>        sample_y_offsets: array<f32>;
@group(0) @binding(4) var<storage, read>        lambda_lut: array<f32>;
@group(0) @binding(5) var<uniform>              options: Options;

const PI  : f32 = 3.14159265359;
const TAU : f32 = 6.28318530718;

fn wang_hash(seed: u32) -> u32 {
    var s = seed;
    s = (s ^ 61u) ^ (s >> 16u);
    s = s * 9u;
    s = s ^ (s >> 4u);
    s = s * 668265261u;
    s = s ^ (s >> 15u);
    return s;
}

fn next_u32(state: ptr<function, u32>) -> u32 {
    (*state) ^= (*state) << 13u;
    (*state) ^= (*state) >> 17u;
    (*state) ^= (*state) << 5u;
    return *state;
}

fn next_f32(state: ptr<function, u32>) -> f32 {
    let v = next_u32(state);
    return f32(v) / 4294967295.0;
}

fn next_standard_normal(state: ptr<function, u32>) -> f32 {
    let u = next_f32(state);
    let v = next_f32(state);
    return sqrt(-2.0 * log(u)) * cos(TAU * v);
}

fn next_poisson(state: ptr<function, u32>, lambda: f32) -> u32 {
    var u = next_f32(state);
    var x: u32 = 0u;
    var prod = exp(-lambda);
    var sum = prod;
    let limit = u32(floor(10000.0 * lambda));
    loop {
        if u <= sum || x >= limit { break; }
        x = x + 1u;
        prod = prod * lambda / f32(x);
        sum = sum + prod;
    }
    return x;
}

fn cell_seed(cell_x: u32, cell_y: u32, offset: u32) -> u32 {
    let period: u32 = 65536u;
    var s = ((cell_y % period) * period + (cell_x % period)) + offset;
    if s == 0u { s = 1u; }
    return s;
}

fn squared_distance(x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    let dx = x1 - x2;
    let dy = y1 - y2;
    return dx * dx + dy * dy;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Bounds check on output image domain
    if global_id.x >= options.output_width || global_id.y >= options.output_height { return; }

    let out_width_f  = f32(options.output_width);
    let out_height_f = f32(options.output_height);
    let in_width_f   = f32(options.input_width);
    let in_height_f  = f32(options.input_height);

    // Continuous input-space position corresponding to this output pixel center
    let x_in = options.input_region_min_x + (f32(global_id.x) + 0.5) *
               ((options.input_region_max_x - options.input_region_min_x) / out_width_f);
    let y_in = options.input_region_min_y + (f32(global_id.y) + 0.5) *
               ((options.input_region_max_y - options.input_region_min_y) / out_height_f);

    // Input->output scale
    let scale_x = (out_width_f  - 1.0) / (options.input_region_max_x - options.input_region_min_x);
    let scale_y = (out_height_f - 1.0) / (options.input_region_max_y - options.input_region_min_y);

    // Grid cell size for grain placement in input space
    let cell_size = 1.0 / ceil(1.0 / options.grain_radius);

    // Compute a conservative max grain radius (for neighborhood search)
    var max_grain_radius = options.grain_radius;
    if options.grain_radius_stddev_factor > 0.0 {
        let lognormal_sigma = sqrt(
            log((options.grain_radius_stddev_factor / options.grain_radius) *
                (options.grain_radius_stddev_factor / options.grain_radius) + 1.0)
        );
        let lognormal_mu = log(options.grain_radius) - lognormal_sigma * lognormal_sigma / 2.0;
        let normal_q = 3.0902; // ~99.8% quantile
        max_grain_radius = exp(lognormal_mu + lognormal_sigma * normal_q);
    }

    var success: u32 = 0u;

    // Monte Carlo samples per output pixel
    var sample_index: u32 = 0u;
    loop {
        if sample_index >= options.monte_carlo_sample_count { break; }

        // Offset the input point by a Gaussian filter standard deviation scaled offset
        let x_shifted = x_in + options.gaussian_filter_stddev * (sample_x_offsets[sample_index] / scale_x);
        let y_shifted = y_in + options.gaussian_filter_stddev * (sample_y_offsets[sample_index] / scale_y);

        // Determine overlapping grain cells to test
        let min_cell_x = i32(floor((x_shifted - max_grain_radius) / cell_size));
        let max_cell_x = i32(floor((x_shifted + max_grain_radius) / cell_size));
        let min_cell_y = i32(floor((y_shifted - max_grain_radius) / cell_size));
        let max_cell_y = i32(floor((y_shifted + max_grain_radius) / cell_size));

        var cell_x: i32 = min_cell_x;
        loop {
            if cell_x > max_cell_x { break; }

            var cell_y: i32 = min_cell_y;
            loop {
                if cell_y > max_cell_y { break; }

                let corner_x = f32(cell_x) * cell_size;
                let corner_y = f32(cell_y) * cell_size;

                // RNG seed per cell
                var rng_state = wang_hash(cell_seed(u32(cell_x), u32(cell_y), options.grain_seed_offset));

                // Look up Poisson intensity (lambda) from the input image at the cell corner
                let sx = clamp(i32(floor(corner_x)), 0, i32(in_width_f  - 1.0));
                let sy = clamp(i32(floor(corner_y)), 0, i32(in_height_f - 1.0));
                let u = input_image[sy * i32(options.input_width) + sx];

                let lut_index = u32(clamp(floor(u * 255.1), 0.0, 255.0));
                let lambda_value = lambda_lut[lut_index];

                let grains_in_cell = select(0u, next_poisson(&rng_state, lambda_value), lambda_value > 0.0);

                // Sample grains in this cell
                var grain_i: u32 = 0u;
                loop {
                    if grain_i >= grains_in_cell { break; }

                    // Grain center uniformly within the cell
                    let x_center = corner_x + cell_size * next_f32(&rng_state);
                    let y_center = corner_y + cell_size * next_f32(&rng_state);

                    // Grain radius (possibly lognormal)
                    var grain_radius = options.grain_radius;
                    if options.grain_radius_stddev_factor > 0.0 {
                        let normal_sample = next_standard_normal(&rng_state);
                        let lognormal_sigma = sqrt(
                            log((options.grain_radius_stddev_factor / options.grain_radius) *
                                (options.grain_radius_stddev_factor / options.grain_radius) + 1.0)
                        );
                        let lognormal_mu = log(options.grain_radius) - lognormal_sigma * lognormal_sigma / 2.0;
                        grain_radius = min(exp(lognormal_mu + lognormal_sigma * normal_sample), max_grain_radius);
                    }

                    // Hit test
                    if squared_distance(x_center, y_center, x_shifted, y_shifted) < grain_radius * grain_radius {
                        success = success + 1u;
                        break;
                    }

                    grain_i = grain_i + 1u;
                }

                cell_y = cell_y + 1;
            }

            cell_x = cell_x + 1;
        }

        sample_index = sample_index + 1u;
    }

    let out_index = global_id.y * options.output_width + global_id.x;
    output_image[out_index] = f32(success) / f32(options.monte_carlo_sample_count);
}
