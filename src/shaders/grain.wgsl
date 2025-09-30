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
@group(0) @binding(1) var<storage, read_write>  output_coverage_counts: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read>        sample_x_offsets: array<f32>;
@group(0) @binding(3) var<storage, read>        sample_y_offsets: array<f32>;
@group(0) @binding(4) var<uniform>              options: Options;

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
    // Bounds check on input image
    if global_id.x >= options.input_width || global_id.y >= options.input_height { return; }

    let input_x = i32(global_id.x);
    let input_y = i32(global_id.y);

    // Clip to active input region
    if f32(input_x) < floor(options.input_region_min_x) ||
       f32(input_x) >= ceil(options.input_region_max_x) ||
       f32(input_y) < floor(options.input_region_min_y) ||
       f32(input_y) >= ceil(options.input_region_max_y) {
        return;
    }

    // Read input intensity
    let input_index = input_y * i32(options.input_width) + input_x;
    let input_intensity = input_image[input_index];

    // Poisson intensity (lambda) setup
    let grain_variance = select(0.0,
                                options.grain_radius_stddev_factor * options.grain_radius_stddev_factor,
                                options.grain_radius_stddev_factor > 0.0);
    let poisson_denominator = PI * (options.grain_radius * options.grain_radius + grain_variance);

    var lambda = 0.0;
    if input_intensity < 1.0 {
        lambda = (1.0 / poisson_denominator) * log(1.0 / (1.0 - input_intensity));
    }

    // RNG seed per input cell
    var rng_state = wang_hash(cell_seed(global_id.x, global_id.y, options.grain_seed_offset));
    let grains_in_cell = select(0u, next_poisson(&rng_state, lambda), lambda > 0.0);

    // Input->output scale (continuous coordinates)
    let scale_x = (f32(options.output_width)  - 1.0) / (options.input_region_max_x - options.input_region_min_x);
    let scale_y = (f32(options.output_height) - 1.0) / (options.input_region_max_y - options.input_region_min_y);

    // Lognormal draw parameters for grain radii (if stddev > 0)
    var lognormal_sigma = 0.0;
    var lognormal_mu = 0.0;
    var max_grain_radius = options.grain_radius;
    if options.grain_radius_stddev_factor > 0.0 {
        lognormal_sigma = sqrt(log((options.grain_radius_stddev_factor / options.grain_radius) *
                                   (options.grain_radius_stddev_factor / options.grain_radius) + 1.0));
        lognormal_mu = log(options.grain_radius) - lognormal_sigma * lognormal_sigma / 2.0;

        // ~99.8% quantile clamp
        let normal_q = 3.0902;
        max_grain_radius = exp(lognormal_mu + lognormal_sigma * normal_q);
    }

    // For each grain in this input cell…
    var grain_index: u32 = 0u;
    loop {
        if grain_index >= grains_in_cell { break; }

        // Grain center within the input cell (jittered in [0,1))
        let grain_center_x_in = f32(input_x) + next_f32(&rng_state);
        let grain_center_y_in = f32(input_y) + next_f32(&rng_state);

        // Sample grain radius
        var grain_radius = options.grain_radius;
        if options.grain_radius_stddev_factor > 0.0 {
            let normal_sample = next_standard_normal(&rng_state);
            grain_radius = min(exp(lognormal_mu + lognormal_sigma * normal_sample), max_grain_radius);
        }

        // Monte Carlo samples per grain
        var sample_index: u32 = 0u;
        loop {
            if sample_index >= options.monte_carlo_sample_count { break; }

            // Shift grain by sample offset (in input space), then project to output space
            let shifted_x_in = grain_center_x_in - (sample_x_offsets[sample_index] / scale_x);
            let shifted_y_in = grain_center_y_in - (sample_y_offsets[sample_index] / scale_y);

            let projected_x = (shifted_x_in - options.input_region_min_x) * scale_x;
            let projected_y = (shifted_y_in - options.input_region_min_y) * scale_y;
            let projected_radius = grain_radius * scale_x;

            let min_x = i32(ceil(projected_x - projected_radius));
            let max_x = i32(floor(projected_x + projected_radius));
            let min_y = i32(ceil(projected_y - projected_radius));
            let max_y = i32(floor(projected_y + projected_radius));

            var out_y: i32 = min_y;
            loop {
                if out_y > max_y { break; }
                if out_y >= 0 && out_y < i32(options.output_height) {
                    var out_x: i32 = min_x;
                    loop {
                        if out_x > max_x { break; }
                        if out_x >= 0 && out_x < i32(options.output_width) {
                            // Pixel center (in output pixel space)
                            let pixel_center_x = f32(out_x) + 0.5;
                            let pixel_center_y = f32(out_y) + 0.5;

                            if squared_distance(pixel_center_x, pixel_center_y, projected_x, projected_y)
                               <= projected_radius * projected_radius {
                                let out_pixel_index = u32(out_y) * options.output_width + u32(out_x);
                                atomicAdd(&output_coverage_counts[out_pixel_index], 1u);
                            }
                        }
                        out_x = out_x + 1;
                    }
                }
                out_y = out_y + 1;
            }

            sample_index = sample_index + 1u;
        }

        grain_index = grain_index + 1u;
    }
}
