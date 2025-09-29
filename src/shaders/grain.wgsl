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

@group(0) @binding(0) var<storage, read> img_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> flags: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> x_offsets: array<f32>;
@group(0) @binding(3) var<storage, read> y_offsets: array<f32>;
@group(0) @binding(4) var<uniform> opts: Options;

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
    return sqrt(-2.0 * log(u)) * cos(6.28318530718 * v);
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

fn cell_seed(x: u32, y: u32, offset: u32) -> u32 {
    let period: u32 = 65536u;
    var s = ((y % period) * period + (x % period)) + offset;
    if s == 0u { s = 1u; }
    return s;
}

fn sq_distance(x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    let dx = x1 - x2;
    let dy = y1 - y2;
    return dx * dx + dy * dy;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= opts.n_in || gid.y >= opts.m_in { return; }

    let j = i32(gid.x);
    let i = i32(gid.y);

    if f32(j) < floor(opts.x_a) || f32(j) >= ceil(opts.x_b) ||
       f32(i) < floor(opts.y_a) || f32(i) >= ceil(opts.y_b) {
        return;
    }

    let idx_in = i * i32(opts.n_in) + j;
    let u = img_in[idx_in];

    let grain_var = select(0.0, opts.sigma_r * opts.sigma_r, opts.sigma_r > 0.0);
    let denom = 3.14159265359 * (opts.grain_radius * opts.grain_radius + grain_var);
    var lambda = 0.0;
    if u < 1.0 {
        lambda = (1.0 / denom) * log(1.0 / (1.0 - u));
    }

    var seed = wang_hash(cell_seed(gid.x, gid.y, opts.grain_seed));
    let n_cell = select(0u, next_poisson(&seed, lambda), lambda > 0.0);

    let s_x = (f32(opts.n_out) - 1.0) / (opts.x_b - opts.x_a);
    let s_y = (f32(opts.m_out) - 1.0) / (opts.y_b - opts.y_a);

    var sigma = 0.0;
    var mu = 0.0;
    var max_radius = opts.grain_radius;
    if opts.sigma_r > 0.0 {
        sigma = sqrt(log((opts.sigma_r / opts.grain_radius)*(opts.sigma_r / opts.grain_radius) + 1.0));
        mu = log(opts.grain_radius) - sigma * sigma / 2.0;
        let normal_q = 3.0902;
        max_radius = exp(mu + sigma * normal_q);
    }

    var g: u32 = 0u;
    loop {
        if g >= n_cell { break; }
        let x_center = f32(j) + next_f32(&seed);
        let y_center = f32(i) + next_f32(&seed);
        var radius = opts.grain_radius;
        if opts.sigma_r > 0.0 {
            let sample_ = next_standard_normal(&seed);
            radius = min(exp(mu + sigma * sample_), max_radius);
        }
        var k: u32 = 0u;
        loop {
            if k >= opts.n_monte_carlo { break; }
            let x_shifted = x_center - (x_offsets[k] / s_x);
            let y_shifted = y_center - (y_offsets[k] / s_y);
            let x_proj = (x_shifted - opts.x_a) * s_x;
            let y_proj = (y_shifted - opts.y_a) * s_y;
            let r_proj = radius * s_x;
            let min_x = i32(ceil(x_proj - r_proj));
            let max_x = i32(floor(x_proj + r_proj));
            let min_y = i32(ceil(y_proj - r_proj));
            let max_y = i32(floor(y_proj + r_proj));
            var oy: i32 = min_y;
            loop {
                if oy > max_y { break; }
                if oy >= 0 && oy < i32(opts.m_out) {
                    var ox: i32 = min_x;
                    loop {
                        if ox > max_x { break; }
                        if ox >= 0 && ox < i32(opts.n_out) {
                            let px = f32(ox) + 0.5;
                            let py = f32(oy) + 0.5;
                            if sq_distance(px, py, x_proj, y_proj) <= r_proj * r_proj {
                                let pixel_idx = u32(oy) * opts.n_out + u32(ox);
                                atomicAdd(&flags[pixel_idx], 1u);
                            }
                        }
                        ox = ox + 1;
                    }
                }
                oy = oy + 1;
            }
            k = k + 1u;
        }
        g = g + 1u;
    }
}
