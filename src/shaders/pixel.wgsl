struct Options {
    m_in: u32,
    n_in: u32,
    m_out: u32,
    n_out: u32,
    n_monte_carlo: u32,
    grain_seed: u32,
    _pad: vec2<u32>,
    grain_radius: f32,
    sigma_r: f32,
    sigma_filter: f32,
    _pad2: f32,
    x_a: f32,
    y_a: f32,
    x_b: f32,
    y_b: f32,
};

@group(0) @binding(0) var<storage, read> img_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> img_out: array<f32>;
@group(0) @binding(2) var<storage, read> x_offsets: array<f32>;
@group(0) @binding(3) var<storage, read> y_offsets: array<f32>;
@group(0) @binding(4) var<storage, read> lambda_lookup: array<f32>;
@group(0) @binding(5) var<uniform> opts: Options;

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
    if gid.x >= opts.n_out || gid.y >= opts.m_out { return; }

    let n_out_f = f32(opts.n_out);
    let m_out_f = f32(opts.m_out);
    let n_in_f = f32(opts.n_in);
    let m_in_f = f32(opts.m_in);

    let x_in = opts.x_a + (f32(gid.x) + 0.5) * ((opts.x_b - opts.x_a) / n_out_f);
    let y_in = opts.y_a + (f32(gid.y) + 0.5) * ((opts.y_b - opts.y_a) / m_out_f);

    let s_x = (n_out_f - 1.0) / (opts.x_b - opts.x_a);
    let s_y = (m_out_f - 1.0) / (opts.y_b - opts.y_a);

    let cell_size = 1.0 / ceil(1.0 / opts.grain_radius);

    var max_radius = opts.grain_radius;
    if opts.sigma_r > 0.0 {
        let sigma = sqrt(log((opts.sigma_r / opts.grain_radius)*(opts.sigma_r / opts.grain_radius) + 1.0));
        let mu = log(opts.grain_radius) - sigma * sigma / 2.0;
        let normal_q = 3.0902;
        max_radius = exp(mu + sigma * normal_q);
    }

    var success: u32 = 0u;
    var i: u32 = 0u;
    loop {
        if i >= opts.n_monte_carlo { break; }
        let x_shifted = x_in + opts.sigma_filter * (x_offsets[i] / s_x);
        let y_shifted = y_in + opts.sigma_filter * (y_offsets[i] / s_y);
        let min_cell_x = i32(floor((x_shifted - max_radius) / cell_size));
        let max_cell_x = i32(floor((x_shifted + max_radius) / cell_size));
        let min_cell_y = i32(floor((y_shifted - max_radius) / cell_size));
        let max_cell_y = i32(floor((y_shifted + max_radius) / cell_size));
        var cx: i32 = min_cell_x;
        loop {
            if cx > max_cell_x { break; }
            var cy: i32 = min_cell_y;
            loop {
                if cy > max_cell_y { break; }
                let corner_x = f32(cx) * cell_size;
                let corner_y = f32(cy) * cell_size;
                var seed = wang_hash(cell_seed(u32(cx), u32(cy), opts.grain_seed));
                let sx = clamp(i32(floor(corner_x)), 0, i32(n_in_f - 1.0));
                let sy = clamp(i32(floor(corner_y)), 0, i32(m_in_f - 1.0));
                let u = img_in[sy * i32(opts.n_in) + sx];
                let idx = u32(clamp(floor(u * 255.1), 0.0, 255.0));
                let curr_lambda = lambda_lookup[idx];
                let n_cell = select(0u, next_poisson(&seed, curr_lambda), curr_lambda > 0.0);
                var g: u32 = 0u;
                loop {
                    if g >= n_cell { break; }
                    let x_center = corner_x + cell_size * next_f32(&seed);
                    let y_center = corner_y + cell_size * next_f32(&seed);
                    var curr_radius = opts.grain_radius;
                    if opts.sigma_r > 0.0 {
                        let sample_ = next_standard_normal(&seed);
                        let sigma = sqrt(log((opts.sigma_r / opts.grain_radius)*(opts.sigma_r / opts.grain_radius) + 1.0));
                        let mu = log(opts.grain_radius) - sigma * sigma / 2.0;
                        curr_radius = min(exp(mu + sigma * sample_), max_radius);
                    }
                    if sq_distance(x_center, y_center, x_shifted, y_shifted) < curr_radius * curr_radius {
                        success = success + 1u;
                        break;
                    }
                    g = g + 1u;
                }
                cy = cy + 1;
            }
            cx = cx + 1;
        }
        i = i + 1u;
    }
    let index = gid.y * opts.n_out + gid.x;
    img_out[index] = f32(success) / f32(opts.n_monte_carlo);
}
