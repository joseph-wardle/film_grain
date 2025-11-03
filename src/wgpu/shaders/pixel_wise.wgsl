struct Uniforms {
    in_w: u32,
    in_h: u32,
    out_w: u32,
    out_h: u32,
    n_samples: u32,
    lanes: u32,
    seed: u32,
    _pad0: u32,
    s: f32,
    delta: f32,
    rm: f32,
    inv_e_pi_r2: f32,
    dist_kind: u32,
    _pad1: u32,
    radius_mean: f32,
    radius_log_mu: f32,
    radius_log_sigma: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<storage, read> lambda_in: array<f32>;
@group(0) @binding(1) var<storage, read> offsets: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> U: Uniforms;
@group(0) @binding(3) var<storage, read_write> out_accum: array<f32>;

fn rotl(x: u32, k: u32) -> u32 {
    return (x << k) | (x >> (32u - k));
}

fn hash3(a: u32, b: u32, c: u32) -> u32 {
    var v = a ^ 0x9E3779B9u;
    v = v + b;
    v = (v ^ (v >> 16u)) * 0x7FEB352Du;
    v = (v ^ (v >> 15u)) * 0x846CA68Bu;
    v = v ^ (v >> 16u);
    v = v + c * 0x9E3779B1u;
    v = (v ^ (v >> 13u)) * 0xC2B2AE35u;
    return v ^ (v >> 16u);
}

fn splitmix_step(state: ptr<function, u32>) -> u32 {
    (*state) = (*state) + 0x9E3779B9u;
    var z = (*state);
    z = (z ^ (z >> 16u)) * 0x85EBCA6Bu;
    z = (z ^ (z >> 13u)) * 0xC2B2AE35u;
    return z ^ (z >> 16u);
}

struct Rng {
    s0: u32,
    s1: u32,
    s2: u32,
    s3: u32,
};

fn rng_init(seed: u32) -> Rng {
    var sm = seed;
    return Rng(
        splitmix_step(&sm),
        splitmix_step(&sm),
        splitmix_step(&sm),
        splitmix_step(&sm),
    );
}

fn rng_next_u32(r: ptr<function, Rng>) -> u32 {
    let result = rotl((*r).s1 * 5u, 7u) * 9u;
    let t = (*r).s1 << 9u;
    (*r).s2 = (*r).s2 ^ (*r).s0;
    (*r).s3 = (*r).s3 ^ (*r).s1;
    (*r).s1 = (*r).s1 ^ (*r).s2;
    (*r).s0 = (*r).s0 ^ (*r).s3;
    (*r).s2 = (*r).s2 ^ t;
    (*r).s3 = rotl((*r).s3, 11u);
    return result;
}

fn uniform01(r: ptr<function, Rng>) -> f32 {
    let value = rng_next_u32(r) >> 8u;
    return f32(value) * (1.0 / 16777216.0);
}

fn box_muller2(r: ptr<function, Rng>) -> vec2<f32> {
    let u1 = max(1e-6, uniform01(r));
    let u2 = uniform01(r);
    let mag = sqrt(-2.0 * log(u1));
    let theta = 6.2831855 * u2;
    return vec2<f32>(mag * cos(theta), mag * sin(theta));
}

fn sample_radius(
    r: ptr<function, Rng>,
    dist: u32,
    mean: f32,
    log_mu: f32,
    log_sigma: f32,
) -> f32 {
    if (dist == 0u) {
        return mean;
    }
    let normal = box_muller2(r).x;
    return exp(normal * log_sigma + log_mu);
}

fn sample_poisson(r: ptr<function, Rng>, lambda: f32) -> u32 {
    if (lambda <= 0.0) {
        return 0u;
    }
    if (lambda < 16.0) {
        let l = exp(-lambda);
        var p = 1.0;
        var k: u32 = 0u;
        loop {
            k = k + 1u;
            p = p * uniform01(r);
            if (p <= l) {
                return k - 1u;
            }
            if (k > 1024u) {
                return k;
            }
        }
    }
    let normal = box_muller2(r).x;
    let approx = floor(lambda + sqrt(lambda) * normal + 0.5);
    return u32(max(approx, 0.0));
}

fn clamp_to_image(ix: i32, iy: i32, w: i32, h: i32) -> vec2<u32> {
    let cx = clamp(ix, 0, w - 1);
    let cy = clamp(iy, 0, h - 1);
    return vec2<u32>(u32(cx), u32(cy));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= U.out_w || gid.y >= U.out_h) {
        return;
    }

    let px = f32(gid.x);
    let py = f32(gid.y);
    let inv_zoom = 1.0 / U.s;
    var sum = 0.0;

    let rm = U.rm;
    let delta = U.delta;
    let in_w_i = i32(U.in_w);
    let in_h_i = i32(U.in_h);

    for (var k: u32 = 0u; k < U.n_samples; k = k + 1u) {
        let off = offsets[k];
        let xg = (px + 0.5) * inv_zoom - off.x;
        let yg = (py + 0.5) * inv_zoom - off.y;

        var i0 = i32(floor((xg - rm) / delta));
        var i1 = i32(floor((xg + rm) / delta));
        var j0 = i32(floor((yg - rm) / delta));
        var j1 = i32(floor((yg + rm) / delta));

        if (i0 > i1 || j0 > j1) {
            continue;
        }

        var hit = false;
        for (var ii = i0; ii <= i1 && !hit; ii = ii + 1) {
            for (var jj = j0; jj <= j1 && !hit; jj = jj + 1) {
                var rng = rng_init(hash3(U.seed, bitcast<u32>(ii), bitcast<u32>(jj)));
                let sx = i32(floor(f32(ii) * delta));
                let sy = i32(floor(f32(jj) * delta));
                let cell = clamp_to_image(sx, sy, in_w_i, in_h_i);
                let idx = cell.y * U.in_w + cell.x;
                let lambda_cell = max(lambda_in[idx], 0.0) * delta * delta;
                let q = sample_poisson(&rng, lambda_cell);
                if (q == 0u) {
                    continue;
                }
                for (var t: u32 = 0u; t < q && !hit; t = t + 1u) {
                    let cx = f32(ii) * delta + uniform01(&rng) * delta;
                    let cy = f32(jj) * delta + uniform01(&rng) * delta;
                    var radius = sample_radius(
                        &rng,
                        U.dist_kind,
                        U.radius_mean,
                        U.radius_log_mu,
                        U.radius_log_sigma,
                    );
                    radius = min(radius, U.rm);
                    if (radius <= 0.0) {
                        continue;
                    }
                    let dx = xg - cx;
                    let dy = yg - cy;
                    if (dx * dx + dy * dy <= radius * radius) {
                        hit = true;
                    }
                }
            }
        }

        if (hit) {
            sum = sum + 1.0;
        }
    }

    let out_idx = gid.y * U.out_w + gid.x;
    out_accum[out_idx] = sum / f32(U.n_samples);
}
