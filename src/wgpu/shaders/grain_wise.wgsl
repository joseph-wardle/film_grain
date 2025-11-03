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
@group(0) @binding(3) var<storage, read_write> bitset: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> out_accum: array<f32>;

fn hash3(a: u32, b: u32, c: u32) -> u32 {
    var x = a ^ (b << 7u) ^ (c << 13u);
    x = x ^ (x >> 17u);
    x = x * 0x9E3779B1u;
    x = x ^ (x >> 15u);
    x = x * 0x85EBCA77u;
    x = x ^ (x >> 13u);
    return x;
}

struct Rng {
    state: u32,
};

fn rng_init(seed: u32) -> Rng {
    return Rng(hash3(seed, 0xC2B2AE35u, 0x27D4EB2Fu));
}

fn rng_next_u32(r: ptr<function, Rng>) -> u32 {
    (*r).state = (*r).state ^ ((*r).state << 13u);
    (*r).state = (*r).state ^ ((*r).state >> 17u);
    (*r).state = (*r).state ^ ((*r).state << 5u);
    return (*r).state;
}

fn uniform01(r: ptr<function, Rng>) -> f32 {
    let value = rng_next_u32(r);
    return f32(value) * (1.0 / 4294967296.0);
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

fn clamp_range(value: i32, min_val: i32, max_val: i32) -> i32 {
    return clamp(value, min_val, max_val);
}

@compute @workgroup_size(8, 8)
fn splat(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= U.in_w || gid.y >= U.in_h) {
        return;
    }

    let ix = i32(gid.x);
    let iy = i32(gid.y);

    var rng = rng_init(hash3(U.seed ^ 0x55AA55AAu, u32(ix), u32(iy)));
    let lam = max(lambda_in[gid.y * U.in_w + gid.x], 0.0);
    let q = sample_poisson(&rng, lam);
    if (q == 0u) {
        return;
    }

    for (var t: u32 = 0u; t < q; t = t + 1u) {
        let cx = f32(ix) + uniform01(&rng);
        let cy = f32(iy) + uniform01(&rng);
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
        let R = radius * U.s;

        for (var k: u32 = 0u; k < U.n_samples; k = k + 1u) {
            let off = offsets[k];
            let tx = cx * U.s + off.x;
            let ty = cy * U.s + off.y;

            let xmin = i32(ceil((tx - R) - 0.5));
            let xmax = i32(floor((tx + R) - 0.5));
            let ymin = i32(ceil((ty - R) - 0.5));
            let ymax = i32(floor((ty + R) - 0.5));

            if (xmax < 0 || ymax < 0 || xmin >= i32(U.out_w) || ymin >= i32(U.out_h)) {
                continue;
            }

            let x0 = clamp_range(xmin, 0, i32(U.out_w) - 1);
            let x1 = clamp_range(xmax, 0, i32(U.out_w) - 1);
            let y0 = clamp_range(ymin, 0, i32(U.out_h) - 1);
            let y1 = clamp_range(ymax, 0, i32(U.out_h) - 1);

            let lane = k >> 5u;
            let mask = 1u << (k & 31u);

            for (var y = y0; y <= y1; y = y + 1) {
                let cy2 = f32(y) + 0.5 - ty;
                let cy2_sq = cy2 * cy2;
                if (cy2_sq > R * R) {
                    continue;
                }
                for (var x = x0; x <= x1; x = x + 1) {
                    let cx2 = f32(x) + 0.5 - tx;
                    if (cx2 * cx2 + cy2_sq <= R * R) {
                        let idx = u32(y) * U.out_w + u32(x);
                        let slot = idx * U.lanes + lane;
                        atomicOr(&bitset[slot], mask);
                    }
                }
            }
        }
    }
}

@compute @workgroup_size(16, 16)
fn reduce(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= U.out_w || gid.y >= U.out_h) {
        return;
    }
    let idx = gid.y * U.out_w + gid.x;
    var count: u32 = 0u;
    for (var lane: u32 = 0u; lane < U.lanes; lane = lane + 1u) {
        let bits = atomicLoad(&bitset[idx * U.lanes + lane]);
        count = count + countOneBits(bits);
    }
    out_accum[idx] = f32(count) / f32(U.n_samples);
}
