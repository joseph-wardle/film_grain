// =================================================================================================
// pixel_wise.wgsl
// Monte-Carlo "pixel-wise" coverage (Algorithms 2 & 3).
// One thread = one output pixel. No atomics. Deterministic RNG.
// Entry points:
//   - @compute fn pixel_main(...)       // writes integer hit counts into `acc`
//   - @compute fn normalize_main(...)   // converts `acc` -> rgba8unorm texture
// =================================================================================================

override WORKGROUP_X : u32 = 16;
override WORKGROUP_Y : u32 = 8;

struct Uniforms {
  in_size     : vec2<u32>;  // input size (m, n)
  out_size    : vec2<u32>;  // output size (sm, sn)
  zoom_s      : f32;        // s
  sigma_px    : f32;        // σ in output pixels (host uses to build ξ)
  mean_r      : f32;        // if radius_kind=0: μ_r ; if 1: μ_log (lognormal)
  std_r       : f32;        // if radius_kind=0: σ_r ; if 1: σ_log (lognormal)
  r_max       : f32;        // clamp (& culling pad)
  cell_size   : f32;        // δ (input pixels)
  n_samples   : u32;        // N
  seed        : u32;        // RNG seed
  grid_dims   : vec2<u32>;  // not used here; kept for uniform parity
  radius_kind : u32;        // 0 = const/normal, 1 = lognormal
  _padX       : vec3<u32>;  // alignment
};

// Bindings for pixel_main
@group(0) @binding(0) var<uniform>            U          : Uniforms;
@group(0) @binding(1) var                      in_tex     : texture_2d<f32>;
@group(0) @binding(2) var<storage, read>       xi         : array<vec2<f32>>;
@group(0) @binding(3) var<storage, read>       lambda_lut : array<f32>;     // 256 entries
@group(0) @binding(4) var<storage, read_write> acc        : array<u32>;     // out_w*out_h

// Bindings for normalize_main
@group(1) @binding(0) var<uniform>            U_norm     : Uniforms;
@group(1) @binding(1) var<storage, read>      acc_in     : array<u32>;
@group(1) @binding(2) var                       out_img   : texture_storage_2d<rgba8unorm, write>;

// --------------------------------- helpers ---------------------------------
fn clamp_i32(x : i32, lo : i32, hi : i32) -> i32 { return min(max(x, lo), hi); }
fn flatten2(ix : u32, iy : u32, nx : u32) -> u32 { return iy * nx + ix; }

fn mix32(x : u32) -> u32 {
  var v = x ^ 0x9E3779B9u;
  v ^= (v >> 16);  v *= 0x7FEB352Du;
  v ^= (v >> 15);  v *= 0x846CA68Bu;
  v ^= (v >> 16);
  return v;
}
fn key4(a : u32, b : u32, c : u32, d : u32) -> u32 {
  return mix32(a ^ (b * 0xA24BAEDCu) ^ (c * 0x9E3779B1u) ^ (d * 0x85EBCA6Bu));
}
fn u32_to_unit_f32(x : u32) -> f32 { return f32(x) * 2.3283064365386963e-10; } // 1/2^32
fn rng01(a:u32,b:u32,c:u32,d:u32)->f32 { return u32_to_unit_f32(key4(a,b,c,d)); }

fn normal2(u1 : f32, u2 : f32) -> vec2<f32> {
  let uu = max(u1, 1e-7);
  let r  = sqrt(-2.0 * log(uu));
  let t  = 6.283185307179586 * u2; // 2π
  return vec2<f32>(r * cos(t), r * sin(t));
}

// radius_kind: 0 = const/normal (mean_r, std_r in linear units)
//               1 = lognormal (mean_r=μ_log, std_r=σ_log)
fn draw_radius(mean_r:f32, std_r:f32, r_max:f32,
               a:u32,b:u32,c:u32,d:u32, radius_kind:u32) -> f32 {
  let z  = normal2(rng01(a,b,c,d), rng01(a^1u,b^3u,c^5u,d^7u)).x;
  let base = mean_r + std_r * z;
  let r_lin = max(base, 0.0);
  let r_logn = exp(base);
  let rr = select(r_lin, r_logn, radius_kind == 1u);
  return min(rr, r_max);
}

fn poisson(lambda : f32, key : u32) -> u32 {
  if (lambda <= 0.0) { return 0u; }
  if (lambda < 12.0) {
    var L = exp(-lambda);
    var k : u32 = 0u;
    var p : f32 = 1.0;
    loop {
      k = k + 1u;
      p = p * (1.0 - rng01(key, k, 0u, 0u));
      if (p <= L) { break; }
      if (k > 1000u) { break; }
    }
    return k - 1u;
  } else {
    let n = normal2(rng01(key,1u,0u,0u), rng01(key,2u,0u,0u)).x;
    let v = lambda + sqrt(lambda) * n;
    return u32(max(v, 0.0));
  }
}

fn luma_bucket_at(px : vec2<i32>) -> u32 {
  let sz  = vec2<i32>(i32(U.in_size.x), i32(U.in_size.y));
  let pxc = vec2<i32>( clamp_i32(px.x, 0, sz.x - 1), clamp_i32(px.y, 0, sz.y - 1) );
  let rgba = textureLoad(in_tex, pxc, 0);
  let y = clamp(dot(rgba.rgb, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
  return u32(round(y * 255.0));
}

// Sample λ for a δ-cell from its center for stability.
fn cell_lambda(cx : i32, cy : i32) -> f32 {
  let center = (vec2<f32>(f32(cx) + 0.5, f32(cy) + 0.5)) * U.cell_size;
  let px = vec2<i32>(i32(floor(center.x)), i32(floor(center.y)));
  return lambda_lut[luma_bucket_at(px)];
}

// --------------------------------- kernels ---------------------------------
@compute @workgroup_size(WORKGROUP_X, WORKGROUP_Y, 1)
fn pixel_main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= U.out_size.x || gid.y >= U.out_size.y) { return; }

  let out_xy  = vec2<u32>(gid.xy);
  let out_idx = flatten2(out_xy.x, out_xy.y, U.out_size.x);
  let y_out   = vec2<f32>(vec2<u32>(out_xy));

  var hits : u32 = 0u;

  for (var k : u32 = 0u; k < U.n_samples; k = k + 1u) {
    let xg = (y_out + xi[k]) / U.zoom_s;

    let r    = U.r_max;
    let cmin = floor((xg - vec2<f32>(r)) / U.cell_size);
    let cmax = floor((xg + vec2<f32>(r)) / U.cell_size);

    var covered = false;

    for (var cy = i32(cmin.y); cy <= i32(cmax.y) && !covered; cy = cy + 1) {
      for (var cx = i32(cmin.x); cx <= i32(cmax.x) && !covered; cx = cx + 1) {
        let key = key4(U.seed, u32(cx), u32(cy), k);
        let lam = cell_lambda(cx, cy);

        let Q = poisson(lam, key);
        for (var j : u32 = 0u; j < Q && !covered; j = j + 1u) {
          let gx  = (f32(cx) + rng01(key, j, 1u, 0u)) * U.cell_size;
          let gy  = (f32(cy) + rng01(key, j, 2u, 0u)) * U.cell_size;
          let rad = draw_radius(U.mean_r, U.std_r, U.r_max, key, j, 3u, 0u, U.radius_kind);

          let dx = xg.x - gx;
          let dy = xg.y - gy;
          covered = (dx*dx + dy*dy) <= (rad*rad);
        }
      }
    }

    if (covered) { hits = hits + 1u; }
  }

  acc[out_idx] = hits;
}

@compute @workgroup_size(WORKGROUP_X, WORKGROUP_Y, 1)
fn normalize_main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= U_norm.out_size.x || gid.y >= U_norm.out_size.y) { return; }
  let idx = flatten2(gid.x, gid.y, U_norm.out_size.x);
  let n   = max(U_norm.n_samples, 1u);
  let v   = f32(acc_in[idx]) / f32(n);
  textureStore(out_img, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(v, v, v, 1.0));
}
