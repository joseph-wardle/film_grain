// =================================================================================================
// grain_wise.wgsl
// Monte-Carlo "grain-wise" with binning + tile draw.
// Entry points:
//   - @compute fn grain_gen_main(...)   // builds δ-grid (counts + items)
//   - @compute fn grain_draw_main(...)  // accumulates hit counts into `acc`
//   - @compute fn normalize_main(...)   // converts `acc` -> rgba8unorm texture
// =================================================================================================

override WORKGROUP_X : u32 = 16;
override WORKGROUP_Y : u32 = 8;
override TILE_W      : u32 = 16;
override TILE_H      : u32 = 16;
override CELL_CAPACITY : u32 = 64; // per δ-cell slots

struct Uniforms {
  in_size     : vec2<u32>;
  out_size    : vec2<u32>;
  zoom_s      : f32;
  sigma_px    : f32;
  mean_r      : f32;        // normal: μ_r ; lognorm: μ_log
  std_r       : f32;        // normal: σ_r ; lognorm: σ_log
  r_max       : f32;
  cell_size   : f32;
  n_samples   : u32;
  seed        : u32;
  grid_dims   : vec2<u32>;  // (#cells_x, #cells_y)
  radius_kind : u32;        // 0 = const/normal, 1 = lognormal
  _padX       : vec3<u32>;
};

struct Grain { center: vec2<f32>; radius: f32; _pad: f32; };

// ---- Bindings ----
// grain_gen_main
@group(0) @binding(0) var<uniform>            U            : Uniforms;
@group(0) @binding(1) var                      in_tex       : texture_2d<f32>;
@group(0) @binding(2) var<storage, read>       lambda_lut   : array<f32>;
@group(0) @binding(3) var<storage, read_write> cell_counts  : array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> cell_items   : array<Grain>;
@group(0) @binding(5) var<storage, read_write> overflow_flag: array<atomic<u32>>; // len=1

// grain_draw_main
@group(1) @binding(0) var<uniform>            U2           : Uniforms;
@group(1) @binding(1) var<storage, read>      xi           : array<vec2<f32>>;
@group(1) @binding(2) var<storage, read>      cell_counts_r: array<u32>;
@group(1) @binding(3) var<storage, read>      cell_items_r : array<Grain>;
@group(1) @binding(4) var<storage, read_write> acc         : array<u32>;

// normalize_main
@group(2) @binding(0) var<uniform>            U_norm       : Uniforms;
@group(2) @binding(1) var<storage, read>      acc_in       : array<u32>;
@group(2) @binding(2) var                       out_img     : texture_storage_2d<rgba8unorm, write>;

// ---- helpers ----
fn clamp_i32(x : i32, lo : i32, hi : i32) -> i32 { return min(max(x, lo), hi); }
fn flatten2(ix : u32, iy : u32, nx : u32) -> u32 { return iy * nx + ix; }

fn mix32(x : u32) -> u32 {
  var v = x ^ 0x9E3779B9u;
  v ^= (v >> 16);  v *= 0x7FEB352Du;
  v ^= (v >> 15);  v *= 0x846CA68Bu;
  v ^= (v >> 16);
  return v;
}
fn key4(a:u32,b:u32,c:u32,d:u32)->u32 {
  return mix32(a ^ (b*0xA24BAEDCu) ^ (c*0x9E3779B1u) ^ (d*0x85EBCA6Bu));
}
fn u32_to_unit_f32(x : u32) -> f32 { return f32(x) * 2.3283064365386963e-10; }
fn rng01(a:u32,b:u32,c:u32,d:u32)->f32 { return u32_to_unit_f32(key4(a,b,c,d)); }

fn normal2(u1:f32,u2:f32)->vec2<f32>{
  let uu = max(u1, 1e-7);
  let r  = sqrt(-2.0 * log(uu));
  let t  = 6.283185307179586 * u2;
  return vec2<f32>(r * cos(t), r * sin(t));
}

// radius_kind: 0 = const/normal ; 1 = lognormal
fn draw_radius(mean_r:f32, std_r:f32, r_max:f32,
               a:u32,b:u32,c:u32,d:u32, radius_kind:u32) -> f32 {
  let z  = normal2(rng01(a,b,c,d), rng01(a^1u,b^3u,c^5u,d^7u)).x;
  let base = mean_r + std_r * z;
  let r_lin  = max(base, 0.0);
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

fn clamp_cell(cx : i32, cy : i32, dims : vec2<u32>) -> vec2<u32> {
  let x = u32(clamp_i32(cx, 0, i32(dims.x) - 1));
  let y = u32(clamp_i32(cy, 0, i32(dims.y) - 1));
  return vec2<u32>(x, y);
}

// ---- Pass A: generate + bin grains ----
@compute @workgroup_size(WORKGROUP_X, WORKGROUP_Y, 1)
fn grain_gen_main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= U.in_size.x || gid.y >= U.in_size.y) { return; }

  let ij  = vec2<u32>(gid.xy);
  let key = key4(U.seed, ij.x, ij.y, 0u);

  let lam = lambda_lut[luma_bucket_at(vec2<i32>(i32(ij.x), i32(ij.y)))];
  let Q = poisson(lam, key);

  for (var q : u32 = 0u; q < Q; q = q + 1u) {
    let x = vec2<f32>(vec2<u32>(ij)) + vec2<f32>(
      rng01(key, q, 1u, 0u),
      rng01(key, q, 2u, 0u)
    );
    let r = draw_radius(U.mean_r, U.std_r, U.r_max, key, q, 3u, 0u, U.radius_kind);

    let c  = floor(x / U.cell_size);
    let cc = clamp_cell(i32(c.x), i32(c.y), U.grid_dims);
    let ci = flatten2(cc.x, cc.y, U.grid_dims.x);

    let slot = atomicAdd(&cell_counts[ci], 1u);
    if (slot < CELL_CAPACITY) {
      let idx = ci * CELL_CAPACITY + slot;
      cell_items[idx].center = x;
      cell_items[idx].radius = r;
      cell_items[idx]._pad   = 0.0;
    } else {
      atomicStore(&cell_counts[ci], CELL_CAPACITY);
      let _ = atomicExchange(&overflow_flag[0u], 1u);
    }
  }
}

// ---- Pass B: per-output-pixel draw ----
@compute @workgroup_size(TILE_W, TILE_H, 1)
fn grain_draw_main(@builtin(workgroup_id) wg_id : vec3<u32>,
                   @builtin(local_invocation_id) lid : vec3<u32>) {

  let tile_origin = wg_id.xy * vec2<u32>(TILE_W, TILE_H);
  let out_xy = tile_origin + lid.xy;
  if (out_xy.x >= U2.out_size.x || out_xy.y >= U2.out_size.y) { return; }

  let out_idx = flatten2(out_xy.x, out_xy.y, U2.out_size.x);
  let y_out   = vec2<f32>(vec2<u32>(out_xy));
  let y_in    = y_out / U2.zoom_s;

  let r_pad = U2.r_max;
  let cmin  = floor((y_in - vec2<f32>(r_pad)) / U2.cell_size);
  let cmax  = floor((y_in + vec2<f32>(r_pad)) / U2.cell_size);

  var hits : u32 = 0u;

  for (var k : u32 = 0u; k < U2.n_samples; k = k + 1u) {
    let shift = xi[k] / U2.zoom_s;
    var covered = false;

    for (var cy = i32(cmin.y); cy <= i32(cmax.y) && !covered; cy = cy + 1) {
      for (var cx = i32(cmin.x); cx <= i32(cmax.x) && !covered; cx = cx + 1) {
        let cc  = clamp_cell(cx, cy, U2.grid_dims);
        let ci  = flatten2(cc.x, cc.y, U2.grid_dims.x);
        let cnt = min(cell_counts_r[ci], CELL_CAPACITY);

        for (var j : u32 = 0u; j < cnt && !covered; j = j + 1u) {
          let g = cell_items_r[ci * CELL_CAPACITY + j];
          let t = g.center + shift;

          let dx = y_in.x - t.x;
          let dy = y_in.y - t.y;
          covered = (dx*dx + dy*dy) <= (g.radius * g.radius);
        }
      }
    }

    if (covered) { hits = hits + 1u; }
  }

  acc[out_idx] = acc[out_idx] + hits; // acc should be zeroed before dispatch
}

// ---- Normalize counts -> rgba8unorm ----
@compute @workgroup_size(WORKGROUP_X, WORKGROUP_Y, 1)
fn normalize_main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= U_norm.out_size.x || gid.y >= U_norm.out_size.y) { return; }
  let idx = flatten2(gid.x, gid.y, U_norm.out_size.x);
  let n   = max(U_norm.n_samples, 1u);
  let v   = f32(acc_in[idx]) / f32(n);
  textureStore(out_img, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(v, v, v, 1.0));
}
