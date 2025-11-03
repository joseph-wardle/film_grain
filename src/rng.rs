use rand::SeedableRng;
use rand::rngs::{SmallRng, StdRng};
use rand_distr::{Distribution, Normal};

const OFFSET_STREAM: u64 = 0x9E37_79B9_7F4A_7C15;
const CELL_STREAM: u64 = 0xA24B_1C30_BEBC_CF59;
const PIXEL_STREAM: u64 = 0x6935_FA5C_55F6_5F1B;

pub fn make_offsets(seed: u64, n: usize, sigma: f32) -> Vec<[f32; 2]> {
    if n == 0 {
        return Vec::new();
    }
    let sigma = sigma.max(f32::EPSILON) as f64;
    let normal = Normal::new(0.0, sigma).expect("normal distribution");
    let mut rng = StdRng::seed_from_u64(mix(seed, OFFSET_STREAM));
    (0..n)
        .map(|_| {
            [
                normal.sample(&mut rng) as f32,
                normal.sample(&mut rng) as f32,
            ]
        })
        .collect()
}

pub fn cell_rng(seed: u64, i: i32, j: i32) -> SmallRng {
    let hashed = mix3(seed, CELL_STREAM, i as i64, j as i64);
    SmallRng::seed_from_u64(hashed)
}

pub fn pixel_rng(seed: u64, i: i32, j: i32) -> SmallRng {
    let hashed = mix3(seed, PIXEL_STREAM, i as i64, j as i64);
    SmallRng::seed_from_u64(hashed)
}

fn mix(seed: u64, stream: u64) -> u64 {
    splitmix64(seed ^ stream)
}

fn mix3(seed: u64, stream: u64, a: i64, b: i64) -> u64 {
    let mut state = seed ^ stream;
    state = splitmix64(state.wrapping_add(a as u64).rotate_left(17));
    splitmix64(state.wrapping_add(b as u64).rotate_left(41))
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
