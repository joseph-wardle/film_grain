//! Deterministic, seedable RNGs for CPU

/// 64-bit SplitMix64 (good stat properties, fast).
#[derive(Clone)]
pub struct SplitMix64 { state: u128 }

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed as u128 }
    }
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut z = (self.state.wrapping_add(0x9E3779B97F4A7C15u128)) as u64;
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15u128);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    #[inline] pub fn next_f32(&mut self) -> f32 {
        let x = self.next_u64() >> 40; // top 24 bits
        (x as f32) / ((1u32 << 24) as f32)
    }
}

/// Standard normal via Box–Muller using two uniforms; deterministic.
pub fn gaussian_pair(rng: &mut SplitMix64) -> (f32, f32) {
    // Avoid 0
    let u1 = (rng.next_f32().max(1e-12)).min(1.0 - 1e-12);
    let u2 = (rng.next_f32().max(1e-12)).min(1.0 - 1e-12);
    let r = (-2.0 * u1.ln()).sqrt();
    let t = 2.0 * std::f32::consts::PI * u2;
    (r * t.cos(), r * t.sin())
}

/// Poisson via inverse transform; optionally pass exp(-λ) to save `exp` calls.
#[inline]
pub fn poisson_inverse(rng: &mut SplitMix64, lambda: f32, exp_neg_lambda: Option<f32>) -> u32 {
    let l = lambda.max(0.0);
    if l == 0.0 { return 0; }
    let mut p = exp_neg_lambda.unwrap_or_else(|| (-l).exp());
    let u = rng.next_f32();
    let mut sum = p;
    let mut x = 0u32;
    let limit = (10000.0 * l).floor() as u32;
    while u > sum && x < limit {
        x += 1;
        p *= l / (x as f32);
        sum += p;
    }
    x
}
