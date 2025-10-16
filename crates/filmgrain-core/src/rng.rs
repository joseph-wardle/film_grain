//! Deterministic, seedable RNGs for CPU

use tracing::instrument;

/// 64-bit SplitMix64 (good stat properties, fast).
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

#[expect(dead_code, reason = "WIP: API not called yet during set-up")]
impl SplitMix64 {
    #[instrument(level = "debug")]
    pub fn new(seed: u64) -> Self {
        tracing::debug!(seed, "initializing SplitMix64 RNG");
        Self { state: seed }
    }

    #[inline]
    #[instrument(level = "trace", skip(self))]
    pub fn next_u64(&mut self) -> u64 {
        let mut z = self.state.wrapping_add(0x9E3779B97F4A7C15u64);
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15u64);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        let result = z ^ (z >> 31);
        tracing::trace!(result, "generated u64 from SplitMix64");
        result
    }

    #[inline]
    #[instrument(level = "trace", skip(self))]
    pub fn next_f32(&mut self) -> f32 {
        let x = self.next_u64() >> 40; // top 24 bits
        let value = (x as f32) / ((1u32 << 24) as f32);
        tracing::trace!(value, "generated f32 uniform from SplitMix64");
        value
    }
}

/// Standard normal via Box–Muller using two uniforms; deterministic.
#[expect(dead_code, reason = "WIP helper, hooked up later")]
#[instrument(level = "debug")]
pub fn gaussian_pair(rng: &mut SplitMix64) -> (f32, f32) {
    // Avoid 0
    let u1 = rng.next_f32().clamp(1e-12, 1.0 - 1e-12);
    let u2 = rng.next_f32().clamp(1e-12, 1.0 - 1e-12);
    let r = (-2.0 * u1.ln()).sqrt();
    let t = 2.0 * std::f32::consts::PI * u2;
    let pair = (r * t.cos(), r * t.sin());
    tracing::debug!(?pair, "generated gaussian pair");
    pair
}

/// Poisson via inverse transform; optionally pass exp(-λ) to save `exp` calls.
#[inline]
#[expect(dead_code, reason = "WIP helper, hooked up later")]
#[instrument(level = "debug", skip(rng))]
pub fn poisson_inverse(rng: &mut SplitMix64, lambda: f32, exp_neg_lambda: Option<f32>) -> u32 {
    let l = lambda.max(0.0);
    if l == 0.0 {
        tracing::debug!("lambda zero; returning 0 draws");
        return 0;
    }
    let mut p = exp_neg_lambda.unwrap_or_else(|| (-l).exp());
    let u = rng.next_f32();
    let mut sum = p;
    let mut x = 0u32;
    let limit = (10000.0 * l).floor() as u32;
    while u > sum && x < limit {
        x += 1;
        p *= l / (x as f32);
        sum += p;
        tracing::trace!(x, sum, p, limit, "poisson inverse iteration");
    }
    tracing::debug!(x, lambda = l, "poisson inverse result");
    x
}
