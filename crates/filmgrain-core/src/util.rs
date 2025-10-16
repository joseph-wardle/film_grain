use crate::config::{EPSILON_GREY_LEVEL, MAX_GREY_LEVEL};
use tracing::{debug, instrument, trace};

/// Precompute λ(i) and exp(-λ(i)) for i∈[0..=255], per §4.4
#[instrument(level = "debug")]
pub fn precompute_lambda_tables(mu_r: f32, sigma_r: f32) -> (Vec<f32>, Vec<f32>) {
    let mut lambda = vec![0.0f32; MAX_GREY_LEVEL + 1];
    let mut exp_lambda = vec![0.0f32; MAX_GREY_LEVEL + 1];

    let denom = std::f32::consts::PI * (mu_r * mu_r + sigma_r * sigma_r);
    debug!(denom, "computed denominator for lambda table");
    for i in 0..=MAX_GREY_LEVEL {
        let u = (i as f32) / (MAX_GREY_LEVEL as f32 + EPSILON_GREY_LEVEL);
        // λ = - (a_g^2 / (π(μ_r^2+σ_r^2))) * ln(1-u), with a_g = 1/ceil(1/μ_r)
        let a_g = 1.0 / (1.0 / mu_r).ceil().max(1.0);
        let lambda_i = -((a_g * a_g) / denom) * (1.0 - u).ln();
        lambda[i] = lambda_i;
        exp_lambda[i] = (-lambda_i).exp();
        trace!(index = i, u, a_g, lambda_i, "computed lambda row");
    }
    debug!("lambda tables generated");
    (lambda, exp_lambda)
}

#[inline]
#[instrument(level = "debug")]
pub fn hash3_u64(x: u64, y: u64, seed: u64) -> u64 {
    let mut z = x.wrapping_mul(0x9E3779B185EBCA87)
        ^ y.wrapping_mul(0xC2B2AE3D27D4EB4F)
        ^ seed;
    z ^= z >> 32; z = z.wrapping_mul(0x9E3779B97F4A7C15);
    z ^= z >> 29; z = z.wrapping_mul(0xBF58476D1CE4E5B9);
    z ^= z >> 32;
    z
}
