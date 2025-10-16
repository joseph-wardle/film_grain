use crate::config::{EPSILON_GREY_LEVEL, MAX_GREY_LEVEL};

/// Precompute λ(i) and exp(-λ(i)) for i∈[0..=255], per §4.4. :contentReference[oaicite:3]{index=3}
#[expect(dead_code, reason = "Used once params/plumbing are connected")]
pub fn precompute_lambda_tables(mu_r: f32, sigma_r: f32) -> (Vec<f32>, Vec<f32>) {
    let mut lambda = vec![0.0f32; MAX_GREY_LEVEL + 1];
    let mut exp_lambda = vec![0.0f32; MAX_GREY_LEVEL + 1];

    let denom = std::f32::consts::PI * (mu_r * mu_r + sigma_r * sigma_r);
    for i in 0..=MAX_GREY_LEVEL {
        let u = (i as f32) / (MAX_GREY_LEVEL as f32 + EPSILON_GREY_LEVEL);
        // λ = - (a_g^2 / (π(μ_r^2+σ_r^2))) * ln(1-u), with a_g = 1/ceil(1/μ_r)
        let a_g = 1.0 / (1.0 / mu_r).ceil().max(1.0);
        let lambda_i = -((a_g * a_g) / denom) * (1.0 - u).ln();
        lambda[i] = lambda_i;
        exp_lambda[i] = (-lambda_i).exp();
    }
    (lambda, exp_lambda)
}
