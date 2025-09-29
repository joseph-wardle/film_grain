use std::f32::consts::PI;
use tracing::{debug, instrument, trace};

/// A simple pseudo–random number generator based on a xorshift algorithm.
/// All random sampling (uniform, Gaussian, Poisson) is done through this PRNG.
#[derive(Debug, Clone)]
pub struct Prng {
    state: u32,
}

impl Prng {
    /// Creates a new PRNG with a given seed (which is first scrambled via the wang–hash).
    pub fn new(seed: u32) -> Self {
        let hashed = Self::wang_hash(seed);
        debug!(seed, hashed, "Initialized PRNG state");
        Self { state: hashed }
    }

    /// Wang hash: a simple hash function to scramble the seed.
    #[instrument(level = "trace")]
    fn wang_hash(seed: u32) -> u32 {
        let mut hash = seed;
        hash = (hash ^ 61) ^ (hash >> 16);
        hash = hash.wrapping_mul(9);
        hash ^= hash >> 4;
        hash = hash.wrapping_mul(668_265_261);
        hash ^= hash >> 15;
        hash
    }

    /// Generates the next random u32 and updates the internal state.
    #[instrument(level = "trace", skip(self))]
    pub fn next_u32(&mut self) -> u32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 17;
        self.state ^= self.state << 5;
        trace!(state = self.state, "Generated next u32");
        self.state
    }

    /// Returns a uniform random number in [0, 1].
    pub fn next_f32(&mut self) -> f32 {
        let value = self.next_u32() as f32 / 4_294_967_295.0;
        trace!(value, "Generated uniform float");
        value
    }

    /// Returns a standard normally distributed random number (mean 0, std dev 1) using Box–Muller.
    #[instrument(level = "trace", skip(self))]
    pub fn next_standard_normal(&mut self) -> f32 {
        let radius_uniform = self.next_f32();
        let angle_uniform = self.next_f32();
        let result = (-2.0 * radius_uniform.ln()).sqrt() * (2.0 * PI * angle_uniform).cos();
        trace!(result, "Generated standard normal sample");
        result
    }

    /// Samples a Poisson–distributed random variable with parameter `lambda`.
    /// An optional precomputed exp(–lambda) may be supplied.
    #[instrument(level = "trace", skip(self))]
    pub fn next_poisson(&mut self, lambda: f32, cached_exponential: Option<f32>) -> u32 {
        let uniform_sample = self.next_f32();
        let mut event_count: u32 = 0;
        let mut poisson_term = match cached_exponential {
            Some(val) if val > 0.0 => val,
            _ => (-lambda).exp(),
        };
        let mut cumulative_probability = poisson_term;
        let iteration_limit = (10000.0 * lambda).floor() as u32;
        while uniform_sample > cumulative_probability && event_count < iteration_limit {
            event_count += 1;
            poisson_term = poisson_term * lambda / (event_count as f32);
            cumulative_probability += poisson_term;
        }
        trace!(
            lambda,
            uniform_sample,
            result = event_count,
            "Generated Poisson sample"
        );
        event_count
    }
}

/// Computes a unique seed for a given cell based on its (x, y) coordinates and a constant offset.
/// The period is 2^16; if the resulting seed is zero, returns 1.
#[instrument(level = "trace")]
pub fn cell_seed(x: u32, y: u32, offset: u32) -> u32 {
    const PERIOD: u32 = 65_536;
    let seed_candidate = ((y % PERIOD) * PERIOD + (x % PERIOD)).wrapping_add(offset);
    if seed_candidate == 0 {
        1
    } else {
        seed_candidate
    }
}
