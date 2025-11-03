use crate::model::Derived;
use crate::params::{Algo, Params};

pub fn choose_algorithm(params: &Params, derived: &Derived) -> Algo {
    match params.algo {
        Algo::Auto => {
            let mean = params.radius_mean.max(1e-6);
            let sigma_ratio = if mean > 0.0 {
                params.radius_stddev / mean
            } else {
                0.0
            };
            let rm_ratio = derived.rm / mean;
            if sigma_ratio < 0.1 && mean < 0.5 && params.n_samples <= 64 {
                Algo::Pixel
            } else if rm_ratio > 8.0 || sigma_ratio > 0.6 || params.n_samples > 96 {
                Algo::Grain
            } else if params.n_samples <= 24 && rm_ratio < 5.0 {
                Algo::Pixel
            } else {
                Algo::Grain
            }
        }
        other => other,
    }
}
