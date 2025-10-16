//! filmgrain-core: library for film grain synthesis (CPU + GPU).
//!
//! Implements the inhomogeneous Boolean model with Monte-Carlo Gaussian filtering
//! in two algorithmic modes: **PixelWise** and **GrainWise**.
//!
//! # Determinism
//! CPU is bit-exact given seed & params. GPU is deterministic for a fixed backend/driver
//! to the extent the backend guarantees, using integer PRNG & explicit barriers.

mod config;
mod error;

}
