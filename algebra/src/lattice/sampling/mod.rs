//! Lattice sampling traits and toy sampling helpers.

use crate::arith::ring::IntegerRing;

pub mod toy;

/// Samples coefficient-ring elements.
pub trait CoeffSampler<R: IntegerRing> {
    /// Sample one coefficient ring element.
    fn sample_coeff<T: grid_std::rand::Rng>(&self, rng: &mut T) -> R;
}
