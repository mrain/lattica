//! Gridland standard library — `no_std` compatibility layer and shared re-exports.
//!
//! This crate provides a common foundation for all Gridland crates:
//! - Re-exports of `rand` for randomness
//! - Optional `rayon` parallel iterators (behind the `parallel` feature)
//! - The [`UniformRand`] trait for sampling random algebraic elements
//! - Common test helpers
//! - `std`-only process utilities for examples and reporting

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

// Re-export core dependencies
pub use rand;

#[cfg(feature = "parallel")]
pub use rayon;

/// Re-export commonly used items from `alloc` for `no_std` environments.
pub mod prelude {
    pub use alloc::boxed::Box;
    pub use alloc::string::String;
    pub use alloc::vec;
    pub use alloc::vec::Vec;
}

#[cfg(feature = "std")]
pub mod process;

/// Trait for sampling a uniformly random element.
///
/// Implementations draw entropy from the provided RNG and return a uniformly
/// distributed value of the implementing type. The exact sampling strategy is
/// left to the implementer but most algebraic types use **rejection sampling**:
/// raw bytes are drawn from the RNG, interpreted as a candidate value, and
/// accepted only if they fall within the valid range. Rejected candidates are
/// discarded and new bytes are drawn until an acceptable value is produced.
///
/// ### Liveness
/// Rejection sampling loops until an acceptable value is found. For well-formed
/// types the rejection probability is bounded away from 1, so the expected
/// number of iterations is constant. Callers should provide an RNG with a
/// uniform distribution; a biased or degenerate RNG can lead to pathological
/// rejection rates.
///
/// ### Borrowing Semantics
/// The RNG is taken as `&mut R`, meaning the caller retains ownership and can
/// continue to use it after the call. The trait is generic over any type that
/// implements `rand::RngExt`, so both `StdRng` and custom generators are accepted.
///
/// ### Examples
/// ```ignore
/// use grid_std::UniformRand;
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// let mut rng = StdRng::seed_from_u64(42);
/// let val: MyFieldType = MyFieldType::rand(&mut rng);
/// ```
pub trait UniformRand: Sized {
    /// Sample a uniformly random element using the given RNG.
    ///
    /// The caller passes a mutable reference to an RNG and receives a uniformly
    /// distributed value of `Self`. See the trait-level documentation for details
    /// on rejection sampling, liveness guarantees, and borrowing semantics.
    fn rand<R: rand::RngExt + ?Sized>(rng: &mut R) -> Self;
}

/// A deterministic RNG for reproducible tests.
///
/// Uses a fixed seed so tests are deterministic across runs.
#[cfg(feature = "std")]
pub fn test_rng() -> rand::rngs::StdRng {
    use rand::SeedableRng;
    rand::rngs::StdRng::seed_from_u64(0xDEAD_BEEF_CAFE_BABE)
}

/// Portable math helpers — uses `std` intrinsics when available, `libm` fallback otherwise.
#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "std")]
#[inline]
pub fn sqrt(x: f64) -> f64 {
    x.sqrt()
}

#[cfg(not(feature = "std"))]
#[inline]
pub fn sqrt(x: f64) -> f64 {
    libm::sqrt(x)
}

#[cfg(feature = "std")]
#[inline]
pub fn log2(x: f64) -> f64 {
    x.log2()
}

#[cfg(not(feature = "std"))]
#[inline]
pub fn log2(x: f64) -> f64 {
    libm::log2(x)
}

#[cfg(feature = "std")]
#[inline]
pub fn ceil(x: f64) -> f64 {
    x.ceil()
}

#[cfg(not(feature = "std"))]
#[inline]
pub fn ceil(x: f64) -> f64 {
    libm::ceil(x)
}

#[cfg(feature = "std")]
#[inline]
pub fn floor(x: f64) -> f64 {
    x.floor()
}

#[cfg(not(feature = "std"))]
#[inline]
pub fn floor(x: f64) -> f64 {
    libm::floor(x)
}

#[cfg(feature = "std")]
#[inline]
pub fn pow(x: f64, y: f64) -> f64 {
    x.powf(y)
}

#[cfg(not(feature = "std"))]
#[inline]
pub fn pow(x: f64, y: f64) -> f64 {
    libm::pow(x, y)
}

#[cfg(feature = "std")]
#[inline]
pub fn cos(x: f64) -> f64 {
    x.cos()
}

#[cfg(not(feature = "std"))]
#[inline]
pub fn cos(x: f64) -> f64 {
    libm::cos(x)
}

#[cfg(feature = "std")]
#[inline]
pub fn sin(x: f64) -> f64 {
    x.sin()
}

#[cfg(not(feature = "std"))]
#[inline]
pub fn sin(x: f64) -> f64 {
    libm::sin(x)
}

#[cfg(feature = "std")]
#[inline]
pub fn hypot(x: f64, y: f64) -> f64 {
    x.hypot(y)
}

#[cfg(not(feature = "std"))]
#[inline]
pub fn hypot(x: f64, y: f64) -> f64 {
    libm::hypot(x, y)
}

#[cfg(feature = "std")]
#[inline]
pub fn exp(x: f64) -> f64 {
    x.exp()
}

#[cfg(not(feature = "std"))]
#[inline]
pub fn exp(x: f64) -> f64 {
    libm::exp(x)
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "std")]
    use super::test_rng;
    #[cfg(feature = "std")]
    use rand::RngExt;

    #[test]
    #[cfg(feature = "std")]
    fn test_rng_deterministic() {
        let mut rng1 = test_rng();
        let mut rng2 = test_rng();
        let a: u64 = rng1.random();
        let b: u64 = rng2.random();
        assert_eq!(a, b);
    }
}
