#![allow(non_snake_case)]

use grid_serialize::{CanonicalDeserialize, CanonicalSerialize};

/// Challenge coefficient shape: maps coefficient value to count.
/// For d=64: {0: 23, ±1: 31, ±2: 10}.
#[derive(Debug, Clone, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct ChallengeShape {
    /// Number of zero coefficients.
    pub zeros: usize,
    /// Number of ±1 coefficients (total of both +1 and -1).
    pub ones: usize,
    /// Number of ±2 coefficients (total of both +2 and -2).
    pub twos: usize,
}

impl ChallengeShape {
    /// Default shape for d=64: 23 zeros, 31 ±1s, 10 ±2s.
    /// Coefficients sum to d (ensuring injectivity via LS18 Cor 1.2).
    pub fn paper_default() -> Self {
        Self {
            zeros: 23,
            ones: 31,
            twos: 10,
        }
    }

    /// Total coefficient count (must equal d for injectivity via LS18 Cor 1.2).
    pub fn degree(&self) -> usize {
        self.zeros + self.ones + self.twos
    }

    /// Sum of absolute values of coefficients: ones + 2·twos.
    pub fn abs_sum(&self) -> usize {
        self.ones + 2 * self.twos
    }

    /// Squared L2 norm τ (||c||²₂): 1²·ones + 2²·twos.
    pub fn tau(&self) -> f64 {
        self.ones as f64 + 4.0 * self.twos as f64
    }
}

impl Default for ChallengeShape {
    fn default() -> Self {
        Self::paper_default()
    }
}

/// Challenge space configuration (§2).
///
/// Defines the coefficient weight distribution for challenge polynomials in R_q.
/// Shape depends on ring degree d; shipped default matches §2 for d=64.
#[derive(Debug, Clone, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
#[allow(non_snake_case)]
pub struct ChallengeProfile {
    /// Coefficient shape: {zeros, ones, twos}.
    pub shape: ChallengeShape,

    /// Operator norm rejection threshold T.
    /// Challenges with ||c||_op > T are rejected during sampling.
    pub T: f64,

    /// Challenge space security: log₂(|C|) collision resistance bits.
    pub space_bits: u8,
}

impl ChallengeProfile {
    /// Squared L2 norm τ = ||c||²₂ (derived from shape).
    /// For d=64: τ = 31·1² + 10·2² = 71.
    pub fn tau(&self) -> f64 {
        self.shape.tau()
    }

    /// Default profile for d=64: τ=71, T=15, 128-bit collision resistance.
    pub fn paper_default() -> Self {
        Self {
            shape: ChallengeShape::paper_default(),
            T: 15.0,
            space_bits: 128,
        }
    }
}

impl Default for ChallengeProfile {
    fn default() -> Self {
        Self::paper_default()
    }
}
