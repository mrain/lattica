//! LaBRADOR protocol errors.

use alloc::string::String;

use grid_transcript::TranscriptError;

/// Errors from the LaBRADOR protocol.
#[derive(Debug)]
pub enum LabradorError {
    /// Transcript operation failed (challenge draw, append, etc.).
    Transcript(TranscriptError),
    /// Folding challenge sampler exhausted all attempts without finding
    /// a polynomial that satisfies the operator norm threshold.
    SamplerExhausted,
    /// CRS dimension is zero — params must have strictly positive dimensions.
    ZeroDimension,
    /// Caller-supplied data is structurally invalid (mismatched shapes, bad params).
    InvalidInput(String),
    /// Prover self-check failed (norm bound exceeded, etc.).
    Prover(String),
    /// Verification equation failed.
    Verification(String),
    /// Internal error (unexpected state).
    Internal(String),
}

impl core::fmt::Display for LabradorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Transcript(e) => write!(f, "transcript error: {e}"),
            Self::SamplerExhausted => {
                write!(f, "folding challenge sampler exhausted all attempts")
            }
            Self::ZeroDimension => write!(f, "CRS dimension is zero"),
            Self::InvalidInput(e) => write!(f, "invalid input: {e}"),
            Self::Prover(e) => write!(f, "prover check failed: {e}"),
            Self::Verification(e) => write!(f, "verification failed: {e}"),
            Self::Internal(e) => write!(f, "internal error: {e}"),
        }
    }
}

impl From<TranscriptError> for LabradorError {
    fn from(e: TranscriptError) -> Self {
        Self::Transcript(e)
    }
}

impl From<String> for LabradorError {
    fn from(e: String) -> Self {
        Self::Verification(e)
    }
}

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "std")]
impl std::error::Error for LabradorError {}
