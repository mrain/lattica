//! Shared error types for the relations crate.

/// Errors raised by relation-system helpers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelationsError {
    /// The supplied parameters are structurally invalid.
    InvalidParameters,
    /// Matrix/vector dimensions do not match the declared shape.
    DimensionMismatch,
    /// The supplied witness is malformed or internally inconsistent.
    InvalidWitness,
    /// Stored witness norm metadata does not match the actual witness.
    WitnessNormMismatch,
    /// The witness exceeds the declared norm bounds.
    WitnessNormExceeded,
}

impl core::fmt::Display for RelationsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidParameters => write!(f, "invalid relation parameters"),
            Self::DimensionMismatch => write!(f, "relation dimensions do not match"),
            Self::InvalidWitness => write!(f, "invalid witness"),
            Self::WitnessNormMismatch => write!(f, "witness norm metadata does not match"),
            Self::WitnessNormExceeded => write!(f, "witness norm bound exceeded"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for RelationsError {}
