//! Shared commitment errors.

use core::fmt;

/// Errors shared by commitment helpers and concrete schemes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommitmentError {
    /// Parameter validation failed.
    InvalidParameters,
    /// Matrix/vector/message shapes do not match.
    DimensionMismatch,
    /// Opening material exceeded the configured norm bound.
    OpeningNormExceeded,
    /// Message encoding was malformed.
    InvalidMessageEncoding,
    /// Opening data was malformed.
    InvalidOpening,
}

impl fmt::Display for CommitmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameters => write!(f, "invalid commitment parameters"),
            Self::DimensionMismatch => write!(f, "dimension mismatch"),
            Self::OpeningNormExceeded => write!(f, "opening norm exceeded configured bound"),
            Self::InvalidMessageEncoding => write!(f, "invalid message encoding"),
            Self::InvalidOpening => write!(f, "invalid opening"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CommitmentError {}
