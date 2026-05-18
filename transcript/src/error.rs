//! Shared transcript errors.

use grid_serialize::SerializationError;

/// Errors raised by transcript helpers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TranscriptError {
    /// Transcript labels must not be empty.
    EmptyLabel,
    /// Challenge output length must be non-zero.
    EmptyChallenge,
    /// Transcript framing metadata exceeded the supported bound.
    LengthOverflow,
    /// Value serialization failed before transcript append.
    Serialization(SerializationError),
    /// Canonical transcript encoding did not emit a valid root event.
    InvalidCanonicalEncoding,
    /// Backend-specific challenge generation failed.
    BackendUnavailable,
    /// Caller-supplied input to a transcript-aware function is structurally invalid.
    InvalidInput(alloc::string::String),
}

impl core::fmt::Display for TranscriptError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyLabel => write!(f, "transcript label must not be empty"),
            Self::EmptyChallenge => write!(f, "transcript challenge length must not be zero"),
            Self::LengthOverflow => {
                write!(f, "transcript length metadata exceeded the supported bound")
            }
            Self::Serialization(err) => write!(f, "transcript serialization failed: {err}"),
            Self::InvalidCanonicalEncoding => {
                write!(
                    f,
                    "canonical transcript encoding did not emit a valid root event"
                )
            }
            Self::BackendUnavailable => write!(f, "transcript backend is not available"),
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
        }
    }
}

impl From<SerializationError> for TranscriptError {
    fn from(value: SerializationError) -> Self {
        Self::Serialization(value)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for TranscriptError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Serialization(err) => Some(err),
            _ => None,
        }
    }
}
