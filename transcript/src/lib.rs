//! Gridland transcript — shared Fiat-Shamir traits, framing, canonical encoding, and backends.
//!
//! This crate provides:
//! - [`error`] for transcript errors
//! - [`encoding`] for canonical framing and domain separation helpers
//! - [`traits`] for the byte-oriented transcript trait
//! - [`field`] for the field-native transcript trait family
//! - [`canonical`] for structured field-native transcript encoding
//! - [`hash`] for transcript backend modules

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod canonical;
pub mod encoding;
pub mod error;
pub mod field;
pub mod hash;
pub mod traits;

pub use canonical::{CanonicalTranscriptEncode, CanonicalTranscriptEncoder};
pub use error::TranscriptError;
pub use field::{FieldTranscript, TranscriptField};
pub use traits::Transcript;
