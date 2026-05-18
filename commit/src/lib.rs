//! Gridland commitment schemes and supporting helpers.
//!
//! This crate provides:
//! - [`error`] for shared commitment errors
//! - [`traits`] for commitment-scheme traits
//! - [`linear`] for shared linear commitment shapes and validation helpers
//! - [`ntt`] for NTT-domain commitment wrappers
//! - [`ajtai`], [`bdlop`], and [`gadget`] for concrete scheme modules

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod ajtai;
pub mod bdlop;
pub mod error;
pub mod gadget;
pub mod linear;
pub mod ntt;
#[doc(hidden)]
pub mod sampling;
pub mod traits;

pub use error::CommitmentError;
pub use ntt::{NttCommitmentScheme, PreparedNttMessage, PreparedNttOpening};
pub use traits::{CommitmentScheme, HomomorphicCommitment};
