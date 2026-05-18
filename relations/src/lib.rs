//! Gridland relations — toy arithmetizations and norm-tracked witnesses.
//!
//! This crate provides:
//! - [`error`] for shared relation errors
//! - [`traits`] for relation-system traits
//! - [`r1cs`] for toy R1CS containers and satisfiability checks
//! - [`ccs`] for toy CCS containers and satisfiability checks
//! - [`demo`] for reusable demo relation builders
//! - [`witness`] for explicit witness norm metadata

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod ccs;
pub mod demo;
pub mod error;
pub mod r1cs;
pub mod traits;
pub mod witness;

pub use error::RelationsError;
pub use traits::{ConstraintSynthesizer, ConstraintSystem};
