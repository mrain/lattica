//! Gridland algebra — modular arithmetic, polynomial rings, and lattice types.
//!
//! This crate contains three sub-modules:
//! - [`arith`] — Modular integer arithmetic over `Z_q` (any modulus)
//! - [`poly`] — Polynomial ring arithmetic over `R_q`
//! - [`lattice`] — Lattice types, sampling, and parameters

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod arith;
pub mod lattice;
pub mod poly;
pub(crate) mod simd;
