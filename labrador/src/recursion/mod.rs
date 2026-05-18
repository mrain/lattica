//! Recursion step (§5.3).
//!
//! Transforms the output of one main protocol level into the witness and
//! target relation for the next level. Three transformations:
//!
//! 1. **Decompose** (`decompose`): z = z⁰ + b·z¹ binary-norm decomposition,
//!    bundle v = t ‖ g ‖ h.
//!
//! 2. **Split** (`split`): split into r' = 2ν + μ vectors of rank n',
//!    zero-padded for uniformity.
//!
//! 3. **Target relation** (`target_relation`): build family G with
//!    K' = κ + κ₁ + κ₂ + 3 constraints, consolidated norm bound β'.

pub mod decompose;
pub mod split;
pub mod target_relation;

pub use decompose::{bundle_v, decompose_z};
pub use split::{compute_next_level_shape, split_last_level_witness, split_witness};
pub use target_relation::{RecursiveTarget, build_target_relation};
