//! R1CS → R reductions (§6).
//!
//! Three reductions from concrete circuit representations to the LaBRADOR
//! principal relation R (§5.1):
//!
//! - `binary_r1cs` — binary R1CS (mod 2), Figure 4, Theorem 6.2
//! - `arith_r1cs` — R1CS mod 2^d+1, Figure 5, Theorem 6.3
//! - `mixed_r1cs` — combined binary + arithmetic, prose §6
//!
//! Dead code and argument count warnings are expected: these are public API
//! entry points used by external callers before invoking the top-level prover.

#![allow(
    dead_code,
    clippy::too_many_arguments,
    clippy::redundant_closure,
    clippy::needless_range_loop,
    clippy::unnecessary_map_or
)]

mod app_ring;
mod arith_r1cs;
mod binary_r1cs;
mod mixed_r1cs;

pub use app_ring::{AppModRing, gf2_to_proof_const, small_app_to_proof_const};
pub use arith_r1cs::{
    ArithR1CSInstance, ArithR1CSReduction, build_arith_r1cs_reduction,
    build_arith_r1cs_reduction_transcript, sample_arith_challenges,
    sample_arith_challenges_transcript, verify_aggregation_rq, verify_naf_coeffs,
    verify_naf_witness,
};
pub use binary_r1cs::{
    BinaryR1CSInstance, BinaryR1CSReduction, build_binary_r1cs_reduction,
    build_binary_r1cs_reduction_transcript, check_g_even, sample_binary_challenges,
    sample_binary_challenges_transcript, verify_binary_r1cs_reduction,
};
pub use mixed_r1cs::{
    MixedR1CSInstance, MixedR1CSReduction, build_mixed_r1cs_reduction,
    build_mixed_r1cs_reduction_transcript, verify_mixed_r1cs_reduction,
};
