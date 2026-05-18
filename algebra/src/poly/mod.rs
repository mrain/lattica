//! Polynomial ring arithmetic over `R_q = Z_q[X] / f(X)`.

pub mod automorphism;
pub mod decomposition;
pub mod ntt;
pub mod ring;
pub mod twisted_ntt;

pub use crate::arith::ntt::NttError;
pub use automorphism::{apply_automorphism, frobenius};
pub use decomposition::{
    gadget_decompose, gadget_decompose_large, gadget_recompose, gadget_recompose_large,
    gadget_vector, gadget_vector_large,
};
pub use ntt::{
    TwistedNttPlan, ntt_forward, ntt_forward_with_plan, ntt_inverse, ntt_inverse_with_plan,
    ntt_plan, poly_mul_ntt, poly_mul_ntt_in_place, poly_mul_ntt_in_place_with_plan,
    twisted_ntt_forward_in_place, twisted_ntt_forward_in_place_with_plan,
    twisted_ntt_inverse_in_place, twisted_ntt_inverse_in_place_with_plan, twisted_ntt_plan,
};
pub use ring::{CyclotomicPolyRing, NegacyclicMulRing, PolyError, PolyRing};
pub use twisted_ntt::{
    TwistedNttPoly, finish_twisted_polys, finish_twisted_polys_with_plan, finish_twisted_ring_mat,
    finish_twisted_ring_mat_with_plan, finish_twisted_ring_vec, finish_twisted_ring_vec_with_plan,
    prepare_twisted_polys, prepare_twisted_polys_with_plan, prepare_twisted_ring_mat,
    prepare_twisted_ring_mat_with_plan, prepare_twisted_ring_vec,
    prepare_twisted_ring_vec_with_plan,
};
