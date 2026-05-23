//! Shared sampling helpers for commitment schemes.

use alloc::vec::Vec;

use grid_algebra::arith::large_modulus::{LargePrimeProfile, LargeRnsProfile};
use grid_algebra::arith::large_prime::LargePrimeField;
use grid_algebra::arith::large_rns::LargeRns;
use grid_algebra::arith::limb::UintLimb;
use grid_algebra::arith::prime::PrimeField;
use grid_algebra::arith::ring::Ring;
use grid_algebra::arith::z2k::Z2K;
use grid_algebra::lattice::sampling::toy::{CBDSampler, LargeCBDSampler};
use grid_algebra::lattice::types::{RingMat, RingVec};
use grid_algebra::poly::ring::{CyclotomicPolyRing, NegacyclicMulRing, PolyRing};
use grid_serialize::{CanonicalDeserialize, CanonicalSerialize, Valid};
use grid_std::UniformRand;

#[doc(hidden)]
pub trait CommitmentSampleRing:
    Ring + CanonicalSerialize + CanonicalDeserialize + Valid + UniformRand
{
    fn sample_short<Rng: grid_std::rand::Rng>(rng: &mut Rng, eta: usize) -> Self;
}

impl<const Q: u64, L: UintLimb> CommitmentSampleRing for PrimeField<Q, L> {
    fn sample_short<Rng: grid_std::rand::Rng>(rng: &mut Rng, eta: usize) -> Self {
        use grid_algebra::lattice::sampling::CoeffSampler;

        CBDSampler::<Self>::new(eta).sample_coeff(rng)
    }
}

impl<const K: u32> CommitmentSampleRing for Z2K<K> {
    fn sample_short<Rng: grid_std::rand::Rng>(rng: &mut Rng, eta: usize) -> Self {
        use grid_algebra::lattice::sampling::CoeffSampler;

        CBDSampler::<Self>::new(eta).sample_coeff(rng)
    }
}

impl<P, const LIMBS: usize> CommitmentSampleRing for LargePrimeField<P, LIMBS>
where
    P: LargePrimeProfile<LIMBS>,
{
    fn sample_short<Rng: grid_std::rand::Rng>(rng: &mut Rng, eta: usize) -> Self {
        use grid_algebra::lattice::sampling::CoeffSampler;

        LargeCBDSampler::<Self>::new(eta).sample_coeff(rng)
    }
}

impl<P, const LIMBS: usize> CommitmentSampleRing for LargeRns<P, LIMBS>
where
    P: LargeRnsProfile<LIMBS>,
{
    fn sample_short<Rng: grid_std::rand::Rng>(rng: &mut Rng, eta: usize) -> Self {
        use grid_algebra::lattice::sampling::CoeffSampler;

        LargeCBDSampler::<Self>::new(eta).sample_coeff(rng)
    }
}

impl<C, const N: usize> CommitmentSampleRing for CyclotomicPolyRing<C, N>
where
    C: NegacyclicMulRing<N> + CommitmentSampleRing + UniformRand + Valid,
{
    fn sample_short<Rng: grid_std::rand::Rng>(rng: &mut Rng, eta: usize) -> Self {
        let coeffs = (0..N)
            .map(|_| C::sample_short(rng, eta))
            .collect::<Vec<_>>();
        Self::try_from_coeffs(&coeffs).expect("coefficient vector is built for the ring degree")
    }
}

pub(crate) fn sample_uniform_mat<R, Rng>(rng: &mut Rng, rows: usize, cols: usize) -> RingMat<R>
where
    R: CommitmentSampleRing,
    Rng: grid_std::rand::Rng,
{
    let len = rows
        .checked_mul(cols)
        .expect("validated commitment dimensions must not overflow");
    RingMat::new(rows, cols, (0..len).map(|_| R::rand(rng)).collect())
}

pub(crate) fn sample_opening_vec<R, Rng>(rng: &mut Rng, len: usize, eta: usize) -> RingVec<R>
where
    R: CommitmentSampleRing,
    Rng: grid_std::rand::Rng,
{
    RingVec::new((0..len).map(|_| R::sample_short(rng, eta)).collect())
}
