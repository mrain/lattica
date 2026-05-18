//! Explicit twisted-NTT polynomial representation for negacyclic arithmetic.

use alloc::vec::Vec;
use core::array::from_fn;
use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use grid_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::arith::ntt::{NTTRing, NttError};
use crate::arith::ring::{IntegerRing, Ring};
use crate::lattice::types::{RingMat, RingVec};
use crate::poly::ntt::{TwistedNttPlan, cached_twisted_plan};
use crate::poly::ring::{CyclotomicPolyRing, NegacyclicMulRing};

/// A polynomial stored in the negacyclic twisted evaluation domain.
///
/// This is an explicit runtime/prepared representation. It is intentionally
/// separate from [`CyclotomicPolyRing`] so coefficient-domain semantics remain
/// obvious throughout the rest of the API.
#[derive(Clone, PartialEq, Eq)]
pub struct TwistedNttPoly<R, const N: usize>
where
    R: IntegerRing + CanonicalSerialize + CanonicalDeserialize,
{
    pub(crate) evals: [R; N],
}

impl<R, const N: usize> TwistedNttPoly<R, N>
where
    R: IntegerRing + NTTRing + CanonicalSerialize + CanonicalDeserialize + Send + Sync + 'static,
{
    fn assert_supported() {
        cached_twisted_plan::<R>(N)
            .expect("TwistedNttPoly requires a coefficient ring with a supported twisted NTT");
    }

    fn from_evals_trusted(evals: [R; N]) -> Self {
        Self { evals }
    }

    /// Construct a twisted-domain polynomial from precomputed evaluation points.
    ///
    /// Used for sampling CRS matrices directly in NTT domain — avoids the
    /// coefficient→NTT conversion entirely since a uniform random polynomial
    /// in coefficient domain maps to a uniform random polynomial in evaluation domain.
    pub fn from_evals(evals: [R; N]) -> Self {
        Self { evals }
    }

    /// Convert a coefficient-domain polynomial into the twisted evaluation domain.
    pub fn from_coeff_poly(poly: &CyclotomicPolyRing<R, N>) -> Result<Self, NttError>
    where
        R: NegacyclicMulRing<N>,
    {
        Self::from_coeff_array(poly.coeff_array().clone())
    }

    /// Convert a coefficient-domain coefficient array into the twisted evaluation domain.
    pub fn from_coeff_array(coeffs: [R; N]) -> Result<Self, NttError> {
        let plan = cached_twisted_plan::<R>(N)?;
        Self::from_coeff_array_with_plan(coeffs, plan.as_ref())
    }

    /// Convert a coefficient-domain coefficient array into the twisted evaluation domain using a
    /// precomputed plan.
    pub fn from_coeff_array_with_plan(
        coeffs: [R; N],
        plan: &TwistedNttPlan<R>,
    ) -> Result<Self, NttError> {
        let mut evals = coeffs;
        plan.forward_in_place(&mut evals)?;
        Ok(Self::from_evals_trusted(evals))
    }

    #[inline(always)]
    pub fn from_coeff_array_with_plan_trusted(coeffs: [R; N], plan: &TwistedNttPlan<R>) -> Self {
        let mut evals = coeffs;
        plan.forward_in_place_trusted(&mut evals);
        Self::from_evals_trusted(evals)
    }

    /// Convert this twisted-domain polynomial back into a coefficient-domain polynomial.
    pub fn to_coeff_poly(&self) -> Result<CyclotomicPolyRing<R, N>, NttError> {
        Ok(CyclotomicPolyRing::from_array(self.to_coeff_array()?))
    }

    /// Convert this twisted-domain polynomial back into a coefficient array.
    pub fn to_coeff_array(&self) -> Result<[R; N], NttError> {
        let plan = cached_twisted_plan::<R>(N)?;
        self.to_coeff_array_with_plan(plan.as_ref())
    }

    /// Convert this twisted-domain polynomial back into a coefficient array using a precomputed
    /// plan.
    pub fn to_coeff_array_with_plan(&self, plan: &TwistedNttPlan<R>) -> Result<[R; N], NttError> {
        let mut coeffs = self.evals.clone();
        plan.inverse_in_place(&mut coeffs)?;
        Ok(coeffs)
    }

    #[inline(always)]
    fn to_coeff_array_with_plan_trusted(&self, plan: &TwistedNttPlan<R>) -> [R; N] {
        let mut coeffs = self.evals.clone();
        plan.inverse_in_place_trusted(&mut coeffs);
        coeffs
    }
}

/// Convert a coefficient-domain slice into twisted-domain polynomials using one cached plan lookup.
pub fn prepare_twisted_polys<R, const N: usize>(
    polys: &[CyclotomicPolyRing<R, N>],
) -> Result<Vec<TwistedNttPoly<R, N>>, NttError>
where
    R: IntegerRing
        + NTTRing
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize
        + Send
        + Sync
        + 'static,
{
    let plan = cached_twisted_plan::<R>(N)?;
    prepare_twisted_polys_with_plan(polys, plan.as_ref())
}

/// Convert a coefficient-domain slice into twisted-domain polynomials using a precomputed plan.
pub fn prepare_twisted_polys_with_plan<R, const N: usize>(
    polys: &[CyclotomicPolyRing<R, N>],
    plan: &TwistedNttPlan<R>,
) -> Result<Vec<TwistedNttPoly<R, N>>, NttError>
where
    R: IntegerRing
        + NTTRing
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize
        + Send
        + Sync
        + 'static,
{
    if plan.len() != N {
        return Err(NttError::LengthMismatch {
            left: plan.len(),
            right: N,
        });
    }
    let mut prepared = Vec::with_capacity(polys.len());
    for poly in polys {
        prepared.push(TwistedNttPoly::from_coeff_array_with_plan_trusted(
            poly.coeff_array().clone(),
            plan,
        ));
    }
    Ok(prepared)
}

/// Convert twisted-domain polynomials back into coefficient form using one cached plan lookup.
pub fn finish_twisted_polys<R, const N: usize>(
    polys: &[TwistedNttPoly<R, N>],
) -> Result<Vec<CyclotomicPolyRing<R, N>>, NttError>
where
    R: IntegerRing
        + NTTRing
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize
        + Send
        + Sync
        + 'static,
{
    let plan = cached_twisted_plan::<R>(N)?;
    finish_twisted_polys_with_plan(polys, plan.as_ref())
}

/// Convert twisted-domain polynomials back into coefficient form using a precomputed plan.
pub fn finish_twisted_polys_with_plan<R, const N: usize>(
    polys: &[TwistedNttPoly<R, N>],
    plan: &TwistedNttPlan<R>,
) -> Result<Vec<CyclotomicPolyRing<R, N>>, NttError>
where
    R: IntegerRing
        + NTTRing
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize
        + Send
        + Sync
        + 'static,
{
    if plan.len() != N {
        return Err(NttError::LengthMismatch {
            left: plan.len(),
            right: N,
        });
    }
    let mut finished = Vec::with_capacity(polys.len());
    for poly in polys {
        finished.push(CyclotomicPolyRing::from_array(
            poly.to_coeff_array_with_plan_trusted(plan),
        ));
    }
    Ok(finished)
}

/// Convert a coefficient-domain vector into the twisted evaluation domain.
pub fn prepare_twisted_ring_vec<R, const N: usize>(
    vector: &RingVec<CyclotomicPolyRing<R, N>>,
) -> Result<RingVec<TwistedNttPoly<R, N>>, NttError>
where
    R: IntegerRing
        + NTTRing
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize
        + Send
        + Sync
        + 'static,
{
    let plan = cached_twisted_plan::<R>(N)?;
    prepare_twisted_ring_vec_with_plan(vector, plan.as_ref())
}

/// Convert a coefficient-domain vector into the twisted evaluation domain using a precomputed
/// plan.
pub fn prepare_twisted_ring_vec_with_plan<R, const N: usize>(
    vector: &RingVec<CyclotomicPolyRing<R, N>>,
    plan: &TwistedNttPlan<R>,
) -> Result<RingVec<TwistedNttPoly<R, N>>, NttError>
where
    R: IntegerRing
        + NTTRing
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize
        + Send
        + Sync
        + 'static,
{
    prepare_twisted_polys_with_plan(vector.entries(), plan).map(RingVec::new)
}

/// Convert a twisted-domain vector back into coefficient form.
pub fn finish_twisted_ring_vec<R, const N: usize>(
    vector: &RingVec<TwistedNttPoly<R, N>>,
) -> Result<RingVec<CyclotomicPolyRing<R, N>>, NttError>
where
    R: IntegerRing
        + NTTRing
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize
        + Send
        + Sync
        + 'static,
{
    let plan = cached_twisted_plan::<R>(N)?;
    finish_twisted_ring_vec_with_plan(vector, plan.as_ref())
}

/// Convert a twisted-domain vector back into coefficient form using a precomputed plan.
pub fn finish_twisted_ring_vec_with_plan<R, const N: usize>(
    vector: &RingVec<TwistedNttPoly<R, N>>,
    plan: &TwistedNttPlan<R>,
) -> Result<RingVec<CyclotomicPolyRing<R, N>>, NttError>
where
    R: IntegerRing
        + NTTRing
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize
        + Send
        + Sync
        + 'static,
{
    finish_twisted_polys_with_plan(vector.entries(), plan).map(RingVec::new)
}

/// Convert a coefficient-domain matrix into the twisted evaluation domain.
pub fn prepare_twisted_ring_mat<R, const N: usize>(
    matrix: &RingMat<CyclotomicPolyRing<R, N>>,
) -> Result<RingMat<TwistedNttPoly<R, N>>, NttError>
where
    R: IntegerRing
        + NTTRing
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize
        + Send
        + Sync
        + 'static,
{
    let plan = cached_twisted_plan::<R>(N)?;
    prepare_twisted_ring_mat_with_plan(matrix, plan.as_ref())
}

/// Convert a coefficient-domain matrix into the twisted evaluation domain using a precomputed
/// plan.
pub fn prepare_twisted_ring_mat_with_plan<R, const N: usize>(
    matrix: &RingMat<CyclotomicPolyRing<R, N>>,
    plan: &TwistedNttPlan<R>,
) -> Result<RingMat<TwistedNttPoly<R, N>>, NttError>
where
    R: IntegerRing
        + NTTRing
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize
        + Send
        + Sync
        + 'static,
{
    prepare_twisted_polys_with_plan(matrix.entries(), plan)
        .map(|entries| RingMat::new(matrix.rows(), matrix.cols(), entries))
}

/// Convert a twisted-domain matrix back into coefficient form.
pub fn finish_twisted_ring_mat<R, const N: usize>(
    matrix: &RingMat<TwistedNttPoly<R, N>>,
) -> Result<RingMat<CyclotomicPolyRing<R, N>>, NttError>
where
    R: IntegerRing
        + NTTRing
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize
        + Send
        + Sync
        + 'static,
{
    let plan = cached_twisted_plan::<R>(N)?;
    finish_twisted_ring_mat_with_plan(matrix, plan.as_ref())
}

/// Convert a twisted-domain matrix back into coefficient form using a precomputed plan.
pub fn finish_twisted_ring_mat_with_plan<R, const N: usize>(
    matrix: &RingMat<TwistedNttPoly<R, N>>,
    plan: &TwistedNttPlan<R>,
) -> Result<RingMat<CyclotomicPolyRing<R, N>>, NttError>
where
    R: IntegerRing
        + NTTRing
        + NegacyclicMulRing<N>
        + CanonicalSerialize
        + CanonicalDeserialize
        + Send
        + Sync
        + 'static,
{
    finish_twisted_polys_with_plan(matrix.entries(), plan)
        .map(|entries| RingMat::new(matrix.rows(), matrix.cols(), entries))
}

impl<R, const N: usize> fmt::Debug for TwistedNttPoly<R, N>
where
    R: IntegerRing + CanonicalSerialize + CanonicalDeserialize,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("TwistedNttPoly").field(&self.evals).finish()
    }
}

impl<R, const N: usize> Ring for TwistedNttPoly<R, N>
where
    R: IntegerRing + NTTRing + CanonicalSerialize + CanonicalDeserialize + Send + Sync + 'static,
{
    fn zero() -> Self {
        Self::assert_supported();
        Self::from_evals_trusted(from_fn(|_| R::zero()))
    }

    fn one() -> Self {
        Self::assert_supported();
        Self::from_evals_trusted(from_fn(|_| R::one()))
    }

    fn dot_product(lhs: &[Self], rhs: &[Self]) -> Self {
        assert_eq!(lhs.len(), rhs.len(), "slice lengths must match");
        let mut acc = from_fn(|_| R::zero());
        const CHUNK: usize = 64;
        if lhs.len() <= 8 || N <= CHUNK {
            for (lhs_poly, rhs_poly) in lhs.iter().zip(rhs.iter()) {
                for ((acc_eval, lhs_eval), rhs_eval) in acc
                    .iter_mut()
                    .zip(lhs_poly.evals.iter())
                    .zip(rhs_poly.evals.iter())
                {
                    *acc_eval += R::mul_ref(lhs_eval, rhs_eval);
                }
            }
            return Self::from_evals_trusted(acc);
        }
        let mut products: [R; CHUNK] = from_fn(|_| R::zero());
        for (lhs_poly, rhs_poly) in lhs.iter().zip(rhs.iter()) {
            for start in (0..N).step_by(CHUNK) {
                let len = (N - start).min(CHUNK);
                let lhs_chunk = &lhs_poly.evals[start..start + len];
                let rhs_chunk = &rhs_poly.evals[start..start + len];
                let acc_chunk = &mut acc[start..start + len];

                products[..len].clone_from_slice(lhs_chunk);
                R::pointwise_mul_assign_slice(&mut products[..len], rhs_chunk);
                R::add_assign_slice(acc_chunk, &products[..len]);
            }
        }
        Self::from_evals_trusted(acc)
    }

    fn add_ref(a: &Self, b: &Self) -> Self {
        a + b
    }

    fn sub_ref(a: &Self, b: &Self) -> Self {
        a - b
    }

    fn mul_ref(a: &Self, b: &Self) -> Self {
        a * b
    }
}

impl<R, const N: usize> Add for TwistedNttPoly<R, N>
where
    R: IntegerRing + NTTRing + CanonicalSerialize + CanonicalDeserialize + Send + Sync + 'static,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        self + &rhs
    }
}

impl<R, const N: usize> Add<&Self> for TwistedNttPoly<R, N>
where
    R: IntegerRing + NTTRing + CanonicalSerialize + CanonicalDeserialize + Send + Sync + 'static,
{
    type Output = Self;

    fn add(self, rhs: &Self) -> Self {
        let mut evals = self.evals;
        R::add_assign_slice(&mut evals, &rhs.evals);
        Self::from_evals_trusted(evals)
    }
}

impl<R, const N: usize> Add<Self> for &TwistedNttPoly<R, N>
where
    R: IntegerRing + NTTRing + CanonicalSerialize + CanonicalDeserialize + Send + Sync + 'static,
{
    type Output = TwistedNttPoly<R, N>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut evals = self.evals.clone();
        R::add_assign_slice(&mut evals, &rhs.evals);
        TwistedNttPoly::from_evals_trusted(evals)
    }
}

impl<R, const N: usize> Add<TwistedNttPoly<R, N>> for &TwistedNttPoly<R, N>
where
    R: IntegerRing + NTTRing + CanonicalSerialize + CanonicalDeserialize + Send + Sync + 'static,
{
    type Output = TwistedNttPoly<R, N>;

    fn add(self, rhs: TwistedNttPoly<R, N>) -> Self::Output {
        let mut evals = self.evals.clone();
        R::add_assign_slice(&mut evals, &rhs.evals);
        TwistedNttPoly::from_evals_trusted(evals)
    }
}

impl<R, const N: usize> AddAssign for TwistedNttPoly<R, N>
where
    R: IntegerRing + NTTRing + CanonicalSerialize + CanonicalDeserialize + Send + Sync + 'static,
{
    fn add_assign(&mut self, rhs: Self) {
        R::add_assign_slice(&mut self.evals, &rhs.evals);
    }
}

impl<R, const N: usize> AddAssign<&Self> for TwistedNttPoly<R, N>
where
    R: IntegerRing + NTTRing + CanonicalSerialize + CanonicalDeserialize + Send + Sync + 'static,
{
    fn add_assign(&mut self, rhs: &Self) {
        R::add_assign_slice(&mut self.evals, &rhs.evals);
    }
}

impl<R, const N: usize> Sub for TwistedNttPoly<R, N>
where
    R: IntegerRing + NTTRing + CanonicalSerialize + CanonicalDeserialize + Send + Sync + 'static,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        self - &rhs
    }
}

impl<R, const N: usize> Sub<&Self> for TwistedNttPoly<R, N>
where
    R: IntegerRing + NTTRing + CanonicalSerialize + CanonicalDeserialize + Send + Sync + 'static,
{
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self {
        let mut evals = self.evals;
        R::sub_assign_slice(&mut evals, &rhs.evals);
        Self::from_evals_trusted(evals)
    }
}

impl<R, const N: usize> Sub<Self> for &TwistedNttPoly<R, N>
where
    R: IntegerRing + NTTRing + CanonicalSerialize + CanonicalDeserialize + Send + Sync + 'static,
{
    type Output = TwistedNttPoly<R, N>;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut evals = self.evals.clone();
        R::sub_assign_slice(&mut evals, &rhs.evals);
        TwistedNttPoly::from_evals_trusted(evals)
    }
}

impl<R, const N: usize> Sub<TwistedNttPoly<R, N>> for &TwistedNttPoly<R, N>
where
    R: IntegerRing + NTTRing + CanonicalSerialize + CanonicalDeserialize + Send + Sync + 'static,
{
    type Output = TwistedNttPoly<R, N>;

    fn sub(self, rhs: TwistedNttPoly<R, N>) -> Self::Output {
        let mut evals = self.evals.clone();
        R::sub_assign_slice(&mut evals, &rhs.evals);
        TwistedNttPoly::from_evals_trusted(evals)
    }
}

impl<R, const N: usize> SubAssign for TwistedNttPoly<R, N>
where
    R: IntegerRing + NTTRing + CanonicalSerialize + CanonicalDeserialize + Send + Sync + 'static,
{
    fn sub_assign(&mut self, rhs: Self) {
        R::sub_assign_slice(&mut self.evals, &rhs.evals);
    }
}

impl<R, const N: usize> SubAssign<&Self> for TwistedNttPoly<R, N>
where
    R: IntegerRing + NTTRing + CanonicalSerialize + CanonicalDeserialize + Send + Sync + 'static,
{
    fn sub_assign(&mut self, rhs: &Self) {
        R::sub_assign_slice(&mut self.evals, &rhs.evals);
    }
}

impl<R, const N: usize> Mul for TwistedNttPoly<R, N>
where
    R: IntegerRing + NTTRing + CanonicalSerialize + CanonicalDeserialize + Send + Sync + 'static,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        self * &rhs
    }
}

impl<R, const N: usize> Mul<&Self> for TwistedNttPoly<R, N>
where
    R: IntegerRing + NTTRing + CanonicalSerialize + CanonicalDeserialize + Send + Sync + 'static,
{
    type Output = Self;

    fn mul(self, rhs: &Self) -> Self {
        let mut evals = self.evals;
        R::pointwise_mul_assign_slice(&mut evals, &rhs.evals);
        Self::from_evals_trusted(evals)
    }
}

impl<R, const N: usize> Mul<Self> for &TwistedNttPoly<R, N>
where
    R: IntegerRing + NTTRing + CanonicalSerialize + CanonicalDeserialize + Send + Sync + 'static,
{
    type Output = TwistedNttPoly<R, N>;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut evals = self.evals.clone();
        R::pointwise_mul_assign_slice(&mut evals, &rhs.evals);
        TwistedNttPoly::from_evals_trusted(evals)
    }
}

impl<R, const N: usize> Mul<TwistedNttPoly<R, N>> for &TwistedNttPoly<R, N>
where
    R: IntegerRing + NTTRing + CanonicalSerialize + CanonicalDeserialize + Send + Sync + 'static,
{
    type Output = TwistedNttPoly<R, N>;

    fn mul(self, rhs: TwistedNttPoly<R, N>) -> Self::Output {
        let mut evals = self.evals.clone();
        R::pointwise_mul_assign_slice(&mut evals, &rhs.evals);
        TwistedNttPoly::from_evals_trusted(evals)
    }
}

impl<R, const N: usize> MulAssign for TwistedNttPoly<R, N>
where
    R: IntegerRing + NTTRing + CanonicalSerialize + CanonicalDeserialize + Send + Sync + 'static,
{
    fn mul_assign(&mut self, rhs: Self) {
        R::pointwise_mul_assign_slice(&mut self.evals, &rhs.evals);
    }
}

impl<R, const N: usize> MulAssign<&Self> for TwistedNttPoly<R, N>
where
    R: IntegerRing + NTTRing + CanonicalSerialize + CanonicalDeserialize + Send + Sync + 'static,
{
    fn mul_assign(&mut self, rhs: &Self) {
        R::pointwise_mul_assign_slice(&mut self.evals, &rhs.evals);
    }
}

impl<R, const N: usize> Neg for TwistedNttPoly<R, N>
where
    R: IntegerRing + NTTRing + CanonicalSerialize + CanonicalDeserialize + Send + Sync + 'static,
{
    type Output = Self;

    fn neg(self) -> Self {
        Self::from_evals_trusted(self.evals.map(Neg::neg))
    }
}

impl<R, const N: usize> Default for TwistedNttPoly<R, N>
where
    R: IntegerRing + NTTRing + CanonicalSerialize + CanonicalDeserialize + Send + Sync + 'static,
{
    fn default() -> Self {
        Self::zero()
    }
}

impl<R, const N: usize> CanonicalSerialize for TwistedNttPoly<R, N>
where
    R: IntegerRing + CanonicalSerialize + CanonicalDeserialize,
{
    fn serialized_size(&self) -> usize {
        self.evals[0].serialized_size() * N
    }

    fn serialize_into(
        &self,
        buf: &mut alloc::vec::Vec<u8>,
    ) -> Result<(), grid_serialize::SerializationError> {
        for eval in &self.evals {
            eval.serialize_into(buf)?;
        }
        Ok(())
    }
}

impl<R, const N: usize> CanonicalDeserialize for TwistedNttPoly<R, N>
where
    R: IntegerRing + CanonicalSerialize + CanonicalDeserialize,
{
    fn deserialize(data: &[u8]) -> Result<(Self, usize), grid_serialize::SerializationError> {
        let elem_size = R::zero().serialized_size();
        if data.len() < elem_size * N {
            return Err(grid_serialize::SerializationError::UnexpectedEnd);
        }
        let mut evals = Vec::with_capacity(N);
        for i in 0..N {
            let slice = &data[i * elem_size..(i + 1) * elem_size];
            evals.push(R::deserialize_exact(slice)?);
        }
        let evals_arr: [R; N] = evals
            .try_into()
            .map_err(|_| grid_serialize::SerializationError::UnexpectedEnd)?;
        Ok((Self { evals: evals_arr }, elem_size * N))
    }
}

impl<R, const N: usize> TwistedNttPoly<R, N>
where
    R: IntegerRing + NTTRing + CanonicalSerialize + CanonicalDeserialize + Send + Sync + 'static,
{
    /// Multiply every evaluation-domain slot by a ring scalar.
    pub fn scalar_mul(&self, scalar: &R) -> Self {
        let mut evals = self.evals.clone();
        R::scalar_mul_slice(&mut evals, scalar);
        Self::from_evals_trusted(evals)
    }

    /// Multiply every evaluation-domain slot by a ring scalar.
    pub fn scalar_mul_assign(&mut self, scalar: &R) {
        R::scalar_mul_slice(&mut self.evals, scalar);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::arith::prime::{GOLDILOCKS_MODULUS, PrimeField};
    use crate::arith::ring::tests::test_ring_axioms;
    use crate::lattice::types::{RingMat, RingVec};
    use grid_std::UniformRand;

    type F17 = PrimeField<17>;
    type F8380417 = PrimeField<8380417>;
    type FGoldilocks = PrimeField<GOLDILOCKS_MODULUS>;
    type Twisted8 = TwistedNttPoly<F17, 8>;
    type TwistedRq23Np8 = TwistedNttPoly<F8380417, 256>;
    type TwistedGoldilocks = TwistedNttPoly<FGoldilocks, 256>;

    fn random_poly<R, const N: usize>() -> CyclotomicPolyRing<R, N>
    where
        R: IntegerRing
            + CanonicalSerialize
            + CanonicalDeserialize
            + grid_serialize::Valid
            + UniformRand,
    {
        let mut rng = grid_std::test_rng();
        CyclotomicPolyRing::from_array(from_fn(|_| R::rand(&mut rng)))
    }

    #[test]
    fn test_round_trip_small() {
        let poly = random_poly::<F17, 8>();
        let twisted = Twisted8::from_coeff_poly(&poly).unwrap();
        assert_eq!(twisted.to_coeff_poly().unwrap(), poly);
    }

    #[test]
    fn test_round_trip_rq23_np8() {
        let poly = random_poly::<F8380417, 256>();
        let twisted = TwistedRq23Np8::from_coeff_poly(&poly).unwrap();
        assert_eq!(twisted.to_coeff_poly().unwrap(), poly);
    }

    #[test]
    fn test_round_trip_goldilocks() {
        let poly = random_poly::<FGoldilocks, 256>();
        let twisted = TwistedGoldilocks::from_coeff_poly(&poly).unwrap();
        assert_eq!(twisted.to_coeff_poly().unwrap(), poly);
    }

    #[test]
    fn test_ring_axioms_small() {
        let a = Twisted8::from_coeff_poly(&random_poly::<F17, 8>()).unwrap();
        let b = Twisted8::from_coeff_poly(&random_poly::<F17, 8>()).unwrap();
        let c = Twisted8::from_coeff_poly(&random_poly::<F17, 8>()).unwrap();
        test_ring_axioms(a, b, c);
    }

    #[test]
    fn test_mul_matches_coefficient_domain() {
        let a = random_poly::<F17, 8>();
        let b = random_poly::<F17, 8>();
        let twisted_product = (Twisted8::from_coeff_poly(&a).unwrap()
            * Twisted8::from_coeff_poly(&b).unwrap())
        .to_coeff_poly()
        .unwrap();
        assert_eq!(twisted_product, a * b);
    }

    #[test]
    fn test_mul_matches_coefficient_domain_goldilocks() {
        let a = random_poly::<FGoldilocks, 256>();
        let b = random_poly::<FGoldilocks, 256>();
        let twisted_product = (TwistedGoldilocks::from_coeff_poly(&a).unwrap()
            * TwistedGoldilocks::from_coeff_poly(&b).unwrap())
        .to_coeff_poly()
        .unwrap();
        assert_eq!(twisted_product, a * b);
    }

    #[test]
    fn test_ringvec_dot_matches_coefficient_domain() {
        let coeff_lhs = RingVec::new(vec![
            random_poly::<F17, 8>(),
            random_poly::<F17, 8>(),
            random_poly::<F17, 8>(),
        ]);
        let coeff_rhs = RingVec::new(vec![
            random_poly::<F17, 8>(),
            random_poly::<F17, 8>(),
            random_poly::<F17, 8>(),
        ]);
        let twisted_lhs = RingVec::new(
            coeff_lhs
                .entries()
                .iter()
                .map(Twisted8::from_coeff_poly)
                .collect::<Result<_, _>>()
                .unwrap(),
        );
        let twisted_rhs = RingVec::new(
            coeff_rhs
                .entries()
                .iter()
                .map(Twisted8::from_coeff_poly)
                .collect::<Result<_, _>>()
                .unwrap(),
        );
        let coeff_dot = coeff_lhs.dot(&coeff_rhs);
        let twisted_dot = twisted_lhs.dot(&twisted_rhs).to_coeff_poly().unwrap();
        assert_eq!(twisted_dot, coeff_dot);
    }

    #[test]
    fn test_ringmat_mul_vec_matches_coefficient_domain() {
        let coeff_mat_entries = (0..6).map(|_| random_poly::<F17, 8>()).collect();
        let coeff_mat = RingMat::new(2, 3, coeff_mat_entries);
        let coeff_vec = RingVec::new(vec![
            random_poly::<F17, 8>(),
            random_poly::<F17, 8>(),
            random_poly::<F17, 8>(),
        ]);
        let twisted_mat = RingMat::new(
            2,
            3,
            coeff_mat
                .entries()
                .iter()
                .map(Twisted8::from_coeff_poly)
                .collect::<Result<_, _>>()
                .unwrap(),
        );
        let twisted_vec = RingVec::new(
            coeff_vec
                .entries()
                .iter()
                .map(Twisted8::from_coeff_poly)
                .collect::<Result<_, _>>()
                .unwrap(),
        );

        let coeff_out = coeff_mat.mul_vec(&coeff_vec);
        let twisted_out = twisted_mat.mul_vec(&twisted_vec);
        let twisted_out_coeff = RingVec::new(
            twisted_out
                .entries()
                .iter()
                .map(TwistedNttPoly::to_coeff_poly)
                .collect::<Result<_, _>>()
                .unwrap(),
        );
        assert_eq!(twisted_out_coeff, coeff_out);
    }

    #[test]
    fn test_prepare_finish_goldilocks_polys() {
        let polys = vec![
            random_poly::<FGoldilocks, 256>(),
            random_poly::<FGoldilocks, 256>(),
            random_poly::<FGoldilocks, 256>(),
        ];

        let prepared = prepare_twisted_polys(&polys).unwrap();
        let finished = finish_twisted_polys(&prepared).unwrap();

        assert_eq!(finished, polys);
    }

    #[test]
    fn test_prepare_finish_goldilocks_polys_with_plan() {
        let polys = vec![
            random_poly::<FGoldilocks, 256>(),
            random_poly::<FGoldilocks, 256>(),
            random_poly::<FGoldilocks, 256>(),
        ];

        let plan = cached_twisted_plan::<FGoldilocks>(256).unwrap();
        let prepared = prepare_twisted_polys_with_plan(&polys, plan.as_ref()).unwrap();
        let finished = finish_twisted_polys_with_plan(&prepared, plan.as_ref()).unwrap();

        assert_eq!(finished, polys);
    }

    #[test]
    fn test_prepare_finish_goldilocks_ring_vec_and_mat() {
        let vector = RingVec::new(vec![
            random_poly::<FGoldilocks, 256>(),
            random_poly::<FGoldilocks, 256>(),
            random_poly::<FGoldilocks, 256>(),
        ]);
        let matrix = RingMat::new(
            2,
            3,
            vec![
                random_poly::<FGoldilocks, 256>(),
                random_poly::<FGoldilocks, 256>(),
                random_poly::<FGoldilocks, 256>(),
                random_poly::<FGoldilocks, 256>(),
                random_poly::<FGoldilocks, 256>(),
                random_poly::<FGoldilocks, 256>(),
            ],
        );

        let prepared_vec = prepare_twisted_ring_vec(&vector).unwrap();
        let finished_vec = finish_twisted_ring_vec(&prepared_vec).unwrap();
        assert_eq!(finished_vec, vector);

        let prepared_mat = prepare_twisted_ring_mat(&matrix).unwrap();
        let finished_mat = finish_twisted_ring_mat(&prepared_mat).unwrap();
        assert_eq!(finished_mat, matrix);
    }

    #[test]
    fn test_deserialize_corrupt_eval() {
        let poly = random_poly::<F8380417, 256>();
        let twisted = TwistedRq23Np8::from_coeff_poly(&poly).unwrap();
        let mut buf = Vec::new();
        twisted.serialize_into(&mut buf).unwrap();

        // Corrupt the high byte of the first element to 0xFF -> value >= modulus
        buf[2] = 0xFF;
        let result = TwistedRq23Np8::deserialize(&buf);
        assert!(
            result.is_err(),
            "deserialization of corrupt data must fail, not silently map to zero"
        );
    }

    #[test]
    fn test_deserialize_truncated() {
        let poly = random_poly::<F17, 8>();
        let twisted = Twisted8::from_coeff_poly(&poly).unwrap();
        let mut buf = Vec::new();
        twisted.serialize_into(&mut buf).unwrap();

        // Truncate by one element
        let elem_size = buf.len() / 8;
        buf.truncate(buf.len() - elem_size);
        let result = Twisted8::deserialize(&buf);
        assert!(
            result.is_err(),
            "deserialization of truncated data must fail"
        );
    }
}
