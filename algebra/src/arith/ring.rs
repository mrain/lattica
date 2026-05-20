//! Core algebraic traits: [`Ring`], [`IntegerRing`], [`Field`].
//!
//! These traits form a hierarchy:
//! ```text
//! Ring  →  IntegerRing  →  Field (inv, div — only when q is prime)
//! ```

use core::fmt::Debug;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A ring with addition, multiplication, and their identities.
///
/// This is the most general algebraic trait. It does NOT require inversion.
pub trait Ring:
    Sized
    + Clone
    + Debug
    + PartialEq
    + Eq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + for<'a> AddAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
    + for<'a> MulAssign<&'a Self>
{
    /// The additive identity (0).
    fn zero() -> Self;

    /// The multiplicative identity (1).
    fn one() -> Self;

    /// Returns `true` if `self` is the additive identity.
    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }

    /// Returns `true` if `self` is the multiplicative identity.
    fn is_one(&self) -> bool {
        *self == Self::one()
    }

    /// Compute `self + self` (doubling).
    fn double(&self) -> Self {
        self.clone() + self
    }

    /// Compute `self * self` (squaring).
    fn square(&self) -> Self {
        self.clone() * self
    }

    /// Add two references, returning an owned value.
    ///
    /// Override on concrete types to use `&T + &T` impls directly (no clone).
    /// Generic code bounded by `R: Ring` calls `R::add_ref(a, b)` instead of
    /// `a.clone() + b.clone()` to avoid cloning when `R` has ref-ref impls.
    fn add_ref(_a: &Self, _b: &Self) -> Self;

    /// Subtract two references, returning an owned value.
    fn sub_ref(_a: &Self, _b: &Self) -> Self;

    /// Multiply two references, returning an owned value.
    fn mul_ref(_a: &Self, _b: &Self) -> Self;

    /// Add `src` into `dst` elementwise.
    #[doc(hidden)]
    fn add_assign_slice(dst: &mut [Self], src: &[Self]) {
        assert_eq!(dst.len(), src.len(), "slice lengths must match");
        for (lhs, rhs) in dst.iter_mut().zip(src.iter()) {
            *lhs += rhs;
        }
    }

    /// Subtract `src` from `dst` elementwise.
    #[doc(hidden)]
    fn sub_assign_slice(dst: &mut [Self], src: &[Self]) {
        assert_eq!(dst.len(), src.len(), "slice lengths must match");
        for (lhs, rhs) in dst.iter_mut().zip(src.iter()) {
            *lhs -= rhs;
        }
    }

    /// Multiply every element in `dst` by `scalar`.
    #[doc(hidden)]
    fn scalar_mul_slice(dst: &mut [Self], scalar: &Self) {
        for value in dst.iter_mut() {
            *value *= scalar;
        }
    }

    /// Multiply `dst` pointwise by `rhs`.
    #[doc(hidden)]
    fn pointwise_mul_assign_slice(dst: &mut [Self], rhs: &[Self]) {
        assert_eq!(dst.len(), rhs.len(), "slice lengths must match");
        for (lhs, rhs) in dst.iter_mut().zip(rhs.iter()) {
            *lhs *= rhs;
        }
    }

    /// Compute the dot product of two slices.
    ///
    /// Default impl clones `lhs` each iteration. Crates implementing `IntegerRing`
    /// should override this with a slice-based version to avoid allocations.
    #[doc(hidden)]
    fn dot_product(lhs: &[Self], rhs: &[Self]) -> Self {
        assert_eq!(lhs.len(), rhs.len(), "slice lengths must match");
        lhs.iter()
            .zip(rhs.iter())
            .fold(Self::zero(), |mut acc, (lhs, rhs)| {
                acc += Self::mul_ref(lhs, rhs);
                acc
            })
    }

    /// Compute `dst += scalar * src` elementwise.
    #[doc(hidden)]
    fn add_assign_scaled_slice(dst: &mut [Self], src: &[Self], scalar: &Self) {
        assert_eq!(dst.len(), src.len(), "slice lengths must match");
        for (dst_value, src_value) in dst.iter_mut().zip(src.iter()) {
            *dst_value += Self::mul_ref(src_value, scalar);
        }
    }

    /// Compute `self^exp` by repeated squaring.
    fn pow(&self, mut exp: u64) -> Self {
        let mut base = self.clone();
        let mut result = Self::one();
        while exp > 0 {
            if exp & 1 == 1 {
                let b = base.clone();
                result *= b;
            }
            base = base.square();
            exp >>= 1;
        }
        result
    }
}

/// An integer ring `Z_q` with a modulus and reduction.
///
/// Extends [`Ring`] with modulus-aware operations.
pub trait IntegerRing: Ring {
    /// The unsigned integer type used to represent the modulus.
    type Uint: Clone + Debug + PartialEq + Eq + PartialOrd + Ord;

    /// The modulus `q`.
    fn modulus() -> Self::Uint;

    /// Create an element from a `u64`, reducing modulo `q`.
    fn from_u64(val: u64) -> Self;

    /// Export the element as a `u64`.
    ///
    /// # Panics
    /// May panic if the internal representation doesn't fit in a `u64`.
    fn to_u64(&self) -> u64;

    /// Centered L2 representative as `f64`.
    ///
    /// For prime fields: the centered integer in `[-q/2, q/2)`.
    /// For power-of-two rings: the signed wrap-around value in `[-2^(K-1), 2^(K-1))`.
    /// Precision degrades above 2^53. Sufficient for norm-bound comparisons
    /// where relative accuracy suffices (LaBRADOR q ≈ 2^32).
    fn lossy_l2_value(&self) -> f64;

    /// Reduce a value modulo `q` (ensure canonical representation).
    fn reduce(&self) -> Self;
}

/// A field `Z_q` where `q` is prime — supports multiplicative inversion.
///
/// Only implemented for prime moduli.
pub trait Field: IntegerRing {
    /// Compute the multiplicative inverse of `self`.
    ///
    /// # Panics
    /// Panics if `self` is zero.
    fn inv(&self) -> Self;

    /// Compute `self / other`.
    ///
    /// Equivalent to `self * other.inv()`.
    fn div(&self, other: &Self) -> Self {
        let inv = other.inv();
        Self::mul_ref(self, &inv)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Generic test that verifies Ring axioms for any type implementing Ring.
    pub fn test_ring_axioms<R: Ring>(a: R, b: R, c: R) {
        // Additive identity
        assert_eq!(a.clone() + &R::zero(), a, "a + 0 = a");
        assert_eq!(R::zero() + &a, a, "0 + a = a");

        // Multiplicative identity
        assert_eq!(a.clone() * &R::one(), a, "a * 1 = a");
        assert_eq!(R::one() * &a, a, "1 * a = a");

        // Additive commutativity: a + b = b + a
        assert_eq!(a.clone() + &b, b.clone() + &a, "a + b = b + a");

        // Multiplicative commutativity: a * b = b * a
        assert_eq!(a.clone() * &b, b.clone() * &a, "a * b = b * a");

        // Additive associativity: (a + b) + c = a + (b + c)
        assert_eq!(
            (a.clone() + &b) + &c,
            a.clone() + &(b.clone() + &c),
            "(a + b) + c = a + (b + c)"
        );

        // Multiplicative associativity: (a * b) * c = a * (b * c)
        assert_eq!(
            (a.clone() * &b) * &c,
            a.clone() * &(b.clone() * &c),
            "(a * b) * c = a * (b * c)"
        );

        // Distributivity: a * (b + c) = a * b + a * c
        assert_eq!(
            a.clone() * &(b.clone() + &c),
            (a.clone() * &b) + &(a.clone() * &c),
            "a * (b + c) = a * b + a * c"
        );

        // Additive inverse: a + (-a) = 0
        assert_eq!(a.clone() + &(-a.clone()), R::zero(), "a + (-a) = 0");

        // Zero element: a * 0 = 0
        assert_eq!(a.clone() * &R::zero(), R::zero(), "a * 0 = 0");
    }

    /// Generic test for IntegerRing operations.
    pub fn test_integer_ring<R: IntegerRing>(val: u64) {
        let a = R::from_u64(val);
        let round_tripped = a.to_u64();
        // The round-trip should give the value reduced mod q
        let _ = round_tripped; // value depends on modulus

        // Reduce should be idempotent
        assert_eq!(a.reduce(), a, "reduce is idempotent");
    }

    /// Generic test for Field operations.
    pub fn test_field_axioms<F: Field>(a: F) {
        if !a.is_zero() {
            let inv_a = a.inv();
            assert_eq!(a.clone() * &inv_a, F::one(), "a * a^(-1) = 1");
            assert_eq!(inv_a * &a, F::one(), "a^(-1) * a = 1");
        }
    }
}
