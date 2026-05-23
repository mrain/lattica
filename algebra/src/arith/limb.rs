//! [`UintLimb`] — sealed trait for the limb (word) type used as the Montgomery backend.
//!
//! Only implemented for `u8`, `u16`, `u32`, and `u64`. Provides the word-level
//! operations needed by `PrimeField` and `Zm`.

/// Sealed so only the four primitive unsigned integers can implement this trait.
pub mod sealed {
    pub trait Sealed {}
    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
}

/// Abstraction over the limb (word) type used as a Montgomery or plain arithmetic backend.
///
/// Only implemented for `u8`, `u16`, `u32`, and `u64`.
///
/// # Safety
///
/// Implementors guarantee that `from_wide_truncate` discards high bits silently
/// (it is semantically a `as` cast from `Wide` to the implementing type).
pub trait UintLimb:
    sealed::Sealed
    + Copy
    + PartialEq
    + Eq
    + PartialOrd
    + Ord
    + core::fmt::Debug
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::AddAssign
    + core::ops::SubAssign
    + core::ops::MulAssign
    + Send
    + Sync
    + 'static
{
    /// The next-wider unsigned integer (`u16` for `u8`, `u32` for `u16`, `u64` for `u32`, `u128` for `u64`).
    type Wide: Copy + 'static;

    /// Number of bits in this limb type.
    const BITS: u32;
    const ZERO: Self;
    const ONE: Self;
    const MAX: Self;

    fn from_u64(v: u64) -> Self;

    fn to_u64(self) -> u64;

    fn to_wide(self) -> Self::Wide;

    fn from_wide_truncate(w: Self::Wide) -> Self;

    fn wrapping_add(self, rhs: Self) -> Self;

    fn wrapping_sub(self, rhs: Self) -> Self;

    fn wrapping_mul(self, rhs: Self) -> Self;

    fn overflowing_add(self, rhs: Self) -> (Self, bool);

    fn wide_add(a: Self::Wide, b: Self::Wide) -> Self::Wide;

    fn wide_mul(a: Self::Wide, b: Self::Wide) -> Self::Wide;

    fn wide_shr(w: Self::Wide, shift: u32) -> Self::Wide;

    fn wide_from_u64(v: u64) -> Self::Wide;

    fn wrapping_neg(self) -> Self;

    fn mod_limb(self, modulus: Self) -> Self;

    fn leading_zeros(self) -> u32;
}

macro_rules! impl_uint_limb {
    ($ty:ty, $wide:ty, $bits:expr) => {
        impl UintLimb for $ty {
            type Wide = $wide;
            const BITS: u32 = $bits;
            const ZERO: Self = 0;
            const ONE: Self = 1;
            const MAX: Self = <$ty>::MAX;

            #[inline(always)]
            fn from_u64(v: u64) -> Self {
                v as $ty
            }

            #[inline(always)]
            fn to_u64(self) -> u64 {
                self as u64
            }

            #[inline(always)]
            fn to_wide(self) -> $wide {
                self as $wide
            }

            #[inline(always)]
            fn from_wide_truncate(w: $wide) -> Self {
                w as $ty
            }

            #[inline(always)]
            fn wrapping_add(self, rhs: Self) -> Self {
                self.wrapping_add(rhs)
            }

            #[inline(always)]
            fn wrapping_sub(self, rhs: Self) -> Self {
                self.wrapping_sub(rhs)
            }

            #[inline(always)]
            fn wrapping_mul(self, rhs: Self) -> Self {
                self.wrapping_mul(rhs)
            }

            #[inline(always)]
            fn overflowing_add(self, rhs: Self) -> (Self, bool) {
                self.overflowing_add(rhs)
            }

            #[inline(always)]
            fn wide_add(a: $wide, b: $wide) -> $wide {
                a.wrapping_add(b)
            }

            #[inline(always)]
            fn wide_mul(a: $wide, b: $wide) -> $wide {
                a.wrapping_mul(b)
            }

            #[inline(always)]
            fn wide_shr(w: $wide, shift: u32) -> $wide {
                w >> shift
            }

            #[inline(always)]
            fn wide_from_u64(v: u64) -> $wide {
                v as $wide
            }

            #[inline(always)]
            fn wrapping_neg(self) -> Self {
                self.wrapping_neg()
            }

            #[inline(always)]
            fn mod_limb(self, modulus: Self) -> Self {
                self % modulus
            }

            #[inline(always)]
            fn leading_zeros(self) -> u32 {
                self.leading_zeros()
            }
        }
    };
}

impl_uint_limb!(u8, u16, 8);
impl_uint_limb!(u16, u32, 16);
impl_uint_limb!(u32, u64, 32);
impl_uint_limb!(u64, u128, 64);
