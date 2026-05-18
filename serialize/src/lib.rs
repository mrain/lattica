//! Gridland serialization — canonical (de)serialization traits for algebraic objects.
//!
//! Canonical encodings in this crate are byte-exact and platform independent:
//! fixed-width integers use little-endian byte order, `usize` is encoded as a
//! fixed-width little-endian `u64`, `f64` is encoded as its IEEE-754 bit pattern
//! in little-endian `u64` form, and variable-length containers use little-endian
//! `u32` length prefixes unless their type documentation says otherwise.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub use grid_serialize_derive::{CanonicalDeserialize, CanonicalSerialize};

use alloc::string::String;
use alloc::vec::Vec;

fn ensure_full_consumption(consumed: usize, total: usize) -> Result<(), SerializationError> {
    if consumed == total {
        Ok(())
    } else {
        Err(SerializationError::InvalidData(
            "trailing bytes after deserialized value".into(),
        ))
    }
}

/// Errors that can occur during serialization/deserialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SerializationError {
    /// The data is too short to deserialize.
    UnexpectedEnd,
    /// The deserialized data fails validity checks.
    InvalidData(String),
}

impl core::fmt::Display for SerializationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnexpectedEnd => write!(f, "unexpected end of data"),
            Self::InvalidData(msg) => write!(f, "invalid data: {msg}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SerializationError {}

/// Canonical serialization into bytes.
pub trait CanonicalSerialize {
    /// Returns the serialized size in bytes.
    fn serialized_size(&self) -> usize;

    /// Serialize `self` into a byte vector.
    fn serialize(&self) -> Result<Vec<u8>, SerializationError> {
        let expected = self.serialized_size();
        let mut buf = Vec::with_capacity(expected);
        let start_len = buf.len();
        self.serialize_into(&mut buf)?;
        debug_assert_eq!(
            buf.len() - start_len,
            expected,
            "serialized_size() must match bytes appended by serialize_into() for {}",
            core::any::type_name::<Self>()
        );
        Ok(buf)
    }

    /// Serialize `self` by appending to the given buffer.
    fn serialize_into(&self, buf: &mut Vec<u8>) -> Result<(), SerializationError>;
}

/// Canonical deserialization from bytes.
pub trait CanonicalDeserialize: Sized {
    /// Deserialize from a byte slice, returning the value and the number of bytes consumed.
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError>;

    /// Deserialize from a byte slice and require exact input consumption.
    fn deserialize_exact(data: &[u8]) -> Result<Self, SerializationError> {
        let (value, consumed) = Self::deserialize(data)?;
        ensure_full_consumption(consumed, data.len())?;
        Ok(value)
    }
}

/// Validity check for deserialized values.
///
/// Some types need additional validation after deserialization
/// (e.g., checking that a value is in range, or that a point is on a curve).
pub trait Valid: CanonicalDeserialize {
    /// Check that `self` satisfies all validity constraints.
    fn is_valid(&self) -> bool;

    /// Validate `self`, returning a diagnostic error on failure.
    fn validate(&self) -> Result<(), SerializationError> {
        if self.is_valid() {
            Ok(())
        } else {
            Err(SerializationError::InvalidData(
                "deserialized value is invalid".into(),
            ))
        }
    }

    /// Deserialize and validate in one step.
    fn deserialize_and_validate(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        let (value, consumed) = Self::deserialize(data)?;
        value.validate()?;
        Ok((value, consumed))
    }

    /// Deserialize, validate, and require exact input consumption.
    fn deserialize_and_validate_exact(data: &[u8]) -> Result<Self, SerializationError> {
        let (value, consumed) = Self::deserialize_and_validate(data)?;
        ensure_full_consumption(consumed, data.len())?;
        Ok(value)
    }
}

// -- Blanket impls for primitive integer types --

macro_rules! impl_int {
    ($t:ty, $bytes:expr) => {
        impl CanonicalSerialize for $t {
            fn serialized_size(&self) -> usize {
                $bytes
            }

            fn serialize_into(&self, buf: &mut Vec<u8>) -> Result<(), SerializationError> {
                buf.extend_from_slice(&self.to_le_bytes());
                Ok(())
            }
        }

        impl CanonicalDeserialize for $t {
            fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
                if data.len() < $bytes {
                    return Err(SerializationError::UnexpectedEnd);
                }
                let mut bytes = [0u8; $bytes];
                bytes.copy_from_slice(&data[..$bytes]);
                Ok((<$t>::from_le_bytes(bytes), $bytes))
            }
        }
    };
}

impl_int!(u8, 1);
impl_int!(u16, 2);
impl_int!(u32, 4);
impl_int!(u64, 8);
impl_int!(u128, 16);

/// f64 is serialized as bit-exact u64 (IEEE 754 representation).
impl CanonicalSerialize for f64 {
    fn serialized_size(&self) -> usize {
        8
    }

    fn serialize_into(&self, buf: &mut Vec<u8>) -> Result<(), SerializationError> {
        buf.extend_from_slice(&self.to_bits().to_le_bytes());
        Ok(())
    }
}

impl CanonicalDeserialize for f64 {
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        if data.len() < 8 {
            return Err(SerializationError::UnexpectedEnd);
        }
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&data[..8]);
        Ok((f64::from_bits(u64::from_le_bytes(bytes)), 8))
    }
}
/// usize is serialized as fixed-width u64 for cross-platform compatibility.
impl CanonicalSerialize for usize {
    fn serialized_size(&self) -> usize {
        8
    }

    fn serialize_into(&self, buf: &mut Vec<u8>) -> Result<(), SerializationError> {
        let val: u64 = (*self)
            .try_into()
            .map_err(|_| SerializationError::InvalidData("usize exceeds u64".into()))?;
        buf.extend_from_slice(&val.to_le_bytes());
        Ok(())
    }
}

impl CanonicalDeserialize for usize {
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        if data.len() < 8 {
            return Err(SerializationError::UnexpectedEnd);
        }
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&data[..8]);
        let val = u64::from_le_bytes(bytes);
        let result = usize::try_from(val).map_err(|_| {
            SerializationError::InvalidData(
                "deserialized usize exceeds platform usize range".into(),
            )
        })?;
        Ok((result, 8))
    }
}

// -- Blanket impls for Vec<T> --

impl<T: CanonicalSerialize> CanonicalSerialize for Vec<T> {
    fn serialized_size(&self) -> usize {
        // 4 bytes for length + sum of element sizes
        4 + self.iter().map(|e| e.serialized_size()).sum::<usize>()
    }

    fn serialize_into(&self, buf: &mut Vec<u8>) -> Result<(), SerializationError> {
        let len = u32::try_from(self.len())
            .map_err(|_| SerializationError::InvalidData("Vec length exceeds u32".into()))?;
        buf.extend_from_slice(&len.to_le_bytes());
        for elem in self {
            elem.serialize_into(buf)?;
        }
        Ok(())
    }
}

impl<T: CanonicalDeserialize> CanonicalDeserialize for Vec<T> {
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        if data.len() < 4 {
            return Err(SerializationError::UnexpectedEnd);
        }
        let len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let mut offset = 4;
        // Use `try_reserve_exact` instead of `Vec::with_capacity(len)` — a
        // maliciously large len would panic or trigger unbounded allocation.
        let mut vec: Vec<T> = Vec::new();
        vec.try_reserve_exact(len)
            .map_err(|_| SerializationError::InvalidData("Vec allocation failed".into()))?;
        for _ in 0..len {
            let (elem, consumed) = T::deserialize(&data[offset..])?;
            offset += consumed;
            vec.push(elem);
        }
        Ok((vec, offset))
    }
}

// -- Blanket impls for strings and arrays --

/// UTF-8 string serialized as a length-prefixed byte vector.
impl CanonicalSerialize for String {
    fn serialized_size(&self) -> usize {
        4 + self.len()
    }

    fn serialize_into(&self, buf: &mut Vec<u8>) -> Result<(), SerializationError> {
        let len = u32::try_from(self.len())
            .map_err(|_| SerializationError::InvalidData("String length exceeds u32".into()))?;
        buf.extend_from_slice(&len.to_le_bytes());
        buf.extend_from_slice(self.as_bytes());
        Ok(())
    }
}

impl CanonicalDeserialize for String {
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        let (bytes, consumed) = Vec::<u8>::deserialize(data)?;
        let value = String::from_utf8(bytes)
            .map_err(|_| SerializationError::InvalidData("invalid UTF-8 string".into()))?;
        Ok((value, consumed))
    }
}

impl<T: CanonicalSerialize, const N: usize> CanonicalSerialize for [T; N] {
    fn serialized_size(&self) -> usize {
        self.iter().map(CanonicalSerialize::serialized_size).sum()
    }

    fn serialize_into(&self, buf: &mut Vec<u8>) -> Result<(), SerializationError> {
        for elem in self {
            elem.serialize_into(buf)?;
        }
        Ok(())
    }
}

impl<T: CanonicalDeserialize, const N: usize> CanonicalDeserialize for [T; N] {
    fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
        let mut offset = 0;
        let mut elems = Vec::new();
        elems
            .try_reserve_exact(N)
            .map_err(|_| SerializationError::InvalidData("array allocation failed".into()))?;
        for _ in 0..N {
            let (elem, consumed) = T::deserialize(&data[offset..])?;
            offset += consumed;
            elems.push(elem);
        }
        let array = elems
            .try_into()
            .map_err(|_| SerializationError::InvalidData("array length mismatch".into()))?;
        Ok((array, offset))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_u64_round_trip() {
        let val: u64 = 0xDEAD_BEEF_CAFE_BABE;
        let bytes = val.serialize().unwrap();
        assert_eq!(bytes.len(), 8);
        let decoded = u64::deserialize_exact(&bytes).unwrap();
        assert_eq!(decoded, val);
    }

    #[test]
    fn test_u32_round_trip() {
        let val: u32 = 0xDEAD_BEEF;
        let bytes = val.serialize().unwrap();
        assert_eq!(bytes.len(), 4);
        let decoded = u32::deserialize_exact(&bytes).unwrap();
        assert_eq!(decoded, val);
    }

    #[test]
    fn test_vec_round_trip() {
        let val: Vec<u64> = vec![1, 2, 3];
        let bytes = val.serialize().unwrap();
        let decoded = Vec::<u64>::deserialize_exact(&bytes).unwrap();
        assert_eq!(decoded, val);
    }

    #[test]
    fn test_byte_array_round_trip() {
        let val: [u8; 32] = [42; 32];
        let bytes = val.serialize().unwrap();
        assert_eq!(bytes.len(), 32);
        let decoded = <[u8; 32]>::deserialize_exact(&bytes).unwrap();
        assert_eq!(decoded, val);
    }

    #[test]
    fn test_generic_array_round_trip() {
        let val: [u16; 3] = [17, 42, 65535];
        let bytes = val.serialize().unwrap();
        assert_eq!(bytes.len(), 6);
        let decoded = <[u16; 3]>::deserialize_exact(&bytes).unwrap();
        assert_eq!(decoded, val);
    }

    #[test]
    fn test_string_round_trip() {
        let val = String::from("gridland");
        let bytes = val.serialize().unwrap();
        assert_eq!(&bytes[..4], &(8u32.to_le_bytes()));
        let decoded = String::deserialize_exact(&bytes).unwrap();
        assert_eq!(decoded, val);
    }

    #[test]
    fn test_string_rejects_invalid_utf8() {
        let bytes = vec![2, 0, 0, 0, 0xff, 0xff];
        let err = String::deserialize_exact(&bytes).unwrap_err();
        assert_eq!(
            err,
            SerializationError::InvalidData("invalid UTF-8 string".into())
        );
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct EvenOnly(u64);

    impl CanonicalDeserialize for EvenOnly {
        fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
            let (value, used) = u64::deserialize(data)?;
            Ok((Self(value), used))
        }
    }

    impl Valid for EvenOnly {
        fn is_valid(&self) -> bool {
            self.0.is_multiple_of(2)
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct DiagnosticValid(u64);

    impl CanonicalDeserialize for DiagnosticValid {
        fn deserialize(data: &[u8]) -> Result<(Self, usize), SerializationError> {
            let (value, used) = u64::deserialize(data)?;
            Ok((Self(value), used))
        }
    }

    impl Valid for DiagnosticValid {
        fn is_valid(&self) -> bool {
            self.0 == 7
        }

        fn validate(&self) -> Result<(), SerializationError> {
            if self.is_valid() {
                Ok(())
            } else {
                Err(SerializationError::InvalidData("expected exactly 7".into()))
            }
        }
    }

    #[test]
    fn test_deserialize_and_validate_rejects_invalid_data() {
        let err = EvenOnly::deserialize_and_validate(&3u64.to_le_bytes()).unwrap_err();
        assert_eq!(
            err,
            SerializationError::InvalidData("deserialized value is invalid".into())
        );
    }

    #[test]
    fn test_deserialize_and_validate_uses_diagnostic_validate() {
        let err = DiagnosticValid::deserialize_and_validate(&8u64.to_le_bytes()).unwrap_err();
        assert_eq!(
            err,
            SerializationError::InvalidData("expected exactly 7".into())
        );
    }

    #[test]
    fn test_deserialize_exact_rejects_trailing_bytes() {
        let mut bytes = 5u64.to_le_bytes().to_vec();
        bytes.push(0xAA);
        let err = u64::deserialize_exact(&bytes).unwrap_err();
        assert_eq!(
            err,
            SerializationError::InvalidData("trailing bytes after deserialized value".into())
        );
    }

    #[test]
    fn test_deserialize_and_validate_exact_rejects_trailing_bytes() {
        let mut bytes = 4u64.to_le_bytes().to_vec();
        bytes.push(0xAA);
        let err = EvenOnly::deserialize_and_validate_exact(&bytes).unwrap_err();
        assert_eq!(
            err,
            SerializationError::InvalidData("trailing bytes after deserialized value".into())
        );
    }
}
