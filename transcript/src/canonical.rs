//! Canonical structured encoding for the field-native transcript surface.

use core::marker::PhantomData;

use grid_serialize::CanonicalSerialize;

use crate::TranscriptError;
use crate::field::{FieldTranscript, TranscriptField, encode_count, encode_count_u64, pack_bytes};

const STRUCT_TAG: u64 = 16;
const SEQUENCE_TAG: u64 = 17;
const VARIANT_TAG: u64 = 18;
const U64_TAG: u64 = 19;
const BYTES_TAG: u64 = 20;

/// Structured object encoding for the field-native transcript surface.
pub trait CanonicalTranscriptEncode<F: TranscriptField> {
    /// Encode `self` under `label` using the shared canonical encoder.
    fn encode<T: FieldTranscript<F>>(
        &self,
        label: &'static [u8],
        encoder: &mut CanonicalTranscriptEncoder<'_, F, T>,
    ) -> Result<(), TranscriptError>;
}

/// Shared helper for canonical field-native transcript encoding.
pub struct CanonicalTranscriptEncoder<'a, F: TranscriptField, T: FieldTranscript<F>> {
    transcript: &'a mut T,
    root_label: &'static [u8],
    saw_root_event: bool,
    _marker: PhantomData<F>,
}

impl<'a, F: TranscriptField, T: FieldTranscript<F>> CanonicalTranscriptEncoder<'a, F, T> {
    pub(crate) fn new(transcript: &'a mut T, root_label: &'static [u8]) -> Self {
        Self {
            transcript,
            root_label,
            saw_root_event: false,
            _marker: PhantomData,
        }
    }

    pub(crate) fn finish(self) -> Result<(), TranscriptError> {
        if self.saw_root_event {
            Ok(())
        } else {
            Err(TranscriptError::InvalidCanonicalEncoding)
        }
    }

    fn note_event(&mut self, label: &'static [u8]) -> Result<(), TranscriptError> {
        if label == self.root_label {
            if self.saw_root_event {
                return Err(TranscriptError::InvalidCanonicalEncoding);
            }
            self.saw_root_event = true;
            return Ok(());
        }

        if !self.saw_root_event {
            return Err(TranscriptError::InvalidCanonicalEncoding);
        }
        Ok(())
    }

    fn tagged_u64s(&mut self, label: &'static [u8], values: &[u64]) -> Result<(), TranscriptError> {
        self.note_event(label)?;
        let encoded: alloc::vec::Vec<F> = values.iter().map(|value| F::from_u64(*value)).collect();
        self.transcript.append_elements(label, &encoded)
    }

    /// Begin a canonically encoded struct.
    pub fn begin_struct(
        &mut self,
        label: &'static [u8],
        field_count: usize,
    ) -> Result<(), TranscriptError> {
        self.tagged_u64s(label, &[STRUCT_TAG, encode_count_u64(field_count)?])
    }

    /// Begin a canonically encoded sequence.
    pub fn begin_sequence(
        &mut self,
        label: &'static [u8],
        len: usize,
    ) -> Result<(), TranscriptError> {
        self.tagged_u64s(label, &[SEQUENCE_TAG, encode_count_u64(len)?])
    }

    /// Record a canonical variant discriminator.
    pub fn variant(
        &mut self,
        label: &'static [u8],
        discriminant: u64,
    ) -> Result<(), TranscriptError> {
        self.tagged_u64s(label, &[VARIANT_TAG, discriminant])
    }

    /// Encode a `u64` value canonically.
    pub fn u64(&mut self, label: &'static [u8], value: u64) -> Result<(), TranscriptError> {
        self.tagged_u64s(label, &[U64_TAG, value])
    }

    /// Encode raw bytes canonically via transcript-field chunks.
    pub fn bytes(&mut self, label: &'static [u8], value: &[u8]) -> Result<(), TranscriptError> {
        self.note_event(label)?;

        let chunks = pack_bytes::<F>(value);
        let mut encoded = alloc::vec::Vec::with_capacity(2 + chunks.len());
        encoded.push(F::from_u64(BYTES_TAG));
        encoded.push(encode_count::<F>(value.len())?);
        encoded.extend(chunks);
        self.transcript.append_elements(label, &encoded)
    }

    /// Encode a field element canonically.
    pub fn field(&mut self, label: &'static [u8], value: &F) -> Result<(), TranscriptError> {
        self.note_event(label)?;
        self.transcript.append_element(label, value)
    }

    /// Encode a slice of field elements canonically.
    pub fn fields(&mut self, label: &'static [u8], values: &[F]) -> Result<(), TranscriptError> {
        self.note_event(label)?;
        self.transcript.append_elements(label, values)
    }

    /// Encode a nested structured value canonically.
    pub fn object<E: CanonicalTranscriptEncode<F>>(
        &mut self,
        label: &'static [u8],
        value: &E,
    ) -> Result<(), TranscriptError> {
        value.encode(label, self)
    }

    /// Encode a `CanonicalSerialize` value through the explicit byte bridge.
    pub fn serialized<S: CanonicalSerialize>(
        &mut self,
        label: &'static [u8],
        value: &S,
    ) -> Result<(), TranscriptError> {
        let payload = value.serialize()?;
        self.bytes(label, &payload)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use grid_algebra::arith::prime::{GOLDILOCKS_MODULUS, PrimeField};
    use grid_algebra::arith::ring::IntegerRing;

    use super::*;

    type Goldilocks = PrimeField<GOLDILOCKS_MODULUS>;

    #[derive(Default)]
    struct RecordingFieldTranscript {
        frames: Vec<Vec<Goldilocks>>,
    }

    impl FieldTranscript<Goldilocks> for RecordingFieldTranscript {
        fn append_preframed_elements(
            &mut self,
            elements: &[Goldilocks],
        ) -> Result<(), TranscriptError> {
            self.frames.push(elements.to_vec());
            Ok(())
        }

        fn challenge_preframed_elements(
            &mut self,
            request_frame: &[Goldilocks],
            out_len: usize,
        ) -> Result<Vec<Goldilocks>, TranscriptError> {
            self.frames.push(request_frame.to_vec());
            Ok((0..out_len).map(|_| Goldilocks::from_u64(0)).collect())
        }
    }

    struct StructEncoding {
        first: Goldilocks,
        second: Goldilocks,
    }

    impl CanonicalTranscriptEncode<Goldilocks> for StructEncoding {
        fn encode<T: FieldTranscript<Goldilocks>>(
            &self,
            label: &'static [u8],
            encoder: &mut CanonicalTranscriptEncoder<'_, Goldilocks, T>,
        ) -> Result<(), TranscriptError> {
            encoder.begin_struct(label, 2)?;
            encoder.field(b"first", &self.first)?;
            encoder.field(b"second", &self.second)
        }
    }

    struct SequenceEncoding {
        values: [Goldilocks; 2],
    }

    impl CanonicalTranscriptEncode<Goldilocks> for SequenceEncoding {
        fn encode<T: FieldTranscript<Goldilocks>>(
            &self,
            label: &'static [u8],
            encoder: &mut CanonicalTranscriptEncoder<'_, Goldilocks, T>,
        ) -> Result<(), TranscriptError> {
            encoder.begin_sequence(label, self.values.len())?;
            encoder.fields(b"values", &self.values)
        }
    }

    struct VariantEncoding {
        value: Goldilocks,
    }

    impl CanonicalTranscriptEncode<Goldilocks> for VariantEncoding {
        fn encode<T: FieldTranscript<Goldilocks>>(
            &self,
            label: &'static [u8],
            encoder: &mut CanonicalTranscriptEncoder<'_, Goldilocks, T>,
        ) -> Result<(), TranscriptError> {
            encoder.variant(label, 1)?;
            encoder.field(b"value", &self.value)
        }
    }

    struct BadEncoding;

    impl CanonicalTranscriptEncode<Goldilocks> for BadEncoding {
        fn encode<T: FieldTranscript<Goldilocks>>(
            &self,
            _label: &'static [u8],
            encoder: &mut CanonicalTranscriptEncoder<'_, Goldilocks, T>,
        ) -> Result<(), TranscriptError> {
            encoder.field(b"child", &Goldilocks::from_u64(1))
        }
    }

    struct DuplicateRootEncoding;

    impl CanonicalTranscriptEncode<Goldilocks> for DuplicateRootEncoding {
        fn encode<T: FieldTranscript<Goldilocks>>(
            &self,
            label: &'static [u8],
            encoder: &mut CanonicalTranscriptEncoder<'_, Goldilocks, T>,
        ) -> Result<(), TranscriptError> {
            encoder.begin_struct(label, 1)?;
            encoder.field(label, &Goldilocks::from_u64(1))
        }
    }

    #[test]
    fn test_struct_and_sequence_are_separated() {
        let mut structured = RecordingFieldTranscript::default();
        let mut sequenced = RecordingFieldTranscript::default();

        structured
            .append_encoded(
                b"root",
                &StructEncoding {
                    first: Goldilocks::from_u64(1),
                    second: Goldilocks::from_u64(2),
                },
            )
            .unwrap();
        sequenced
            .append_encoded(
                b"root",
                &SequenceEncoding {
                    values: [Goldilocks::from_u64(1), Goldilocks::from_u64(2)],
                },
            )
            .unwrap();

        assert_ne!(structured.frames[0], sequenced.frames[0]);
    }

    #[test]
    fn test_variant_and_struct_are_separated() {
        let mut variant = RecordingFieldTranscript::default();
        let mut structured = RecordingFieldTranscript::default();

        variant
            .append_encoded(
                b"root",
                &VariantEncoding {
                    value: Goldilocks::from_u64(9),
                },
            )
            .unwrap();
        structured
            .append_encoded(
                b"root",
                &StructEncoding {
                    first: Goldilocks::from_u64(9),
                    second: Goldilocks::from_u64(0),
                },
            )
            .unwrap();

        assert_ne!(variant.frames[0], structured.frames[0]);
    }

    #[test]
    fn test_u64_and_field_are_separated() {
        let mut encoded_u64 = RecordingFieldTranscript::default();
        let mut encoded_field = RecordingFieldTranscript::default();

        {
            let mut encoder = CanonicalTranscriptEncoder::new(&mut encoded_u64, b"root");
            encoder.u64(b"root", 7).unwrap();
            encoder.finish().unwrap();
        }
        {
            let mut encoder = CanonicalTranscriptEncoder::new(&mut encoded_field, b"root");
            encoder.field(b"root", &Goldilocks::from_u64(7)).unwrap();
            encoder.finish().unwrap();
        }

        assert_ne!(encoded_u64.frames, encoded_field.frames);
    }

    #[test]
    fn test_bytes_are_length_sensitive() {
        let mut lhs = RecordingFieldTranscript::default();
        let mut rhs = RecordingFieldTranscript::default();

        {
            let mut encoder = CanonicalTranscriptEncoder::new(&mut lhs, b"root");
            encoder.bytes(b"root", b"abc").unwrap();
            encoder.finish().unwrap();
        }
        {
            let mut encoder = CanonicalTranscriptEncoder::new(&mut rhs, b"root");
            encoder.bytes(b"root", b"abc\0").unwrap();
            encoder.finish().unwrap();
        }

        assert_ne!(lhs.frames, rhs.frames);
    }

    #[test]
    fn test_nested_object_order_changes_encoding() {
        let mut lhs = RecordingFieldTranscript::default();
        let mut rhs = RecordingFieldTranscript::default();

        lhs.append_encoded(
            b"root",
            &StructEncoding {
                first: Goldilocks::from_u64(1),
                second: Goldilocks::from_u64(2),
            },
        )
        .unwrap();
        rhs.append_encoded(
            b"root",
            &StructEncoding {
                first: Goldilocks::from_u64(2),
                second: Goldilocks::from_u64(1),
            },
        )
        .unwrap();

        assert_ne!(lhs.frames, rhs.frames);
    }

    #[test]
    fn test_root_label_boundary_is_enforced() {
        let mut transcript = RecordingFieldTranscript::default();
        assert_eq!(
            transcript.append_encoded(b"root", &BadEncoding),
            Err(TranscriptError::InvalidCanonicalEncoding)
        );
    }

    #[test]
    fn test_root_label_must_appear_exactly_once() {
        let mut transcript = RecordingFieldTranscript::default();
        assert_eq!(
            transcript.append_encoded(b"root", &DuplicateRootEncoding),
            Err(TranscriptError::InvalidCanonicalEncoding)
        );
    }
}
