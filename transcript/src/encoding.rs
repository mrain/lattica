//! Canonical transcript framing and domain separation.

use alloc::vec::Vec;

use crate::TranscriptError;

const APPEND_DOMAIN_TAG: u8 = 0x00;
const CHALLENGE_DOMAIN_TAG: u8 = 0x01;

fn encode_frame(
    domain_tag: u8,
    label: &'static [u8],
    payload: &[u8],
) -> Result<Vec<u8>, TranscriptError> {
    if label.is_empty() {
        return Err(TranscriptError::EmptyLabel);
    }

    let mut out = Vec::with_capacity(1 + 8 + 8 + label.len() + payload.len());
    out.push(domain_tag);
    out.extend_from_slice(&(label.len() as u64).to_le_bytes());
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.extend_from_slice(label);
    out.extend_from_slice(payload);
    Ok(out)
}

/// Frame an append operation for deterministic transcript input.
pub fn frame_append(label: &'static [u8], payload: &[u8]) -> Result<Vec<u8>, TranscriptError> {
    encode_frame(APPEND_DOMAIN_TAG, label, payload)
}

/// Frame a challenge request for deterministic transcript input.
pub fn frame_challenge_request(
    label: &'static [u8],
    out_len: usize,
) -> Result<Vec<u8>, TranscriptError> {
    if out_len == 0 {
        return Err(TranscriptError::EmptyChallenge);
    }
    encode_frame(CHALLENGE_DOMAIN_TAG, label, &(out_len as u64).to_le_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_append_domain_separates_from_challenge() {
        let append = frame_append(b"msg", b"abc").unwrap();
        let challenge = frame_challenge_request(b"msg", 3).unwrap();
        assert_ne!(append, challenge);
    }

    #[test]
    fn test_frame_append_encodes_lengths() {
        let frame = frame_append(b"msg", b"abc").unwrap();
        assert_eq!(frame[0], APPEND_DOMAIN_TAG);
        assert_eq!(u64::from_le_bytes(frame[1..9].try_into().unwrap()), 3);
        assert_eq!(u64::from_le_bytes(frame[9..17].try_into().unwrap()), 3);
    }

    #[test]
    fn test_frame_rejects_empty_label() {
        assert_eq!(frame_append(b"", b"abc"), Err(TranscriptError::EmptyLabel));
    }
}
