use grid_serialize::{CanonicalDeserialize, CanonicalSerialize};

/// Johnson-Lindenstrauss projection configuration (§4, Lemma 4.1).
///
/// Entry distribution is fixed: `Pr[0]=1/2, Pr[±1]=1/4` (Lemma 4.1, GHL21).
/// 256 rows, 128-bit tail bounds, √30 lower factor, √128 verification threshold,
/// √337 upper tail.
#[derive(Debug, Clone, Copy, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct JLProfile {
    /// Projection output dimension (256).
    pub rows: usize,

    /// Tail-bound security level (128 bits).
    /// Documents the `2^(-security_bits)` failure probability of Lemma 4.1/4.2.
    pub security_bits: u8,

    /// Lemma 4.1 lower bound factor (√30).
    /// Used by `norm_slack()` to compute `verify_factor / lower_factor`.
    pub lower_factor: f64,

    /// Protocol verification threshold factor (√128).
    /// Verifier checks ||p|| ≤ √128·β.
    pub verify_factor: f64,

    /// Lemma 4.1 upper tail factor (√337).
    /// Guarantees ||p|| > √337·||w|| with prob < 2⁻¹²⁸ (not used in verification).
    pub tail_upper: f64,
}

impl JLProfile {
    /// Default profile matching Lemma 4.1 (GHL21): 256 rows, 128-bit security.
    pub fn paper_default() -> Self {
        Self {
            rows: 256,
            security_bits: 128,
            lower_factor: grid_std::sqrt(30.0),
            verify_factor: grid_std::sqrt(128.0),
            tail_upper: grid_std::sqrt(337.0),
        }
    }

    /// Validate structural invariants of this profile.
    ///
    /// The 2^-128 JL tail bound from Lemma 4.1 is only proven for exactly 256 rows
    /// and 128-bit security. Custom row counts or security levels require externally
    /// justified tail parameters.
    ///
    /// Returns `true` if: rows == 256, security_bits == 128, all factors are positive
    /// finite, and lower < verify < tail_upper.
    pub fn validate(&self) -> bool {
        self.rows == 256
            && self.security_bits == 128
            && self.lower_factor > 0.0
            && self.lower_factor.is_finite()
            && self.verify_factor > 0.0
            && self.verify_factor.is_finite()
            && self.tail_upper > 0.0
            && self.tail_upper.is_finite()
            && self.lower_factor < self.verify_factor
            && self.verify_factor < self.tail_upper
    }

    /// Norm slack factor √(128/30) ≈ 2.07.
    /// Propagates into Module-SIS norm requirements at recursed levels (Remark 5.2).
    pub fn norm_slack(&self) -> f64 {
        self.verify_factor / self.lower_factor
    }
}

impl Default for JLProfile {
    fn default() -> Self {
        Self::paper_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paper_default_validates() {
        assert!(JLProfile::paper_default().validate());
    }

    #[test]
    fn test_invalid_profiles_rejected() {
        let mut base = JLProfile::paper_default();
        let cases = [
            ("zero rows", {
                base.rows = 0;
                base
            }),
            ("zero verify_factor", {
                base = JLProfile::paper_default();
                base.verify_factor = 0.0;
                base
            }),
            ("nan lower_factor", {
                base = JLProfile::paper_default();
                base.lower_factor = f64::NAN;
                base
            }),
            ("inverted factors", {
                base = JLProfile::paper_default();
                core::mem::swap(&mut base.lower_factor, &mut base.verify_factor);
                base
            }),
        ];

        for (name, profile) in &cases {
            assert!(!profile.validate(), "{} should be rejected", name);
        }
    }
}
