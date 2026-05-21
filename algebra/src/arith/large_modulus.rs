//! Companion profile traits for large-modulus backends.

const fn mul_limbs_by_word<const LIMBS: usize>(
    value: [u64; LIMBS],
    word: u64,
) -> ([u64; LIMBS], u64) {
    let mut result = [0u64; LIMBS];
    let mut carry = 0u128;
    let mut i = 0;
    while i < LIMBS {
        let product = (value[i] as u128) * (word as u128) + carry;
        result[i] = product as u64;
        carry = product >> 64;
        i += 1;
    }
    (result, carry as u64)
}

const fn limbs_equal<const LIMBS: usize>(lhs: [u64; LIMBS], rhs: [u64; LIMBS]) -> bool {
    let mut i = 0;
    while i < LIMBS {
        if lhs[i] != rhs[i] {
            return false;
        }
        i += 1;
    }
    true
}

const fn rns_profile_shape_is_valid<const LIMBS: usize>(
    moduli: [u64; LIMBS],
    modulus: [u64; LIMBS],
    prefix_products: [[u64; LIMBS]; LIMBS],
) -> bool {
    let mut running = [0u64; LIMBS];
    if LIMBS > 0 {
        running[0] = 1;
    }

    let mut i = 0;
    while i < LIMBS {
        if !limbs_equal(prefix_products[i], running) {
            return false;
        }
        let (next, carry) = mul_limbs_by_word(running, moduli[i]);
        if carry != 0 {
            return false;
        }
        running = next;
        i += 1;
    }

    limbs_equal(running, modulus)
}

/// Compile-time metadata for a fixed-limb large-prime profile.
pub trait LargePrimeProfile<const LIMBS: usize> {
    /// Modulus in little-endian limb order.
    const MODULUS: [u64; LIMBS];
    /// `R^2 mod q` in little-endian limb order for Montgomery conversion.
    const MONT_R2: [u64; LIMBS];
    /// `-q^{-1} mod 2^64` for Montgomery reduction.
    const MONT_NEG_INV: u64;
    /// `floor(2^(64 * LIMBS) / q)` for canonical Barrett reduction into Montgomery form.
    const BARRETT_MU: [u64; LIMBS];
    /// Largest `s` such that `2^s` divides `q - 1`.
    const TWO_ADICITY: usize;
    /// Primitive `2^TWO_ADICITY`-th root of unity in canonical little-endian limb order.
    const TWO_ADIC_ROOT: [u64; LIMBS];
}

/// Compile-time metadata for a fixed-profile large-RNS backend.
pub trait LargeRnsProfile<const LIMBS: usize> {
    /// Pairwise-coprime component moduli in the fixed residue order.
    const MODULI: [u64; LIMBS];
    /// Largest `s` such that `2^s` divides every `m_i - 1`.
    const TWO_ADICITY: usize;
    /// Primitive `2^TWO_ADICITY`-th root of unity in residue order.
    const TWO_ADIC_ROOT: [u64; LIMBS];
    /// Composite modulus product in little-endian limb order.
    const MODULUS: [u64; LIMBS];
    /// Prefix products where `PREFIX_PRODUCTS[i] = product_{j < i} MODULI[j]`.
    const PREFIX_PRODUCTS: [[u64; LIMBS]; LIMBS];
    /// Garner inverses where `GARNER_INVERSES[i][j] = m_j^(-1) mod m_i` for `j < i`.
    const GARNER_INVERSES: [[u64; LIMBS]; LIMBS];
    /// Compile-time validation that the fixed profile's prefix products and composite modulus
    /// agree with the component moduli and fit the declared canonical limb width.
    const PROFILE_VALID: () = assert!(rns_profile_shape_is_valid(
        Self::MODULI,
        Self::MODULUS,
        Self::PREFIX_PRODUCTS,
    ));
    /// Opaque metadata handle type for cached CRT / Garner helpers.
    type Metadata;
}
