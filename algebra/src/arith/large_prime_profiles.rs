//! Fixed large-prime profile constants for the first shipping targets.

use super::large_modulus::LargePrimeProfile;

/// BN254 scalar field profile.
pub struct Bn254FrProfile;

impl LargePrimeProfile<4> for Bn254FrProfile {
    const MODULUS: [u64; 4] = [
        0x43e1_f593_f000_0001,
        0x2833_e848_79b9_7091,
        0xb850_45b6_8181_585d,
        0x3064_4e72_e131_a029,
    ];
    const MONT_R2: [u64; 4] = [
        0x1bb8_e645_ae21_6da7,
        0x53fe_3ab1_e35c_59e3,
        0x8c49_833d_53bb_8085,
        0x0216_d0b1_7f4e_44a5,
    ];
    const MONT_NEG_INV: u64 = 0xc2e1_f593_efff_ffff;
    const BARRETT_MU: [u64; 4] = [0x5, 0x0, 0x0, 0x0];
    const TWO_ADICITY: usize = 28;
    const TWO_ADIC_ROOT: [u64; 4] = [
        0x9bd6_1b6e_725b_19f0,
        0x402d_111e_4111_2ed4,
        0x00e0_a7eb_8ef6_2abc,
        0x2a3c_09f0_a58a_7e85,
    ];
}

/// BN254 base field profile.
pub struct Bn254FqProfile;

impl LargePrimeProfile<4> for Bn254FqProfile {
    const MODULUS: [u64; 4] = [
        0x3c20_8c16_d87c_fd47,
        0x9781_6a91_6871_ca8d,
        0xb850_45b6_8181_585d,
        0x3064_4e72_e131_a029,
    ];
    const MONT_R2: [u64; 4] = [
        0xf32c_fc5b_538a_fa89,
        0xb5e7_1911_d445_01fb,
        0x47ab_1eff_0a41_7ff6,
        0x06d8_9f71_cab8_351f,
    ];
    const MONT_NEG_INV: u64 = 0x87d2_0782_e486_6389;
    const BARRETT_MU: [u64; 4] = [0x5, 0x0, 0x0, 0x0];
    const TWO_ADICITY: usize = 1;
    const TWO_ADIC_ROOT: [u64; 4] = [
        0x3c20_8c16_d87c_fd46,
        0x9781_6a91_6871_ca8d,
        0xb850_45b6_8181_585d,
        0x3064_4e72_e131_a029,
    ];
}

/// BLS12-381 scalar field profile.
pub struct Bls12_381FrProfile;

impl LargePrimeProfile<4> for Bls12_381FrProfile {
    const MODULUS: [u64; 4] = [
        0xffff_ffff_0000_0001,
        0x53bd_a402_fffe_5bfe,
        0x3339_d808_09a1_d805,
        0x73ed_a753_299d_7d48,
    ];
    const MONT_R2: [u64; 4] = [
        0xc999_e990_f3f2_9c6d,
        0x2b6c_edcb_8792_5c23,
        0x05d3_1496_7254_398f,
        0x0748_d9d9_9f59_ff11,
    ];
    const MONT_NEG_INV: u64 = 0xffff_fffe_ffff_ffff;
    const BARRETT_MU: [u64; 4] = [0x2, 0x0, 0x0, 0x0];
    const TWO_ADICITY: usize = 32;
    const TWO_ADIC_ROOT: [u64; 4] = [
        0x1b78_8f50_0b91_2f1f,
        0xc402_4ff2_70b3_e094,
        0x0fd5_6dc8_d168_d6c0,
        0x0212_d79e_5b41_6b6f,
    ];
}

/// BLS12-381 base field profile.
pub struct Bls12_381FqProfile;

impl LargePrimeProfile<6> for Bls12_381FqProfile {
    const MODULUS: [u64; 6] = [
        0xb9fe_ffff_ffff_aaab,
        0x1eab_fffe_b153_ffff,
        0x6730_d2a0_f6b0_f624,
        0x6477_4b84_f385_12bf,
        0x4b1b_a7b6_434b_acd7,
        0x1a01_11ea_397f_e69a,
    ];
    const MONT_R2: [u64; 6] = [
        0xf4df_1f34_1c34_1746,
        0x0a76_e6a6_09d1_04f1,
        0x8de5_476c_4c95_b6d5,
        0x67eb_88a9_939d_83c0,
        0x9a79_3e85_b519_952d,
        0x1198_8fe5_92ca_e3aa,
    ];
    const MONT_NEG_INV: u64 = 0x89f3_fffc_fffc_fffd;
    const BARRETT_MU: [u64; 6] = [0x9, 0x0, 0x0, 0x0, 0x0, 0x0];
    const TWO_ADICITY: usize = 1;
    const TWO_ADIC_ROOT: [u64; 6] = [
        0xb9fe_ffff_ffff_aaaa,
        0x1eab_fffe_b153_ffff,
        0x6730_d2a0_f6b0_f624,
        0x6477_4b84_f385_12bf,
        0x4b1b_a7b6_434b_acd7,
        0x1a01_11ea_397f_e69a,
    ];
}
