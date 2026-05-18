//! Fixed large-RNS profile constants for the first shipping target.

use super::large_modulus::LargeRnsProfile;

/// First reviewed 3-limb large-RNS shipping profile.
pub struct Rns3V0Profile;

impl LargeRnsProfile<3> for Rns3V0Profile {
    const MODULI: [u64; 3] = [
        0x0000_1000_01d0_0001,
        0x0000_1000_03b0_0001,
        0x0000_1000_0450_0001,
    ];
    const TWO_ADICITY: usize = 20;
    const TWO_ADIC_ROOT: [u64; 3] = [
        0x0000_027a_5925_d381,
        0x0000_0a5b_4405_efa4,
        0x0000_0845_904c_0ef3,
    ];
    const MODULUS: [u64; 3] = [0xb01e_9700_09d0_0001, 0x0009_d001_e970_1e0c, 0x10];
    const PREFIX_PRODUCTS: [[u64; 3]; 3] = [
        [0x1, 0x0, 0x0],
        [0x0000_1000_01d0_0001, 0x0, 0x0],
        [0x0006_cf00_0580_0001, 0x0000_0000_0100_0058, 0x0],
    ];
    const GARNER_INVERSES: [[u64; 3]; 3] = [
        [0, 0, 0],
        [0x0000_0800_01e0_888b, 0, 0],
        [0x0000_0e00_03cc_6669, 0x0000_0800_0241_99a1, 0],
    ];
    type Metadata = ();
}
