//! Modular integer arithmetic over `Z_q`.
//!
//! Supports prime, power-of-two, and composite moduli.

pub mod bigint;
pub mod composite;
pub mod gf2;
pub mod large_modulus;
pub mod large_prime;
pub mod large_prime_profiles;
pub mod large_rns;
pub mod large_rns_profiles;
pub mod large_zm;
pub mod large_zm_profiles;
pub mod limb;
pub mod ntt;
pub mod prime;
pub mod ring;
pub mod rns;
pub mod z2k;
pub mod zm;

pub use gf2::GF2;
pub use large_modulus::{LargePrimeProfile, LargeRnsProfile};
pub use large_prime::{Bls12_381Fq, Bls12_381Fr, Bn254Fq, Bn254Fr, LargePrimeField};
pub use large_prime_profiles::{
    Bls12_381FqProfile, Bls12_381FrProfile, Bn254FqProfile, Bn254FrProfile,
};
pub use large_rns::{LargeRns, Rns3V0};
pub use large_rns_profiles::Rns3V0Profile;
pub use large_zm::LargeZm;
pub use large_zm_profiles::Fermat64Profile;
pub use limb::UintLimb;
pub use ntt::NTTRing;
pub use ring::{Field, IntegerRing, Ring};
pub use zm::Zm;

/// Fermat ring `Z/(2^64 + 1)Z` for the arithmetic R1CS reduction path.
pub type FermatRing64 = LargeZm<Fermat64Profile, 2>;
