//! Internal SIMD backend scaffolding.

#[cfg(target_arch = "aarch64")]
pub(crate) mod aarch64;
#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2;
pub(crate) mod dispatch;
pub(crate) mod montgomery;
