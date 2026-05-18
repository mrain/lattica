//! SIMD backend selection helpers.

/// The backend chosen for SIMD-dispatchable operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Backend {
    Scalar,
    Avx2,
    Neon,
}

#[cfg(feature = "std")]
fn env_override() -> Option<Backend> {
    match std::env::var("GRID_SIMD").ok().as_deref() {
        Some("scalar") => Some(Backend::Scalar),
        Some("auto") | None => None,
        Some("avx2") => Some(Backend::Avx2),
        Some("neon") => Some(Backend::Neon),
        Some(_) => Some(Backend::Scalar),
    }
}

#[cfg(not(feature = "std"))]
fn env_override() -> Option<Backend> {
    None
}

/// Select the active backend for this build/runtime.
pub(crate) fn selected_backend() -> Backend {
    if let Some(backend) = env_override() {
        return match backend {
            Backend::Scalar => Backend::Scalar,
            #[cfg(target_arch = "x86_64")]
            Backend::Avx2 => {
                if crate::simd::avx2::u64_arith::avx2_available() {
                    Backend::Avx2
                } else {
                    Backend::Scalar
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            Backend::Avx2 => Backend::Scalar,
            #[cfg(target_arch = "aarch64")]
            Backend::Neon => {
                if crate::simd::aarch64::neon_available() {
                    Backend::Neon
                } else {
                    Backend::Scalar
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            Backend::Neon => Backend::Scalar,
        };
    }

    #[cfg(target_arch = "x86_64")]
    if crate::simd::avx2::u64_arith::avx2_available() {
        return Backend::Avx2;
    }

    #[cfg(target_arch = "aarch64")]
    if crate::simd::aarch64::neon_available() {
        return Backend::Neon;
    }

    Backend::Scalar
}
