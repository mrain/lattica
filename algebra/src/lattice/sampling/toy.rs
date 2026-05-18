//! Correctness-first toy samplers.

use core::array::from_fn;
use core::marker::PhantomData;

use crate::arith::bigint::BigUint;
use crate::arith::large_modulus::LargeCanonicalRing;
use crate::arith::ring::IntegerRing;
use crate::lattice::sampling::CoeffSampler;
use crate::lattice::types::{RingMat, RingVec};
use crate::poly::ring::CyclotomicPolyRing;

fn encode_signed_word<R: IntegerRing<Uint = u64>>(x: i64) -> R {
    if x >= 0 {
        R::from_u64(x as u64)
    } else {
        let abs = (-x) as u64;
        let modulus = R::modulus();
        if modulus == 0 {
            R::from_u64(abs.wrapping_neg())
        } else {
            let reduced = abs % modulus;
            if reduced == 0 {
                R::zero()
            } else {
                R::from_u64(modulus - reduced)
            }
        }
    }
}

fn encode_signed_large<R, const N: usize>(x: i64) -> R
where
    R: IntegerRing<Uint = BigUint<N>> + LargeCanonicalRing<Canonical = BigUint<N>>,
{
    if x >= 0 {
        return R::from_small_u64(x as u64);
    }

    let abs = (-x) as u64;
    let modulus = R::modulus_canonical();
    let reduced = modulus
        .try_to_u64()
        .map_or(abs, |modulus_u64| abs % modulus_u64);
    debug_assert!(
        reduced == 0 || BigUint::<N>::from_u64(reduced) < modulus,
        "large signed encoding expects the sampled magnitude to fit below the backend modulus"
    );
    if reduced == 0 {
        R::zero()
    } else {
        let (canonical, borrow) = modulus.sub_small(reduced);
        assert!(
            !borrow,
            "signed sample magnitude exceeded the large-modulus ring"
        );
        R::from_canonical(&canonical)
    }
}

fn sample_uniform_word<R, T>(rng: &mut T) -> R
where
    R: IntegerRing<Uint = u64>,
    T: grid_std::rand::RngExt,
{
    let modulus = R::modulus();
    if modulus == 0 {
        return R::from_u64(rng.random());
    }

    let reject = (u64::MAX % modulus).wrapping_add(1) % modulus;
    let upper = u64::MAX - reject;
    loop {
        let sample: u64 = rng.random();
        if sample <= upper {
            return R::from_u64(sample % modulus);
        }
    }
}

fn sample_uniform_large<R, T, const N: usize>(rng: &mut T) -> R
where
    R: IntegerRing<Uint = BigUint<N>> + LargeCanonicalRing<Canonical = BigUint<N>>,
    T: grid_std::rand::RngExt,
{
    let modulus = R::modulus_canonical();
    let modulus_bits = modulus.bits() as usize;
    let highest_limb = (modulus_bits - 1) / 64;
    let top_bits = modulus_bits % 64;
    let top_mask = if top_bits == 0 {
        u64::MAX
    } else {
        (1u64 << top_bits) - 1
    };
    loop {
        let sample = BigUint {
            limbs: from_fn(|i| {
                if i > highest_limb {
                    0
                } else {
                    let limb: u64 = rng.random();
                    if i == highest_limb {
                        limb & top_mask
                    } else {
                        limb
                    }
                }
            }),
        };
        if sample < modulus {
            return R::from_canonical(&sample);
        }
    }
}

/// Uniform sampler over `[0, q)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct UniformSampler<R> {
    _marker: PhantomData<R>,
}

impl<R> UniformSampler<R> {
    /// Construct a new sampler.
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<R: IntegerRing<Uint = u64>> CoeffSampler<R> for UniformSampler<R> {
    fn sample_coeff<T: grid_std::rand::RngExt>(&self, rng: &mut T) -> R {
        sample_uniform_word::<R, T>(rng)
    }
}

/// Uniform sampler over `{-1, 0, 1}` modulo `q`.
#[derive(Debug, Clone, Copy, Default)]
pub struct TernarySampler<R> {
    _marker: PhantomData<R>,
}

impl<R> TernarySampler<R> {
    /// Construct a new sampler.
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<R: IntegerRing<Uint = u64>> CoeffSampler<R> for TernarySampler<R> {
    fn sample_coeff<T: grid_std::rand::RngExt>(&self, rng: &mut T) -> R {
        match rng.random_range(0..3usize) {
            0 => R::zero(),
            1 => R::one(),
            _ => encode_signed_word::<R>(-1),
        }
    }
}

/// Centered binomial sampler with parameter `eta`.
#[derive(Debug, Clone, Copy)]
pub struct CBDSampler<R> {
    eta: usize,
    _marker: PhantomData<R>,
}

impl<R> CBDSampler<R> {
    /// Construct a new CBD sampler.
    pub fn new(eta: usize) -> Self {
        assert!(eta > 0, "eta must be positive");
        Self {
            eta,
            _marker: PhantomData,
        }
    }

    /// Return the configured `eta`.
    pub fn eta(&self) -> usize {
        self.eta
    }
}

impl<R: IntegerRing<Uint = u64>> CoeffSampler<R> for CBDSampler<R> {
    fn sample_coeff<T: grid_std::rand::RngExt>(&self, rng: &mut T) -> R {
        let mut left = 0i64;
        let mut right = 0i64;
        for _ in 0..self.eta {
            left += i64::from(rng.random_bool(0.5));
            right += i64::from(rng.random_bool(0.5));
        }
        encode_signed_word::<R>(left - right)
    }
}

/// Approximate discrete Gaussian sampler with a finite tail cutoff.
#[derive(Debug, Clone, Copy)]
pub struct ApproxGaussianSampler<R> {
    sigma: f64,
    tail_cut: usize,
    _marker: PhantomData<R>,
}

impl<R> ApproxGaussianSampler<R> {
    /// Construct a new sampler.
    pub fn new(sigma: f64, tail_cut: usize) -> Self {
        assert!(
            sigma.is_finite() && sigma > 0.0,
            "sigma must be finite and positive"
        );
        assert!(tail_cut > 0, "tail_cut must be positive");
        Self {
            sigma,
            tail_cut,
            _marker: PhantomData,
        }
    }

    /// Return the configured sigma.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Return the tail cutoff.
    pub fn tail_cut(&self) -> usize {
        self.tail_cut
    }
}

impl<R: IntegerRing<Uint = u64>> CoeffSampler<R> for ApproxGaussianSampler<R> {
    fn sample_coeff<T: grid_std::rand::RngExt>(&self, rng: &mut T) -> R {
        loop {
            let signed = rng.random_range(-(self.tail_cut as i64)..=(self.tail_cut as i64));
            let exponent = -((signed * signed) as f64) / (2.0 * self.sigma * self.sigma);
            let accept_prob = grid_std::exp(exponent);
            if rng.random::<f64>() < accept_prob {
                return encode_signed_word::<R>(signed);
            }
        }
    }
}

/// Uniform sampler over `[0, q)` for `BigUint`-backed large-modulus rings.
#[derive(Debug, Clone, Copy, Default)]
pub struct LargeUniformSampler<R> {
    _marker: PhantomData<R>,
}

impl<R> LargeUniformSampler<R> {
    /// Construct a new sampler.
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<R, const N: usize> CoeffSampler<R> for LargeUniformSampler<R>
where
    R: IntegerRing<Uint = BigUint<N>> + LargeCanonicalRing<Canonical = BigUint<N>>,
{
    fn sample_coeff<T: grid_std::rand::RngExt>(&self, rng: &mut T) -> R {
        sample_uniform_large::<R, T, N>(rng)
    }
}

/// Uniform sampler over `{-1, 0, 1}` modulo `q` for `BigUint`-backed large-modulus rings.
#[derive(Debug, Clone, Copy, Default)]
pub struct LargeTernarySampler<R> {
    _marker: PhantomData<R>,
}

impl<R> LargeTernarySampler<R> {
    /// Construct a new sampler.
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<R, const N: usize> CoeffSampler<R> for LargeTernarySampler<R>
where
    R: IntegerRing<Uint = BigUint<N>> + LargeCanonicalRing<Canonical = BigUint<N>>,
{
    fn sample_coeff<T: grid_std::rand::RngExt>(&self, rng: &mut T) -> R {
        match rng.random_range(0..3usize) {
            0 => R::zero(),
            1 => R::one(),
            _ => encode_signed_large::<R, N>(-1),
        }
    }
}

/// Centered binomial sampler with parameter `eta` for `BigUint`-backed large-modulus rings.
#[derive(Debug, Clone, Copy)]
pub struct LargeCBDSampler<R> {
    eta: usize,
    _marker: PhantomData<R>,
}

impl<R> LargeCBDSampler<R> {
    /// Construct a new CBD sampler.
    pub fn new(eta: usize) -> Self {
        assert!(eta > 0, "eta must be positive");
        Self {
            eta,
            _marker: PhantomData,
        }
    }

    /// Return the configured `eta`.
    pub fn eta(&self) -> usize {
        self.eta
    }
}

impl<R, const N: usize> CoeffSampler<R> for LargeCBDSampler<R>
where
    R: IntegerRing<Uint = BigUint<N>> + LargeCanonicalRing<Canonical = BigUint<N>>,
{
    fn sample_coeff<T: grid_std::rand::RngExt>(&self, rng: &mut T) -> R {
        let mut left = 0i64;
        let mut right = 0i64;
        for _ in 0..self.eta {
            left += i64::from(rng.random_bool(0.5));
            right += i64::from(rng.random_bool(0.5));
        }
        encode_signed_large::<R, N>(left - right)
    }
}

/// Approximate discrete Gaussian sampler with a finite tail cutoff for `BigUint`-backed large-modulus rings.
#[derive(Debug, Clone, Copy)]
pub struct LargeApproxGaussianSampler<R> {
    sigma: f64,
    tail_cut: usize,
    _marker: PhantomData<R>,
}

impl<R> LargeApproxGaussianSampler<R> {
    /// Construct a new sampler.
    pub fn new(sigma: f64, tail_cut: usize) -> Self {
        assert!(
            sigma.is_finite() && sigma > 0.0,
            "sigma must be finite and positive"
        );
        assert!(tail_cut > 0, "tail_cut must be positive");
        Self {
            sigma,
            tail_cut,
            _marker: PhantomData,
        }
    }

    /// Return the configured sigma.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Return the tail cutoff.
    pub fn tail_cut(&self) -> usize {
        self.tail_cut
    }
}

impl<R, const N: usize> CoeffSampler<R> for LargeApproxGaussianSampler<R>
where
    R: IntegerRing<Uint = BigUint<N>> + LargeCanonicalRing<Canonical = BigUint<N>>,
{
    fn sample_coeff<T: grid_std::rand::RngExt>(&self, rng: &mut T) -> R {
        loop {
            let signed = rng.random_range(-(self.tail_cut as i64)..=(self.tail_cut as i64));
            let exponent = -((signed * signed) as f64) / (2.0 * self.sigma * self.sigma);
            let accept_prob = grid_std::exp(exponent);
            if rng.random::<f64>() < accept_prob {
                return encode_signed_large::<R, N>(signed);
            }
        }
    }
}

/// Sample a vector of coefficients.
pub fn sample_vec<R, S, T>(sampler: &S, rng: &mut T, n: usize) -> RingVec<R>
where
    R: IntegerRing,
    S: CoeffSampler<R>,
    T: grid_std::rand::RngExt,
{
    RingVec::new((0..n).map(|_| sampler.sample_coeff(rng)).collect())
}

/// Sample a matrix of coefficients.
pub fn sample_mat<R, S, T>(sampler: &S, rng: &mut T, rows: usize, cols: usize) -> RingMat<R>
where
    R: IntegerRing,
    S: CoeffSampler<R>,
    T: grid_std::rand::RngExt,
{
    let entry_count = rows
        .checked_mul(cols)
        .expect("matrix shape overflowed usize");
    RingMat::new(
        rows,
        cols,
        (0..entry_count)
            .map(|_| sampler.sample_coeff(rng))
            .collect(),
    )
}

/// Sample a cyclotomic polynomial by sampling coefficients independently.
pub fn sample_poly<R, S, T, const N: usize>(sampler: &S, rng: &mut T) -> CyclotomicPolyRing<R, N>
where
    R: crate::poly::ring::NegacyclicMulRing<N>,
    S: CoeffSampler<R>,
    T: grid_std::rand::RngExt,
{
    CyclotomicPolyRing::from_array(from_fn(|_| sampler.sample_coeff(rng)))
}

#[cfg(test)]
fn centered_u64<R: IntegerRing<Uint = u64>>(value: &R) -> i64 {
    let x = value.to_u64();
    let modulus = R::modulus();
    if modulus == 0 || x <= modulus / 2 {
        x as i64
    } else {
        x as i64 - modulus as i64
    }
}

#[cfg(test)]
fn centered_big<R, const N: usize>(value: &R) -> i64
where
    R: IntegerRing<Uint = BigUint<N>> + LargeCanonicalRing<Canonical = BigUint<N>>,
{
    let canonical = value.to_canonical();
    if let Some(small) = canonical.try_to_u64() {
        small as i64
    } else {
        let modulus = R::modulus_canonical();
        let (distance, borrow) = modulus.sub_with_borrow(&canonical);
        assert!(!borrow, "canonical sample must be below the modulus");
        -(distance
            .try_to_u64()
            .expect("sample tail should fit in u64") as i64)
    }
}

#[cfg(test)]
fn chi_square_within_critical(observed: &[usize], expected: f64, critical: f64) -> bool {
    let statistic = observed
        .iter()
        .map(|&count| {
            let diff = count as f64 - expected;
            (diff * diff) / expected
        })
        .sum::<f64>();
    statistic < critical
}

#[cfg(test)]
fn gaussian_weight(x: i64, sigma: f64) -> f64 {
    grid_std::exp(-((x * x) as f64) / (2.0 * sigma * sigma))
}

#[cfg(test)]
fn truncated_gaussian_mass(sigma: f64, tail_cut: usize) -> f64 {
    let tail = tail_cut as i64;
    (-tail..=tail).map(|x| gaussian_weight(x, sigma)).sum()
}

#[cfg(test)]
fn empirical_frequency(values: &[i64], x: i64) -> f64 {
    values.iter().filter(|&&value| value == x).count() as f64 / values.len() as f64
}

#[cfg(test)]
fn expected_frequency(x: i64, sigma: f64, tail_cut: usize) -> f64 {
    gaussian_weight(x, sigma) / truncated_gaussian_mass(sigma, tail_cut)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith::large_prime::Bn254Fr;
    use crate::arith::large_rns::Rns3V0;
    use crate::arith::prime::PrimeField;
    use crate::arith::ring::IntegerRing;
    use crate::poly::ring::PolyRing;

    type F17 = PrimeField<17>;
    type F12289 = PrimeField<12289>;

    #[test]
    fn test_uniform_sampler_range_and_balance() {
        let sampler = UniformSampler::<F17>::new();
        let mut rng = grid_std::test_rng();
        let mut counts = [0usize; 17];
        for _ in 0..17_000 {
            let x = sampler.sample_coeff(&mut rng);
            let value = x.to_u64();
            assert!(value < 17);
            counts[value as usize] += 1;
        }
        assert!(chi_square_within_critical(&counts, 1_000.0, 40.79));
    }

    #[test]
    fn test_ternary_sampler_support() {
        let sampler = TernarySampler::<F17>::new();
        let mut rng = grid_std::test_rng();
        for _ in 0..10_000 {
            let x = sampler.sample_coeff(&mut rng).to_u64();
            assert!(matches!(x, 0 | 1 | 16));
        }
    }

    #[test]
    fn test_cbd_sampler_bounds_and_mean() {
        let sampler = CBDSampler::<F17>::new(4);
        let mut rng = grid_std::test_rng();
        let mut sum = 0i64;
        for _ in 0..10_000 {
            let x = sampler.sample_coeff(&mut rng);
            let centered = centered_u64(&x).abs();
            assert!(centered <= 4);
            sum += centered_u64(&x);
        }
        assert!(sum.abs() < 1_000);
    }

    #[test]
    fn test_gaussian_sampler_tail_and_shape() {
        let sampler = ApproxGaussianSampler::<F12289>::new(3.0, 8);
        let mut rng = grid_std::test_rng();
        let values = (0..40_000)
            .map(|_| centered_u64(&sampler.sample_coeff(&mut rng)))
            .collect::<Vec<_>>();
        let mean = values.iter().sum::<i64>() as f64 / values.len() as f64;
        assert!(values.iter().all(|value| value.abs() <= 8));
        assert!(mean.abs() < 0.2);

        for x in 0..=4 {
            let empirical = empirical_frequency(&values, x);
            let expected = expected_frequency(x, 3.0, 8);
            assert!((empirical - expected).abs() < 0.02);
        }
    }

    #[test]
    fn test_gaussian_sampler_rejects_non_finite_sigma() {
        let result = std::panic::catch_unwind(|| {
            let _ = ApproxGaussianSampler::<F17>::new(f64::INFINITY, 4);
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_sampling_helpers() {
        let sampler = UniformSampler::<F17>::new();
        let mut rng = grid_std::test_rng();
        let vec = sample_vec(&sampler, &mut rng, 4);
        assert_eq!(vec.len(), 4);
        let mat = sample_mat(&sampler, &mut rng, 2, 3);
        assert_eq!(mat.rows(), 2);
        assert_eq!(mat.cols(), 3);
        let poly = sample_poly::<F17, _, _, 8>(&sampler, &mut rng);
        assert_eq!(
            <crate::poly::ring::CyclotomicPolyRing<F17, 8> as PolyRing>::degree(),
            8
        );
        assert_eq!(poly.coeffs().len(), 8);
    }

    #[test]
    fn test_large_uniform_sampler_outputs_canonical_values() {
        let sampler = LargeUniformSampler::<Bn254Fr>::new();
        let mut rng = grid_std::test_rng();
        let modulus = Bn254Fr::modulus_canonical();
        for _ in 0..512 {
            let sample = sampler.sample_coeff(&mut rng);
            assert!(sample.to_canonical() < modulus);
        }
    }

    #[test]
    fn test_large_signed_samplers_preserve_small_centered_values() {
        let mut rng = grid_std::test_rng();

        let ternary = LargeTernarySampler::<Bn254Fr>::new();
        for _ in 0..4096 {
            assert!(matches!(
                centered_big(&ternary.sample_coeff(&mut rng)),
                -1..=1
            ));
        }

        let cbd = LargeCBDSampler::<Rns3V0>::new(4);
        for _ in 0..4096 {
            assert!(centered_big(&cbd.sample_coeff(&mut rng)).abs() <= 4);
        }

        let gaussian = LargeApproxGaussianSampler::<Bn254Fr>::new(3.0, 8);
        for _ in 0..4096 {
            assert!(centered_big(&gaussian.sample_coeff(&mut rng)).abs() <= 8);
        }
    }

    #[test]
    fn test_large_sampling_helpers() {
        let sampler = LargeUniformSampler::<Bn254Fr>::new();
        let mut rng = grid_std::test_rng();
        let vec = sample_vec(&sampler, &mut rng, 4);
        assert_eq!(vec.len(), 4);

        let mat = sample_mat(&sampler, &mut rng, 2, 3);
        assert_eq!(mat.rows(), 2);
        assert_eq!(mat.cols(), 3);
    }
}
