//! Demo relation builders used by examples and integration checks.

use alloc::vec;
use alloc::vec::Vec;
use core::array::from_fn;

use grid_algebra::arith::ring::Ring;
use grid_algebra::lattice::params::{NormBound, NormStats};
use grid_algebra::lattice::types::{RingMat, RingVec};
use grid_algebra::poly::ring::{CyclotomicPolyRing, NegacyclicMulRing};

use crate::r1cs::{R1csInstance, R1csWitness};
use crate::witness::WitnessNormBounds;

#[derive(Clone)]
struct ConstraintRow<R: Ring> {
    a: Vec<(usize, R)>,
    b: Vec<(usize, R)>,
    c: Vec<(usize, R)>,
}

struct CircuitBuilder<R: Ring> {
    private_witness: Vec<R>,
    constraints: Vec<ConstraintRow<R>>,
}

impl<R: Ring> CircuitBuilder<R> {
    fn new() -> Self {
        Self {
            private_witness: Vec::new(),
            constraints: Vec::new(),
        }
    }

    fn alloc_private(&mut self, value: R) -> usize {
        let idx = self.private_witness.len() + 2;
        self.private_witness.push(value);
        idx
    }

    fn constrain_linear(&mut self, value: R, terms: Vec<(usize, R)>) -> usize {
        let idx = self.alloc_private(value);
        self.constraints.push(ConstraintRow {
            a: terms,
            b: vec![(0, R::one())],
            c: vec![(idx, R::one())],
        });
        idx
    }

    fn constrain_mul(&mut self, lhs: Vec<(usize, R)>, rhs: Vec<(usize, R)>, value: R) -> usize {
        let idx = self.alloc_private(value);
        self.constraints.push(ConstraintRow {
            a: lhs,
            b: rhs,
            c: vec![(idx, R::one())],
        });
        idx
    }

    fn constrain_public_output(&mut self, idx: usize) {
        self.constraints.push(ConstraintRow {
            a: vec![(idx, R::one())],
            b: vec![(0, R::one())],
            c: vec![(1, R::one())],
        });
    }

    fn into_artifacts(self, public_output: R) -> FibonacciR1csDemo<R>
    where
        R: grid_algebra::lattice::params::NormedRing,
    {
        let num_constraints = self.constraints.len();
        let num_variables = self.private_witness.len() + 2;

        let mut a_entries = Vec::with_capacity(num_constraints * num_variables);
        let mut b_entries = Vec::with_capacity(num_constraints * num_variables);
        let mut c_entries = Vec::with_capacity(num_constraints * num_variables);

        for row in &self.constraints {
            a_entries.extend(dense_row(&row.a, num_variables));
            b_entries.extend(dense_row(&row.b, num_variables));
            c_entries.extend(dense_row(&row.c, num_variables));
        }

        let private_witness = RingVec::new(self.private_witness);
        let witness_bounds = WitnessNormBounds {
            private_witness: NormBound::from_stats(&NormStats::compute(&private_witness)),
        };
        let witness = R1csWitness::new(private_witness);
        let public_inputs = RingVec::new(vec![public_output.clone()]);
        let a = RingMat::new(num_constraints, num_variables, a_entries);
        let b = RingMat::new(num_constraints, num_variables, b_entries);
        let c = RingMat::new(num_constraints, num_variables, c_entries);
        let instance = R1csInstance::new(public_inputs, witness_bounds, a, b, c).unwrap();

        FibonacciR1csDemo {
            target_index: 0,
            output: public_output,
            instance,
            witness,
        }
    }
}

fn dense_row<R: Ring>(terms: &[(usize, R)], cols: usize) -> Vec<R> {
    let mut row = vec![R::zero(); cols];
    for (idx, coeff) in terms {
        row[*idx] += coeff;
    }
    row
}

fn msb_bits(n: usize) -> Vec<bool> {
    assert!(n > 0, "Fibonacci target must be non-zero");
    let top_bit = usize::BITS as usize - 1 - n.leading_zeros() as usize;
    (0..=top_bit)
        .rev()
        .map(|shift| ((n >> shift) & 1) == 1)
        .collect()
}

fn const_poly<C, const N: usize>(value: u64) -> CyclotomicPolyRing<C, N>
where
    C: NegacyclicMulRing<N>,
{
    CyclotomicPolyRing::from_array(from_fn(|idx| {
        if idx == 0 {
            C::from_u64(value)
        } else {
            C::zero()
        }
    }))
}

/// Demo artifacts for a Fibonacci R1CS instance and satisfying witness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FibonacciR1csDemo<R: Ring> {
    /// Target Fibonacci index encoded by the circuit.
    pub target_index: usize,
    /// Public output `fib(target_index)` in the selected ring.
    pub output: R,
    /// Dense R1CS statement.
    pub instance: R1csInstance<R>,
    /// Satisfying witness for the statement.
    pub witness: R1csWitness<R>,
}

/// Build a fast-doubling Fibonacci R1CS demo over a constant-polynomial ring backend.
///
/// The generated circuit proves knowledge of the intermediate fast-doubling values used to compute
/// `fib(target_index)` and exposes the final value as the single public input.
pub fn fibonacci_r1cs_constant_poly<C, const N: usize>(
    target_index: usize,
) -> FibonacciR1csDemo<CyclotomicPolyRing<C, N>>
where
    C: NegacyclicMulRing<N, Canonical = u64>,
{
    type Poly<C0, const N0: usize> = CyclotomicPolyRing<C0, N0>;

    let zero = const_poly::<C, N>(0);
    let one = const_poly::<C, N>(1);
    let two = const_poly::<C, N>(2);

    let mut builder = CircuitBuilder::<Poly<C, N>>::new();
    let mut a_idx = builder.constrain_linear(zero.clone(), Vec::new());
    let mut a_val = zero;
    let mut b_idx = 0usize;
    let mut b_val = one.clone();

    for bit in msb_bits(target_index) {
        let t_val = (b_val.clone() * &two) - &a_val;
        let t_idx = builder.constrain_linear(
            t_val.clone(),
            vec![(b_idx, two.clone()), (a_idx, -one.clone())],
        );

        let c_val = a_val.clone() * &t_val;
        let c_idx = builder.constrain_mul(
            vec![(a_idx, one.clone())],
            vec![(t_idx, one.clone())],
            c_val.clone(),
        );

        let a_sq_val = a_val.square();
        let a_sq_idx = builder.constrain_mul(
            vec![(a_idx, one.clone())],
            vec![(a_idx, one.clone())],
            a_sq_val.clone(),
        );

        let b_sq_val = b_val.square();
        let b_sq_idx = builder.constrain_mul(
            vec![(b_idx, one.clone())],
            vec![(b_idx, one.clone())],
            b_sq_val.clone(),
        );

        let d_val = a_sq_val + &b_sq_val;
        let d_idx = builder.constrain_linear(
            d_val.clone(),
            vec![(a_sq_idx, one.clone()), (b_sq_idx, one.clone())],
        );

        if bit {
            let e_val = c_val + &d_val;
            let e_idx = builder.constrain_linear(
                e_val.clone(),
                vec![(c_idx, one.clone()), (d_idx, one.clone())],
            );
            a_idx = d_idx;
            a_val = d_val;
            b_idx = e_idx;
            b_val = e_val;
        } else {
            a_idx = c_idx;
            a_val = c_val;
            b_idx = d_idx;
            b_val = d_val;
        }
    }

    builder.constrain_public_output(a_idx);
    let output = a_val;
    let mut demo = builder.into_artifacts(output.clone());
    demo.target_index = target_index;
    demo.output = output;
    demo
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::ConstraintSystem;
    use grid_algebra::arith::prime::PrimeField;
    use grid_algebra::arith::ring::IntegerRing;
    use grid_algebra::poly::ring::PolyRing;

    type Poly8 = CyclotomicPolyRing<PrimeField<97>, 8>;

    #[test]
    fn test_fibonacci_demo_satisfies_r1cs() {
        let demo = fibonacci_r1cs_constant_poly::<PrimeField<97>, 8>(10);

        assert!(demo.instance.is_satisfied(&demo.witness).unwrap());
        assert_eq!(demo.output.coeff(0), PrimeField::<97>::from_u64(55));
        assert_eq!(demo.instance.public_inputs.len(), 1);
        assert_eq!(
            demo.instance.num_variables,
            1 + demo.instance.public_inputs.len() + demo.witness.private_witness.len()
        );
        let _typed: Poly8 = demo.output;
    }
}
