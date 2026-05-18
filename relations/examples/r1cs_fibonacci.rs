use grid_algebra::arith::bigint::BigUint;
use grid_algebra::arith::large_prime::Bn254Fr;
use grid_algebra::arith::ring::{IntegerRing, Ring};
use grid_algebra::lattice::params::{LargeNormBound, LargeNormStats};
use grid_algebra::lattice::types::{RingMat, RingVec};
use grid_relations::r1cs::{R1csInstance, R1csWitness};
use grid_relations::witness::{LargeWitnessNormBounds, LargeWitnessNorms};
use grid_relations::{ConstraintSystem, RelationsError};
use grid_std::process::ProcessMemorySnapshot;
use std::time::Instant;

type F = Bn254Fr;
type LargeNorm = BigUint<8>;
type FibInstance = R1csInstance<F, LargeWitnessNormBounds<LargeNorm>>;
type FibWitness = R1csWitness<F, LargeWitnessNorms<LargeNorm>>;

const TARGET_INDEX: usize = 10_000;

#[derive(Clone)]
struct ConstraintRow {
    a: Vec<(usize, F)>,
    b: Vec<(usize, F)>,
    c: Vec<(usize, F)>,
}

struct CircuitBuilder {
    private_witness: Vec<F>,
    constraints: Vec<ConstraintRow>,
}

impl CircuitBuilder {
    fn new() -> Self {
        Self {
            private_witness: Vec::new(),
            constraints: Vec::new(),
        }
    }

    fn alloc_private(&mut self, value: F) -> usize {
        let idx = self.private_witness.len() + 2;
        self.private_witness.push(value);
        idx
    }

    fn constrain_linear(&mut self, value: F, terms: Vec<(usize, F)>) -> usize {
        let idx = self.alloc_private(value);
        self.constraints.push(ConstraintRow {
            a: terms,
            b: vec![(0, F::one())],
            c: vec![(idx, F::one())],
        });
        idx
    }

    fn constrain_mul(&mut self, lhs: Vec<(usize, F)>, rhs: Vec<(usize, F)>, value: F) -> usize {
        let idx = self.alloc_private(value);
        self.constraints.push(ConstraintRow {
            a: lhs,
            b: rhs,
            c: vec![(idx, F::one())],
        });
        idx
    }

    fn constrain_public_output(&mut self, idx: usize) {
        self.constraints.push(ConstraintRow {
            a: vec![(idx, F::one())],
            b: vec![(0, F::one())],
            c: vec![(1, F::one())],
        });
    }

    fn into_instance_and_witness(self, public_output: F) -> (FibInstance, FibWitness) {
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
        let witness_bounds = LargeWitnessNormBounds {
            private_witness: LargeNormBound::from_stats(&LargeNormStats::compute(&private_witness)),
        };
        let witness = FibWitness::new(private_witness);
        let public_inputs = RingVec::new(vec![public_output]);
        let a = RingMat::new(num_constraints, num_variables, a_entries);
        let b = RingMat::new(num_constraints, num_variables, b_entries);
        let c = RingMat::new(num_constraints, num_variables, c_entries);
        let instance = FibInstance::new(public_inputs, witness_bounds, a, b, c).unwrap();

        (instance, witness)
    }
}

fn dense_row(terms: &[(usize, F)], cols: usize) -> Vec<F> {
    let mut row = vec![F::zero(); cols];
    for &(idx, coeff) in terms {
        row[idx] += coeff;
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

fn fibonacci_10000_r1cs() -> (FibInstance, FibWitness, F) {
    let mut builder = CircuitBuilder::new();
    let mut a_idx = builder.constrain_linear(F::zero(), Vec::new());
    let mut a_val = F::zero();
    let mut b_idx = 0usize;
    let mut b_val = F::one();

    for bit in msb_bits(TARGET_INDEX) {
        let t_val = (b_val + b_val) - a_val;
        let t_idx =
            builder.constrain_linear(t_val, vec![(b_idx, F::from_u64(2)), (a_idx, -F::one())]);

        let c_val = a_val * t_val;
        let c_idx = builder.constrain_mul(vec![(a_idx, F::one())], vec![(t_idx, F::one())], c_val);

        let a_sq_val = a_val * a_val;
        let a_sq_idx =
            builder.constrain_mul(vec![(a_idx, F::one())], vec![(a_idx, F::one())], a_sq_val);

        let b_sq_val = b_val * b_val;
        let b_sq_idx =
            builder.constrain_mul(vec![(b_idx, F::one())], vec![(b_idx, F::one())], b_sq_val);

        let d_val = a_sq_val + b_sq_val;
        let d_idx =
            builder.constrain_linear(d_val, vec![(a_sq_idx, F::one()), (b_sq_idx, F::one())]);

        if bit {
            let e_val = c_val + d_val;
            let e_idx = builder.constrain_linear(e_val, vec![(c_idx, F::one()), (d_idx, F::one())]);
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
    let (instance, witness) = builder.into_instance_and_witness(output);
    (instance, witness, output)
}

fn main() -> Result<(), RelationsError> {
    let mem_start = ProcessMemorySnapshot::capture();
    let build_start = Instant::now();
    let (instance, witness, output) = fibonacci_10000_r1cs();
    let build_time = build_start.elapsed();
    let mem_build = ProcessMemorySnapshot::capture();
    let check_start = Instant::now();
    let satisfied = instance.is_satisfied(&witness)?;
    let check_time = check_start.elapsed();
    let mem_check = ProcessMemorySnapshot::capture();

    println!("R1CS Fibonacci example");
    println!("  target index: {TARGET_INDEX}");
    println!("  field: BN254 scalar field");
    println!("  public output fib({TARGET_INDEX}) mod r = {output}");
    println!("  constraints: {}", instance.num_constraints);
    println!("  variables: {}", instance.num_variables);
    println!("  build_time: {:?}", build_time);
    println!("  build_memory: {}", mem_build.describe_since(&mem_start));
    println!("  satisfiability_time: {:?}", check_time);
    println!(
        "  satisfiability_memory: {}",
        mem_check.describe_since(&mem_build)
    );
    println!("  total_memory: {}", mem_check.describe_since(&mem_start));

    match satisfied {
        true => {
            println!("  witness: satisfied");
            Ok(())
        }
        false => {
            println!("  witness: unsatisfied");
            Ok(())
        }
    }
}
