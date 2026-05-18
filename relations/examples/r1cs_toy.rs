use grid_algebra::arith::prime::PrimeField;
use grid_algebra::arith::ring::{IntegerRing, Ring};
use grid_algebra::lattice::params::NormBound;
use grid_algebra::lattice::types::{RingMat, RingVec};
use grid_relations::r1cs::{R1csInstance, R1csWitness};
use grid_relations::witness::WitnessNormBounds;
use grid_relations::{ConstraintSystem, RelationsError};

type F17 = PrimeField<17>;

fn toy_r1cs_instance() -> R1csInstance<F17> {
    // Witness layout: z = [1, y, x, w], with the constraint x * w = y.
    let public_inputs = RingVec::new(vec![F17::from_u64(15)]);
    let witness_bounds = WitnessNormBounds {
        private_witness: NormBound {
            max_l2_sq: 34,
            max_linf: 5,
        },
    };
    let a = RingMat::new(
        1,
        4,
        vec![F17::zero(), F17::zero(), F17::one(), F17::zero()],
    );
    let b = RingMat::new(
        1,
        4,
        vec![F17::zero(), F17::zero(), F17::zero(), F17::one()],
    );
    let c = RingMat::new(
        1,
        4,
        vec![F17::zero(), F17::one(), F17::zero(), F17::zero()],
    );

    R1csInstance::new(public_inputs, witness_bounds, a, b, c).unwrap()
}

fn report(
    label: &str,
    instance: &R1csInstance<F17>,
    witness: &R1csWitness<F17>,
) -> Result<(), RelationsError> {
    match instance.is_satisfied(witness) {
        Ok(true) => {
            println!("{label}: satisfied");
            Ok(())
        }
        Ok(false) => {
            println!("{label}: unsatisfied");
            Ok(())
        }
        Err(RelationsError::WitnessNormExceeded) => {
            println!("{label}: rejected by witness norm bound");
            Ok(())
        }
        Err(err) => Err(err),
    }
}

fn main() -> Result<(), RelationsError> {
    let instance = toy_r1cs_instance();

    let satisfying = R1csWitness::new(RingVec::new(vec![F17::from_u64(3), F17::from_u64(5)]));
    let unsatisfied = R1csWitness::new(RingVec::new(vec![F17::from_u64(1), F17::from_u64(5)]));
    let over_norm = R1csWitness::new(RingVec::new(vec![F17::from_u64(7), F17::from_u64(5)]));

    println!("R1CS toy example");
    println!("  public input y = 15");
    println!("  private witness is [x, w]");
    report("  satisfying witness [3, 5]", &instance, &satisfying)?;
    report("  unsatisfied witness [1, 5]", &instance, &unsatisfied)?;
    report("  over-norm witness [7, 5]", &instance, &over_norm)?;

    Ok(())
}
