use grid_algebra::arith::prime::PrimeField;
use grid_algebra::arith::ring::{IntegerRing, Ring};
use grid_algebra::lattice::params::NormBound;
use grid_algebra::lattice::types::RingVec;
use grid_algebra::poly::ring::CyclotomicPolyRing;
use grid_commit::CommitmentScheme;
use grid_commit::ajtai::{AjtaiCommitmentScheme, AjtaiParams};
use grid_commit::bdlop::{BdlopCommitmentScheme, BdlopParams};
use grid_commit::gadget::{GadgetCommitmentScheme, GadgetParams};
use grid_commit::linear::CommitmentDimensions;
use grid_serialize::CanonicalSerialize;

type F17 = PrimeField<17>;
type Rq23Np8 = CyclotomicPolyRing<PrimeField<8380417>, 256>;

fn small_dims() -> CommitmentDimensions {
    CommitmentDimensions {
        message_len: 2,
        opening_len: 2,
        commitment_len: 2,
    }
}

fn rq23_np8_dims() -> CommitmentDimensions {
    CommitmentDimensions {
        message_len: 4,
        opening_len: 4,
        commitment_len: 4,
    }
}

fn ajtai_small() -> AjtaiParams {
    AjtaiParams {
        dims: small_dims(),
        opening_eta: 1,
        opening_bound: NormBound {
            max_l2_sq: 2,
            max_linf: 1,
        },
        security_bits: 128,
    }
}

fn bdlop_small() -> BdlopParams {
    BdlopParams {
        dims: small_dims(),
        opening_eta: 1,
        opening_bound: NormBound {
            max_l2_sq: 2,
            max_linf: 1,
        },
        security_bits: 128,
    }
}

fn gadget_small() -> GadgetParams {
    GadgetParams {
        dims: small_dims(),
        opening_eta: 1,
        opening_bound: NormBound {
            max_l2_sq: 2,
            max_linf: 1,
        },
        base: 4,
        digits: 3,
        security_bits: 128,
    }
}

fn ajtai_rq23_np8() -> AjtaiParams {
    AjtaiParams {
        dims: rq23_np8_dims(),
        opening_eta: 2,
        opening_bound: NormBound {
            max_l2_sq: 4096,
            max_linf: 2,
        },
        security_bits: 128,
    }
}

fn bdlop_rq23_np8() -> BdlopParams {
    BdlopParams {
        dims: rq23_np8_dims(),
        opening_eta: 2,
        opening_bound: NormBound {
            max_l2_sq: 4096,
            max_linf: 2,
        },
        security_bits: 128,
    }
}

fn gadget_rq23_np8() -> GadgetParams {
    GadgetParams {
        dims: rq23_np8_dims(),
        opening_eta: 2,
        opening_bound: NormBound {
            max_l2_sq: 4096,
            max_linf: 2,
        },
        base: 256,
        digits: 3,
        security_bits: 128,
    }
}

fn print_ajtai_f17() {
    let mut rng = grid_std::test_rng();
    let scheme = AjtaiCommitmentScheme::<F17>::setup(&mut rng, &ajtai_small()).unwrap();
    let message = RingVec::new(vec![F17::from_u64(5), F17::from_u64(9)]);
    let (commitment, opening) = scheme.commit(&message, &mut rng).unwrap();
    println!(
        "ajtai/f17 key={} commitment={} opening={}",
        scheme.key().serialized_size(),
        commitment.serialized_size(),
        opening.serialized_size()
    );
}

fn print_ajtai_rq23_np8() {
    let mut rng = grid_std::test_rng();
    let scheme = AjtaiCommitmentScheme::<Rq23Np8>::setup(&mut rng, &ajtai_rq23_np8()).unwrap();
    let message = RingVec::new(vec![
        Rq23Np8::one(),
        Rq23Np8::zero(),
        Rq23Np8::one(),
        Rq23Np8::zero(),
    ]);
    let (commitment, opening) = scheme.commit(&message, &mut rng).unwrap();
    println!(
        "ajtai/rq23_np8 key={} commitment={} opening={}",
        scheme.key().serialized_size(),
        commitment.serialized_size(),
        opening.serialized_size()
    );
}

fn print_bdlop_f17() {
    let mut rng = grid_std::test_rng();
    let scheme = BdlopCommitmentScheme::<F17>::setup(&mut rng, &bdlop_small()).unwrap();
    let message = RingVec::new(vec![F17::from_u64(5), F17::from_u64(9)]);
    let (commitment, opening) = scheme.commit(&message, &mut rng).unwrap();
    println!(
        "bdlop/f17 key={} commitment={} opening={}",
        scheme.key().serialized_size(),
        commitment.serialized_size(),
        opening.serialized_size()
    );
}

fn print_bdlop_rq23_np8() {
    let mut rng = grid_std::test_rng();
    let scheme = BdlopCommitmentScheme::<Rq23Np8>::setup(&mut rng, &bdlop_rq23_np8()).unwrap();
    let message = RingVec::new(vec![
        Rq23Np8::one(),
        Rq23Np8::zero(),
        Rq23Np8::one(),
        Rq23Np8::zero(),
    ]);
    let (commitment, opening) = scheme.commit(&message, &mut rng).unwrap();
    println!(
        "bdlop/rq23_np8 key={} commitment={} opening={}",
        scheme.key().serialized_size(),
        commitment.serialized_size(),
        opening.serialized_size()
    );
}

fn print_gadget_f17() {
    let mut rng = grid_std::test_rng();
    let scheme = GadgetCommitmentScheme::<F17>::setup(&mut rng, &gadget_small()).unwrap();
    let message = RingVec::new(vec![F17::from_u64(5), F17::from_u64(6)]);
    let (commitment, opening) = scheme.commit(&message, &mut rng).unwrap();
    println!(
        "gadget/f17 a_open={} g_matrix={} commitment={} opening={}",
        scheme.a_open().serialized_size(),
        scheme.g_matrix().serialized_size(),
        commitment.serialized_size(),
        opening.serialized_size()
    );
}

fn print_gadget_rq23_np8() {
    let mut rng = grid_std::test_rng();
    let scheme = GadgetCommitmentScheme::<Rq23Np8>::setup(&mut rng, &gadget_rq23_np8()).unwrap();
    let message = RingVec::new(vec![
        Rq23Np8::one(),
        Rq23Np8::zero(),
        Rq23Np8::one(),
        Rq23Np8::zero(),
    ]);
    let (commitment, opening) = scheme.commit(&message, &mut rng).unwrap();
    println!(
        "gadget/rq23_np8 a_open={} g_matrix={} commitment={} opening={}",
        scheme.a_open().serialized_size(),
        scheme.g_matrix().serialized_size(),
        commitment.serialized_size(),
        opening.serialized_size()
    );
}

fn main() {
    print_ajtai_f17();
    print_ajtai_rq23_np8();
    print_bdlop_f17();
    print_bdlop_rq23_np8();
    print_gadget_f17();
    print_gadget_rq23_np8();
}
