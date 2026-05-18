use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use grid_algebra::arith::ring::IntegerRing;
use grid_transcript::hash::{
    GoldilocksPoseidon2Field, GoldilocksPoseidon2Transcript, ShakeTranscript,
};
use grid_transcript::{FieldTranscript, Transcript};
use std::hint::black_box;

fn bench_poseidon2_absorb_and_challenge(c: &mut Criterion) {
    let mut group = c.benchmark_group("poseidon2_transcript");
    for input_len in [4usize, 32usize] {
        group.bench_with_input(
            BenchmarkId::new("challenge_fields", input_len),
            &input_len,
            |b, &len| {
                let payload: Vec<_> = (0..len)
                    .map(|index| GoldilocksPoseidon2Field::from_u64(index as u64 + 1))
                    .collect();
                b.iter(|| {
                    let mut transcript = GoldilocksPoseidon2Transcript::goldilocks_t12();
                    transcript
                        .append_elements(black_box(b"payload"), black_box(&payload))
                        .unwrap();
                    black_box(transcript.challenge_elements(b"challenge", 4).unwrap());
                });
            },
        );
    }
    group.finish();
}

fn bench_shake_absorb_and_challenge(c: &mut Criterion) {
    let mut group = c.benchmark_group("shake_transcript");
    for input_len in [32usize, 256usize] {
        group.bench_with_input(
            BenchmarkId::new("challenge_bytes", input_len),
            &input_len,
            |b, &len| {
                let payload: Vec<u8> = (0..len).map(|index| index as u8).collect();
                b.iter(|| {
                    let mut transcript = ShakeTranscript::new();
                    transcript
                        .append_bytes(black_box(b"payload"), black_box(&payload))
                        .unwrap();
                    black_box(transcript.challenge_bytes(b"challenge", 32).unwrap());
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_poseidon2_absorb_and_challenge,
    bench_shake_absorb_and_challenge
);
criterion_main!(benches);
