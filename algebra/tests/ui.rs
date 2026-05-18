use std::fs;
use std::path::PathBuf;
use std::process::Command;

struct UiCase {
    rel_path: &'static str,
    panic_message: &'static str,
    trigger: &'static str,
}

const UI_CASES: &[UiCase] = &[
    UiCase {
        rel_path: "prime_field/composite_modulus.rs",
        panic_message: "PrimeField modulus must be an odd prime",
        trigger: "PrimeField::<15>::MODULUS",
    },
    UiCase {
        rel_path: "prime_field/even_modulus.rs",
        panic_message: "PrimeField modulus must be an odd prime",
        trigger: "PrimeField::<18>::MODULUS",
    },
    UiCase {
        rel_path: "prime_field/too_large_modulus.rs",
        panic_message: "PrimeField modulus must be an odd prime",
        trigger: "PrimeField::<18446744073709551615>::MODULUS",
    },
    UiCase {
        rel_path: "prime_field/too_small_modulus.rs",
        panic_message: "PrimeField modulus must be an odd prime",
        trigger: "PrimeField::<1>::MODULUS",
    },
    UiCase {
        rel_path: "poly/invalid_degree_non_power_of_two.rs",
        panic_message: "cyclotomic degree must be a non-zero power of two",
        trigger: "CyclotomicPolyRing::<PrimeField<17>, 3>::DEGREE",
    },
    UiCase {
        rel_path: "poly/invalid_degree_zero.rs",
        panic_message: "cyclotomic degree must be a non-zero power of two",
        trigger: "CyclotomicPolyRing::<PrimeField<17>, 0>::DEGREE",
    },
];

fn algebra_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn workspace_root() -> PathBuf {
    algebra_root()
        .parent()
        .expect("algebra crate should live under the workspace root")
        .to_path_buf()
}

fn harness_root() -> PathBuf {
    workspace_root().join("target/ui-harness")
}

fn target_dir() -> PathBuf {
    workspace_root().join("target/ui-harness-target")
}

fn case_name(rel_path: &str) -> String {
    rel_path
        .chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' => ch,
            _ => '_',
        })
        .collect()
}

fn write_harness_crate(case: &UiCase) -> PathBuf {
    let crate_dir = harness_root().join(case_name(case.rel_path));
    if crate_dir.exists() {
        fs::remove_dir_all(&crate_dir).expect("failed to clear previous UI harness crate");
    }
    fs::create_dir_all(crate_dir.join("src")).expect("failed to create UI harness crate");

    let manifest = format!(
        "[package]\nname = \"{}\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\n[workspace]\n\n[dependencies]\ngrid-algebra = {{ path = \"{}\" }}\n",
        case_name(case.rel_path),
        algebra_root().display()
    );
    fs::write(crate_dir.join("Cargo.toml"), manifest).expect("failed to write UI harness manifest");

    let fixture = algebra_root().join("tests/ui").join(case.rel_path);
    fs::copy(&fixture, crate_dir.join("src/main.rs")).expect("failed to copy UI fixture");
    crate_dir
}

fn run_case(case: &UiCase) -> std::process::Output {
    let crate_dir = write_harness_crate(case);
    Command::new("cargo")
        .arg("check")
        .arg("--quiet")
        .arg("--offline")
        .arg("--color")
        .arg("never")
        .arg("--manifest-path")
        .arg(crate_dir.join("Cargo.toml"))
        .arg("--target-dir")
        .arg(target_dir())
        .current_dir(workspace_root())
        .output()
        .expect("failed to run cargo check for UI fixture")
}

fn stderr(output: &std::process::Output) -> String {
    String::from_utf8_lossy(&output.stderr).replace("\r\n", "\n")
}

fn assert_compile_fail(case: &UiCase) {
    let output = run_case(case);
    let stderr = stderr(&output);

    assert!(
        !output.status.success(),
        "expected compile failure for {}\nstdout:\n{}\nstderr:\n{}",
        case.rel_path,
        String::from_utf8_lossy(&output.stdout),
        stderr,
    );
    assert!(
        stderr.contains(case.panic_message),
        "missing panic message for {}\nstderr:\n{}",
        case.rel_path,
        stderr,
    );
    assert!(
        stderr.contains(case.trigger),
        "missing const trigger for {}\nstderr:\n{}",
        case.rel_path,
        stderr,
    );
}

#[test]
fn ui() {
    for case in UI_CASES {
        assert_compile_fail(case);
    }
}
