use iristrust::io::write_json;
use iristrust::{CodePlane, EyeSide, IrisTemplate, TEMPLATE_VERSION};
use serde_json::Value;
use std::process::Command;

fn run(arguments: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_iristrust"))
        .args(arguments)
        .output()
        .unwrap()
}
#[test]
fn native_cli_runs_image_template_matching_aggregation_end_to_end() {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("demo");
    let result = run(&["demo", "--output-dir", output.to_str().unwrap()]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(report["rotation_match"]["distance"], 0.0);
    assert_eq!(report["rotation_match"]["rotation"], -5);
    let template = output.join("template.json");
    let reloaded = output.join("from-disk.json");
    let geometry = output.join("geometry.json");
    let eye = output.join("eye.png");
    let extraction = run(&[
        "extract",
        eye.to_str().unwrap(),
        "--eye",
        "left",
        "--geometry",
        geometry.to_str().unwrap(),
        "--output",
        reloaded.to_str().unwrap(),
    ]);
    assert!(
        extraction.status.success(),
        "{}",
        String::from_utf8_lossy(&extraction.stderr)
    );
    let aggregated = output.join("cli-aggregate.json");
    let rotated = output.join("rotated.json");
    let aggregation = run(&[
        "aggregate",
        reloaded.to_str().unwrap(),
        rotated.to_str().unwrap(),
        "--output",
        aggregated.to_str().unwrap(),
    ]);
    assert!(
        aggregation.status.success(),
        "{}",
        String::from_utf8_lossy(&aggregation.stderr)
    );
    let result = run(&[
        "match",
        template.to_str().unwrap(),
        aggregated.to_str().unwrap(),
    ]);
    assert!(result.status.success());
    let compared: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(compared["distance"], 0.0);
    let result = run(&["inspect", template.to_str().unwrap()]);
    assert!(result.status.success());
    let metadata: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert!(metadata.get("code").is_none());
    assert!(
        run(&["demo", "--output-dir", output.to_str().unwrap()])
            .status
            .code()
            .unwrap()
            != 0
    );
}

#[test]
fn invalid_cli_inputs_fail_with_nonzero_status() {
    assert!(
        !run(&["match", "missing-a.json", "missing-b.json"])
            .status
            .success()
    );
    assert!(!run(&["--threads", "0", "config"]).status.success());
    assert!(
        !run(&[
            "extract",
            "missing.png",
            "--eye",
            "left",
            "--output",
            "unused.json",
            "--geometry",
            "a.json",
            "--model",
            "b.onnx"
        ])
        .status
        .success()
    );
}

#[test]
fn aggregate_rejects_normalized_duplicate_output_and_report_without_artifacts() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input.json");
    let output = temp.path().join("output.json");
    let alias = temp.path().join(".").join("output.json");
    let template = IrisTemplate {
        version: TEMPLATE_VERSION.into(),
        encoding_profile: "test-profile-v1".into(),
        eye_side: EyeSide::Left,
        planes: vec![CodePlane {
            rows: 1,
            cols: 1,
            code: vec![0],
            mask: vec![3],
        }],
    };
    write_json(&input, &template).unwrap();

    let result = run(&[
        "aggregate",
        input.to_str().unwrap(),
        "--output",
        output.to_str().unwrap(),
        "--report",
        alias.to_str().unwrap(),
        "--min-overlap",
        "1",
    ]);
    assert!(!result.status.success());
    assert!(!output.exists());
}

#[test]
fn extract_rejects_normalized_duplicate_output_and_report_before_reading_inputs() {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("output.json");
    let alias = temp.path().join(".").join("output.json");

    let result = run(&[
        "extract",
        "missing.png",
        "--eye",
        "left",
        "--output",
        output.to_str().unwrap(),
        "--report",
        alias.to_str().unwrap(),
        "--geometry",
        "missing-geometry.json",
    ]);
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("duplicate output path"));
    assert!(!output.exists());
}
