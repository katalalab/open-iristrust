#![cfg(feature = "onnx")]
use iristrust::segmentation::{OnnxSegmenter, SegmentationConfig, Segmenter};

/// Model-dependent test is explicit and is run by the dedicated ONNX CI job.
#[test]
#[ignore = "requires pinned 56 MB model; run scripts/fetch-model first"]
fn official_model_segments_anonymized_eye() {
    let model = std::env::var_os("IRISTRUST_MODEL")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| "models/iris-semseg.onnx".into());
    let image =
        iristrust::io::load_image(std::path::Path::new("tests/fixtures/anonymized.png")).unwrap();
    let mut segmenter = OnnxSegmenter::new(model, SegmentationConfig::default()).unwrap();
    let masks = segmenter.segment(&image).unwrap();
    masks.validate().unwrap();
    for (label, mask) in [
        ("pupil", &masks.pupil),
        ("iris", &masks.iris),
        ("eyeball", &masks.eyeball),
    ] {
        let count = mask.data.iter().filter(|&&x| x).count();
        assert!(count > 100, "{label} is empty");
        assert!(count < image.data.len() / 2, "{label} covers too much");
        eprintln!("{label}: {count} pixels");
    }
}

#[test]
#[ignore = "requires pinned model; dedicated CI runs this complete CLI path"]
fn official_model_cli_extract_match_and_aggregate_end_to_end() {
    use std::process::Command;
    let model = std::env::var_os("IRISTRUST_MODEL")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| "models/iris-semseg.onnx".into());
    let temp = tempfile::tempdir().unwrap();
    let a = temp.path().join("a.json");
    let b = temp.path().join("b.json");
    let aggregate = temp.path().join("aggregate.json");
    let binary = env!("CARGO_BIN_EXE_iristrust");
    for target in [&a, &b] {
        let output = Command::new(binary)
            .args([
                "extract",
                "tests/fixtures/anonymized.png",
                "--eye",
                "left",
                "--model",
            ])
            .arg(&model)
            .arg("--output")
            .arg(target)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert!(report["valid_bits"].as_u64().unwrap() > 4096);
    }
    let output = Command::new(binary)
        .arg("match")
        .arg(&a)
        .arg(&b)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let matched: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(matched["distance"], 0.0);
    let output = Command::new(binary)
        .arg("aggregate")
        .arg(&a)
        .arg(&b)
        .arg("--output")
        .arg(&aggregate)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let original: iristrust::IrisTemplate = iristrust::io::read_json(&a).unwrap();
    let aggregated: iristrust::IrisTemplate = iristrust::io::read_json(&aggregate).unwrap();
    let matcher = iristrust::Matcher::new(iristrust::MatcherConfig::default()).unwrap();
    assert_eq!(
        matcher.compare(&original, &aggregated).unwrap().distance,
        0.0
    );
    eprintln!(
        "ONNX CLI E2E: {} valid bits; repeated extraction and aggregation distance 0",
        original.valid_bits()
    );
}
