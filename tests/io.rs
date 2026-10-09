use iristrust::io::{OutputTransaction, load_image, read_json, save_image, write_json};
use iristrust::{CodePlane, EyeSide, GrayImage, IrisTemplate, TEMPLATE_VERSION};

#[test]
fn serialization_roundtrip_preserves_packed_words() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("template.json");
    let t = IrisTemplate {
        version: TEMPLATE_VERSION.into(),
        encoding_profile: "test-profile-v1".into(),
        eye_side: EyeSide::Left,
        planes: vec![CodePlane {
            rows: 1,
            cols: 32,
            code: vec![u64::MAX],
            mask: vec![u64::MAX],
        }],
    };
    write_json(&path, &t).unwrap();
    let decoded: IrisTemplate = read_json(&path).unwrap();
    decoded.validate().unwrap();
    assert_eq!(t, decoded);
    assert!(
        write_json(&path, &t).is_err(),
        "outputs must not overwrite implicitly"
    );
}

#[test]
fn missing_and_invalid_images_fail() {
    let dir = tempfile::tempdir().unwrap();
    assert!(load_image(&dir.path().join("missing.png")).is_err());
    let path = dir.path().join("invalid.png");
    std::fs::write(&path, b"not an image").unwrap();
    assert!(load_image(&path).is_err());
}

#[test]
fn failed_image_encoding_leaves_no_output_or_staged_json() {
    let dir = tempfile::tempdir().unwrap();
    let json = dir.path().join("template.json");
    let image = dir.path().join("unsupported.extension");
    let template = IrisTemplate {
        version: TEMPLATE_VERSION.into(),
        encoding_profile: "test-profile-v1".into(),
        eye_side: EyeSide::Left,
        planes: vec![CodePlane {
            rows: 1,
            cols: 1,
            code: vec![0],
            mask: vec![0],
        }],
    };
    let grayscale = GrayImage::new(1, 1, vec![127]).unwrap();

    let mut transaction = OutputTransaction::new([json.as_path(), image.as_path()]).unwrap();
    transaction.stage_json(&json, &template).unwrap();
    assert!(transaction.stage_image(&image, &grayscale).is_err());
    drop(transaction);

    assert!(!json.exists());
    assert!(!image.exists());
    assert!(save_image(&image, &grayscale).is_err());
    assert!(!image.exists());
}
