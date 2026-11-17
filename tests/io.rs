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

fn directory_entries(dir: &std::path::Path) -> Vec<std::ffi::OsString> {
    let mut names = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect::<Vec<_>>();
    names.sort();
    names
}

#[test]
fn commit_with_unstaged_output_publishes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let first = dir.path().join("first.json");
    let second = dir.path().join("second.json");
    let mut transaction = OutputTransaction::new([first.as_path(), second.as_path()]).unwrap();
    transaction.stage_json(&first, &1).unwrap();

    assert!(transaction.commit().is_err());
    assert!(directory_entries(dir.path()).is_empty());
}

#[test]
fn failed_publish_rolls_back_earlier_outputs_and_staged_files() {
    let dir = tempfile::tempdir().unwrap();
    let first = dir.path().join("first.json");
    let second = dir.path().join("second.json");
    let third = dir.path().join("third.json");
    let mut transaction =
        OutputTransaction::new([first.as_path(), second.as_path(), third.as_path()]).unwrap();
    for path in [&first, &second, &third] {
        transaction.stage_json(path, &1).unwrap();
    }
    std::fs::write(&second, b"created concurrently").unwrap();

    assert!(transaction.commit().is_err());
    assert_eq!(directory_entries(dir.path()), vec!["second.json"]);
    assert_eq!(std::fs::read(&second).unwrap(), b"created concurrently");
}
