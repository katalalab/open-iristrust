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

#[test]
fn failed_later_publication_does_not_delete_published_path() {
    let temp_root = std::env::temp_dir().canonicalize().unwrap();
    let dir = tempfile::tempdir_in(&temp_root).unwrap();
    assert!(dir.path().canonicalize().unwrap().starts_with(&temp_root));
    let first = dir.path().join("SYNTHETIC_PRIVATE_FIRST.json");
    let second = dir.path().join("SYNTHETIC_PRIVATE_SECOND.json");
    let mut transaction = OutputTransaction::new([&first, &second]).unwrap();
    transaction.stage_json(&first, &"synthetic first").unwrap();
    transaction
        .stage_json(&second, &"synthetic second")
        .unwrap();
    std::fs::write(&second, "synthetic conflict").unwrap();

    let error = transaction.commit().unwrap_err();
    assert!(
        first.is_file(),
        "a published path must not be removed by rollback: {error}"
    );
    assert!(!format!("{error}").contains("SYNTHETIC_PRIVATE"));
    assert!(!format!("{error:?}").contains("SYNTHETIC_PRIVATE"));
    match &error {
        iristrust::Error::PartialPublish(partial) => {
            assert_eq!(partial.published_paths(), &[first.canonicalize().unwrap()]);
            assert_eq!(partial.cause().kind(), std::io::ErrorKind::AlreadyExists);
        }
        other => panic!("expected structured partial publication, got {other:?}"),
    }
    assert_eq!(
        std::fs::read_to_string(&second).unwrap(),
        "synthetic conflict"
    );
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 2);
}

#[test]
fn failure_before_any_publication_preserves_conflict_and_cleans_staging() {
    let temp_root = std::env::temp_dir().canonicalize().unwrap();
    let dir = tempfile::tempdir_in(&temp_root).unwrap();
    assert!(dir.path().canonicalize().unwrap().starts_with(&temp_root));
    let first = dir.path().join("first.json");
    let second = dir.path().join("second.json");
    let mut transaction = OutputTransaction::new([&first, &second]).unwrap();
    transaction.stage_json(&first, &"synthetic first").unwrap();
    transaction
        .stage_json(&second, &"synthetic second")
        .unwrap();
    std::fs::write(&first, "synthetic conflict").unwrap();

    let error = transaction.commit().unwrap_err();
    assert!(matches!(error, iristrust::Error::Io(_)));
    assert_eq!(
        std::fs::read_to_string(&first).unwrap(),
        "synthetic conflict"
    );
    assert!(!second.exists());
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
}

#[test]
fn incomplete_staging_returns_invalid_input_without_publishing() {
    let temp_root = std::env::temp_dir().canonicalize().unwrap();
    let dir = tempfile::tempdir_in(&temp_root).unwrap();
    assert!(dir.path().canonicalize().unwrap().starts_with(&temp_root));
    let first = dir.path().join("first.json");
    let second = dir.path().join("second.json");
    let mut transaction = OutputTransaction::new([&first, &second]).unwrap();
    transaction.stage_json(&first, &"synthetic first").unwrap();
    assert!(matches!(
        transaction.commit(),
        Err(iristrust::Error::InvalidInput(_))
    ));
    assert!(!first.exists() && !second.exists());
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
}
