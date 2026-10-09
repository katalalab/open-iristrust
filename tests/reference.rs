use iristrust::filters::{FilterBank, FilterBankConfig};
use iristrust::{EyeSide, IrisTemplate, NormalizedIris};
use serde::Deserialize;

#[derive(Deserialize)]
struct Kernel {
    real: Vec<f64>,
    imag: Vec<f64>,
}
#[derive(Deserialize)]
struct Response {
    real: Vec<f64>,
    imag: Vec<f64>,
    mask_real: Vec<f64>,
    mask_imag: Vec<f64>,
}
#[derive(Deserialize)]
struct Reference {
    normalized: NormalizedIris,
    kernels: Vec<Kernel>,
    responses: Vec<Response>,
    template: IrisTemplate,
}

#[test]
fn gabor_and_encoded_bits_match_pinned_upstream_reference() {
    let reference: Reference =
        serde_json::from_str(include_str!("fixtures/upstream-features.json")).unwrap();
    let bank = FilterBank::new(FilterBankConfig::default()).unwrap();
    for (actual, expected) in bank.kernels().iter().zip(&reference.kernels) {
        assert_eq!(actual.real, expected.real);
        assert_eq!(actual.imag, expected.imag);
    }
    let response = bank.respond(&reference.normalized).unwrap();
    let mut max_error = 0_f64;
    for (actual, expected) in response.iter().zip(&reference.responses) {
        for (a, b) in [
            (&actual.real, &expected.real),
            (&actual.imag, &expected.imag),
            (&actual.mask_real, &expected.mask_real),
            (&actual.mask_imag, &expected.mask_imag),
        ] {
            assert_eq!(a.len(), b.len());
            for (&a, &b) in a.iter().zip(b) {
                max_error = max_error.max((a - b).abs());
            }
        }
    }
    assert!(max_error < 1e-6, "float32 reference max error {max_error}");
    let encoded = bank.encode(&reference.normalized, EyeSide::Left).unwrap();
    assert_eq!(
        encoded.planes, reference.template.planes,
        "all phase and mask bits must agree"
    );
    eprintln!(
        "upstream reference: exact kernels and {} phase/mask bits; max response error {max_error:.9}",
        encoded
            .planes
            .iter()
            .map(|p| p.bit_len().unwrap() * 2)
            .sum::<usize>()
    );
}
