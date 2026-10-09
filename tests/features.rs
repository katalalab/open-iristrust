use iristrust::encoder::IrisEncoder;
use iristrust::filters::{FilterBank, FilterBankConfig, FilterResponse, GaborConfig, GaborKernel};
use iristrust::{EyeSide, GrayImage, Mask, NormalizedIris};

fn normalized(width: usize, height: usize, pixels: Vec<u8>, mask: Vec<bool>) -> NormalizedIris {
    NormalizedIris {
        image: GrayImage::new(width, height, pixels).unwrap(),
        mask: Mask::new(width, height, mask).unwrap(),
    }
}

#[test]
fn gabor_kernel_matches_independently_calculated_golden_values() {
    let kernel = GaborKernel::new(&GaborConfig {
        kernel_width: 5,
        kernel_height: 5,
        sigma_phi: 1.2,
        sigma_rho: 1.5,
        theta_degrees: 0.0,
        wavelength: 3.0,
        dc_correction: false,
        fixed_point: false,
    })
    .unwrap();

    assert_eq!(kernel.real.len(), 25);
    assert!((kernel.real[0] + 0.027_978_623_9).abs() < 1e-9);
    assert!((kernel.real[12] - 0.545_861_753_2).abs() < 1e-9);
    assert!((kernel.imag[7] + 0.411_926_494_4).abs() < 1e-9);
    assert!((kernel.norm_real - 1.0).abs() < 1e-12);
    assert!((kernel.norm_imag - 1.0).abs() < 1e-12);
}

#[test]
fn encoder_packs_phase_sign_and_each_component_mask_independently() {
    let response = FilterResponse {
        rows: 1,
        cols: 2,
        real: vec![-0.25, 0.25],
        imag: vec![0.25, -0.25],
        mask_real: vec![0.94, 0.92],
        mask_imag: vec![0.92, 0.94],
    };
    let plane = IrisEncoder::new(0.93)
        .unwrap()
        .encode_planes(&[response])
        .unwrap()
        .remove(0);

    assert!(!plane.get_bit(0, 0, 0, false).unwrap());
    assert!(plane.get_bit(0, 0, 1, false).unwrap());
    assert!(plane.get_bit(0, 1, 0, false).unwrap());
    assert!(!plane.get_bit(0, 1, 1, false).unwrap());
    assert!(plane.get_bit(0, 0, 0, true).unwrap());
    assert!(!plane.get_bit(0, 0, 1, true).unwrap());
    assert!(!plane.get_bit(0, 1, 0, true).unwrap());
    assert!(plane.get_bit(0, 1, 1, true).unwrap());
}

#[test]
fn convolution_wraps_across_the_angular_boundary() {
    let config = FilterBankConfig {
        filters: vec![GaborConfig {
            kernel_width: 5,
            kernel_height: 5,
            sigma_phi: 1.5,
            sigma_rho: 1.5,
            theta_degrees: 90.0,
            wavelength: 3.0,
            dc_correction: false,
            fixed_point: false,
        }],
        probe_rows: 2,
        probe_cols: 8,
        fragile_thresholds: None,
        ..FilterBankConfig::default()
    };
    let bank = FilterBank::new(config).unwrap();
    let mut pixels = vec![0; 8 * 5];
    for row in 0..5 {
        pixels[row * 8 + 7] = 255;
    }
    let iris = normalized(8, 5, pixels, vec![true; 8 * 5]);

    let response = bank.respond(&iris).unwrap().remove(0);
    assert!(response.real[0].abs() > 1e-9 || response.imag[0].abs() > 1e-9);
}

#[test]
fn default_fragile_validation_masks_constant_images() {
    let bank = FilterBank::new(FilterBankConfig::default()).unwrap();
    let iris = normalized(64, 16, vec![127; 64 * 16], vec![true; 64 * 16]);

    let template = bank.encode(&iris, EyeSide::Right).unwrap();
    assert_eq!(template.valid_bits(), 0);
}

#[test]
fn malformed_gabor_and_tiny_normalized_inputs_fail() {
    let invalid = GaborConfig {
        sigma_phi: f64::NAN,
        ..GaborConfig::default()
    };
    assert!(GaborKernel::new(&invalid).is_err());

    let bank_config = FilterBankConfig {
        probe_cols: 1,
        ..FilterBankConfig::default()
    };
    assert!(FilterBank::new(bank_config).is_err());

    let bank = FilterBank::new(FilterBankConfig::default()).unwrap();
    let tiny = normalized(1, 1, vec![0], vec![true]);
    assert!(bank.respond(&tiny).is_err());
}
