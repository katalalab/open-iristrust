# Open Iristrust

Native Rust iris recognition, built by Katalalab from the algorithms and public model in [Worldcoin Open IRIS](https://github.com/worldcoin/open-iris).

The library and CLI implement local ONNX segmentation, largest-component geometry and radial ellipse fitting, rubber-sheet normalization, parallel Gabor probes, fragile-bit refinement, phase encoding, rotational masked Hamming matching, and aligned multi-frame majority voting. Packed `u64` templates let matching use XOR, AND and population counts. Kernels and inference sessions can be reused across frames.

This is an independent Rust design. It has no Python runtime or Python subprocess. ONNX Runtime is a native dependency enabled by the default `onnx` feature; `--no-default-features` builds the Rust image-processing, encoding and matching core without it. Nothing in extraction or matching connects to a network. Downloading the optional model is a separate, explicit step.

## Quick start

Requires Rust 1.88 or newer. Build a release binary:

```sh
cargo build --locked --release
cargo run --locked --release -- demo --output-dir out/demo
```

The demo writes a synthetic eye image, geometry, normalized image, templates and an end-to-end report. Its five-column rotation must match at distance `0` and rotation `-5`.

Fetch the pinned, checksum-verified official model:

```sh
# Linux/macOS
sh scripts/fetch-model.sh
# Windows PowerShell
./scripts/fetch-model.ps1
```

Extract from the included upstream anonymized test eye:

```sh
mkdir -p out
cargo run --release -- extract tests/fixtures/anonymized.png --eye left \
  --model models/iris-semseg.onnx --output out/eye.json --report out/quality.json
cargo run --release -- match out/eye.json out/eye.json
cargo run --release -- inspect out/eye.json
cargo run --release -- aggregate out/eye.json out/eye.json --output out/aggregate.json
```

Use the same command on one line in PowerShell; its continuation character is a backtick. Outputs refuse to overwrite existing files. `--threads 4` controls both native workers and ONNX CPU inference. `--denoise` enables bilateral model preprocessing; it is disabled by default for throughput. Infrared eye imagery is the intended input.

## Library

```rust,no_run
use iristrust::{EyeSide, IrisPipeline, PipelineConfig, Matcher, MatcherConfig};
use iristrust::segmentation::{OnnxSegmenter, SegmentationConfig};

# fn main() -> iristrust::Result<()> {
let image = iristrust::io::load_image(std::path::Path::new("eye.png"))?;
let mut segmenter = OnnxSegmenter::new("models/iris-semseg.onnx", SegmentationConfig::default())?;
let pipeline = IrisPipeline::new(PipelineConfig::default())?;
let result = pipeline.extract(&image, EyeSide::Left, &mut segmenter)?;
let matcher = Matcher::new(MatcherConfig::default())?;
let comparison = matcher.compare(&result.template, &result.template)?;
assert_eq!(comparison.distance, 0.0);
# Ok(())
# }
```

Embedding applications can implement the `Segmenter` trait, supply `GeometryMasks`, or provide validated circle/ellipse `Geometry`. `extract_from_masks` and `extract_with_geometry` require no ONNX feature. `encode` handles externally normalized images and optional validity masks. `config` prints all default pipeline settings; `extract --config path.json` applies an edited configuration. JSON rasters contain `width`, `height` and row-major `data`; see the public Rust types and `docs/design.md`.

## Template and scoring contract

- Format: `iristrust-v1`, with eye side, encoding profile, and code/mask planes containing two phase bits per sample. Bits are packed little-endian within `u64` words.
- The default template has two `16 × 256 × 2` bit planes: 16,384 code bits plus 16,384 mask bits (4 KiB of packed words before JSON metadata).
- Encoding profiles include Gabor/probe/mask settings and, for extraction, normalization settings. Different profiles, eye sides or layouts are rejected. External normalized input is a separate profile domain; it is not assumed to share extraction provenance.
- Distance is differing valid bits divided by jointly valid bits, minimized over allowed angular rotations. Default rotation search is ±15 columns and minimum overlap is 1,024 bits. Zero/insufficient overlap is an error. Equal scores select zero, then smallest absolute rotation, with negative before positive.
- Aggregation aligns to the first frame, accepts distances up to the configured limit, and votes among valid bits. Ties are masked out; insufficient frames/overlap fail. The default `0.35` aggregation cutoff is a configurable engineering default, **not a calibrated identity threshold**.
- Quality gates check visibility, pupil/iris ratio, center displacement and a radial eccentricity proxy. Native sharpness is an RMS angular-gradient metric; its default minimum is `0` because no dataset calibration is supplied. Scores do not provide identity probabilities or liveness detection.

Generated templates and normalized images are biometric data. The product writes them only to explicitly requested local output paths and does not send them elsewhere.

## Verification and performance

```sh
cargo fmt --all -- --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked
cargo test --locked --no-default-features
# Requires the downloaded model; never treat the ordinary ignored test as evidence.
cargo test --locked --test onnx -- --ignored --nocapture
cargo run --locked --release --example benchmark -- 100
```

Tests cover native CLI end-to-end execution, malformed inputs, masks, geometry containment, reference features, rotation against a per-bit oracle, incompatible profiles, alignment and majority voting. A dedicated CI job downloads the pinned model and runs actual inference and extraction. [Verification details](docs/verification.md) distinguish what was measured from what remains unverified.

Selected stages are tested against Open IRIS commit `123890d4852067ff5c573e6cb32e4c00fd5261cd` (v1.11.2). The fixture covers exact fixed-point Gabor kernels and all phase/mask bits for one upstream normalized eye; response tolerance is `1e-6`. That does **not** establish end-to-end bit compatibility or biometric accuracy.

The Python DAG, Orb/TensorRT integrations, legacy Python callbacks, and alternative nonlinear/perspective normalizers are not copied into this independent architecture. Rust traits and typed stages provide extension points. Native fitted geometry, bilinear sampling and quality metrics differ from the Python pipeline, so complete Python API or template compatibility is not claimed.

## License and attribution

MIT. Original algorithm source and derived test fixtures retain the Worldcoin Foundation notice in [LICENSE](LICENSE). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream and model provenance. Open Iristrust is maintained by Katalalab and is not affiliated with Worldcoin. Model weights are downloaded separately under their upstream model licenses.
