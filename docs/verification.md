# Verification record

Date: 2026-10-09 (Asia/Tokyo). Implementation target: Windows x86-64, Rust 1.99.0, release profile with thin LTO. Source reference: Open IRIS 1.11.2 at `123890d4852067ff5c573e6cb32e4c00fd5261cd`.

## Scope of evidence

The Rust product performs segmentation with native ONNX Runtime and implements the downstream processing in Rust. No Python interpreter is invoked by the product. Python is used only by the optional developer fixture generator and comparison benchmark.

Reference fixture generation used NumPy 2.4.6 and the pinned upstream source. Fixed-point Gabor kernels match exactly. The complete 16,384 phase bits and 16,384 mask bits of the chosen normalized upstream fixture match exactly; maximum response error was `8.1e-8` (float32 reference), below the asserted `1e-6` tolerance. The fixture is 400 × 100 pixels, not the Rust default 512 × 128.

The official model ran locally on `tests/fixtures/anonymized.png` (640 × 480 pixels). Without bilateral preprocessing it predicted 12,755 pupil, 38,404 iris and 76,384 eyeball pixels. These counts demonstrate an executed inference on one fixture; they are not an accuracy study. Native geometry, normalization and encoding produced a template with over 12,000 valid bits. Model-dependent tests are explicitly ignored by the ordinary test command and explicitly executed by the dedicated model test command and CI job.

Native CLI verification follows image creation → PNG reload → normalization → encoding → JSON reload → five-column rotation comparison → aligned aggregation → comparison. A separate ONNX CLI test follows actual image inference → template output → repeat extraction → matching → aggregation → readback. Regression tests include invalid profiles, malformed probabilities, local contour crossings, bilinear mask leakage, insufficient overlap, invalid aggregation settings, atomic output staging and duplicate output aliases.

## Measured release performance

Hardware: Intel Core i9-14900KS, 24 physical cores / 32 logical processors; Windows. Rust used 4 Rayon workers. Each result includes kernel convolution, fragile-bit refinement and phase encoding. Inference, image I/O and geometry are excluded from this stage benchmark.

| Stage | Input / trials | Median | p95 |
| --- | --- | ---: | ---: |
| Rust Gabor + refinement + encoding | 400 × 100, two filters, 16 × 256 probes, 100 trials after 5 warmups | 2.6555 ms | 3.5366 ms |
| Upstream Python same feature stages | same fixture / filter / probe settings, 20 trials after 3 warmups | 89.0559 ms | 95.0372 ms |
| Rust rotational packed matching | two planes, 31 candidate rotations, 100 trials | 39.5 µs | 122.0 µs |

The measured feature-stage median ratio is about 33.5×. These two benchmarks were run sequentially in the same session after the release build finished. This is a local measurement, not a general speed guarantee. Neither number measures complete image-to-template latency or identity search at scale.

Commands:

```text
cargo run --locked --release --example benchmark -- 100
python scripts/benchmark-reference.py /path/to/pinned/open-iris
```

## Reproduce

```text
cargo fmt --all -- --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked
cargo test --locked --no-default-features
sh scripts/fetch-model.sh
cargo test --locked --test onnx -- --ignored --nocapture
cargo run --locked --release -- demo --output-dir out/verification-demo
```

Windows model download: `./scripts/fetch-model.ps1`.

Local Windows results: 46 regular tests passed with default features and with the ONNX feature disabled; the two model-dependent tests were additionally executed and both passed (no skips in that command). Formatting, Clippy with warnings as errors, and workflow static validation passed. The actual model CLI run produced 12,326 valid bits after the stricter bilinear validity fix; repeated extraction and aggregation both compared at distance 0.

The GitHub workflow runs native tests, lint and release builds on Linux/Windows, model-dependent E2E on Linux, and core tests on Rust 1.88. Successful workflow runs are the evidence for those environments; local Windows results alone do not establish them.

## Deliberate differences and unverified claims

- Geometry: largest connected component and radial ellipse fit rather than the Python contour-refinement/extrapolation chain.
- Normalization: default bilinear radial-bin centers; invalid pixels receive zero intensity and invalid masks. The Python default linear normalizer uses nearest sampling and radial endpoints.
- Quality: native RMS angular sharpness and eccentricity proxy; thresholds need a representative dataset calibration.
- Encoding profiles: checked configuration fingerprints and separate extraction/external-normalized domains; this is not the upstream Python base64 template schema.
- Matching: raw masked Hamming distance; `normalized=true` is explicitly rejected because no normalized-distance calibration is provided.
- Aggregation: typed aligned voting rather than the Python DAG/callback system.
- Not verified: complete Python API/template compatibility, TensorRT or Orb deployment, biometric FAR/FRR, liveness/anti-spoofing, large population performance, or production identity thresholds.
