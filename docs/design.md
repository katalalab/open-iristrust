# Open Iristrust — design and acceptance

Operator request: rebuild Worldcoin Open IRIS in Rust, publish under Katalalab.
Operator preference: independent Rust design and performance, rather than Python API compatibility.

Upstream reference: `worldcoin/open-iris`, commit `123890d4852067ff5c573e6cb32e4c00fd5261cd`, MIT.

## Architecture

Native Rust library and CLI. A local ONNX model segments grayscale eye images into eyeball, iris, pupil and eyelashes. Segmentation can also be supplied as masks; geometry or a normalized image can be supplied directly. Boundary estimation uses robust native geometry; rubber-sheet normalization, Gabor probes, phase encoding, quality evaluation, rotational masked Hamming matching and multi-frame alignment/majority voting are Rust implementations. ONNX Runtime is optional. No Python subprocess in the product.

The template format is versioned `iristrust-v1`: two phase bits per sample and masks packed into u64 words. Templates also carry a SHA-256 fingerprint of the encoding configuration, including normalization for extraction. Left/right eyes and different encoding profiles cannot be compared. Zero or insufficient overlap is an error, never a perfect match. Scores are distances, not identity probabilities. Native geometry and normalization make this format distinct from upstream v0.1; no end-to-end bit compatibility is promised without measured evidence.

Geometry uses equal angular samples starting at positive x around each boundary's center. The iris must be convex and strictly contain the pupil. Native orientation is the iris PCA major-axis angle; offgaze is a radial eccentricity proxy, not calibrated gaze estimation. Bilinear validity requires every nonzero-weight pixel tap to be visible and free of noise. Normalization uses radial-bin centers and zero intensity for invalid pixels. These are deliberate differences from upstream.

## Implementation plan

- [ ] Validated raster/geometry/template types and bounded file loading.
- [ ] Segmentation masks, robust boundary estimation, normalized sampling and quality metrics.
- [ ] Gabor kernels, parallel sparse convolution, fragile-bit refinement and phase encoding.
- [ ] Packed XOR/popcount matching, angular rotation and aligned multi-frame majority vote.
- [ ] ONNX inference, local model acquisition with checksum verification, reusable sessions.
- [ ] CLI extract/match/aggregate/inspect and a local end-to-end example.
- [ ] Reference fixtures, native and ONNX end-to-end tests, release benchmarks.
- [ ] Documentation, MIT attribution, Linux/Windows CI, independent review and public repository.

## Acceptance

Run formatting, Clippy, Rust tests, native CLI extraction → serialization → matching → aggregation, and ONNX inference on the upstream anonymized eye. Compare selected stages against upstream numerical fixtures and publish measured results, failures and differences. Benchmark release builds with actual dimensions and thread counts. Confirm public repository head and its CI results. Do not report production biometric accuracy, cross-template compatibility or anti-spoofing without evidence.

Rollback: preserve upstream checkout unchanged; new project and new remote have no effect on existing repositories. Keep downloaded models and temporary reference environments ignored.
