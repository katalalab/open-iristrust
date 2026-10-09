# Third-party notices

## Worldcoin Open IRIS

Source: https://github.com/worldcoin/open-iris

Reference revision: `123890d4852067ff5c573e6cb32e4c00fd5261cd`, Open IRIS 1.11.2.

License: MIT, Copyright (c) 2023 Worldcoin Foundation. The complete notice is retained in `LICENSE`.

Gabor generation, sparse probe response, fragile-bit refinement, phase encoding and masked Hamming matching use the upstream algorithms. Native geometry estimation, normalized sampling, typed orchestration, serialization and interfaces are independently implemented for this Rust design.

`tests/fixtures/anonymized.png` is copied from upstream `tests/e2e_tests/pipelines/mocks/inputs/anonymized.png`. `tests/fixtures/upstream-features.json` is generated from upstream's trusted `normalized_iris.pickle` fixture and algorithm outputs by `scripts/generate-reference.py`. Only the pinned trusted checkout is unpickled; the product never reads Python pickles.

## Iris semantic-segmentation model

Source: https://huggingface.co/Worldcoin/iris-semantic-segmentation

Revision: `e6c8fbb8e2e7e024a58818530a6e3ae5c005f47a`

Artifact: `iris_semseg_upp_scse_mobilenetv2.onnx`, 56,105,729 bytes.

SHA-256: `1228a1c760791749fceeedf9322e24f098aac80234368e1beee156e1e8fa57e7`

The model repository supplies MIT and Apache-2.0 licenses: [MIT](https://huggingface.co/Worldcoin/iris-semantic-segmentation/blob/e6c8fbb8e2e7e024a58818530a6e3ae5c005f47a/LICENSE-MIT.md), [Apache-2.0](https://huggingface.co/Worldcoin/iris-semantic-segmentation/blob/e6c8fbb8e2e7e024a58818530a6e3ae5c005f47a/LICENSE-APACHE.md). Weights are neither checked into nor redistributed by this repository.

## Rust dependencies

Dependency versions are recorded in `Cargo.lock`. `ort` is a Rust wrapper around the separately licensed native ONNX Runtime (Microsoft, MIT). The optional build-time binary download is performed by `ort-sys`; consumers distributing a binary must preserve all relevant dependency notices and the native runtime library alongside it. A plain executable copy does not bundle ONNX Runtime automatically.
