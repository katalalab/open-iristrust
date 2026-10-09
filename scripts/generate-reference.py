"""Developer-only golden fixture generation; never invoked by the Rust product.

Usage: python scripts/generate-reference.py /path/to/pinned/open-iris
Install upstream requirements into an isolated development environment first.
Only unpickle the trusted checkout at the pinned upstream revision.
"""
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

upstream = Path(sys.argv[1]).resolve()
revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
expected_revision = "123890d4852067ff5c573e6cb32e4c00fd5261cd"
if revision != expected_revision:
    raise SystemExit("upstream revision is not the pinned reference")
sys.path.insert(0, str(upstream / "src"))
import iris
from iris.nodes.iris_response_refinement.fragile_bits_refinement import FragileBitRefinement

with (upstream / "tests/e2e_tests/nodes/iris_response/mocks/conv_filter_bank/normalized_iris.pickle").open("rb") as file:
    normalized = pickle.load(file)
bank = iris.ConvFilterBank(maskisduplicated=False)
responses = bank(normalized)
refined = FragileBitRefinement(value_threshold=[0.0001, 0.275, 0.08726646259971647], fragile_type="polar", maskisduplicated=False)(responses)
encoded = iris.IrisEncoder()(refined)

def raster(array):
    return {"width": array.shape[1], "height": array.shape[0], "data": array.flatten().tolist()}

def words(array):
    bits = array.reshape(-1)
    result = [0] * ((len(bits) + 63) // 64)
    for i, bit in enumerate(bits):
        if bit:
            result[i // 64] |= 1 << (i % 64)
    return result

fixture = {
    "provenance": {"repository": "https://github.com/worldcoin/open-iris", "revision": revision,
                   "iris_version": iris.__version__, "numpy_version": np.__version__,
                   "source": "tests/e2e_tests/nodes/iris_response/mocks/conv_filter_bank/normalized_iris.pickle",
                   "maskisduplicated": False, "license": "MIT"},
    "normalized": {"image": raster(normalized.normalized_image), "mask": raster(normalized.normalized_mask)},
    "kernels": [{"real": f.kernel_values.real.flatten().tolist(), "imag": f.kernel_values.imag.flatten().tolist()}
                for f in bank.params.filters],
    "responses": [{"rows": a.shape[0], "cols": a.shape[1], "real": a.real.flatten().tolist(),
                   "imag": a.imag.flatten().tolist(), "mask_real": m.real.flatten().tolist(),
                   "mask_imag": m.imag.flatten().tolist()}
                  for a, m in zip(responses.iris_responses, responses.mask_responses)],
    "template": {"version": "iristrust-v1", "encoding_profile": "upstream-reference", "eye_side": "left", "planes": [
        {"rows": code.shape[0], "cols": code.shape[1], "code": words(code), "mask": words(mask)}
        for code, mask in zip(encoded.iris_codes, encoded.mask_codes)]},
}
destination = Path(__file__).resolve().parent.parent / "tests/fixtures/upstream-features.json"
destination.write_text(json.dumps(fixture, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
print(f"Generated {destination.name}: {destination.stat().st_size} bytes; upstream {revision}")
