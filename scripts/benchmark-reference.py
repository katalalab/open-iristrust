"""Development-only same-input benchmark, requires the pinned upstream checkout."""
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(sys.argv[1]).resolve() / "src"))
import iris
from iris.nodes.iris_response_refinement.fragile_bits_refinement import FragileBitRefinement

fixture = json.loads((Path(__file__).resolve().parent.parent / "tests/fixtures/upstream-features.json").read_text())
image = fixture["normalized"]["image"]
mask = fixture["normalized"]["mask"]
normalized = iris.NormalizedIris(normalized_image=np.array(image["data"], dtype=np.uint8).reshape(image["height"], image["width"]),
                                 normalized_mask=np.array(mask["data"], dtype=bool).reshape(mask["height"], mask["width"]))
bank = iris.ConvFilterBank(maskisduplicated=False)
refinement = FragileBitRefinement(value_threshold=[0.0001, 0.275, 0.08726646259971647], fragile_type="polar", maskisduplicated=False)
encoder = iris.IrisEncoder()
def encode():
    return encoder(refinement(bank(normalized)))
for _ in range(3):
    encode()
elapsed = []
for _ in range(20):
    start = time.perf_counter()
    encode()
    elapsed.append((time.perf_counter() - start) * 1000)
print(json.dumps({"iterations": 20, "iris_version": iris.__version__, "numpy_version": np.__version__,
                  "normalized_width": image["width"], "normalized_height": image["height"],
                  "gabor_encode_ms_p50": statistics.median(elapsed), "gabor_encode_ms_p95": sorted(elapsed)[18]}, indent=2))
