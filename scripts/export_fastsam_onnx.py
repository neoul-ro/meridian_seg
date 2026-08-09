#!/usr/bin/env python3
"""Export weights/FastSAM.pt -> weights/FastSAM-<imgsz>.onnx.

Uses the vendored ultralytics 8.0.120 inside FastSAM_official, i.e. the same
code fastsam_node.py runs, so the exported graph matches production.

    python3 scripts/export_fastsam_onnx.py

Two things here are deliberate and easy to get wrong:

1. imgsz is a [h, w] LIST, not an int. fastsam/model.py:78 does
   `if args.imgsz == DEFAULT_CFG.imgsz: args.imgsz = self.model.args['imgsz']`
   and DEFAULT_CFG.imgsz is 640 -- so passing the int 640 is indistinguishable
   from passing nothing and silently becomes the trained 1024, which then
   shape-mismatches the node's imgsz=640 inference. A list compares unequal,
   so the requested size survives.

2. The ONNX stays FP32. FP16 is applied later by `trtexec --fp16` when building
   the Jetson engine; baking it in here would only lose range. simplify=False
   because TensorRT's builder already does constant folding and layer fusion,
   so onnxsim buys nothing and drags in onnxruntime-gpu.

A rectangular size is usually what you want. ultralytics runs a .pt with
rectangular inference, so a 640x480 camera frame reaches the model as 480x640 --
307,200 pixels. A square 640x640 engine forces the same frame to 409,600 pixels,
a third of which is letterbox padding, and that padding also shifts the detection
set. Exporting at 480x640 matches the .pt path exactly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = PACKAGE_ROOT / "weights" / "FastSAM.pt"
FASTSAM_REPO = Path.home() / "FastSAM_official"
OPSET = 17  # supported by TRT 8.5 (JetPack 5.x) through TRT 10.x (JetPack 6.x)
# 480x640 matches a 640x480 camera under rectangular inference; 640 and 1024 are
# the square sizes (1024 is what FastSAM was trained at).
DEFAULT_SIZES = ((480, 640), (640, 640), (1024, 1024))

sys.path.insert(0, str(FASTSAM_REPO))
from fastsam import FastSAM  # noqa: E402


def stem_for(height: int, width: int) -> str:
    """FastSAM-640 for square, FastSAM-480x640 for rectangular."""
    if height == width:
        return f"{WEIGHTS.stem}-{height}"
    return f"{WEIGHTS.stem}-{height}x{width}"


def export_one(height: int, width: int) -> Path:
    model = FastSAM(str(WEIGHTS))
    produced = Path(
        model.export(
            format="onnx",
            imgsz=[height, width],
            opset=OPSET,
            simplify=False,
            half=False,
            dynamic=False,
            batch=1,
        )
    )
    target = produced.with_name(f"{stem_for(height, width)}.onnx")
    produced.replace(target)
    return target


def parse_size(token: str) -> tuple[int, int]:
    parts = [int(piece) for piece in token.replace("x", ",").split(",")]
    if len(parts) == 1:
        return parts[0], parts[0]
    if len(parts) == 2:
        return parts[0], parts[1]
    raise argparse.ArgumentTypeError(f"expected N or HxW, got {token!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imgsz", type=parse_size, nargs="*", default=None,
                        help="sizes to export as N or HxW (default: "
                             "480x640, 640, 1024)")
    parser.add_argument("--weights", type=Path, default=None,
                        help="checkpoint to export (default: weights/FastSAM.pt). "
                             "The stem drives the output name, so FastSAM-s.pt "
                             "yields FastSAM-s-<imgsz>.onnx.")
    return parser.parse_args()


def main() -> int:
    global WEIGHTS

    args = parse_args()
    if args.weights is not None:
        WEIGHTS = args.weights.expanduser().resolve()
    if not WEIGHTS.exists():
        print(f"checkpoint not found: {WEIGHTS}", file=sys.stderr)
        return 1
    print(f"checkpoint: {WEIGHTS}")

    for height, width in (args.imgsz or DEFAULT_SIZES):
        print(f"\n=== exporting {height}x{width} opset={OPSET} fp32 static "
              f"batch=1 ===")
        print(f"exported -> {export_one(height, width)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
