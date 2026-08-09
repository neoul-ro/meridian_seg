#!/usr/bin/env python3
"""Build FastSAM-<imgsz>.engine from FastSAM-<imgsz>.onnx. FP16, static shape.

    python3 scripts/build_engine.py

No arguments needed. Detects the installed TensorRT, builds through the Python
API when bindings are importable and falls back to trtexec when they are not
(a venv on Jetson without --system-site-packages), prepends the metadata header
ultralytics needs, then deserializes the result and prints the real I/O tensors
as proof.

Run this on the machine that will run inference. A TensorRT engine is compiled
and kernel-autotuned against one specific GPU architecture, TensorRT version and
CUDA version -- the builder times candidate kernels on the actual device, so an
engine from a different GPU cannot be deserialized. The .onnx is the portable
artifact; the .engine never is.

On Jetson, put the board in its top power mode first (this script will not sudo):

    sudo nvpmodel -m 0 && sudo jetson_clocks
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wrap_engine_for_ultralytics import has_header, wrap  # noqa: E402

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = PACKAGE_ROOT / "weights"
TRTEXEC_CANDIDATES = (
    Path("/usr/src/tensorrt/bin/trtexec"),
    Path("/usr/local/tensorrt/bin/trtexec"),
)


def import_trt():
    try:
        import tensorrt  # noqa: PLC0415

        return tensorrt
    except ImportError:
        return None


def find_trtexec() -> Path | None:
    for candidate in TRTEXEC_CANDIDATES:
        if candidate.is_file():
            return candidate
    found = shutil.which("trtexec")
    return Path(found) if found else None


def build_with_api(trt, onnx: Path, workspace_mib: int) -> tuple[bytes, tuple[int, int]]:
    """Serialize an FP16 engine from onnx, plus the (h, w) it was built for.

    Works on TensorRT 8 and 10+.
    """
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    # EXPLICIT_BATCH was mandatory in TRT 8 and removed in TRT 10, where it is
    # the only behaviour.
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)

    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        for i in range(parser.num_errors):
            print(f"  onnx parse error: {parser.get_error(i)}", file=sys.stderr)
        sys.exit("failed to parse the onnx graph")

    # Take the size from the graph rather than a flag, so the metadata header can
    # never disagree with what the engine actually accepts.
    shape = tuple(network.get_input(0).shape)
    if len(shape) != 4:
        sys.exit(f"expected a 4D NCHW input, got {shape}")
    imgsz = (int(shape[2]), int(shape[3]))

    config = builder.create_builder_config()
    # max_workspace_size in TRT 8.0-8.3, set_memory_pool_limit from 8.4 on.
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, workspace_mib << 20
        )
    else:
        config.max_workspace_size = workspace_mib << 20

    if not builder.platform_has_fast_fp16:
        print("  this device has no fast fp16 -- building fp32 instead")
    else:
        config.set_flag(trt.BuilderFlag.FP16)

    print(f"  building for input {shape} ({imgsz[0] * imgsz[1]:,} px)")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        sys.exit(
            "the builder returned nothing. Out of memory is the usual cause -- "
            "retry with a smaller --workspace."
        )
    return bytes(serialized), imgsz


def build_with_trtexec(trtexec: Path, onnx: Path, raw: Path,
                       trt_version: tuple[int, int], workspace_mib: int) -> bytes:
    flags = [f"--onnx={onnx}", f"--saveEngine={raw}", "--fp16"]
    # The workspace flag was renamed in TRT 8.4; the old spelling is gone in 10.
    if trt_version >= (8, 4):
        flags.append(f"--memPoolSize=workspace:{workspace_mib}")
    else:
        flags.append(f"--workspace={workspace_mib}")
    if trt_version >= (8, 6):
        flags.append("--builderOptimizationLevel=3")

    print(f"\n$ {trtexec} {' '.join(flags)}\n")
    result = subprocess.run([str(trtexec), *flags])
    if result.returncode != 0:
        sys.exit(f"trtexec failed with exit code {result.returncode}")
    if not raw.is_file():
        sys.exit(f"trtexec reported success but {raw} is missing")
    blob = raw.read_bytes()
    raw.unlink()
    return blob


def trtexec_version(trtexec: Path) -> tuple[int, int]:
    try:
        out = subprocess.run([str(trtexec), "--version"], capture_output=True,
                             text=True, timeout=60)
        text = (out.stdout + out.stderr).replace("[", " ").replace("]", " ")
        for token in text.split():
            if token.count(".") >= 2 and token[0].isdigit():
                parts = token.split(".")
                return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return (8, 6)  # conservative: the flags this picks work on 8.4 through 10


def verify(trt, engine_path: Path) -> int:
    import json

    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as handle, trt.Runtime(logger) as runtime:
        meta_len = int.from_bytes(handle.read(4), byteorder="little")
        metadata = json.loads(handle.read(meta_len).decode("utf-8"))
        engine = runtime.deserialize_cuda_engine(handle.read())

    if engine is None:
        print(f"\nFAILED to deserialize {engine_path}")
        return 1

    print(f"\nverified {engine_path.name}")
    print(f"  metadata: task={metadata['task']} imgsz={metadata['imgsz']} "
          f"stride={metadata['stride']}")
    if hasattr(engine, "num_io_tensors"):  # TRT 10+
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            is_in = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            print(f"  {'in ' if is_in else 'out'} {name} "
                  f"{tuple(engine.get_tensor_shape(name))} "
                  f"{engine.get_tensor_dtype(name)}")
    else:  # TRT 8.x
        for i in range(engine.num_bindings):
            is_in = engine.binding_is_input(i)
            print(f"  {'in ' if is_in else 'out'} {engine.get_binding_name(i)} "
                  f"{tuple(engine.get_binding_shape(i))} "
                  f"{engine.get_binding_dtype(i)}")
    return 0


def imgsz_from_name(onnx: Path) -> tuple[int, int]:
    """Recover (h, w) from FastSAM-480x640.onnx / FastSAM-640.onnx.

    Only needed on the trtexec path, where there is no parser to ask.
    """
    tail = onnx.stem.rsplit("-", 1)[-1]
    try:
        if "x" in tail:
            height, width = (int(piece) for piece in tail.split("x"))
            return height, width
        side = int(tail)
        return side, side
    except ValueError:
        sys.exit(
            f"cannot tell the input size from {onnx.name}. Name it "
            "FastSAM-<h>x<w>.onnx or FastSAM-<n>.onnx, or pass --imgsz."
        )


def resolve_onnx(explicit: Path | None, requested: str | None) -> Path:
    if explicit is not None:
        return explicit
    if requested is not None:
        parts = [int(piece) for piece in requested.replace("x", ",").split(",")]
        height, width = (parts[0], parts[0]) if len(parts) == 1 else parts[:2]
        stem = f"FastSAM-{height}" if height == width else f"FastSAM-{height}x{width}"
        return WEIGHTS_DIR / f"{stem}.onnx"
    # 480x640 matches a 640x480 camera under rectangular inference, which is what
    # the .pt path does, so prefer it when it exists.
    for name in ("FastSAM-480x640.onnx", "FastSAM-640.onnx"):
        candidate = WEIGHTS_DIR / name
        if candidate.is_file():
            return candidate
    found = sorted(WEIGHTS_DIR.glob("*.onnx"))
    return found[0] if found else WEIGHTS_DIR / "FastSAM-480x640.onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imgsz", default=None,
                        help="which FastSAM-<imgsz>.onnx to build, N or HxW "
                             "(default: 480x640 if present, else 640)")
    parser.add_argument("--onnx", type=Path, default=None,
                        help="explicit .onnx path, overrides --imgsz lookup")
    parser.add_argument("--workspace", type=int, default=2048,
                        help="builder workspace cap in MiB (default: 2048)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    onnx = resolve_onnx(args.onnx, args.imgsz)
    if not onnx.is_file():
        available = sorted(p.name for p in WEIGHTS_DIR.glob("*.onnx"))
        sys.exit(f"onnx not found: {onnx}\n  available in {WEIGHTS_DIR}: {available}")
    engine = onnx.with_suffix(".engine")

    trt = import_trt()
    trtexec = find_trtexec()

    print(f"onnx         {onnx.name} ({onnx.stat().st_size / 1e6:.1f} MB)")
    print(f"engine       {engine.name}  (FP16, static shape)")

    if trt is not None:
        print(f"TensorRT     {trt.__version__} (python API)")
        print("\nbuilding -- kernel autotuning takes several minutes\n")
        blob, imgsz = build_with_api(trt, onnx, args.workspace)
    elif trtexec is not None:
        version = trtexec_version(trtexec)
        imgsz = imgsz_from_name(onnx)
        print(f"TensorRT     {version[0]}.{version[1]} (trtexec, no python bindings)")
        print("\nbuilding -- kernel autotuning takes several minutes\n")
        blob = build_with_trtexec(trtexec, onnx, onnx.with_suffix(".raw.engine"),
                                  version, args.workspace)
    else:
        sys.exit(
            "no TensorRT found -- neither python bindings nor trtexec.\n"
            "  x86:    pip install tensorrt-cu12\n"
            "  Jetson: it ships with JetPack at /usr/src/tensorrt/bin/trtexec.\n"
            "          Inside a venv, python bindings need --system-site-packages;\n"
            "          `pip install tensorrt` does not work on Jetson."
        )

    engine.write_bytes(blob if has_header(blob) else wrap(blob, imgsz))
    print(f"\nwrote {engine}  ({engine.stat().st_size / 1e6:.1f} MB, "
          f"metadata header + engine)")

    rc = verify(trt, engine) if trt is not None else 0
    if trt is None:
        print("\nskipping verification: no tensorrt python bindings here "
              "(the engine itself is fine)")

    if rc == 0:
        # seg_node takes no image_size: it reads the engine's own input shape and
        # derives the letterbox geometry from the first frame it receives.
        print("\nrun the node against it:")
        print("  ros2 run meridian_seg seg_node --ros-args \\")
        print(f"    -p model_path:={engine}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
