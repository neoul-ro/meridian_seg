#!/usr/bin/env python3
"""Prepend the ultralytics metadata header to a trtexec-built .engine file.

Run this ON THE JETSON, right after trtexec. Pure stdlib -- no onnx, no torch,
no ultralytics -- so it works on a bare JetPack image. build_engine_jetson.py
imports wrap() from here, so normally you do not need to call this by hand.

Why this is needed
------------------
ultralytics reads an engine as [4-byte LE length][JSON metadata][engine]:

    meta_len = int.from_bytes(f.read(4), byteorder='little')
    metadata = json.loads(f.read(meta_len).decode('utf-8'))
    model = runtime.deserialize_cuda_engine(f.read())

A raw `trtexec --saveEngine` file has no header, so those first 4 bytes are
engine payload. Modern ultralytics (8.4.x, nn/backends/tensorrt.py:53) catches
the resulting UnicodeDecodeError and recovers, but the vendored 8.0.120 in
FastSAM_official (nn/autobackend.py:164) does not -- json.loads() blows up on
garbage. Writing the header satisfies both, so it is always the safe move.

The values mirror what the ONNX export embedded (task=segment, stride=32,
names={0: 'object'}), which is what ultralytics reads back to set up
letterboxing and postprocessing.

Usage
-----
    python3 wrap_engine_for_ultralytics.py \
        --engine FastSAM-640.raw.engine \
        --out    FastSAM-640.engine \
        --imgsz  640
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Mirrors ultralytics 8.0.120 Exporter.metadata for this checkpoint. The loader
# int()s stride/batch and eval()s imgsz/names, so the string forms are expected.
METADATA = {
    "description": "Ultralytics YOLOv8x-seg model (untrained)",
    "author": "Ultralytics",
    "license": "AGPL-3.0 https://ultralytics.com/license",
    "version": "8.0.120",
    "stride": "32",
    "task": "segment",
    "batch": "1",
    "names": "{0: 'object'}",
}


def has_header(raw: bytes) -> bool:
    """True if raw already starts with a metadata header.

    A wrapped file opens with a small length followed by '{'. A serialized
    engine effectively never does, so this is a safe double-wrap guard.
    """
    head = int.from_bytes(raw[:4], byteorder="little", signed=True)
    return 0 < head < 4096 and raw[4:5] == b"{"


def wrap(raw: bytes, imgsz: int | tuple[int, int]) -> bytes:
    """Return raw with the ultralytics metadata header prepended.

    imgsz may be a single int for a square engine or an (h, w) pair.
    """
    height, width = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
    meta = dict(METADATA, imgsz=f"[{height}, {width}]")
    blob = json.dumps(meta).encode("utf-8")
    return len(blob).to_bytes(4, byteorder="little", signed=True) + blob + raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, type=Path,
                        help="raw engine from trtexec --saveEngine")
    parser.add_argument("--out", required=True, type=Path,
                        help="output engine with metadata header")
    parser.add_argument("--imgsz", required=True,
                        help="input size the engine was built for, N or HxW")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    raw = args.engine.read_bytes()
    if not raw:
        print(f"empty engine file: {args.engine}")
        return 1
    if has_header(raw):
        print(f"{args.engine} already has a metadata header -- nothing to do")
        return 1

    parts = [int(piece) for piece in args.imgsz.replace("x", ",").split(",")]
    imgsz = parts[0] if len(parts) == 1 else (parts[0], parts[1])

    args.out.write_bytes(wrap(raw, imgsz))
    print(f"wrote {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")
    print(f"  imgsz={imgsz} task=segment stride=32")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
