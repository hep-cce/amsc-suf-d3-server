#!/usr/bin/env python3
"""Check TorchScript (.pt) and ONNX (.onnx) models for invalid BatchNorm nodes.

Usage:
    python check_batchnorm.py model.pt [model2.onnx ...]
    python check_batchnorm.py cuda_models/*/1/model.*

Exit codes:
    0 — all models OK
    1 — at least one model has an invalid BatchNorm node
"""

import argparse
import sys
from pathlib import Path


def check_torchscript(path):
    import torch

    model = torch.jit.load(path, map_location="cpu")
    graph_str = str(model.inlined_graph)
    lines = graph_str.split("\n")

    constants = {}
    for line in lines:
        stripped = line.strip()
        if "prim::Constant[value=" in stripped and ":" in stripped:
            var = stripped.split(":")[0].strip()
            val_start = stripped.index("value=") + len("value=")
            val_end = stripped.index("]", val_start)
            constants[var] = stripped[val_start:val_end]

    findings = []
    for i, line in enumerate(lines):
        if "aten::batch_norm(" not in line:
            continue
        args_str = line[line.index("(") + 1 : line.rindex(")")]
        args = [a.strip() for a in args_str.split(",")]
        if len(args) < 6:
            continue
        training_var = args[5]
        training_val = constants.get(training_var)
        if training_val == "1":
            findings.append(
                {
                    "line": i + 1,
                    "var": training_var,
                    "output": line.strip().split("=")[0].strip(),
                }
            )
    return findings


def check_onnx(path):
    import onnx

    model = onnx.load(path)
    opset = next((o.version for o in model.opset_import if not o.domain), 0)
    findings = []
    for node in model.graph.node:
        if node.op_type != "BatchNormalization":
            continue
        training_mode = next(
            (attr.i for attr in node.attribute if attr.name == "training_mode"), 0
        )
        if training_mode == 1:
            findings.append({"node": node.name, "opset": opset, "reason": "training"})
        elif opset >= 14 and len(node.output) != 1:
            findings.append(
                {"node": node.name, "opset": opset, "reason": "inference_outputs"}
            )
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", help="Paths to .pt or .onnx model files")
    args = parser.parse_args()

    has_error = False

    for model_path in args.models:
        p = Path(model_path)
        if not p.exists():
            print(f"SKIP  {p} (not found)")
            continue

        suffix = p.suffix.lower()
        if suffix == ".pt":
            findings = check_torchscript(str(p))
            if findings:
                has_error = True
                print(f"FAIL  {p}: {len(findings)} batch_norm node(s) with training=True")
                for f in findings[:3]:
                    print(f"        IR line {f['line']}: {f['output']}")
                if len(findings) > 3:
                    print(f"        ... and {len(findings) - 3} more")
            else:
                print(f"OK    {p}")

        elif suffix == ".onnx":
            findings = check_onnx(str(p))
            if findings:
                has_error = True
                opset = findings[0]["opset"]
                print(
                    f"FAIL  {p}: {len(findings)} invalid BatchNormalization "
                    f"node(s) (opset {opset})"
                )
                training = [f for f in findings if f["reason"] == "training"]
                inference_outputs = [
                    f for f in findings if f["reason"] == "inference_outputs"
                ]
                if training:
                    print(
                        f"        {len(training)} node(s) have training_mode=1"
                    )
                if inference_outputs:
                    print(
                        f"        {len(inference_outputs)} inference node(s) have "
                        "more than one output"
                    )
            else:
                print(f"OK    {p}")

        else:
            print(f"SKIP  {p} (unsupported format)")

    return 1 if has_error else 0


if __name__ == "__main__":
    sys.exit(main())
