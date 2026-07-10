#!/usr/bin/env python3
"""Convert the repository's ParticleNet TorchScript models to ONNX."""

from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch


INPUT_NAMES = (
    "pf_points__0",
    "pf_features__1",
    "pf_mask__2",
    "sv_points__3",
    "sv_features__4",
    "sv_mask__5",
)


@dataclass(frozen=True)
class ModelSpec:
    pf_features: int
    sv_features: int
    output: str


MODELS = {
    "particlenet_PT": ModelSpec(25, 11, "softmax__0"),
    "particlenet_AK8_MD-2prong_PT": ModelSpec(20, 11, "softmax__0"),
    "particlenet_AK8_MassRegression_PT": ModelSpec(25, 11, "output__0"),
}


def make_inputs(
    spec: ModelSpec, batch: int, pf_particles: int, sv_particles: int
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(12345)
    randn = lambda *shape: torch.randn(*shape, generator=generator)
    return (
        randn(batch, 2, pf_particles),
        randn(batch, spec.pf_features, pf_particles),
        torch.ones(batch, 1, pf_particles),
        randn(batch, 2, sv_particles),
        randn(batch, spec.sv_features, sv_particles),
        torch.ones(batch, 1, sv_particles),
    )


def dynamic_axes(output_name: str) -> dict[str, dict[int, str]]:
    axes = {name: {0: "batch"} for name in INPUT_NAMES}
    for name in INPUT_NAMES[:3]:
        axes[name][2] = "pf_particles"
    for name in INPUT_NAMES[3:]:
        axes[name][2] = "sv_particles"
    axes[output_name] = {0: "batch"}
    return axes


def verify(
    model: torch.jit.ScriptModule, onnx_path: Path, spec: ModelSpec
) -> None:
    inputs = make_inputs(spec, batch=2, pf_particles=37, sv_particles=11)
    with torch.no_grad():
        expected = model(*inputs).detach().cpu().numpy()

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    actual = session.run(
        [spec.output],
        {name: tensor.numpy() for name, tensor in zip(INPUT_NAMES, inputs)},
    )[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def update_config(repository: Path, name: str, digest: str) -> None:
    config_path = repository / name / "config.pbtxt"
    config = config_path.read_text()
    config, platform_updates = re.subn(
        r'platform: "(?:pytorch_libtorch|onnxruntime_onnx)"',
        'platform: "onnxruntime_onnx"',
        config,
        count=1,
    )
    config, checksum_updates = re.subn(
        r'key: "MD5:1/model\.(?:pt|onnx)"\s+value: "[0-9a-f]{32}"',
        f'key: "MD5:1/model.onnx"\n      value: "{digest}"',
        config,
        count=1,
    )
    if platform_updates != 1 or checksum_updates != 1:
        raise RuntimeError(f"could not update {config_path}")
    config_path.write_text(config)


def convert(
    repository: Path,
    name: str,
    opset: int,
    check_outputs: bool,
    update_model_config: bool,
) -> None:
    spec = MODELS[name]
    version_dir = repository / name / "1"
    torchscript_path = version_dir / "model.pt"
    onnx_path = version_dir / "model.onnx"
    temporary_path = version_dir / "model.onnx.tmp"

    model = torch.jit.load(str(torchscript_path), map_location="cpu").eval()
    sample = make_inputs(spec, batch=1, pf_particles=32, sv_particles=8)
    torch.onnx.export(
        model,
        sample,
        str(temporary_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=list(INPUT_NAMES),
        output_names=[spec.output],
        dynamic_axes=dynamic_axes(spec.output),
        dynamo=False,
    )

    converted = onnx.load(str(temporary_path))
    onnx.checker.check_model(converted)
    temporary_path.replace(onnx_path)

    if check_outputs:
        verify(model, onnx_path, spec)

    digest = hashlib.md5(onnx_path.read_bytes()).hexdigest()
    if update_model_config:
        update_config(repository, name, digest)
    print(f"{name}: {onnx_path} MD5={digest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository", type=Path, default=Path("cuda_models"), help="model repository"
    )
    parser.add_argument(
        "--models", nargs="+", choices=MODELS, default=list(MODELS), help="models to convert"
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--skip-output-check",
        action="store_true",
        help="skip the TorchScript versus ONNX Runtime numerical comparison",
    )
    parser.add_argument(
        "--skip-config-update",
        action="store_true",
        help="do not switch config.pbtxt to ONNX Runtime or update its checksum",
    )
    args = parser.parse_args()

    for name in args.models:
        convert(
            args.repository,
            name,
            args.opset,
            not args.skip_output_check,
            not args.skip_config_update,
        )


if __name__ == "__main__":
    main()
