#!/usr/bin/env python3
"""Convert the repository's TensorFlow CUDA models to ONNX."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf


@dataclass(frozen=True)
class ModelSpec:
    source: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


MODELS = {
    "deepmet": ModelSpec(
        "model.graphdef",
        ("input", "input_cat0", "input_cat1", "input_cat2"),
        ("output/BiasAdd",),
    ),
    "deeptau_2018v2p5": ModelSpec(
        "model.graphdef",
        (
            "input_tau",
            "input_inner_egamma",
            "input_inner_muon",
            "input_inner_hadrons",
            "input_outer_egamma",
            "input_outer_muon",
            "input_outer_hadrons",
        ),
        ("main_output/Softmax",),
    ),
    "snbamsc_2dcnn_u": ModelSpec(
        "model.savedmodel", ("zero_padding2d_input",), ("dense_1",)
    ),
    "snbamsc_2dcnn_v": ModelSpec(
        "model.savedmodel", ("zero_padding2d_input",), ("dense_1",)
    ),
    "snbamsc_2dcnn_z": ModelSpec(
        "model.savedmodel", ("zero_padding2d_1_input",), ("dense_3",)
    ),
}


def tf2onnx_command(
    source: Path, target: Path, spec: ModelSpec, opset: int
) -> list[str]:
    command = [sys.executable, "-m", "tf2onnx.convert"]
    if source.suffix == ".graphdef":
        command += [
            "--graphdef",
            str(source),
            "--inputs",
            ",".join(f"{name}:0" for name in spec.inputs),
            "--outputs",
            ",".join(f"{name}:0" for name in spec.outputs),
            "--rename-inputs",
            ",".join(spec.inputs),
            "--rename-outputs",
            ",".join(spec.outputs),
        ]
    else:
        command += [
            "--saved-model",
            str(source),
            "--signature_def",
            "serving_default",
        ]
    return command + ["--opset", str(opset), "--output", str(target)]


def make_inputs(session: ort.InferenceSession) -> dict[str, np.ndarray]:
    generator = np.random.default_rng(12345)
    return {
        item.name: (
            np.zeros((2, *(int(size) for size in item.shape[1:])), dtype=np.float32)
            if item.name.startswith("input_cat")
            else generator.standard_normal(
                (2, *(int(size) for size in item.shape[1:])), dtype=np.float32
            )
        )
        for item in session.get_inputs()
    }


def tensorflow_outputs(
    source: Path, spec: ModelSpec, inputs: dict[str, np.ndarray]
) -> list[np.ndarray]:
    if source.suffix != ".graphdef":
        signature = tf.saved_model.load(str(source)).signatures["serving_default"]
        result = signature(
            **{name: tf.convert_to_tensor(value) for name, value in inputs.items()}
        )
        return [result[name].numpy() for name in spec.outputs]

    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(source.read_bytes())
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name="")
    feed = {
        graph.get_tensor_by_name(f"{name}:0"): value for name, value in inputs.items()
    }
    try:
        feed[graph.get_tensor_by_name(
            "batch_normalization_1/keras_learning_phase:0"
        )] = False
    except KeyError:
        pass
    with tf.compat.v1.Session(graph=graph) as session:
        return session.run(
            [graph.get_tensor_by_name(f"{name}:0") for name in spec.outputs], feed
        )


def verify(source: Path, target: Path, spec: ModelSpec) -> None:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(target), sess_options=options, providers=["CPUExecutionProvider"]
    )
    inputs = make_inputs(session)
    expected = tensorflow_outputs(source, spec, inputs)
    actual = session.run(list(spec.outputs), inputs)
    for expected_output, actual_output in zip(expected, actual):
        np.testing.assert_allclose(actual_output, expected_output, rtol=1e-4, atol=1e-5)


def update_config(repository: Path, name: str, digest: str) -> None:
    config_path = repository / name / "config.pbtxt"
    config = config_path.read_text()
    config, platform_updates = re.subn(
        r'platform: "(?:tensorflow_graphdef|tensorflow_savedmodel|onnxruntime_onnx)"',
        'platform: "onnxruntime_onnx"',
        config,
        count=1,
    )
    checksum = f'key: "MD5:1/model.onnx"\n      value: "{digest}"'
    config, checksum_updates = re.subn(
        r'key: "MD5:1/model\.(?:graphdef|onnx)"\s+value: "[0-9a-f]{32}"',
        checksum,
        config,
        count=1,
    )
    if not checksum_updates:
        config = config.rstrip() + f"""

model_repository_agents {{
  agents [{{
    name: "checksum"
    parameters {{
      {checksum}
    }}
  }}]
}}
"""
    if platform_updates != 1:
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
    source = version_dir / spec.source
    target = version_dir / "model.onnx"
    temporary = version_dir / "model.onnx.tmp"
    temporary.unlink(missing_ok=True)
    subprocess.run(tf2onnx_command(source, temporary, spec, opset), check=True)
    onnx.checker.check_model(onnx.load(str(temporary)))
    if check_outputs:
        verify(source, temporary, spec)
    temporary.replace(target)

    digest = hashlib.md5(target.read_bytes()).hexdigest()
    if update_model_config:
        update_config(repository, name, digest)
    print(f"{name}: {target} MD5={digest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository", type=Path, default=Path("cuda_models"), help="model repository"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=list(MODELS),
        help="models to convert",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--skip-output-check", action="store_true")
    parser.add_argument("--skip-config-update", action="store_true")
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
