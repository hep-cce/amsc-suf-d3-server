# Model conversions

## ParticleNet TorchScript to ONNX

Run the converter from the repository root in an environment containing
PyTorch 2.12:

```bash
python3 -m pip install -r converions/requirements-particlenet.txt
python3 converions/convert_particlenet.py
```

The converter writes `model.onnx` beside each source `model.pt`, validates the
ONNX graph, tests dynamic batch and particle dimensions, compares ONNX Runtime
output with TorchScript output on CPU, and updates the Triton platform and
checksum in `config.pbtxt`.

The source `.pt` files are intentionally retained for reproducibility and
numerical comparisons.

## TensorFlow to ONNX

Run the converter with Python 3.11:

```bash
python3 -m pip install -r converions/requirements-tensorflow.txt
python3 converions/convert_tensorflow.py
```

The converter handles both frozen GraphDefs and SavedModels, preserves the
Triton input and output names, validates each ONNX graph, compares its output
with TensorFlow for a dynamic batch, and updates the Triton platform and
checksum. The TensorFlow sources are retained for reproducibility.
