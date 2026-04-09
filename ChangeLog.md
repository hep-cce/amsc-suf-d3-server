# Changelog

## [v0.2.0] - 2026-04-03

### New Features
- **Supernova models**: Added three new 2D CNN supernova detection models (`snbamsc_2dcnn_u`, `snbamsc_2dcnn_v`, `snbamsc_2dcnn_z`) served via TensorFlow SavedModel backend with GPU instance groups.

### Improvements
- **NuGraph2**: Updated `config.pbtxt` to enable GPU execution by adding an explicit `instance_group` with `KIND_GPU`, replacing the previously commented-out environment path configuration.
- **NuGraph2 model**: Enhanced `model.py` with better handling of graph data loading and inference.
- **Dockerfile**:
  - Consolidated `numba` installation into the main PyTorch/PyG dependency install step.
  - Removed outdated `LABEL` directives.
  - Added a post-install script that patches `nugraph` source files: replaces `__call__` with `forward` in classes inheriting `BaseTransform`, ensuring compatibility with newer PyTorch Geometric transforms API.

## [v0.1.0] - 2026-04-01

Initial release supporting both GNN (GNN4Pixel / DoubleMetricLearning) and NuGraph2 models on a Triton Inference Server backend.
