# Changelog

All notable changes to this task will be documented in this file.

## [1.0.0] - 2026-02-24

### Added
- MMOE多任务学习完整示例
- 支持CTR/CVR双任务训练
- ONNX模型导出功能
- 预测结果CSV保存

### Fixed
- RuntimeError: tensor dimension mismatch (city embed_dim: 16→32)
- KeyError: y data format (dict→2D numpy array)
- AttributeError: y needs numpy array not list
- RuntimeError: model_path directory not exists (removed parameter)
- AttributeError: predict bug (manual inference loop)

### Dependencies
- torch_rechub
- numpy
- pandas
- torch
- joblib
