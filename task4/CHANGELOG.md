# Changelog

All notable changes to this task will be documented in this file.

## [1.1.0] - 2026-02-24

### Updated
- 数据生成：使用sigmoid函数生成有意义的标签（Task1: age+income, Task2: city+age）
- Task2信号增强：city系数0.5, age系数0.15
- 模型参数：n_expert=16, expert_dims=[256,128], tower_dims=[128,64], lr=0.001, n_epoch=200

### Fixed
- AUC从~0.5提升至0.72/0.80

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
