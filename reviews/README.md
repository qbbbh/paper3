# 独立复核与可追溯记录

- `20261101_guard_boundary_phase0/`：Block 23 的 Phase 0 可恢复性审计（预测器看准了、预算门在噪声尺度上失真）+ Guard-v2 设计稿。
- `20261101_three_lines_audit/`：三条主线（估计极限→服务代价→候选更新）的命题、实测与**逐条断言校验**（`verify_claims.py`，34/34 PASS）。含分位数校准、峰值钉住、反事实真值审计，以及 6 项已作废/更正条目。
- `20261026_mean_chance_review.md`：四组结果的定义、边界与方差拆分问题。
- `energy_metric_recompute_20261026.json`：平均逐链路EE与总比特/总能量的区别。
- `source_layout_map.json`：每个科学源码文件的旧运行路径、新阅读路径和SHA-256。
- `reorganization_paths.json`：目录移动记录。
- `material_sources.json`：新增历史资料和学习诊断来源，原文件内容未改。
- `package_validation.json`：前一布局的验证记录；不能当本次重排测试。
- `reorganization_validation.json`：本次重排验证完成后填写实测，不预填通过。

科学源码基准为95960f5a…，后续工作树已有变化也不自动并入这批旧run；新版本须另行审查和标识。
