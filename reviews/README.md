# 独立复核与可追溯记录

- `20261102_guard_r_feasibility/`：Guard-R 候选可行性恢复（历史记录）。**已被外部复核纠错（控制组非等预算、判据用错、预算口径混用），不作为结论引用**；纠正版见 `20261102_feasible_first_joint/`。
- `20261102_three_lines_positioning/`：三条主线的定位说明（v1）——每条主线能写什么、不能写什么、三处必须更正的表述。
- `20261102_guard_validation_seed2026102801/`：Guard 独立信道种子验证。**判决：不通过，停止 Guard-v2**（rho 窗口未复现；真正该采纳的候选本来就预测预算通过，卡在预测 min EE）。
- `20261101_guard_boundary_phase0/`：Block 23 的 Phase 0 + 源码核对 + 标定分析。结论：预测看见了、干扰图无漏边、预算余量的预测分布已标定（sd(z)=1.03），因此应以 rho 为风险的分布判定替代点估计（决策曲线见 calibration.txt）。
- `20261101_three_lines_audit/`：三条主线（估计极限→服务代价→候选更新）的命题、实测与**逐条断言校验**（`verify_claims.py`，47/47 PASS）。含分位数校准、峰值钉住、反事实真值审计，以及 6 项已作废/更正条目。
- `20261026_mean_chance_review.md`：四组结果的定义、边界与方差拆分问题。
- `energy_metric_recompute_20261026.json`：平均逐链路EE与总比特/总能量的区别。
- `source_layout_map.json`：每个科学源码文件的旧运行路径、新阅读路径和SHA-256。
- `reorganization_paths.json`：目录移动记录。
- `material_sources.json`：新增历史资料和学习诊断来源，原文件内容未改。
- `package_validation.json`：前一布局的验证记录；不能当本次重排测试。
- `reorganization_validation.json`：本次重排验证完成后填写实测，不预填通过。

科学源码基准为95960f5a…，后续工作树已有变化也不自动并入这批旧run；新版本须另行审查和标识。
