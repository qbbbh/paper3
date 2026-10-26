# Paper3：V2V RF/VLC 科研审查与可复现实验

这是待审查的开发快照，不是已完成论文或已证明正确的算法。先读 `MODEL_SPEC.md`、`PROJECT_STATUS.md` 和 `reviews/20261026_mean_chance_review.md`。

## 阅读顺序
1. MODEL_SPEC.md：用户确认的设定与尚未批准的研究扩展。
2. references/INDEX.md：参数来源和适用边界；不上传全文PDF。
3. code/：原样保存的基础层、数值分配器、学习代码及测试。
4. experiments/evidence/：四组10车、100块开发结果，保留原始CSV、NPZ、summary与压缩内层日志。
5. reviews/：独立复核，优先于 archive/ 中未更正的历史表述。
6. RUNBOOK.md：只运行已批准版本的执行流程。

`code/README.md` 是来源快照的说明，其中历史内容须与本仓库规格、复核记录核对。50/100车学习代码只作为开发资料，不代表验收通过。

首次检查：`python tools/verify_snapshot.py`。单测：在 code/ 下执行 `python -m pytest -q`。依赖记录见 requirements.txt；上传前本机验证结果见 reviews/package_validation.json。

GitHub负责版本和审查记录，不能保证Chat、Codex或Work自动遵守规则。正式执行还需固定源码、配置和数据，执行前后复算哈希。当前没有自动启动正式实验的工作流。
