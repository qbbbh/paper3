# 改进前结果与研究转变

本目录保留新增功率协调阶段之前的跨规模分析及原数值，解释后来为何改进连续功率求解。它不是新版主方法的结果目录。

- [旧跨规模报告](10车与50车_统一性能功率统计及差距分析_20261110.md)：比较原10车和50车结果，记录功率、模式使用、服务与数值搜索差距。
- `results.json`、`performance.csv`、`paired.csv`、`mode_power.csv`及`10车50车_核心统计.csv`：保留原比较与统计口径。
- `saved_physics_summary.json`、`same_weak_link_physics_50.csv`、`weakest_links.csv`：保留固定连接和模式下的弱链路信号、干扰及功率诊断。
- `rf_uniform_power_doubling_posthoc_20261110.json`：旧动作、固定信道下的RF功率缩放诊断，不是新版算法结果，也不是重新仿真所得的独立场景。

研究过程从“50车原功率搜索不足”推进到“保留原动作，增加连续功率候选，再用原预测评价与约束决定是否接受”。最新效果见[新版主方法报告](../../reports/10车与50车_功率改进全量核查_20261110.md)和[新版同拓扑报告](../../reports/同拓扑消融_补算复用与全量结论.md)。旧诊断仍可说明改进动机，但不能替代新版结果，也不证明新版在每个场景或指标上都占优。

旧分析与报告生成脚本已从当前发布目录移除，数据和报告未改；实现可从[固定历史提交](https://github.com/qbbbh/paper3/tree/6ae23c04c091edee70e228cae39d72efb2af04a8/evidence/paper3_20261110/baseline_records/pre_guard_cross_scale)追溯。原来源及SHA-256保留在[来源清单](../../SOURCE_MANIFEST.json)中。
