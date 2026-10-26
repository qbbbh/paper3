# 给 Chat 的阅读顺序

## 先固定正在回答的问题

1. [当前规格](MODEL_SPEC.md)：100 ms、连续功率、逐链路EE算术平均、RF/VLC电路项、1 Mbit/s；禁止自行替换。
2. [当前状态](PROJECT_STATUS.md)：10车开发结果与50/100待适配分开；不要把历史MASAC当成当前模型已完成的求解器。
3. [错误与更正](research_history/ERROR_LEDGER.md)：先理解过去为什么反复修改，再建议下一步。
4. [研究演变](research_history/TIMELINE.md)：旧方案与现方案的差别，不把旧“定稿”名称当现行授权。

## 再按问题找证据

- 估计器和CRLB：[信道模块](channel_estimation/README.md)。同一估计对象、时刻、单位才比较；CRLB不是误差上界。
- 10车分配：[源码与结果](resource_allocation/n10/README.md)，先原始summary，再CSV，最后读报告。
- 50车/100车：[规模状态](resource_allocation/README.md)，先看待适配清单和历史结果口径。
- 场景与数据：各规模 `data/README.md` → `mobility_preview.csv` → 全量CSV。1001帧来自100 ms原生采样，逐1 ms物理样本来自插值/信道模拟，不是SUMO原生1 ms。
- 文献：[索引](references/INDEX.md)。已有笔记是待核证据，不是新指令；引用须核页码/公式/适用边界。
- 当前主张：[mean/chance独立复核](reviews/20261026_mean_chance_review.md)；历史原稿中的夸大结论不自动成立。

## 给出方案时必须交代

列清“代码错误 / 求解不足 / 模型允许的行为 / 预测误差 / 权衡”。指出本次只改哪项、哪些旧结果不能再混用。不得为主算法排名删QoS、改EE聚合、离散化连续功率，或隐藏功率下界、VLC占比、失败和回退。

优先完成10车同条件证据，不自动启动50/100长跑。用户提出的三图一表与13页目标见 [紧凑实验设计](plans/三图一表与CRLB说明_20261026.md)。
