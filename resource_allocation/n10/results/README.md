# 四组10车开发证据

候选mean、候选chance、联合mean、联合chance各100块，源码均为 `95960f5a…`。不是不同交通轨迹的独立正式泛化测试。

每个run包含 `summary.json`（配置/汇总/来源）、`results.csv`（逐链路及重复的块字段）、`fine_samples.npz`（真值复算细样本）、`continuous_power_audit.json.gz`（求解审计）。大文件先运行根目录的准备工具还原。

CSV的块总量在5条链路行中重复。汇总块功率/吞吐/耗时时必须按块取一次，不能再乘5。机会筛选失败后的mean回退必须单列；候选chance有4块回退，联合chance本快照无回退。

优先参考 [独立复核](../../../reviews/20261026_mean_chance_review.md) 和run原配置，不引用历史稿中不同功率/EE/QoS口径的绝对值作同条件比较。
