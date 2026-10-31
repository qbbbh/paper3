# 风险界与不确定性感知资源分配：待核文献清单（2026-10-27）

本清单由 AI2 转述、经本次整理补上"能支持什么 / 不能支持什么"两列。
**状态一律为"待核"**：题名、venue、年份、卷期、页码、公式编号在正式引用前必须逐条
核对原文，按 [AGENTS.md](../AGENTS.md) 与 [INDEX.md](INDEX.md) 的要求处理。
本文件不是"这些结论已被证实"，只是"这些文献值得按什么顺序去核实"。

---

## 1. 直接对口的 V2V + 不完美 CSI

### [1] Robust Resource Allocation for Vehicular Communications With Imperfect CSI

- 作者：Wu, Liu, Yang, Quek
- 出处：IEEE Transactions on Wireless Communications, 2021
- DOI：https://doi.org/10.1109/TWC.2021.3070894
- 与本项目的关系：**最接近的一篇**。V2X 场景下的不完美 CSI、V2V 服务可靠性与资源分配。

| 能支持 | 不能支持 |
|---|---|
| "误差统计 → V2V 可靠服务约束"的建模范式 | 你的 RF/VLC 复合链路结构（本文无 VLC） |
| 已知误差分布与分布不完全已知两种情况的分层处理 | 你的"块内不轮换、块平均速率"协议 |
| 用一阶/二阶统计量构造分布鲁棒机会约束、用支持集降低保守性 | 你的 `mean_k(r_k/p_k)` 目标（本文不是逐链路 EE 算术平均） |

**借鉴点**：误差分布不完全已知时如何降保守，以及鲁棒保障的代价怎么量化。

---

## 2. 干扰网络中的误差集合鲁棒优化

### [2] Robust Beamforming in Interference Channels With Imperfect Transmitter Channel Information

- 出处：Signal Processing, 2012
- DOI：https://doi.org/10.1016/j.sigpro.2012.03.014
- 与本项目的关系：多条链路同频并行，误差落在椭球不确定集内，对集合内最坏情况做稳健速率与波束成形。

| 能支持 | 不能支持 |
|---|---|
| "误差不只影响本链路，也影响**其他链路的干扰**"这一结构性论点 | VLC 平方律信道（本文是 RF 线性增益） |
| 椭球不确定集 + 最坏情况的建模范式 | 你的逐链路 EE 目标与复合链路选择 |

**借鉴点**：这正是本项目拒绝"给每条链路独立加一个固定 dB 裕度"的理论依据。

---

## 3. 数据驱动的不确定集（替代人为 ±2 dB）

### [3] Data-Driven Robust Resource Allocation for Vehicular Communications via Deep Support Vector Clustering

- 出处：IEEE Transactions on Cognitive Communications and Networking, 2026
- DOI：https://doi.org/10.1109/TCCN.2026.3677218
- 与本项目的关系：从历史信道样本学习不确定集，再在其上优化资源；并研究如何降低传统不确定集的保守性。

| 能支持 | 不能支持 |
|---|---|
| "不确定集不必人为固定，可由数据与信道统计结构建立" | 你的块级条件轨迹预测库（本文不是轨迹预测） |
| 数据驱动降保守性的路线 | 你的 VLC 模式与物理层 |

**借鉴点**：如果将来要把"σ 族"换成"从数据学出来的误差集"，这是入口。

---

## 4. 多步 CSI 预测的风险控制

### [4] TRACE-CRC: Trajectory-Adaptive Conformal Risk Control for Multi-Step Channel State Information Prediction

- 作者：Rezaei 等
- 出处：Proceedings of Machine Learning Research, vol. 329
- 链接：https://proceedings.mlr.press/v329/rezaei26a.html
- 与本项目的关系：对**多步 CSI 预测轨迹**做联合不确定性与风险控制，而不只是单时刻误差区间。

| 能支持 | 不能支持 |
|---|---|
| "对预测轨迹做校准"的方法学先例 | 你的 RF/VLC 复合链路与逐链路 EE 目标 |
| 风险控制（而非仅区间覆盖）的表述方式 | 你的仿真数据可交换性（交通时间序列有自相关） |

**重要含义**：**"对 CSI 预测区间做校准"本身已不是空白研究。** 本项目的差异必须落在
"校准后的服务风险如何改变 RF/VLC 复合链路选择、连续功率与最低通信对 EE"，
而不是"我们把区间校准得更准"。

---

## 5. 最低个体 EE 优化（E4 的理论基础）

### [5] 最低个体能效 / 最小化最大功耗类的分式规划工作

- 作者：Zappone 等
- 出处：arXiv:1503.04609 — https://arxiv.org/abs/1503.04609
- 与本项目的关系：为"最低个体 EE 优化"提供成熟的非凸功率控制理论基础。

| 能支持 | 不能支持 |
|---|---|
| max-min 型 EE 问题的非凸性与常用解法（分式规划/SCA） | "在平均 EE 损失预算 δ 下改善最低 EE"的具体算法 |
| 最低 EE 作为公平指标的合法性 | 你的复合链路（匹配/方向/模式）离散层 |

**注意**：本项目 E4 的目标是**带 δ 预算的 max-min**，不是纯 max-min。
纯 max-min 会牺牲平均 EE，δ 的作用就是把这个牺牲框住。

---

## 6. 图神经网络用于可扩展功率控制（E5-100 车）

### [6] A Graph Neural Network Approach for Scalable Wireless Power Control

- 作者：Shen 等
- 出处：arXiv:1907.08487 — https://arxiv.org/abs/1907.08487
- 与本项目的关系：用干扰图表达邻居耦合，做可扩展的功率控制。

| 能支持 | 不能支持 |
|---|---|
| 图结构在干扰网络中的可扩展性论证 | 你的复合链路（模式/方向）离散动作 |
| GNN 作为 100 车编码器的合理性 | 你的不确定性输入（本文无 CSI 不确定性） |

**借鉴点**：本项目的 v2 编码器在干扰关系上另加"最弱边摘要"与"不确定性特征"，
差异要写清，不能只说"我们也用了 GNN"。

---

## 7. Conformal Prediction 的方法与前提

### [7] A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification

- 作者：Angelopoulos, Bates
- 出处：arXiv:2107.07511 — https://arxiv.org/abs/2107.07511

| 能支持 | 不能支持 |
|---|---|
| 边际覆盖的构造与"无分布假设"的准确含义 | 时间相关数据上的直接套用 |
| 有限样本次序统计量的正确用法 | 交通轨迹的自相关场景下的覆盖保证 |

### [8] Conformal Risk Control

- 作者：Angelopoulos 等
- 出处：arXiv:2208.02814（式 (4)、§2.1、§2.3）
- 链接：https://arxiv.org/html/2208.02814v4

| 能支持 | 不能支持 |
|---|---|
| 单调损失下由覆盖控制推广到风险控制的框架 | "CRC 一定比 Conformal 更强"（0/1 损失下二者重合） |
| 为什么改损失函数会改变风险目标 | 固定动作上的单调性推广到重新优化后的策略风险 |

### [9] Data-Driven Chance Constrained Optimization under Wasserstein Ambiguity Sets

- 作者：Hota, Cherukuri, Lygeros
- 出处：arXiv:1805.06729 — https://arxiv.org/abs/1805.06729

| 能支持 | 不能支持 |
|---|---|
| 一维 W1 球下 $\sup \Pr(S>c)$ 的精确计算 | 完整信道不确定集下的混合资源分配 DRO |
| 半径作为保守性旋钮的含义 | "真实分布一定在球内"（当前无证明） |

---

## 核验清单（正式引用前逐条打勾）

- [ ] [1] 卷期页码、误差模型的具体假设、"分布不完全已知"那一节用的是几阶统计量
- [ ] [2] 不确定集是椭球还是盒式；最坏情况是针对 SINR 还是速率
- [ ] [3] 2026 年的 DOI 是否已正式见刊；不确定集的学习对象是信道还是干扰
- [ ] [4] PMLR 卷号与页码；"轨迹自适应"具体指什么；是否要求可交换性
- [ ] [5] arXiv 编号对应的是哪一篇正式论文；是否含逐链路 EE 的算术平均
- [ ] [6] 图结构定义（节点/边/消息）；是否处理离散模式选择
- [ ] [7][8] 有限样本公式的精确形式与适用条件（本项目用的是块级校准单元）
- [ ] [9] 一维情形的上确界公式与半径定义（本项目用的是自然对数残差尺度）

## 与本项目结论的直接对应

| 本项目结论 | 依赖哪条文献的哪一部分 | 不能声称的部分 |
|---|---|---|
| 不能给每条链路独立加固定裕度 | [2] 干扰链路也会偏移 | 本文没有 VLC |
| 不确定集可以由数据学出来 | [3] | 本文不是块级轨迹预测 |
| 校准本身不是创新点 | [4][7][8] | 本文没有 RF/VLC 与 EE 目标 |
| 最低 EE 需要 δ 预算 | [5] | 本文不是带预算的 max-min |
| 整块可靠要另立联合约束 | [8][9] 的风险目标区分 | 本项目尚未实现联合约束 |
