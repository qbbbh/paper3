# 核心阅读路线与引用边界

|问题|优先材料|要查什么|
|---|---|---|
|与原Paper3的继承关系|旧写作包参考文献.bib的R01：Optimization algorithm for energy efficiency in vehicular heterogeneous VLC/RF systems based on channel estimation；doi:10.1016/j.aeue.2025.155769|原模型/DIJCPO与当前接收者、方向、模式、连续功率及EE定义的差别；不是独立外部基线|
|车用RF/VLC拓扑|Chen & Wang，Topology Control in Hybrid VLC/RF Vehicular Ad-Hoc Network；doi:10.1109/TWC.2019.2960229|拓扑及物理假设，不能直接借来证明现有光传播参数|
|车用异构动态拓扑|Coordination Game Theory-Based Adaptive Topology Control for Hybrid VLC/RF VANET；doi:10.1109/TCOMM.2021.3077950|拓扑协同与当前集中配对的区别|
|过时CSI与异构资源分配|Proactive Resource Allocation for Aggregated VLC/RF Networks With Outdated CSI（本地参考清单）|CSI老化如何进入分配；其聚合网络与当前互斥RF/VLC模式是否相同|
|RF/VLC信道选择|Optimal Channel Selection in Hybrid RF/VLC Networks: A Multi-Armed Bandit Approach（目录L0001）|在线信道选择、信息假设及候选对照|
|多比值能效优化|Shen & Yu，Fractional Programming for Communication Systems—Part I；https://arxiv.org/abs/1802.10192|二次变换适用条件；不能把sum-of-ratios当单比值Dinkelbach|
|动态估计理论界|Tichavský等，Posterior Cramér–Rao Bounds for Discrete-Time Nonlinear Filtering；doi:10.1109/78.668800|完整动态状态的适用信息矩阵；当前代码未实现完整PCRB|
|可见光观测CRLB|PARAMETER ESTIMATION IN AN LED-CAMERA VISIBLE LIGHT COMMUNICATION SYSTEM（既有CRLB核读笔记）|相机观测与当前PD导频观测并非同一对象，不可直接移植数值界|
|功耗来源|RF Wang et al.表2、VLC Mayahi表5.2，见parameter_sources.md|参数物理归属、是否包含照明/RX/功放效率，避免重复计能|
|学习与图扩展|目录的V2V/RL/GNN分类、历史MASAC验收与探针|同规模同目标对照；图结构并不自动解决奖励/采样/回退缺陷|

R01–R03的书目信息来自已存在的写作包，完整条目在 `reading_notes/manuscript/21_既有论文框架与公式依据/参考文献.bib`。本次是资料整理，不代表重新核验了全部论文全文。没有可靠DOI的条目保持题名检索线索，绝不补造引用。
