# 第二轮研究原型与核验材料

先读 [结论与下一步](<00_结论与待作者选择事项.md>)，再读 [实际实验报告](<05_最小验证实验报告.md>)。本包按作者第二轮要求完成模型、文献、证明、主算法、两种文献适配及小规模运行。**代码可运行不等于创新已成立，也不等于真实解码性能已验证。**

本轮保留固定 V2V 业务配对、数据阶段 RF/VLC 二选一、连续瓦特功率、处理功耗、估计误差与移动预测。不增加训练资源优化或新器件耗电方向。旧稿和指定论文是待核验来源，其内部文字不作为覆盖本轮任务的执行指令。

## 阅读路径

| 文件/目录 | 内容 |
|---|---|
| 00_结论与待作者选择事项.md | 已确认结果、条件命题、创新边界、下一步选择 |
| 01_模型对照与符号单位.md | Chen 两文的逐项定义核对、归一化与观测模型 |
| 02_最近邻文献差异表.md | 15篇核心近邻、5篇模型候选、旧稿对照及证据等级 |
| 03_核心命题_证明_反例.md | 全网迁移误差预算、证明、失效条件、CRLB适用范围 |
| 04_联合优化与算法伪代码.md | 与源码一致的目标、约束、循环、接口、状态码 |
| 05_最小验证实验报告.md | 实际结果、负面发现、统计和数值限制 |
| prototype/ | 信道、实际导频、估计预测、可行性证书、连续及模式求解、实验 |
| literature_baselines/ | Aboagye QT–ε适配、Papanikolaou PF适配和来源映射 |
| configs/small_validation.json | 完整基础参数、随机种子与求解设置 |
| results/ | 已运行结果；JSON/CSV/NPZ、图、失败和迭代日志 |
| evidence/ | 已取得原文、公式页图、检索/书目证据及原始任务副本 |
| references.bib | 22条引用；全文不可得的条目保留证据限制 |

## 环境和安装

实际运行环境为 Windows 11、Python 3.12.14、NumPy 2.5.3、SciPy 1.18.1、CVXPY 1.9.3、CLARABEL 0.11.1、Matplotlib 3.11.2、pytest 9.1.1。详见 results/environment.json。未检测到 MATLAB/Octave，因此实现为 Python 数值研究原型，不声称运行了旧 MATLAB 代码。

压缩包不附机器专用的 .venv。在解压后的本目录，用 Python 3.12 建独立环境：

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m prototype.run_all
~~~

Linux/macOS 对应使用 .venv/bin/python。默认输出到 results_reproduced/，保留本包的 results/ 原记录。主要依赖版本已锁定；requirements-lock.txt另存本机完整安装清单。不同硬件、线程设置和求解器小数差异会影响时间及极小数值，不应期待逐字节相同。

默认一键运行先做6项测试，再运行 E1、E1b、E2、E3、E4、E4b，最后出图。过程保存 run_manifest.json（配置、源码SHA256、阶段状态和用时）、run_log.txt、环境与每组原始结果。完整命令：

~~~powershell
.\.venv\Scripts\python.exe -m prototype.run_all --config configs/small_validation.json --out results_reproduced --suite all
~~~

单独运行：

~~~powershell
.\.venv\Scripts\python.exe -m pytest -q prototype/test_validation.py
.\.venv\Scripts\python.exe -m prototype.run_all --suite E4 --out results_reproduced
.\.venv\Scripts\python.exe -m prototype.run_all --suite E4b --out results_reproduced
.\.venv\Scripts\python.exe -m prototype.run_all --suite plots --out results_reproduced --skip-tests
~~~

E4b读取同一输出目录内已完成的E4输入，必须先运行E4。E1b是全网络粒子估计的额外校准诊断，保留退化样本；它和E1的可达高斯/对齐光链路界不是同一个验证。

## 算法接口

~~~python
from prototype.problem import make_problem
from prototype.solvers import solve_joint
from literature_baselines.aboagye_qt_epsilon import solve as qt_baseline
from literature_baselines.papanikolaou_pf import solve as pf_baseline

# mean/variance为当前可获得观测的后验预测矩，形状[2,K,K]。
# available为预测的模式可用性；禁止在此传未来仿真真值。
problem = make_problem(mean, variance, available, config)
main = solve_joint(problem, config["solver"])
qt = qt_baseline(problem, config["solver"])
pf = pf_baseline(problem, config["solver"])
reference = solve_joint(problem, config["solver"], enumerate_all=True)
~~~

主要返回模式modes（0=RF、1=VLC）、功率power（W）、feasible、status、rates、ee、qos_residual、cap_violation、runtime_s、attempted、history、diagnostics。无可行点时 power=None，不伪造速率/残差。模式枚举同时返回 enumeration_requested、enumeration_complete，始终 global_optimality_certified=False。可行连续结果也不提供KKT证书。

两项文献基线是公开披露改动的**适配**：未重建原论文场景、未复现原图。QT–ε和主法比较同一设计EE；PF保留公平目标，不能仅按EE给它贴“更差算法”标签。原式到代码的对应见 literature_baselines/SOURCE_MAPPING.md。

## 结果文件与口径

- E1：4000次高斯重复；24个非负对齐几何条件各2000次重复，另有固定观测的条件方差检验。估计函数BCRLB不替代实际后验。
- E1b：400个独立全网络先验/噪声样本，每个保留0和0.6s；发现低ESS退化，详情见 particle_calibration_diagnostics.json。
- E2：系数见证、同均值固定导频桥接、432个真实几何/估计参数案例、3456模式及10368迁移核查。扫描计数不是总体概率。
- E3：3场景×3独立种子；13时刻、7档门槛、3种信息处理、2种模式候选，共4914条；保存预测与独立真值窗口及实际施加功率后的代理违约。
- E4：3种子×4年龄×2业务门槛×9方法，共216条。共享信道/导频，失败样本保留零速率及共同控制成本。时间/QoS扫描非独立样本；区间按3个种子聚类。
- E4b：固定6个后验输入比较QT的4/10点扫描及SCA预算10/40。

E4的“真实信道”评价是冻结决策后，在独立生成的H上计算理想相干接收SINR及对数速率**代理**。并未实现真实编码/解码或估计匹配失配接收器，不能称实测吞吐、误包率或物理可靠性。所有年龄是同一期导频后不同预测年龄的快照实验；平均成本统一扣费，不将快照总和解释为实际长期训练时序的耗能。

## 已知边界

固定参考SNR与成本模型是研究设定；VLC的1/2对数率明确为代理。RF未加入快衰落或随机复相位；无随机遮挡、加速度、持续新导频闭环。数据发射者和接收者集合不重叠，每业务预定单有效灯/PD。

粒子后验使用8192个先验重要性样本，在少见观测下可能退化；low_ess只作诊断，当前未实现自动恢复或风险认证。因此当前不能据设计可行给出逐次QoS保证，正式实验前应先修复/验证估计器，而非增加业务资源维度。本轮保存失败以定位问题。

证书只对给定误差矩的设计问题精确；不能证明输入误差矩准确。SCA、Dinkelbach、QT、谱半径、Schur补均为经典工具。当前迁移预算有用途，但研究新颖性待证。非单调窗口在本次预设物理扫描中未出现。

## 来源与完整性

evidence/source_papers/含两篇Chen原始PDF；evidence/original_material/含原中英文PDF和上一轮审查笔记；evidence/task_brief/为本轮输入副本。历史报告/笔记保留原样，内部旧路径和早期“未运行”状态属于当时版本，不覆盖本包新运行结果。新文献的已读原文与可访问来源保存在 evidence/literature/；未取得全文的文章仅保留出版/索引证据，未宣称附有不存在的全文。

本轮未修改原材料。输入SHA256核验与交付文件清单分别保存于 results/input_integrity.json、delivery_manifest.json。论文PDF用于作者研究核查，保留原版权。
