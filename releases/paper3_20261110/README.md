# Paper3：2026-11-10 求解器与复算版本

这是三组 1000 周期数值结果对应的当前求解器分发目录，用于独立代码审查和复现。2026-11-11 清理移除了独立旧开发脚本及旧版源码副本；保留的 `code/` 文件仍与 `代码/公平功率求解改进_20261110/code` 逐字节一致，求解器和模型没有改写。正式实验的完整冻结快照与当前精简分发包分别登记，不能混用源码聚合哈希。完整结果、理论边界和写作材料由仓库主索引另行提供。

## 当前版本与旧版

- 当前完整入口：`code/run_allocation.py`，必须显式选择 `--maxmin-refine-solver guarded`，并启用 `--maxmin-power-refine`。
- 原 CLI 默认仍是 `nm`。直接省略该选项不会自动启用本次追加阶段。默认方法列表还含 `policy`，不能把裸默认命令当成本轮数值配置；请使用下方固定配置入口。
- `guarded` 完整执行原初始分配及 NM 公平阶段，再对其最终动作追加联合功率候选和完整条件预测筛选。它没有替换原初始分配，也没有读取事后真实 EE 选择动作。
- 匹配 11 月 6 日正式实验的 [72 文件原始源码快照](https://github.com/qbbbh/paper3/tree/6ae23c04c091edee70e228cae39d72efb2af04a8/releases/paper3_20261110/baseline_20261106/code) 保留在固定 Git 提交 `6ae23c04c091edee70e228cae39d72efb2af04a8`，当前树不再重复放置 `baseline_20261106/`。其聚合 SHA-256 为 `f51a16b010614c6ab6b053ac7d54a7c6f30defbf9f16650548a22bc030b85815`；[历史源码清单](provenance/baseline_source_files.json) 保留来源摘要，当前入口不要求检出这些历史文件。
- 新旧共有文件中只有 `p3learn/allocation.py`、`p3learn/bottleneck_power.py`、`run_allocation.py` 改变；新增文件另列于 [改动清单](provenance/source_changes_from_20261106.json)。SCA 导数修正不影响旧正式 NM 分支，不能据此声称已经解释了旧 50 车的全部差距。

正式冻结的 [完整 78 文件快照](https://github.com/qbbbh/paper3/tree/6ae23c04c091edee70e228cae39d72efb2af04a8/releases/paper3_20261110/code) 合计 894,314 字节，聚合 SHA-256 为 `1c6ef8efbd5e86336c3015b911063b946130bc84f8cefd9f3c42550bcabc5387`；其 [原来源清单](provenance/source_files.json) 和三个原 replay manifest 均保持不动。

当前分发 `code/` 为 **69 个 Python 文件、790,028 字节**，聚合 SHA-256 为 `db61bbcd8be6ddb78c8fda1db9bd53961d89a909735c23e4653266ff5eaa333d`。[分发清单](provenance/distribution_sources.json) 明确记录保留文件、移除文件及它们在原 78 文件清单中的摘要。两种聚合均采用原 `run_allocation.source_hash()` 的路径加原始字节算法；从精简包重新运行时会记录新的分发聚合，不能把它追认为原实验的完整源码聚合。

本轮移除 `analyze_bc_checkpoints.py`、`blk60_20261028.py`、五个 `probe_*.py`、`run_development_suite.py` 和配套旧 `test_probe_selection.py`。保留原输出审计、理论检查入口、现行求解模块及其实际导入依赖；没有通过删测试掩盖现行错误。

## 环境与小规模验证

原实测环境是 Python 3.10.19、CPU PyTorch 2.0.0、Windows；[requirements.txt](requirements.txt) 固定原环境的六项依赖，并补 `threadpoolctl==3.1.0`。建议用 Python 3.10 的独立环境。下列命令从本目录执行，Linux 和 Windows 均不需要原项目 E 盘路径：

```sh
python -m pip install -r requirements.txt
python tools/check_release.py
python tools/check_release.py --tests
```

第一项检查当前分发源码与原冻结来源的逐文件一致性、固定配置，以及 Git 中直接保存的全部 12 个交通输入；不需要下载历史源码或执行旧数据还原。`--tests` 再执行 39 个已有小测试，覆盖不等功率导数、预测预算与服务筛选、保留旧解、后备标记、完整入口与追加入口、保存文件损坏检查、能耗和几何快照。测试不读取本地原运行目录，不启动完整仿真。

发布整理时，这些测试在上述 Windows 环境中 **39 项通过**；有一个既有 SLSQP 中间步骤越界后裁剪的警告。没有可用 Linux 执行环境，因此不声称已在 Linux 实跑或跨操作系统逐位复现。路径适配仅使用 `pathlib` 的仓库相对目录。仅旧 probe 专用测试随对应脚本移出分发，其他原测试保留；不宣称全套无失败：已知三个旧 CLI 错误文案匹配测试在原版也失败。本次不为改变这些历史测试而修改科学源码。

清理后另建只含本 release 与 `data/` 的隔离副本，再次得到 **39 项通过**，10A/10B/50B 三组预览全部通过；没有借用旧目录、历史数据还原结果或原项目运行数据。记录见 [清理后验证](provenance/cleanup_validation.json)，前一发布验证另存 [历史记录](provenance/release_validation_before_cleanup.json)。

`reference/geometry_expected.npz` 是仓库原几何快照的逐字节副本。几何测试需要仓库已有的 10/50/100 车数据；100 车仅作为旧几何兼容性测试输入，不能计作本轮新结果。其余所选测试使用程序内构造的数据。`tools/check_release.py` 只适配场景目录定位，不修改模型或测试。

## 三组固定配置与数据

配置从原 `summary.json` 读取，并与复算 manifest 逐项对照；没有凭记忆补参数。

| 组别 | 车辆 | 交通种子 | 信道种子 | 预测种子 | 条件轨迹数 | 初始联合搜索轮数 | 原 NM 迭代/重启 | 原完整运行线程 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [10A](configs/10A.json) | 10 | 20261019 | 2026102401 | 2026102501 | 32 | 8 | 240 / 2 | 1 |
| [10B](configs/10B.json) | 10 | 20261019 | 2026102801 | 2026102501 | 32 | 8 | 240 / 2 | 1 |
| [50B](configs/50B.json) | 50 | 20261019 | 2026102801 | 2026102501 | 16 | 4 | 120 / 1 | 2 |

10A/B 共享交通轨迹，区别是信道实现；不是两条独立交通轨迹。本次是在同一批保存场景上评价追加阶段，没有增加独立种子。本次复算每个进程用 1 线程，区别于表内的历史完整运行线程。

共同设置：100 ms 配置周期，RF 电输入 0.01–20 W、VLC 电输入 0.1–30 W；发送电路项分别为 0.1159/1.745 W；预测均值计算链路 EE，校准下分位检查固定 1 Mbit/s 要求，校准指数 1.35，预测平均 EE 下降预算 0.10。完整参数以 JSON 为准，其中 `saved_config.maxmin_refine_solver="nm"` 如实保存历史值，`release_override` 明确本次使用 `guarded`。

当前输入位于仓库 [data/n10、data/n50、data/n100](../../data/README.md)，12 个原始文件均直接由 Git 保存，字节与迁移前相同。旧发布布局的 50/100 车 `mobility.npz` 曾只在本机还原；现已修正新克隆缺文件的问题，不再要求运行 `restore_data.py`。文件列表和 SHA-256 见 [交通清单](provenance/traffic_files.json) 及 [数据来源清单](../../data/INPUT_MANIFEST.json)。配置中的历史绝对路径仅是来源记录，不参与下面入口的目录定位。

## 查看配置或重新运行

先预览，不启动仿真：

```sh
python tools/run_case.py --case 10A
python tools/run_case.py --case 10B
python tools/run_case.py --case 50B
```

该入口校验源码、数据与物理参数，加载原保存配置，仅将求解选项改为 `guarded`；目录定位改用同一仓库的数据。执行仍调用未改动的 `code/run_allocation.py`。重新计算一个周期可用：

```sh
python tools/run_case.py --case 10A --blocks 1 --out .runtime/review_10A_block0 --execute
```

完整 1000 周期需省略 `--blocks 1`，并使用新的输出目录；50 车完整流程可能耗时很多小时。此发布整理仅运行小测试和配置预览，未再进行科研仿真。前缀小运行只能验证入口，不能替代完整结果，也不能称独立测试集。

## 保存状态复算与计时边界

`code/replay_saved.py` 完整保留。它从原完整初始分配及 NM 最终动作接续计算，因此本次省去了未改变部分的重复求解。原预测样本和执行数组的全量恢复已核验；实际真值仅在动作返回后用于评价。

**当前仓库仅凭交通文件不能运行完整保存状态复算。** 还需原 10A/10B/50B 运行的 `summary.json`、逐块 `block_checkpoints`、`fine_samples.npz`、恢复状态及 commit 校验文件；50B 还依赖补跑目录的逐块完整恢复状态。`SOURCE_RUNS` 保留历史相邻目录布局，原 summary 中的交通路径也需在恢复环境正确映射。本发布不悄悄放宽恢复哈希门禁，也不把缺失的本地记录生成伪替代。需要在原环境或经审查的数据路径适配后执行。

11 月 5 日 50 车主批与 11 月 6 日副本的驱动保存机制存在版本差异；50 车恢复的具体来源和两份原 summary 摘要见 [50B manifest](provenance/replay_manifest_50B.json)。不要拿本目录字节哈希追认另一版本的完整驱动。

追加阶段时间是实测；“历史完整决策时间＋本次追加时间”是新完整时间的组合估算，不是新环境下实测端到端耗时。仿真未将求解等待加入车辆轨迹，不能据 100 ms 配置周期声称在线实时决策。

## 审查边界

新方法保留原动作，仅接受预测最低 EE 提高且预测预算、校准服务条件通过的候选。原动作可行时，返回动作保持原预测约束；原动作来自均值后备时，仍须独立记录最终下分位是否通过。预测非下降不代表真实逐周期非下降，也不保证实际平均 EE 损失不超过 10%。局部候选生成不提供完整问题的全局最优或驻点收敛证明。

本轮是纯数值评价，没有训练策略、没有策略更新，也没有使用学习检查点。主驱动仍导入 `p3learn/masac.py`、`models.py`、`features.py`、`credit.py`，Torch 还用于线程、随机状态及环境记录，因此这些实际依赖保留；固定案例入口拒绝 `train` 与 `policy`。不能把旧学习结果混入本轮 10/50 车结果。原实验说明的逐字节副本见 [来源说明](provenance/original_experiment_readme.md)，其中本机路径及相对链接仅是历史上下文。
