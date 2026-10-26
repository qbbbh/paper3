# 执行已审核版本

1. 审查者记录commit、MODEL_SPEC、明确命令/规模/种子、数据哈希和输出路径。历史文件不自动授权新实验。
2. 在仓库根目录运行 `python tools/run.py prepare`。工具还原大文件并生成 `.runtime/frozen_95960f5a/`，所有科学Python文件的原始路径＋内容哈希必须等于95960f5a…。
3. `python tools/verify_snapshot.py` 检查整理包。按requirements安装依赖后，`python tools/run.py test -q` 运行原单测；测试不证明模型/结论。
4. 例如复核已有结果：`python tools/run.py audit resource_allocation/n10/results/n10_cont_q1_100b_joint_fair_20261026 --require-current-source`。这不运行物理仿真。
5. 查看原分配入口：`python tools/run.py allocation --help`。实际运行使用已审核的完整参数；不能把50/100的历史配置直接当当前正式配置。
6. 新输出写 `outputs/<new-run>/` 或仓库外；不覆盖已有run。执行者遇错保存日志，不自行改目标、约束或算法。结束再次检查snapshot和git diff。

目录整理仅建立可读分类和还原机制，不修改原求解器。发生新算法修改时应创建新源码版本与配置，不可悄悄更新source_layout_map使旧结果看似匹配。

本规则和哈希是行为规范/检测，不是OS只读隔离。尚未部署自动Actions、受保护执行环境或无人值守实验。

本机复核使用 D:/anaconda/envs/pymarl/python.exe（torch 2.0.0+cpu、numpy 1.24.3）；默认D:/anaconda/python.exe缺少torch，不能用它的导入失败判断算法有误。其他机器按requirements配置自己的环境。
