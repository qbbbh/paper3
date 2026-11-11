# 当前发布版本的准备与验证

当前入口位于 `releases/paper3_20261110`。新克隆直接读取 `data/` 中的输入文件。

```bash
python -m pip install -r requirements.txt
python tools/verify_snapshot.py
python releases/paper3_20261110/tools/check_release.py --tests
python releases/paper3_20261110/tools/run_case.py --case 10A
python releases/paper3_20261110/tools/run_case.py --case 10B
python releases/paper3_20261110/tools/run_case.py --case 50B
```

case预览不执行仿真。确实要执行时，在case命令上加 `--execute --out <从未使用的新目录>`；可用 `--blocks 1` 作短检查。不要在环境安装阶段自动执行1000周期或覆盖正式结果。工具先核对交通/建筑输入及固定配置，必要时还原仓库已有分段数据。

新方法显式选 `--maxmin-refine-solver guarded`；原NM默认仍保留。逐次凸近似只生成候选，最后由完整条件预测接受；不要把较快的追加阶段计时当作完整求解时间。

历史保存动作复算工具位于当前code/replay_saved.py。它仍要求未上传的原run目录、检查点与校验文件，读取路径见其SOURCE_RUNS。公开包提供完整紧凑结果表和固定原始数组样例；这些不等于恢复旧全过程所需的大体积文件。新的完整仿真可由固定输入重新计算，任意历史位置的快速续算则需要相应恢复状态。

在Windows原环境验证了发布校验、配置预览及39个小测试。没有配置Linux实跑环境，不宣称已完成Linux端到端安装或全程仿真。环境配置若失败，应记录具体安装/命令错误；不能据此修改科学模型让脚本通过。

需要复核历史源码或历史运行方法时，可检出[上一版固定提交](https://github.com/qbbbh/paper3/tree/6ae23c04c091edee70e228cae39d72efb2af04a8)并使用该提交的说明。
