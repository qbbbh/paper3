# 只执行已审核代码

1. 审查者先记录目标commit、MODEL_SPEC版本、完整命令、数据哈希、依赖、输出目录和时间预算。未批准的模型方案不得由执行者自行选择。
2. 在独立检出目录执行 `python tools/restore_data.py`（首次还原大文件），再执行 `python tools/verify_snapshot.py`；保存 `git rev-parse HEAD` 和环境版本。
3. 安装requirements后先在code/运行 `python -m pytest -q`；这只验证测试，不证明论文结论。
4. 在code/运行 `python audit_allocation_outputs.py ../experiments/evidence/<run目录> --require-current-source` 复核已有数据，无需重新仿真。
5. 正式运行只能使用审查批准的命令。新的输出写仓库外目录；禁止覆盖旧run、修改源码或为跑通而改参数。错误保存并停止。
6. 结束再次验证snapshot与git diff，并保存完整日志、CSV/必要NPZ、配置、源码与数据哈希。允许临时缓存，不允许改已哈希输入。

这是行为规范和篡改检测，不是OS权限隔离。要强制只读，应由宿主机将源码及数据以只读方式挂载、输出单独可写；尚未部署该环境，也未配置自动Actions实验。

四组证据的continuous_power_audit.json以无损gzip压缩，按需解压到临时目录。归档variance脚本含旧绝对路径且推导未验收，仅供复核，禁止当正式结果生成器。
