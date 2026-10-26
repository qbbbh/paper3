# 共享信道、观测与估计

|位置|内容|
|---|---|
|src/p3base/channel.py、fading.py、geometry.py、scene.py|物理信道、快慢衰落、遮挡、位置和端口；包含真值模拟基础，不等同于估计器|
|src/p3base/pilots.py、noise.py|导频时序与噪声|
|src/p3base/estimator.py、kalman_mmse.py、rf_two_state.py|估计状态和双状态RF更新|
|src/forecast.py|条件轨迹库，用后验计算未来块服务分布|
|src/p3base/block.py、link_eval.py、energy.py、config.py、contract.py|共享块评价、SINR/速率、能耗和接口约定|
|entrypoints/|基础统计、估计曲线、CRLB与块预测验收原始入口|
|results/baseline/|10车3→8历史估计CSV、CRLB校验JSON；不能与后来模型跨版本混用|

这是实际源码归类，不是将“真实信道”和“估计信道”混成一份输入。决策只能用当时可获得的观测/后验，真值用于执行评价。

## 估计数据怎样读

估计曲线CSV含 `time_s`、RF/VLC真值、估计、区间、观测时间、观测值、可见性和端口。`*_obs_time_s` 与当前 `time_s` 不一定相同，不能直接把陈旧观测误差当同期测量噪声。文件名 two_state 表示双状态版本；未带该后缀的文件是旧对照。

CRLB校验中的MSE/CRLB约1来自特定合成高斯子问题；不等于完整双状态后验/未来服务达界。CRLB是方差下界，不能当保守误差上界。14%/86%的旧置零方差实验尚非严格全方差分解，见 reviews 与 research_history/ERROR_LEDGER.md。

该目录目前没有50/100完整信道估计正式结果，不复制10车结果改名。
