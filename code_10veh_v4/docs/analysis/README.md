# analysis：只读统计与绘图脚本（2026-10-28）

本目录**不在 `code/**/*.py` 的源码哈希范围内**，加脚本不会改变任何批次的
`source_sha256`。

全部脚本只读 `code/runs/` 下的产物，不写 runs，不调用 PHY。

| 脚本 | 作用 | 输出 |
|---|---|---|
| `report_10veh_v1.py` | 按 `<tag>:<method>` 打印统一统计行（均逐链路 EE、块内最低 EE、全网平均电功率、VLC 使用率、1 mW 贴地与其违速率、真值违约、回退、adopt、worst20、adopt 块损失） | 终端 |
| `plot_10veh_results_v1.py` | δ 曲线图（效率—公平权衡、逐块最低 EE 的 CDF）+ 汇总 CSV | `fig_delta_curve.png`、`delta_curve.csv` |
| `plot_delta_cross_v1.py` | 两条信道实现的 δ 对照（均 EE / 最低 EE / 全网平均电功率） | `fig_delta_cross.png` |
| `estimation_figure_v1.py` | 信道估计质量：因果预测 vs 陈旧观测，按 CSI 年龄分箱 | `estimation_links.csv`、`coverage.csv`、`fig_estimation.png` |
| `estimation_sweep_v1.py` | 导频周期扫描（RF 1–40 ms、VLC 50–800 ms） | `sweep.csv`、`fig_sweep.png` |
| `verify_network_ee_and_matching_20261028.py` | 复算两件事：① 含 P_c 的网络 EE `ΣR/Σ(P_c+p)T`（同时给出仅 p 的口径作对照，两者不能混）② 逐块比较 reference 与 maxmin 的选中链路 `(tx,rx,mode)` 是否相同 | 终端 |

## 两个口径陷阱（2026-10-28 被 agent1 纠正后写入）

- **本文的口径是均逐链路 EE `mean_k(r_k/(P_c+p_k))`，不报网络 EE `ΣR/Σ(P_c+p)`。**
  后者按功率加权，会被少数高功率链路主导，排序与均 EE 不同，不能用来讲节能。
  审计器的 `cumulative_network_ee_bit_per_j` 连分母都不对（用 `total_data_energy_j`，
  只有 `p·T`，不含 P_c），**不引用**。该字段只保留在 `verify_network_ee_and_matching`
  脚本里作反例，不参与任何结论。
- VLC 链路份额是聚合量，**抓不到同模式内部的边重排**。δ=.05/.10 的第 95 块换了 3 条
  RF 边的端点，份额仍是 26.2%。要判断"匹配动没动"必须逐块比 `(tx,rx,mode)` 集合。

## 用法

解释器一律 `-B -I`，路径写成参数，不 `cd` 进数据目录：

```powershell
$P = "D:\anaconda\envs\pymarl\python.exe"
$R = "E:\博士\课题论文\第3篇-修改版\代码\开发_10车机制定型_v4_20261027\code\runs"

& $P -B -I report_10veh_v1.py $R e6_B_d010_20261028:maxmin
& $P -B -I plot_delta_cross_v1.py $R <out_dir>
& $P -B -I verify_network_ee_and_matching_20261028.py $R e5_D_chance_d005_20261028 e6_A_chance_d100_20261028

# 估计类脚本要代码树（找 n10 场景从 code.parent 开始）
& $P -B -I estimation_figure_v1.py --code code --out <out_dir>
& $P -B -I estimation_sweep_v1.py  --code code --out <out_dir>
```

## 已知陷阱

- `find_scene_root` 必须传 `code.parent`，不是 `code`。
- `_predict_copy` 在 `t0 == t1` 时仍需传 `doppler`/`tau` 两个位置参数（传 `None`）。
- RF 的 CSI 年龄恒为 2 ms、VLC 为 5–55 ms，按年龄分箱对 RF 无意义，只能画整体 CDF。
- 中文字体依赖 `SimHei`/`Microsoft YaHei`；缺字体时图能出但方块化。
