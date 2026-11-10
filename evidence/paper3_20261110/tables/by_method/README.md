# 逐方法周期小表

每份CSV含同一方法的全部1000个计划周期，保留失败；字段与原完整表完全相同。每份约0.4–0.5MB，方便仅有网页访问能力的审核者读取。

[证据入口](../../README.md) · [字段与单位](../SCHEMA.json) · [拆分来源与哈希](INDEX.json)

| 方法 | 10A | 10B | 50B |
|---|---|---|---|
| 初始分配 (`reference`) | [10A](10A_reference_periods.csv) | [10B](10B_reference_periods.csv) | [50B](50B_reference_periods.csv) |
| 旧主方法 (`main_old`) | [10A](10A_main_old_periods.csv) | [10B](10B_main_old_periods.csv) | [50B](50B_main_old_periods.csv) |
| 新版主方法 (`main_new`) | [10A](10A_main_new_periods.csv) | [10B](10B_main_new_periods.csv) | [50B](50B_main_new_periods.csv) |
| 旧同拓扑 (`same_topo_old`) | [10A](10A_same_topo_old_periods.csv) | [10B](10B_same_topo_old_periods.csv) | [50B](50B_same_topo_old_periods.csv) |
| 新版同拓扑 (`same_topo_new`) | [10A](10A_same_topo_new_periods.csv) | [10B](10B_same_topo_new_periods.csv) | [50B](50B_same_topo_new_periods.csv) |
| 无连续精修消融 (`no_refine`) | [10A](10A_no_refine_periods.csv) | [10B](10B_no_refine_periods.csv) | [50B](50B_no_refine_periods.csv) |
| Aboagye对比 (`aboagye_matching_qt`) | [10A](10A_aboagye_matching_qt_periods.csv) | [10B](10B_aboagye_matching_qt_periods.csv) | [50B](50B_aboagye_matching_qt_periods.csv) |
| Zappone对比 (`zappone_mmee_fp_fixed`) | [10A](10A_zappone_mmee_fp_fixed_periods.csv) | [10B](10B_zappone_mmee_fp_fixed_periods.csv) | [50B](50B_zappone_mmee_fp_fixed_periods.csv) |

失败周期没有可执行动作，EE字段留空，服务达标为False。相对初始分配的损失或预测预算检查在初始分配无有效结果时也留空，不应按零补齐。

原完整周期表仍保留：[10A](../10A_periods.csv)、[10B](../10B_periods.csv)、[50B](../50B_periods.csv)。小表没有筛选有利周期，也没有重新统计或修改数值。
