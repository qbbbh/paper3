from pathlib import Path
import json
import pandas as pd

OUT=Path(__file__).resolve().parent
d=json.loads((OUT/'results.json').read_text(encoding='utf-8'))
phy=json.loads((OUT/'saved_physics_summary.json').read_text(encoding='utf-8'))
names={'reference':'平均EE初始分配','maxmin':'本文完整方法','same_topo_maxmin':'固定模式连续细化','no_refine':'无公平阶段连续细化','aboagye_matching_qt':'Aboagye适配','zappone_mmee_fp_fixed':'Zappone适配'}
order=list(names);groups=['10A','10B','50B']
s={(x['dataset'],x['method']):x for x in d['summary']}
p={(x['dataset'],x['method'],x['mode']):x for x in d['by_mode']}
def table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
parts=['# 10车与50车：统一性能、模式使用及功率统计\n\n日期：2026-11-10。仅分析已完成保存结果，未修改算法、参数或输入，未启动新仿真。\n',
'''## 1. 核心判断

10车A/B中，本文方法相对Zappone适配的平均EE和最低链路EE均值均更高；50车中，平均EE更高，但最低链路EE均值更低。两者不能合并为“本文全面领先”或“50车所有指标都不如对照”。

50车的直接差别在相同通信模式下的功率分配。两者VLC使用率均为65.301%，全部999个共同成功周期的车辆连接、方式与端口逐链路一致。本文最弱链路在755个周期为RF；Zappone对这些同一条RF链路提供了更有效的信号与干扰组合。

求解设置也发生了明显变化：10车优化5个功率变量，每起点240次迭代上限、2个额外重启；50车优化25个变量，却降为120次迭代上限、1个额外重启。与此对应，满足本文约束而预测最低EE优于本文的Zappone候选，在10车A/B仅有2/0个周期，在50车有347个周期。可以确认存在数值搜索不足；降低搜索预算造成了多少差距，仍需保存状态上的控制对照才能量化。

## 2. 统计对象、单位和复核

- 10A和10B各1000个计划周期，共享10车交通轨迹、采用两个信道实现；50B是另一车辆规模下的1000个计划周期。保留三组分别报告，不把车辆规模和场景变化等同为单一自变量实验。
- EE主表仅统计各方法成功返回完整动作的周期，包含实际速率未达标周期；失败单独报告。平均EE为每周期各活动链路EE的算术平均；最低EE为每周期最弱链路EE，再跨周期平均。
- 方法间变化用共同成功周期的两个均值之比计算。它不是逐周期相对增益的平均。
- “服务达标率”是计划周期中成功得到动作且全部链路实际速率达到1 Mbit/s的比例，分母均为1000。
- VLC使用率为成功周期中VLC活动链路数／全部活动链路数。每周期链路数固定为5或25，因此也等于成功周期VLC链路比例的均值。它不是VLC数据量占比，也不是“至少有一条VLC的周期比例”。
- 功率变量p是发送支路电输入功率。RF辐射功率为0.1p，VLC发射光功率为0.25p；固定电路功耗不包含在p中，计算EE时加上RF 0.1159 W或VLC 1.745 W。
- RF/VLC功率分布分别对该方式的活动链路—周期记录统计。均值、分位数不把未使用该方式的车辆记零。总功耗则逐周期对全部活动发送支路求和，再跨成功周期平均。
- 数据吞吐量占比采用跨成功周期的数据量之比。通信周期等长，所以也可由保存的周期平均速率求和得到。
- 本轮独立复算全部逐链路EE、每周期均值和最小值、实际服务计数，与原始记录一致。50车两批的初始动作和指标精确一致，保存的1000周期数组对齐审计通过，合并时剔除重复初始分配。

旧10车主表将失败周期记零，本表统一改为成功周期EE。例如10B本文平均EE由原总体口径386.883变为387.271 Mbit/J；这是分母变化，没有重跑或调整算法。
''']
core=[]
for group in groups:
 parts.append('## 3'+{'10A':'A','10B':'B','50B':'C'}[group]+'. '+group+'性能\n')
 rows=[]
 for method in order:
  r=s[group,method]
  rows.append([names[method],r['success'],f"{r['mean_ee']:.3f}",f"{r['min_ee']:.3f}",f"{r['service_success_pct']:.1f}%",r['failed'],f"{r['qos_bad_blocks']}／{r['qos_bad_links']}",f"{r['time_mean_s']:.3f}"])
  core.append({'数据组':group,'方法':names[method],'成功周期数':r['success'],'平均链路EE_Mbit每J':r['mean_ee'],'周期最低EE均值_Mbit每J':r['min_ee'],'计划周期服务达标率_pct':r['service_success_pct'],'失败周期数':r['failed'],'VLC链路使用率_pct':r['vlc_link_pct'],'RF电输入平均功率_mW':1000*r['rf_input_power_mean_w'],'VLC电输入平均功率_W':r['vlc_input_power_mean_w'],'RF平均辐射功率_mW':1000*r['rf_emitted_power_mean_w'],'VLC平均光功率_W':r['vlc_emitted_power_mean_w'],'全部发送支路平均总功耗_W':r['mean_total_electrical_w'],'完整决策平均时间_s':r['time_mean_s']})
 parts.append(table(['方法','成功周期','平均EE','最低EE均值','服务达标率','失败','实际违约周期／链路','完整决策均时/s'],rows)+'\n')

parts.append('## 4. VLC使用率及RF/VLC功率分布\n\nRF列使用mW、VLC列使用W，均为电输入p。括号内为中位数／第90百分位，用于观察均值是否被少数高功率链路拉高。\n')
for group in groups:
 rows=[]
 for m in order:
  r=s[group,m];rf=p[group,m,'RF'];vlc=p[group,m,'VLC']
  rows.append([names[m],f"{r['vlc_link_pct']:.2f}%",f"{rf['input_power_mean_w']*1000:.2f}（{rf['input_power_median_w']*1000:.2f}／{rf['input_power_p90_w']*1000:.2f}）",f"{vlc['input_power_mean_w']:.3f}（{vlc['input_power_median_w']:.3f}／{vlc['input_power_p90_w']:.3f}）",f"{rf['at_lower_pct']:.2f}%",f"{vlc['at_upper_pct']:.3f}%"])
 parts.append('### '+group+'\n\n'+table(['方法','VLC使用率','RF均值（中位／P90）mW','VLC均值（中位／P90）W','RF处于10 mW下限','VLC处于30 W上限'],rows)+'\n')
parts.append('''RF上限为20 W，除Aboagye适配中极少数记录外，表内方法没有触及该上限；VLC触顶也只占很小比例。现有结果不支持把总体差距归为发射功率上限不足。

本文方法的平均RF辐射功率在10A、10B、50B分别为4.329、4.593、3.278 mW；平均VLC发射光功率分别为0.358、0.356、0.208 W。不能直接把表中电输入功率当成辐射功率或光功率。
''')
parts.append('## 5. 总功耗与传输数据组成\n\n总输入功率为全部活动链路的p之和，总发送支路功耗为Pc+p之和。VLC数据占比为跨成功周期实际传输数据量的比例。\n')
rows=[]
for group in groups:
 for m in order:
  r=s[group,m];rows.append([group,names[m],f"{r['mean_total_input_w']:.3f}",f"{r['mean_total_electrical_w']:.3f}",f"{r['mean_total_rate_Mbps']:.2f}",f"{r['vlc_rate_share_pct']:.2f}%"])
parts.append(table(['组别','方法','总输入功率/W','总发送支路功耗/W','总速率均值/Mbit每s','VLC数据占比'],rows)+'\n')
parts.append('''当前50车场景中，本文VLC链路占65.30%，承担95.60%的数据量；但最弱EE仍在75.58%的成功周期由RF链路决定。因此，VLC承担较多数据，并不能自动保证最低链路EE较高。这是本批数据的结构，不据此宣称车辆数增加必然提高VLC使用率。
''')
parts.append('## 6. 共同成功周期上的改善幅度\n\n正数表示本文方法高于对照；负数表示低于对照。\n')
rows=[]
for group in groups:
 for x in d['paired']:
  if x['dataset']==group:rows.append([group,names[x['other']],x['n'],f"{x['mean_ee_change_pct']:+.3f}%",f"{x['min_ee_change_pct']:+.3f}%"])
parts.append(table(['组别','对照','共同成功周期','平均EE变化','最低EE均值变化'],rows)+'\n')
parts.append('''10车对Aboagye的相对增益与旧失败记零总体表不同，本表使用共同成功周期。对Zappone，10A/B的平均EE分别高5.340%/4.459%，最低EE分别高4.166%/4.018%；50B平均EE高3.381%，最低EE低23.898%。

## 7. 50车差距的分层解释

### 7.1 已确认：差距来自相同模式下的功率决策

50车本文、固定模式细化和Zappone适配的最终车辆连接、方式及端口一致，VLC比例也完全相同。本文最终模式与初始分配在999个成功周期全部一致。模式使用比例不能解释它与Zappone的最低EE差距。

以下在两种方法的同一批24975条活动链路—周期记录上比较：

| 对象 | 本文 | Zappone适配 |
|---|---:|---:|
| RF平均输入功率 | 32.776 mW | 43.187 mW |
| RF输入功率中位数 | 15.575 mW | 约10.01 mW |
| VLC平均输入功率 | 0.831 W | 0.780 W |
| RF链路平均实际速率 | 49.350 Mbit/s | 43.545 Mbit/s |
| 全部发送支路平均总功耗 | 43.337 W | 42.608 W |

Zappone相对本文上调约33.00%的RF链路功率，下调48.98%，其余基本相同。平均RF功率增加而中位数下降，说明它将较大的RF功率集中到部分链路；不能概括成所有RF链路一起升功率。其VLC平均功率略低，全部发送支路总功耗也略低。

### 7.2 已确认：主要瓶颈是部分RF链路，Zappone对这些链路的协调更有效

按本文方案每周期的实际最弱EE链路取样：999个成功周期中755条为RF、244条为VLC。对755条RF链路，保持车辆、方式、端口和真实信道不变，比较它们在Zappone方案中的表现：

| 同一批RF最弱链路 | 本文 | Zappone适配 |
|---|---:|---:|
| 平均电输入功率 | 134.386 mW | 242.194 mW |
| 平均实际速率 | 5.197 Mbit/s | 13.019 Mbit/s |
| 同一条链路的平均EE | 35.165 Mbit/J | 61.597 Mbit/J |

其中604/755条链路的EE提高。61.597是对“本文最弱RF链路”这一固定子集的结果，不是Zappone每周期最低EE的均值。

进一步读取已保存的逐毫秒信号、干扰和噪声，得到这755条链路的Zappone／本文比值中位数：功率及接收信号2.451、干扰0.947、噪声1.000、周期内SINR均值3.005。452条链路的干扰下降，628条链路的SINR提高。这直接说明信号增强与其他发送者功率协调共同改善了部分瓶颈链路。

对本文全部8666条RF链路—周期记录，周期平均干扰／噪声的中位数为41.30，93.26%的记录中干扰大于噪声。这支持研究RF功率之间的相互影响，不能据此建议无差别提高所有RF功率。VLC对应的干扰／噪声中位数为0，只有约1.02%的记录干扰大于噪声，符合本批方向性、端口及遮挡条件下较少同模式光干扰的结果；不外推为一般VLC网络结论。

### 7.3 已确认：已有更优可行方案，预算约束不足以解释全部差距
''')
rows=[]
for group in groups:
 z=d['zappone_diagnostic'][group]
 rows.append([group,z['power_dimensions'],z['nm_iteration_limit'],z['nm_additional_restarts'],z['z_pass_both'],z['z_feasible_predicted_better']])
parts.append(table(['组别','功率变量维数','每起点迭代上限','额外重启数','Zappone通过本文两项预测约束','其中预测最低EE更高'],rows)+'\n')
parts.append('''两项预测约束指平均EE至少保留初始结果的90%，以及全链路经校准的速率下分位数达到1 Mbit/s；候选功率与结构合法性已通过原输出检查及本轮同模式核对。

10车中本文几乎覆盖了Zappone能够提供的更优可行候选，50车则有347个周期未做到。这一判断比较的是算法实际优化的预测最低EE，而不只比较执行后的真实EE，因此不能全归于预测误差。

### 7.4 合理解释：维数增加，同时削减了无导数局部搜索预算

主方法用Nelder–Mead搜索完整条件预测指标，50车的功率变量由5个变为25个，每起点迭代上限却减半，起点从3个变为2个。Zappone适配则以广义分式规划与带速率导数的SCA子问题调整功率，再通过共同条件预测器检查候选。两者利用功率耦合信息的方式不同。

这些实现差异和347个更优可行候选共同支持“高维局部搜索不充分”这一解释。但当前正式运行在Nelder–Mead且关闭弱链路修复时，没有将逐起点终止诊断保存到结果：驱动只在修复或其他求解器分支保存该诊断。因此，尚不能宣称所有周期都耗尽迭代，也不能给出仅增加迭代次数会恢复多少性能的数值结论。

### 7.5 预测偏差存在，但不是排名变化的全部原因
''')
rows=[]
for group in groups:
 z=d['zappone_diagnostic'][group]
 rows.append([group,f"{z['proposed_predicted_min_mean']:.3f}",f"{z['z_predicted_min_mean']:.3f}",f"{z['proposed_actual_min_mean']:.3f}",f"{z['z_actual_min_mean']:.3f}"])
parts.append(table(['组别','本文预测最低EE均值','Zappone预测最低EE均值','本文实际最低EE均值','Zappone实际最低EE均值'],rows)+'\n')
parts.append('''50车在预测指标上已经是43.132对55.065，执行后为37.045对48.678。因此排名变化在决策阶段就已出现。预测与实际的偏差会影响最终幅度，但不能将当前差距归为未来真实信道恰好对本文不利。两种规模的预测样本数还不同（32与16），目前没有单因素对照证明样本数变化贡献了多少差距。

## 8. 下一步的有界判断

现有数据支持保留业务、物理模型、EE定义及公平问题。优先检查连续功率求解的初始点、维数适配和终止情况，而不以改变功率边界、降低基线质量或更换指标处理排名。

若后续开展验证，宜固定少量已保存周期及同一预测样本，分别检查增加搜索预算、加入已有可行起点的影响，再判断是否需要调整求解器。这是后续验证建议，本轮没有执行新优化或新信道仿真，也没有将Zappone方案并入本文方法制造新的结果。

计算方面，50车完整方法平均107.804秒，第一阶段初始分配约92.309秒；即使公平求解得到改进，第一阶段仍是当前主要时间开销。10车与50车的运行设置、交通状态和预测样本数不同，应报告实测代价，不直接拟合纯规模复杂度。

## 9. 文件与来源

- [中文核心统计CSV](<E:/博士/课题论文/第3篇-修改版/代码/跨规模结果分析_20261110/10车50车_核心统计.csv>)。
- [完整性能及服务统计](<E:/博士/课题论文/第3篇-修改版/代码/跨规模结果分析_20261110/performance.csv>)。
- [分模式功率、分位数、上下限及速率统计](<E:/博士/课题论文/第3篇-修改版/代码/跨规模结果分析_20261110/mode_power.csv>)。
- [共同成功周期比较](<E:/博士/课题论文/第3篇-修改版/代码/跨规模结果分析_20261110/paired.csv>)。
- [全部数值、输入SHA-256与配置](<E:/博士/课题论文/第3篇-修改版/代码/跨规模结果分析_20261110/results.json>)。
- [同一最弱链路的物理量核对](<E:/博士/课题论文/第3篇-修改版/代码/跨规模结果分析_20261110/saved_physics_summary.json>)。

原始来源为10车A/B完整100秒运行、50车主批和50车对照补跑；每个来源的绝对路径、配置和原始文件哈希保存在数值文件的provenance中。对应科学源码、运行目录和原始结果均未修改。
''')
pd.DataFrame(core).to_csv(OUT/'10车50车_核心统计.csv',index=False,encoding='utf-8-sig')
target=OUT/'10车与50车_统一性能功率统计及差距分析_20261110.md'
target.write_text('\n'.join(parts),encoding='utf-8')
print(target)
