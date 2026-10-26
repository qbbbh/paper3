# 车辆遮挡感知的 RF/VLC 异构 V2V 通信资源分配论文思路汇总

## 0. 当前方案的核心判断

这篇旧论文不建议继续沿着“单一 V2V-VLC 信道建模 + 链路选择规则”的方向扩展，因为原始方案的主要创新集中在行人遮挡概率建模、LOS/NLOS 功率分析和启发式链路选择上，理论深度和资源分配属性不足。

更合适的升级方向是：

> 将原来的单一 V2V-VLC 遮挡建模论文，改造成“车辆遮挡感知的 RF/VLC 异构 V2V 通信链路选择与功率联合优化论文”。

不建议继续强行接入贝叶斯 Stackelberg 博弈。原因是：

1. “行人随机遮挡 + 恶意 RF 干扰 + 博弈论”会引入过多外部扰动源，场景复杂但主线不聚焦；
2. 行人不是理性博弈主体，若同时引入恶意车辆，问题设定会显得拼接；
3. 这篇论文更适合做“小规模、可建模、可解释、理论推导较强”的 RF/VLC 异构资源分配论文；
4. 后续 GNN-RL 论文可以承担“多车辆动态拓扑 + 分布式智能资源分配”的任务，避免两篇论文重复。

---

## 1. 推荐题目

### 中文题目候选

**题目 1：车辆遮挡感知的 RF/VLC 异构 V2V 通信链路选择与功率联合优化方法**

优点：稳妥，突出资源分配，不会让审稿人期待“恶意攻击/安全博弈”。

**题目 2：面向车辆遮挡与同频干扰的 RF/VLC 异构 V2V 通信资源分配方法**

优点：同时包含 VLC 遮挡和 RF 干扰，场景完整，但不引入恶意攻击。

**题目 3：车辆遮挡感知的 RF/VLC 异构 V2V 通信抗干扰资源分配方法**

优点：听起来更有冲突性，但“抗干扰”容易让人期待更强的干扰建模或鲁棒理论。

### 英文题目候选

**Option 1**

Vehicle-Blockage-Aware Joint Link Selection and Power Allocation for RF/VLC Heterogeneous V2V Communications

**Option 2**

Energy-Efficient Resource Allocation for RF/VLC Heterogeneous V2V Communications under Vehicle Blockage and Co-Channel Interference

**Option 3**

CRLB-Induced Energy Efficiency Bound for Vehicle-Blockage-Aware RF/VLC Heterogeneous V2V Communications

如果主打理论公式，推荐 Option 3；如果主打完整通信优化，推荐 Option 1 或 Option 2。

---

## 2. 新论文主线

原始论文主线：

> 行人遮挡 → V2V-VLC LOS/NLOS 建模 → 链路选择 → EGC → BER/中断概率/CF 改善

建议改为：

> 车辆遮挡 → VLC 链路可用性下降  
> RF 同频干扰 → RF 链路 SINR 下降  
> RF/VLC 异构互补 → 链路选择与功率联合优化  
> CRLB 诱导理论界 → 揭示遮挡概率、信道估计误差与 EE 性能之间的极限关系

最终形成的新逻辑：

1. 单一 VLC 链路虽然不受 RF 电磁干扰影响，但严重依赖 LOS 传输，容易受到前车、邻车或变道车辆遮挡；
2. 单一 RF 链路具有较好的非视距通信能力，但容易受到同频车辆干扰；
3. RF/VLC 异构通信能够利用 VLC 的高容量和 RF 的覆盖稳定性实现互补；
4. 在车辆遮挡和 RF 干扰共存条件下，需要联合优化链路选择与发射功率；
5. 为增强理论深度，引入信道估计 CRLB，并进一步推导 CRLB 诱导的 EE 上界和不可避免 EE 损失下界。

---

## 3. 场景设定

### 3.1 通信场景

考虑小规模 V2V 通信场景，包括若干对车车通信链路。每个通信对同时具备 RF 和 VLC 两种候选链路：

- RF 链路：可用于非视距或 VLC 遮挡时的补充传输，但存在同频干扰；
- VLC 链路：由车辆前照灯 LED 和接收端 PD 构成，具有较高方向性和抗 RF 干扰能力，但容易受到车辆遮挡。

### 3.2 不再采用的复杂设定

不建议采用以下组合：

> 行人随机遮挡 + 恶意 RF 干扰车辆 + 贝叶斯 Stackelberg 博弈

主要原因：

- 外部不良影响过多，系统模型不聚焦；
- 行人遮挡与恶意干扰属于两类不同机制，不利于统一建模；
- 会分散论文贡献点，使“RF/VLC 资源分配”主线变弱。

### 3.3 推荐采用的干扰与遮挡机制

建议采用：

> 车辆遮挡 + RF 同频干扰

其中：

- 车辆遮挡影响 VLC 链路可用性；
- RF 同频干扰影响 RF 链路 SINR；
- 两者共同决定 RF/VLC 异构链路选择与功率控制策略。

---

## 4. 车辆遮挡建模

不要建立过于复杂的“车辆变道行为模型”。更建议建立“车辆几何遮挡模型”或“遮挡概率模型”。

### 4.1 VLC 链路遮挡状态变量

定义第 \(i\) 条 VLC 链路的可用状态为：

\[
\chi_i^{\mathrm{VLC}} =
\begin{cases}
1, & \text{VLC LOS link is available},\\
0, & \text{VLC LOS link is blocked}.
\end{cases}
\]

其中，\(\chi_i^{\mathrm{VLC}}\) 由车辆相对位置、车辆尺寸、LED-PD 视场角以及邻车遮挡情况决定。

### 4.2 VLC 链路遮挡概率

定义第 \(i\) 条 VLC 链路的车辆遮挡概率为：

\[
P_{i}^{\mathrm{blk}}
=
f(d_i^{x},d_i^{y},w_v,l_v,h_v,\theta_{\mathrm{FOV}})
\]

其中：

| 符号 | 含义 |
|---|---|
| \(d_i^{x}\) | 发射车与接收车之间的纵向距离 |
| \(d_i^{y}\) | 发射车与接收车之间的横向偏移 |
| \(w_v\) | 车辆宽度 |
| \(l_v\) | 车辆长度 |
| \(h_v\) | 车辆高度 |
| \(\theta_{\mathrm{FOV}}\) | PD 视场角 |
| \(P_i^{\mathrm{blk}}\) | VLC LOS 链路被车辆遮挡的概率 |

VLC 链路可用概率为：

\[
a_i^{\mathrm{VLC}}=1-P_i^{\mathrm{blk}}
\]

RF 链路不受车辆 LOS 遮挡限制，可定义：

\[
a_i^{\mathrm{RF}}=1
\]

---

## 5. RF/VLC 异构信道与 SINR 模型

### 5.1 RF 链路 SINR

第 \(i\) 条 RF 链路的 SINR 可表示为：

\[
\gamma_i^{\mathrm{RF}}
=
\frac{
P_i^{\mathrm{RF}}|h_i^{\mathrm{RF}}|^2
}{
I_i^{\mathrm{RF}}+\sigma_{\mathrm{RF}}^2
}
\]

其中：

\[
I_i^{\mathrm{RF}}
=
\sum_{k\neq i}
P_k^{\mathrm{RF}} |g_{k,i}^{\mathrm{RF}}|^2
\]

表示其他车辆 RF 链路对第 \(i\) 条链路造成的同频干扰。

### 5.2 VLC 链路 SINR

第 \(i\) 条 VLC 链路的 SINR 可表示为：

\[
\gamma_i^{\mathrm{VLC}}
=
\frac{
P_i^{\mathrm{VLC}}|h_i^{\mathrm{VLC}}|^2
}{
\sigma_{\mathrm{VLC}}^2+\sigma_{\mathrm{ISI},i}^{2}
}
\]

考虑车辆遮挡后的有效 VLC 速率应乘以链路可用概率：

\[
\bar{R}_i^{\mathrm{VLC}}
=
(1-P_i^{\mathrm{blk}})
B_{\mathrm{VLC}}\log_2(1+\gamma_i^{\mathrm{VLC}})
\]

RF 链路速率为：

\[
R_i^{\mathrm{RF}}
=
B_{\mathrm{RF}}\log_2(1+\gamma_i^{\mathrm{RF}})
\]

---

## 6. 链路选择与功率联合优化问题

### 6.1 决策变量

定义链路选择变量：

\[
x_i^{\mathrm{RF}},x_i^{\mathrm{VLC}}\in \{0,1\}
\]

定义功率分配变量：

\[
P_i^{\mathrm{RF}},P_i^{\mathrm{VLC}}
\]

如果每个通信对只能选择一种链路，则：

\[
x_i^{\mathrm{RF}}+x_i^{\mathrm{VLC}}\leq 1
\]

如果允许 RF/VLC 聚合传输，则：

\[
x_i^{\mathrm{RF}}+x_i^{\mathrm{VLC}}\leq 2
\]

### 6.2 遮挡感知异构有效速率

\[
\bar R_i
=
x_i^{\mathrm{RF}}B_{\mathrm{RF}}\log_2(1+\gamma_i^{\mathrm{RF}})
+
x_i^{\mathrm{VLC}}(1-P_i^{\mathrm{blk}})
B_{\mathrm{VLC}}\log_2(1+\gamma_i^{\mathrm{VLC}})
\]

### 6.3 能效目标函数

建议以系统能效最大化作为主目标：

\[
\eta
=
\frac{
\sum_{i=1}^{N}\bar R_i
}{
\sum_{i=1}^{N}
\left(
x_i^{\mathrm{RF}}P_i^{\mathrm{RF}}
+
x_i^{\mathrm{VLC}}P_i^{\mathrm{VLC}}
\right)
+P_c
}
\]

其中，\(P_c\) 表示电路功耗或固定功耗。

### 6.4 优化问题

\[
\begin{aligned}
\max_{\mathbf{x},\mathbf{p}} \quad
&
\eta
\\
\mathrm{s.t.}\quad
&
0\leq P_i^{\mathrm{RF}}\leq P_{\max}^{\mathrm{RF}},
\\
&
0\leq P_i^{\mathrm{VLC}}\leq P_{\max}^{\mathrm{VLC}},
\\
&
x_i^{\mathrm{RF}},x_i^{\mathrm{VLC}}\in\{0,1\},
\\
&
x_i^{\mathrm{RF}}+x_i^{\mathrm{VLC}}\leq 1
\quad \text{or} \quad
x_i^{\mathrm{RF}}+x_i^{\mathrm{VLC}}\leq 2,
\\
&
\bar R_i\geq R_i^{\min},
\\
&
P_i^{\mathrm{blk}}\leq P_{\mathrm{blk}}^{\max}
\quad \text{if VLC is selected}.
\end{aligned}
\]

该问题属于混合整数非线性分式规划问题。可采用以下方法求解：

1. 链路选择与功率分配交替优化；
2. Dinkelbach 方法处理能效分式目标；
3. SCA 方法处理非凸速率项；
4. 穷举或松弛链路选择变量；
5. 低复杂度启发式算法作为工程可行方案。

---

## 7. 核心理论创新：车辆遮挡感知的 CRLB-诱导 EE 上界

这是当前方案中最适合作为理论创新点的部分。

### 7.1 需要避免的错误表述

不建议直接写：

> EE 性能的克拉美罗下界

原因是 CRLB 是针对参数估计误差的下界，而不是直接针对 EE 这种系统性能指标的下界。

更严谨的表述是：

> 先推导车辆遮挡条件下 RF/VLC 信道估计误差的 CRLB，再将该误差极限传播到 SINR、速率和 EE 中，从而得到 CRLB 诱导的 EE 上界与不可避免 EE 损失下界。

---

## 8. 车辆遮挡下 VLC 信道估计 CRLB

### 8.1 VLC 导频观测模型

设第 \(i\) 条 VLC 链路的导频观测模型为：

\[
y_i[n]
=
\chi_i^{\mathrm{VLC}}
R_{\mathrm{pd}}
P_{p}^{\mathrm{VLC}}
h_i^{\mathrm{VLC}}
s[n]
+
z_i[n]
\]

其中：

| 符号 | 含义 |
|---|---|
| \(\chi_i^{\mathrm{VLC}}\) | VLC 链路是否可用的随机变量 |
| \(R_{\mathrm{pd}}\) | PD 响应度 |
| \(P_{p}^{\mathrm{VLC}}\) | VLC 导频光功率 |
| \(h_i^{\mathrm{VLC}}\) | VLC 信道增益 |
| \(s[n]\) | 导频符号 |
| \(z_i[n]\) | VLC 观测噪声 |

若：

\[
z_i[n]\sim \mathcal{N}(0,\sigma_{\mathrm{VLC}}^2)
\]

并定义导频能量为：

\[
E_s=\sum_{n=1}^{N_p}s^2[n]
\]

### 8.2 基于平均可观测导频数的 Fisher 信息

由于车辆遮挡会减少 VLC 链路的有效可观测导频数量，因此 Fisher 信息可近似写为：

\[
\mathcal{I}(h_i^{\mathrm{VLC}})
=
\frac{
(1-P_i^{\mathrm{blk}})
(R_{\mathrm{pd}}P_p^{\mathrm{VLC}})^2E_s
}{
\sigma_{\mathrm{VLC}}^2
}
\]

因此，VLC 信道估计误差满足：

\[
\mathrm{MSE}(\hat h_i^{\mathrm{VLC}})
\geq
\varepsilon_{i,\mathrm{CRLB}}^{2,\mathrm{VLC}}
=
\frac{
\sigma_{\mathrm{VLC}}^2
}{
(1-P_i^{\mathrm{blk}})
(R_{\mathrm{pd}}P_p^{\mathrm{VLC}})^2E_s
}
\]

### 8.3 物理含义

该式说明：

1. 车辆遮挡概率 \(P_i^{\mathrm{blk}}\) 越大，可用导频越少；
2. Fisher 信息随 \(1-P_i^{\mathrm{blk}}\) 减小而降低；
3. 信道估计误差下界随遮挡概率升高而增大；
4. 车辆遮挡不仅降低 VLC 链路可用速率，也会恶化信道估计精度。

---

## 9. RF/VLC 异构 SINR 的 CRLB 修正模型

将信道估计误差视为等效自干扰，可得到 CRLB 修正后的 RF SINR：

\[
\gamma_{i,\mathrm{CRLB}}^{\mathrm{RF}}
=
\frac{
P_i^{\mathrm{RF}}|\hat h_i^{\mathrm{RF}}|^2
}{
I_i^{\mathrm{RF}}
+
\sigma_{\mathrm{RF}}^2
+
P_i^{\mathrm{RF}}\varepsilon_{i,\mathrm{CRLB}}^{2,\mathrm{RF}}
}
\]

CRLB 修正后的 VLC SINR 为：

\[
\gamma_{i,\mathrm{CRLB}}^{\mathrm{VLC}}
=
\frac{
P_i^{\mathrm{VLC}}|\hat h_i^{\mathrm{VLC}}|^2
}{
\sigma_{\mathrm{VLC}}^2
+
\sigma_{\mathrm{ISI},i}^{2}
+
P_i^{\mathrm{VLC}}\varepsilon_{i,\mathrm{CRLB}}^{2,\mathrm{VLC}}
}
\]

其中：

\[
\varepsilon_{i,\mathrm{CRLB}}^{2,\mathrm{VLC}}
=
\frac{
\sigma_{\mathrm{VLC}}^2
}{
(1-P_i^{\mathrm{blk}})
(R_{\mathrm{pd}}P_p^{\mathrm{VLC}})^2E_s
}
\]

这一步是理论推导的关键连接点：

> CRLB 不是直接作用于 EE，而是先约束信道估计误差，再通过 SINR 影响速率和 EE。

---

## 10. CRLB 诱导的 EE 上界

### 10.1 CRLB 修正后的异构有效速率

\[
\bar R_{i,\mathrm{CRLB}}
=
x_i^{\mathrm{RF}}B_{\mathrm{RF}}
\log_2
(1+\gamma_{i,\mathrm{CRLB}}^{\mathrm{RF}})
+
x_i^{\mathrm{VLC}}(1-P_i^{\mathrm{blk}})B_{\mathrm{VLC}}
\log_2
(1+\gamma_{i,\mathrm{CRLB}}^{\mathrm{VLC}})
\]

### 10.2 CRLB 诱导的系统 EE 上界

由于任意无偏信道估计器的估计误差不能低于 CRLB，在“估计误差等效为 SINR 自干扰”的建模下，系统可达到的 EE 存在如下理论上界：

\[
\boxed{
\eta^{\star}
\leq
\eta_{\mathrm{CRLB}}^{\mathrm{UB}}
=
\max_{\mathbf{x},\mathbf{p}}
\frac{
\sum_{i=1}^{N}
\left[
x_i^{\mathrm{RF}}B_{\mathrm{RF}}
\log_2
\left(
1+
\frac{
P_i^{\mathrm{RF}}|\hat h_i^{\mathrm{RF}}|^2
}{
I_i^{\mathrm{RF}}
+
\sigma_{\mathrm{RF}}^2
+
P_i^{\mathrm{RF}}\varepsilon_{i,\mathrm{CRLB}}^{2,\mathrm{RF}}
}
\right)
+
x_i^{\mathrm{VLC}}
(1-P_i^{\mathrm{blk}})
B_{\mathrm{VLC}}
\log_2
\left(
1+
\frac{
P_i^{\mathrm{VLC}}|\hat h_i^{\mathrm{VLC}}|^2
}{
\sigma_{\mathrm{VLC}}^2
+
\sigma_{\mathrm{ISI},i}^{2}
+
P_i^{\mathrm{VLC}}\varepsilon_{i,\mathrm{CRLB}}^{2,\mathrm{VLC}}
}
\right)
\right]
}
{
\sum_{i=1}^{N}
\left(
x_i^{\mathrm{RF}}P_i^{\mathrm{RF}}
+
x_i^{\mathrm{VLC}}P_i^{\mathrm{VLC}}
\right)
+
P_c
}
}
\]

### 10.3 理论含义

该公式揭示：

1. 车辆遮挡概率 \(P_i^{\mathrm{blk}}\) 同时影响 VLC 可用速率和 VLC 信道估计下界；
2. CRLB 通过等效估计误差进入 SINR 分母，导致速率和 EE 上界下降；
3. 即使采用最优链路选择和功率分配，系统 EE 仍受到信道可估计性的理论限制；
4. 这为论文提供了比普通资源分配更强的理论支撑。

---

## 11. 不可避免 EE 损失下界

### 11.1 完美 CSI 下的链路速率

对于任意链路类型 \(m\in\{\mathrm{RF},\mathrm{VLC}\}\)，定义：

\[
R_{i,\mathrm{PCSI}}^{m}
=
a_i^mB_m
\log_2
\left(
1+
\frac{
P_i^m |h_i^m|^2
}{
D_i^m
}
\right)
\]

其中：

\[
a_i^{\mathrm{RF}}=1
\]

\[
a_i^{\mathrm{VLC}}=1-P_i^{\mathrm{blk}}
\]

\[
D_i^{\mathrm{RF}}=I_i^{\mathrm{RF}}+\sigma_{\mathrm{RF}}^2
\]

\[
D_i^{\mathrm{VLC}}=\sigma_{\mathrm{VLC}}^2+\sigma_{\mathrm{ISI},i}^{2}
\]

### 11.2 CRLB 修正后的链路速率

\[
R_{i,\mathrm{CRLB}}^{m}
=
a_i^mB_m
\log_2
\left(
1+
\frac{
P_i^m |h_i^m|^2
}{
D_i^m+P_i^m\varepsilon_{i,\mathrm{CRLB}}^{2,m}
}
\right)
\]

### 11.3 由 CRLB 导致的速率损失

\[
\Delta R_i^m
=
R_{i,\mathrm{PCSI}}^{m}
-
R_{i,\mathrm{CRLB}}^{m}
\]

整理可得：

\[
\boxed{
\Delta R_i^m
=
a_i^mB_m
\log_2
\left[
\frac{
(D_i^m+P_i^m|h_i^m|^2)
(D_i^m+P_i^m\varepsilon_{i,\mathrm{CRLB}}^{2,m})
}{
D_i^m
(D_i^m+P_i^m|h_i^m|^2+P_i^m\varepsilon_{i,\mathrm{CRLB}}^{2,m})
}
\right]
}
\]

### 11.4 不可避免 EE 损失下界

若给定资源分配策略 \(\pi=\{\mathbf{x},\mathbf{p}\}\)，总功耗为：

\[
P_{\mathrm{tot}}^{\pi}
=
\sum_{i=1}^{N}
\left(
x_i^{\mathrm{RF}}P_i^{\mathrm{RF}}
+
x_i^{\mathrm{VLC}}P_i^{\mathrm{VLC}}
\right)
+
P_c
\]

则由信道估计 CRLB 导致的 EE 损失满足：

\[
\boxed{
\Delta\eta^{\pi}
\geq
\frac{
\sum_{i=1}^{N}
\sum_{m\in\{\mathrm{RF},\mathrm{VLC}\}}
x_i^m
\Delta R_i^m
}{
P_{\mathrm{tot}}^{\pi}
}
}
\]

该式可作为论文中的“推论 1”：

> 由于信道估计误差不能低于 CRLB，RF/VLC 异构系统在车辆遮挡条件下存在不可避免的 EE 损失；且该损失随遮挡概率、估计误差下界和发射功率增大而增加。

---

## 12. 链路切换临界遮挡概率

为了进一步增强理论解释性，可以推导 RF/VLC 链路选择边界。

### 12.1 RF 链路 EE

\[
\eta_i^{\mathrm{RF}}
=
\frac{
B_{\mathrm{RF}}
\log_2
\left(
1+
\frac{
P_i^{\mathrm{RF}}|\hat h_i^{\mathrm{RF}}|^2
}{
I_i^{\mathrm{RF}}
+
\sigma_{\mathrm{RF}}^2
+
P_i^{\mathrm{RF}}\varepsilon_{i,\mathrm{CRLB}}^{2,\mathrm{RF}}
}
\right)
}{
P_i^{\mathrm{RF}}+P_c^{\mathrm{RF}}
}
\]

### 12.2 VLC 链路 EE

\[
\eta_i^{\mathrm{VLC}}(P_i^{\mathrm{blk}})
=
\frac{
(1-P_i^{\mathrm{blk}})
B_{\mathrm{VLC}}
\log_2
\left(
1+
\frac{
P_i^{\mathrm{VLC}}|\hat h_i^{\mathrm{VLC}}|^2
}{
\sigma_{\mathrm{VLC}}^2
+
\sigma_{\mathrm{ISI},i}^{2}
+
P_i^{\mathrm{VLC}}
\frac{
\sigma_{\mathrm{VLC}}^2
}{
(1-P_i^{\mathrm{blk}})
(R_{\mathrm{pd}}P_p^{\mathrm{VLC}})^2E_s
}
}
\right)
}{
P_i^{\mathrm{VLC}}+P_c^{\mathrm{VLC}}
}
\]

### 12.3 链路切换边界

定义临界遮挡概率：

\[
\boxed{
P_{i,\mathrm{th}}^{\mathrm{blk}}
=
\left\{
P_i^{\mathrm{blk}}
\ \middle|\
\eta_i^{\mathrm{VLC}}(P_i^{\mathrm{blk}})
=
\eta_i^{\mathrm{RF}}
\right\}
}
\]

链路选择规则可写为：

\[
\begin{cases}
\text{select VLC}, & P_i^{\mathrm{blk}}<P_{i,\mathrm{th}}^{\mathrm{blk}},\\
\text{select RF}, & P_i^{\mathrm{blk}}\geq P_{i,\mathrm{th}}^{\mathrm{blk}}.
\end{cases}
\]

如果允许聚合传输，则可以进一步定义：

\[
\text{select RF/VLC aggregation}
\]

用于 \(P_i^{\mathrm{blk}}\) 接近阈值、RF 干扰中等且两类链路均可用的过渡区域。

---

## 13. 可包装成论文的理论贡献

建议将理论部分包装为：

### 定理 1：车辆遮挡感知的 VLC 信道估计 CRLB

在给定车辆遮挡概率 \(P_i^{\mathrm{blk}}\) 和 VLC 导频观测模型的条件下，第 \(i\) 条 VLC 链路的任意无偏信道估计器满足：

\[
\mathrm{MSE}(\hat h_i^{\mathrm{VLC}})
\geq
\frac{
\sigma_{\mathrm{VLC}}^2
}{
(1-P_i^{\mathrm{blk}})
(R_{\mathrm{pd}}P_p^{\mathrm{VLC}})^2E_s
}
\]

### 定理 2：CRLB 诱导的 RF/VLC 异构系统 EE 上界

在估计误差等效为 SINR 自干扰的条件下，系统可达到的 EE 不超过：

\[
\eta_{\mathrm{CRLB}}^{\mathrm{UB}}
\]

该上界由链路选择变量、功率分配变量、RF 干扰、VLC 遮挡概率和信道估计 CRLB 共同决定。

### 推论 1：不可避免 EE 损失下界

相对于完美 CSI 情况，由信道估计极限导致的 EE 损失不低于：

\[
\Delta\eta^{\pi}
\geq
\frac{
\sum_{i=1}^{N}
\sum_{m\in\{\mathrm{RF},\mathrm{VLC}\}}
x_i^m
\Delta R_i^m
}{
P_{\mathrm{tot}}^{\pi}
}
\]

### 推论 2：RF/VLC 链路切换临界遮挡概率

存在临界遮挡概率 \(P_{i,\mathrm{th}}^{\mathrm{blk}}\)，使得：

\[
\eta_i^{\mathrm{VLC}}(P_i^{\mathrm{blk}})
=
\eta_i^{\mathrm{RF}}
\]

当遮挡概率超过该阈值时，系统应从 VLC 切换至 RF 或采用 RF/VLC 聚合传输。

---

## 14. 算法设计建议

### 14.1 算法名称候选

**Blockage-Aware CRLB-Constrained Energy-Efficient Resource Allocation Algorithm**

可简称为：

**BAC-EEA**

或：

**CRLB-BARA**

Blockage-Aware Resource Allocation with CRLB-induced EE Bound

### 14.2 算法框架

1. 输入车辆位置、车辆尺寸、RF/VLC 信道参数、干扰功率、噪声功率；
2. 计算每条 VLC 链路的车辆遮挡概率 \(P_i^{\mathrm{blk}}\)；
3. 根据遮挡概率计算 VLC 信道估计 CRLB；
4. 计算 RF/VLC 的 CRLB 修正 SINR；
5. 根据临界遮挡概率进行候选链路筛选；
6. 固定链路选择，利用 Dinkelbach 方法或 SCA 优化功率；
7. 固定功率，更新链路选择；
8. 迭代直到 EE 收敛。

### 14.3 伪代码结构

```text
Algorithm: Vehicle-Blockage-Aware CRLB-Induced EE Resource Allocation

Input:
    Vehicle positions, RF/VLC channel parameters, noise power,
    maximum transmit power, minimum rate constraint.

Output:
    Link selection variables x_RF, x_VLC,
    power allocation P_RF, P_VLC.

1. Initialize link selection and transmit power.
2. For each V2V pair i:
       Calculate vehicle blockage probability P_blk_i.
       Calculate VLC channel estimation CRLB.
       Calculate CRLB-corrected RF/VLC SINR.
3. Determine candidate RF/VLC links using blockage threshold.
4. Repeat:
       Fix link selection and update power allocation.
       Fix power allocation and update link selection.
       Update system EE.
   Until convergence.
5. Return optimal link selection and power allocation.
```

---

## 15. 仿真设计建议

### 15.1 对比算法

至少设置以下对比方法：

1. **VLC-only**：只使用 VLC，不考虑 RF 备份；
2. **RF-only**：只使用 RF；
3. **RSS-based RF/VLC selection**：基于接收信号强度的传统链路选择；
4. **Blockage-aware selection without CRLB**：考虑遮挡但不考虑信道估计极限；
5. **Proposed CRLB-induced EE optimization**：本文方法。

### 15.2 关键仿真图

建议至少做以下图：

1. EE vs. 车辆遮挡概率 \(P_i^{\mathrm{blk}}\)；
2. EE vs. RF 同频干扰功率；
3. EE vs. V2V 通信距离；
4. EE vs. VLC 导频功率；
5. EE vs. 信道估计误差；
6. RF/VLC 链路选择比例 vs. 遮挡概率；
7. 临界遮挡概率 \(P_{i,\mathrm{th}}^{\mathrm{blk}}\) 随 RF 干扰功率变化；
8. 系统 EE 收敛曲线；
9. CRLB 理论上界与实际算法 EE 的差距；
10. 不可避免 EE 损失下界随遮挡概率变化。

### 15.3 预期结果逻辑

预期应体现：

1. 车辆遮挡概率升高时，VLC-only 性能快速下降；
2. RF 干扰增强时，RF-only 性能下降；
3. RF/VLC 异构方法在两类不确定性下更稳定；
4. 考虑 CRLB 的方法相比不考虑估计误差的方法更鲁棒；
5. 所提算法 EE 接近 CRLB 诱导上界，但不会超过该上界；
6. 临界遮挡概率随 RF 干扰增强而升高，因为 RF 越差，系统越倾向于保留 VLC。

---

## 16. 论文结构建议

### 1. Introduction

建议突出三点：

1. V2V 通信对可靠性和能效有较高要求；
2. VLC 和 RF 具有互补性，但各自存在局限；
3. 现有 RF/VLC 异构资源分配研究多关注链路选择或功率控制，较少从“车辆遮挡导致的信道可估计性退化”角度推导 EE 理论极限。

### 2. Related Work

分为三类：

1. V2V-VLC 信道建模与遮挡分析；
2. RF/VLC 异构通信资源分配；
3. 信道估计误差、CRLB 与能效优化。

### 3. System Model

包括：

1. RF/VLC 异构 V2V 场景；
2. 车辆遮挡概率模型；
3. RF 同频干扰模型；
4. VLC LOS/NLOS 信道模型；
5. RF/VLC 速率和 EE 定义。

### 4. CRLB-Induced EE Bound

这是理论核心章节，包括：

1. 车辆遮挡下 VLC 信道估计 CRLB；
2. CRLB 修正 RF/VLC SINR；
3. CRLB 诱导 EE 上界；
4. 不可避免 EE 损失下界；
5. 链路切换临界遮挡概率。

### 5. Proposed Resource Allocation Algorithm

包括：

1. 优化问题建模；
2. 分式规划转换；
3. 链路选择与功率控制交替优化；
4. 收敛性与复杂度分析。

### 6. Simulation Results

包括：

1. 参数设置；
2. 基线算法；
3. 性能对比；
4. 理论上界验证；
5. 消融实验。

### 7. Conclusion

强调：

1. 建立车辆遮挡感知 RF/VLC 异构通信模型；
2. 推导 CRLB 诱导 EE 上界；
3. 设计链路选择与功率联合优化方法；
4. 仿真验证所提方法在车辆遮挡和 RF 干扰下的能效优势。

---

## 17. 可写成论文贡献的版本

建议贡献点写成三条，不要太多：

1. **车辆遮挡感知 RF/VLC 异构 V2V 通信模型。**  
   构建同时考虑 VLC 车辆遮挡和 RF 同频干扰的异构 V2V 通信模型，将车辆几何遮挡概率引入 VLC 链路可用性与有效速率表征。

2. **CRLB 诱导的 EE 理论界。**  
   推导车辆遮挡条件下 VLC 信道估计 CRLB，并进一步建立 CRLB 修正的 RF/VLC SINR 模型，得到系统 EE 上界和不可避免 EE 损失下界，从理论上揭示遮挡概率、信道估计精度与异构链路能效之间的耦合关系。

3. **链路选择与功率联合优化算法。**  
   面向系统 EE 最大化，构建混合整数非线性分式规划问题，并提出低复杂度交替优化算法，实现车辆遮挡和 RF 同频干扰条件下的 RF/VLC 链路选择与功率控制。

---

## 18. 需要注意的风险

### 18.1 CRLB 不能直接叫“EE 的克拉美罗下界”

推荐表述：

> CRLB-induced EE upper bound

或：

> CRLB-constrained EE performance bound

中文可写：

> CRLB 诱导的 EE 上界

或：

> 信道估计极限约束下的 EE 性能界

### 18.2 车辆变道不要建得太复杂

不要把题目写成“车辆变道行为建模”，否则审稿人可能要求：

1. 变道决策模型；
2. 轨迹预测；
3. 交通流数据；
4. 真实驾驶行为验证。

建议写“车辆遮挡感知”，聚焦通信链路遮挡，而不是交通行为建模。

### 18.3 不要和后续 GNN-RL 论文重复

这篇论文应定位为：

> 小规模、可解释、理论推导型、优化方法型。

后续 GNN-RL 论文定位为：

> 多车辆动态拓扑、图结构感知、分布式智能资源分配。

两篇分工必须明确。

---

## 19. 推荐最终定位

最终推荐定位为：

> 本文面向车辆遮挡与 RF 同频干扰共存的 V2V 通信场景，构建 RF/VLC 异构链路模型，并将车辆遮挡概率引入 VLC 链路可用性与信道估计精度分析。在此基础上，推导车辆遮挡条件下的 VLC 信道估计 CRLB，进一步建立 CRLB 诱导的系统 EE 上界和不可避免 EE 损失下界。最后，针对系统 EE 最大化问题，设计链路选择与功率联合优化算法，实现车辆遮挡感知的 RF/VLC 异构 V2V 资源分配。

---

## 20. 一句话总结

这篇论文最适合从原来的：

> 行人干扰下的 MIMO V2V-VLC 链路选择性能分析

升级为：

> 车辆遮挡感知的 RF/VLC 异构 V2V 能效资源分配，并以 CRLB 诱导 EE 上界作为核心理论创新。

