# 第3篇节点图式 SINR 与统一符号设计

> 核心原则：第3篇不再把系统写成“链路集合 \(\mathcal L\)”上的优化问题，而是写成**车辆节点图**上的分布式资源分配问题。车辆是图节点，节点之间的可通信关系、干扰关系、可见光视距关系作为边或邻接矩阵存在；SINR 是车辆节点动作产生的通信质量指标，而不是独立链路编号 \(\ell\) 的属性。

## 1. 图建模主线

| 含义 | 建议符号 | 说明 |
|---|---|---|
| 车辆节点集合 | \(\mathcal V(t)=\{1,\dots,N(t)\}\) | 第3篇所有智能体均为车辆节点 |
| 车辆节点 | \(i,j,k\) | \(i\) 通常表示决策车辆，\(j\) 表示其目标接收车辆或邻居车辆，\(k\) 表示潜在干扰车辆 |
| 动态车辆图 | \(\mathcal G_t=(\mathcal V(t),\mathcal E_t)\) | 由车辆位置、FOV、通信距离和干扰关系共同决定 |
| 通信可达邻接 | \(A_{ij}^m(t)\in\{0,1\}\) | 车辆 \(i\) 采用方式 \(m\) 时，是否可与车辆 \(j\) 通信 |
| 干扰邻接 | \(C_{kj}^{m,c}(t)\in\{0,1\}\) | 车辆 \(k\) 是否在方式 \(m\)、资源块/信道 \(c\) 上干扰车辆 \(j\) |
| 节点特征 | \(\mathbf x_i(t)\) | 位置、速度、方向、队列、信道估计、邻居统计、历史干扰等 |
| 边特征 | \(\mathbf e_{ij}(t)\) | 距离、相对角度、VLC FOV、RF 路损、估计误差、遮挡状态等 |
| 车辆邻域 | \(\mathcal N_i^m(t)=\{j:A_{ij}^m(t)=1\}\) | GNN 聚合和候选接收节点集合 |
| 接收目标集合 | \(\mathcal R_i(t)\subseteq \mathcal N_i^{m_i}(t)\) | 单播时为一个车辆，广播/安全消息时为多个邻居 |

## 2. 节点动作设计

第3篇建议把资源分配动作写成车辆节点动作：

\[
a_i(t)=\big(m_i(t),c_i(t),P_i(t)\big)
\]

其中：

| 含义 | 建议符号 | 说明 |
|---|---|---|
| 通信方式 | \(m_i(t)\in\mathcal M\) | \(\mathcal M=\{\mathrm{VLC},\mathrm{RF}\}\) |
| 信道/资源块 | \(c_i(t)\in\mathcal C_{m_i}\) | RF 可为频谱资源块，VLC 可为光信道/时隙 |
| 发射功率 | \(P_i(t)\) | 如需区分方式，可写 \(P_i^m(t)\)，但正文优先用 \(P_i(t)\) |
| 是否启用某方式 | \(x_i^m(t)\in\{0,1\}\) | 若需要混合整数表达，可令 \(\sum_m x_i^m(t)=1\) |
| 中继候选节点 | \(r_i(t)\) | 仅在需要中继时使用；不是链路编号 |

不建议再使用：

| 不建议符号 | 替代写法 |
|---|---|
| \(\ell=(i,j)\) | 直接使用车辆节点 \(i,j\) 与邻接矩阵 \(A_{ij},C_{ij}\) |
| \(x_\ell^m\) | \(x_i^m\) 或 \(m_i\) |
| \(P_\ell^m\) | \(P_i^m\) 或 \(P_i\) |
| \(\Gamma_\ell^m\) | \(\Gamma_i^m\) 或 \(\Gamma_i^{\mathrm{eq}}\) |
| \(R_\ell^m\) | \(R_i^m\) 或 \(R_i\) |

## 3. 节点式 SINR

### 3.1 接收节点层面的瞬时质量

为了保留物理意义，可以在公式内部使用 \(i,j,k\) 表示“车辆 \(i\) 的数据在车辆 \(j\) 处被接收”，但不引入链路对象 \(\ell\)：

\[
\gamma_{ij}^{m}(t)=
\frac{
P_i(t)\left|\hat h_{ij}^{m}(t)\right|^2
}{
\sigma_m^2
+\sum_{k\in\mathcal V(t)\setminus\{i\}}
C_{kj}^{m,c_i}(t)P_k(t)\left|\hat h_{kj}^{m}(t)\right|^2
+P_i(t)\varepsilon_{ij,\mathrm{CRLB}}^{2,m}(t)
}.
\]

这里的重点不是定义“链路 \((i,j)\)”，而是说明车辆图中节点 \(j\) 对车辆 \(i\) 消息的接收质量。

### 3.2 车辆节点级 SINR

第3篇正文建议使用节点级指标：

\[
\Gamma_i^{m_i}(t)=
\operatorname{Agg}_{j\in\mathcal R_i(t)}
\gamma_{ij}^{m_i}(t).
\]

聚合函数按任务选：

| 场景 | 建议聚合 |
|---|---|
| 单播 V2V | \(\operatorname{Agg}=\gamma_{iu_i}^{m_i}\)，其中 \(u_i(t)\) 是车辆 \(i\) 的目标接收节点 |
| 安全广播 | \(\operatorname{Agg}=\min_{j\in\mathcal R_i}\gamma_{ij}^{m_i}\)，保证最差邻居可靠性 |
| 吞吐优先 | \(\operatorname{Agg}=\frac{1}{|\mathcal R_i|}\sum_{j\in\mathcal R_i}\gamma_{ij}^{m_i}\) |
| 加权业务 | \(\operatorname{Agg}=\sum_{j\in\mathcal R_i}w_{ij}\gamma_{ij}^{m_i}\) |

这样后续 GNN/MASAC 可以直接把 \(\Gamma_i^{m_i}(t)\) 作为车辆节点 \(i\) 的局部通信质量，不需要再维护链路编号。

## 4. CRLB 修正项进入 SINR 分母

第3篇可以保留你准备放入大论文的核心设计：信道估计误差下界进入 SINR 分母，作为估计误差导致的等效自干扰。

| 含义 | 建议符号 | 说明 |
|---|---|---|
| 真实节点间信道 | \(h_{ij}^m(t)\) | 车辆 \(i\) 到车辆 \(j\)，通信方式 \(m\) |
| 信道估计值 | \(\hat h_{ij}^m(t)\) | 由导频/MMSE/Kalman 等估计得到 |
| 估计误差 | \(\Delta h_{ij}^m(t)=h_{ij}^m(t)-\hat h_{ij}^m(t)\) | 不作为主优化变量 |
| CRLB 误差下界 | \(\varepsilon_{ij,\mathrm{CRLB}}^{2,m}(t)\) | 可由 Fisher 信息矩阵或等效观测模型得到 |
| 节点聚合 CRLB | \(\varepsilon_{i,\mathrm{CRLB}}^{2,m}(t)\) | 若正文只写节点指标，可对 \(j\in\mathcal R_i\) 聚合 |

CRLB 修正后的节点级 SINR：

\[
\Gamma_{i,\mathrm C}^{m_i}(t)=
\operatorname{Agg}_{j\in\mathcal R_i(t)}
\frac{
P_i(t)\left|\hat h_{ij}^{m_i}(t)\right|^2
}{
\sigma_{m_i}^2
+\sum_{k\ne i}
C_{kj}^{m_i,c_i}(t)P_k(t)\left|\hat h_{kj}^{m_i}(t)\right|^2
+P_i(t)\varepsilon_{ij,\mathrm{CRLB}}^{2,m_i}(t)
}.
\]

如果正文需要更简洁的节点表达，可写成：

\[
\Gamma_{i,\mathrm C}^{m_i}(t)=
\frac{
P_i(t)G_i^{m_i}(t)
}{
\sigma_{m_i}^2+I_i^{m_i}(t)+P_i(t)\varepsilon_{i,\mathrm{CRLB}}^{2,m_i}(t)
},
\]

其中：

\[
G_i^{m_i}(t)=\operatorname{Agg}_{j\in\mathcal R_i}\left|\hat h_{ij}^{m_i}(t)\right|^2,
\quad
I_i^{m_i}(t)=\operatorname{Agg}_{j\in\mathcal R_i}
\sum_{k\ne i}C_{kj}^{m_i,c_i}(t)P_k(t)\left|\hat h_{kj}^{m_i}(t)\right|^2.
\]

这就是第3篇最推荐的正文写法：**节点 \(i\) 的信号增益、干扰和 CRLB 修正项全部是车辆图上的节点/邻域聚合量**。

## 5. VLC/RF 等效 SINR

如果 VLC 与 RF 带宽不同，为了统一资源分配目标，可以定义节点级等效 SINR：

\[
\Gamma_i^{\mathrm{eq}}(t)=
\left(1+\Gamma_{i,\mathrm C}^{m_i}(t)\right)^{B_{m_i}/B_0}-1,
\]

其中 \(B_0\) 是统一参考带宽，通常可取 \(B_{\mathrm{RF}}\) 或仿真中的基准带宽。

节点速率写成：

\[
R_i(t)=B_0\log_2\left(1+\Gamma_i^{\mathrm{eq}}(t)\right).
\]

如果不需要跨制式等效，也可以直接写：

\[
R_i^{m_i}(t)=B_{m_i}\log_2\left(1+\Gamma_{i,\mathrm C}^{m_i}(t)\right).
\]

建议全文只保留一种主写法。若第3篇强调 VLC/RF 统一调度，优先使用 \(\Gamma_i^{\mathrm{eq}}\)。

## 6. Dinkelbach 能效优化的节点式写法

系统能效：

\[
\eta(\mathbf a,\mathbf P)=
\frac{\sum_{i\in\mathcal V}R_i(t)}
\sum_{i\in\mathcal V}P_i(t)+\sum_{i\in\mathcal V}P_{c,i}^{m_i}}.
\]

Dinkelbach 第 \(s\) 次迭代的子问题：

\[
\max_{\{a_i,P_i\}_{i\in\mathcal V}}
\sum_{i\in\mathcal V}R_i(t)
-q^{(s)}
\left(
\sum_{i\in\mathcal V}P_i(t)+\sum_{i\in\mathcal V}P_{c,i}^{m_i}
\right).
\]

建议约束写成节点约束：

| 约束 | 节点式表达 |
|---|---|
| 功率约束 | \(0\le P_i(t)\le P_{i,\max}^{m_i}\) |
| 通信方式选择 | \(m_i(t)\in\{\mathrm{VLC},\mathrm{RF}\}\) 或 \(\sum_m x_i^m=1\) |
| SINR 约束 | \(\Gamma_{i,\mathrm C}^{m_i}(t)\ge \Gamma_{i,\min}\) |
| 速率约束 | \(R_i(t)\ge R_{i,\min}\) |
| 干扰约束 | \(I_i^{m_i}(t)\le I_{i,\max}\) |
| 图连通性约束 | \(\sum_j A_{ij}^{m_i}(t)\ge d_i^{\min}\) |

## 7. GNN-MASAC 符号承接

GNN 的输入：

\[
\mathbf X_t=[\mathbf x_1(t),\dots,\mathbf x_{N(t)}(t)]^\top,
\quad
\mathbf E_t=\{\mathbf e_{ij}(t):A_{ij}(t)=1\}.
\]

节点嵌入：

\[
\mathbf z_i(t)=\mathrm{GNN}_\omega\left(\mathbf x_i(t),\{\mathbf x_j(t),\mathbf e_{ij}(t):j\in\mathcal N_i(t)\}\right).
\]

MASAC 策略：

\[
a_i(t)\sim \pi_{\theta_i}\left(a_i\mid o_i(t),\mathbf z_i(t)\right).
\]

建议局部奖励：

\[
r_i(t)=
\omega_R R_i(t)
-\omega_P P_i(t)
-\omega_I I_i^{m_i}(t)
-\omega_Q\left[\Gamma_{i,\min}-\Gamma_{i,\mathrm C}^{m_i}(t)\right]^+
-\omega_H H_i(t),
\]

其中 \(H_i(t)\) 可表示切换代价、队列惩罚或 AoI 惩罚。若使用 Shapley 信用分配，建议记为 \(\rho_i(t)\)，避免和 SINR、功率、Dinkelbach 参数冲突。

## 8. 恶意干扰博弈章节的衔接

后续恶意干扰章节仍保持节点图写法：

| 含义 | 建议符号 |
|---|---|
| 可疑/恶意节点概率 | \(p_j^{\mathrm{mal}}(t)\) |
| 恶意干扰功率 | \(P_j^{\mathrm{jam}}(t)\) |
| 干扰图 | \(\mathcal G_t^{\mathrm{jam}}=(\mathcal V,\mathcal E_t^{\mathrm{jam}})\) |
| 干扰邻接 | \(J_{ji}(t)\) 或 \(C_{ji}^{\mathrm{jam}}(t)\) |
| 正常车辆效用 | \(U_i^{\mathrm{veh}}(t)\) |
| 干扰方效用 | \(U_j^{\mathrm{jam}}(t)\) |
| Stackelberg 策略 | \(\pi_i^{\mathrm L}\), \(\pi_j^{\mathrm F}\) |

恶意干扰进入节点级 SINR 分母：

\[
\Gamma_{i,\mathrm C}^{m_i}(t)=
\frac{
P_i(t)G_i^{m_i}(t)
}{
\sigma_{m_i}^2
+I_i^{m_i}(t)
+I_i^{\mathrm{jam}}(t)
+P_i(t)\varepsilon_{i,\mathrm{CRLB}}^{2,m_i}(t)
},
\]

\[
I_i^{\mathrm{jam}}(t)=
\sum_{j\in\mathcal V}
p_j^{\mathrm{mal}}(t)J_{ji}(t)P_j^{\mathrm{jam}}(t)g_{ji}^{\mathrm{jam}}(t).
\]

这样第3篇的 GNN-MASAC 和后续干扰博弈可以共用同一套节点图符号。

## 9. 推荐全文固定的核心符号

| 符号 | 固定含义 |
|---|---|
| \(i,j,k\) | 车辆节点索引 |
| \(\mathcal G_t\) | 动态车辆图 |
| \(A_{ij}^m(t)\) | 通信可达邻接 |
| \(C_{ij}^{m,c}(t)\) | 同方式/同资源干扰邻接 |
| \(\mathbf x_i(t)\) | 车辆节点特征 |
| \(\mathbf e_{ij}(t)\) | 图边特征 |
| \(m_i(t)\) | 车辆 \(i\) 的通信方式 |
| \(c_i(t)\) | 车辆 \(i\) 的资源块/信道 |
| \(P_i(t)\) | 车辆 \(i\) 的发射功率 |
| \(\hat h_{ij}^m(t)\) | 车辆 \(i\) 到车辆 \(j\) 的估计信道 |
| \(\varepsilon_{i,\mathrm{CRLB}}^{2,m}(t)\) | 节点级 CRLB 估计误差下界 |
| \(\Gamma_{i,\mathrm C}^{m_i}(t)\) | CRLB 修正后的节点级 SINR |
| \(\Gamma_i^{\mathrm{eq}}(t)\) | 节点级等效 SINR |
| \(R_i(t)\) | 节点速率 |
| \(\eta\) | 系统能效 |
| \(q^{(s)}\) | Dinkelbach 第 \(s\) 次迭代参数 |
| \(\mathbf z_i(t)\) | GNN 节点嵌入 |
| \(a_i(t)\) | 节点动作 |
| \(r_i(t)\) | 节点奖励 |
| \(p_j^{\mathrm{mal}}(t)\) | 节点恶意概率 |

## 10. 写作时的一句话定位

可以在论文中这样交代：

> Unlike link-indexed formulations, the proposed model treats each vehicle as a graph node. The communication mode, channel resource, and transmit power are selected as node actions, while the feasible communication relation and co-channel interference are encoded by dynamic graph adjacency matrices. The CRLB-induced channel estimation error is incorporated into the denominator of the node-level equivalent SINR, enabling a unified graph-based formulation for RF/VLC heterogeneous resource allocation.

对应中文意思：

> 本文不再将 RF/VLC 异构通信建模为链路编号集合上的优化问题，而是将车辆视为动态图节点。通信方式、信道资源和发射功率是车辆节点动作；通信可达性和同资源干扰由动态图邻接矩阵描述；信道估计 CRLB 修正项进入节点级等效 SINR 分母，从而为 RF/VLC 异构车辆网络资源分配提供统一的图建模框架。
