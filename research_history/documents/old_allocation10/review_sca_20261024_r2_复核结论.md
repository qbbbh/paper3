# 10车 SCA + Dinkelbach 第二轮独立复核

日期：2026-10-24。仅复核和运行测试，没有修改分配器源码。

## 结论

具备继续短时开发验证的条件，三处原阻断错误已修复；尚不能标为完整验收通过。还需修正求解失败被标为 stationary/success，以及求解器详细状态没有写入最终 actions.json 两处问题。不需要换掉 SLSQP，也不需要因此恢复 QoS 硬约束。

## 独立实跑

使用 D:/anaconda/python.exe，参数 -X utf8 -B -I。

- test_handcalc：12/12；test_rate_model：4/4；test_pool：5/5。
- test_sca：6/6；test_stages：5/5；test_dinkelbach：7/7。合计39/39。
- 解析速率梯度相对差 4.2257e-9；SCA 界梯度相对差 1.6102e-9。
- 普通差式目标改善136.6538（单位Mbit/s，测试输出标签mbps_w不准确）；Stage B夹具最低EE从7.0451提高至45.3949 Mbit/J。
- run_allocation10.py --blocks 5 --self-check --out diagnostics/review_sca_20261024_r2 --tag independent_review_r2。
- 五块均返回ok；单块预测与优化约0.32–0.36秒，程序总耗时4.475秒。5块只覆盖0.5秒物理时间。
- 与predict_block的速率最大相对差2.763e-16。
- 五块全部五条RF链路、每条功率0.001W。
- 实际块EE均值37482.997 Mbit/J；实际块最低链路EE均值5766.222 Mbit/J。仅开发短跑，无算法排名结论。
- q_mbps=0，q_eval回退到0，因此blocks_ok_actual=5没有正速率门限的QoS证明意义。

## 仍存在的状态错误及复现

allocation10/inner.py第398行无条件solved=True，第400行进入_accept。没有根据si中的solver_ok和violation判断是否真正解好子问题。第439/463行再把stationary当作success。

在tests/test_sca.py的RF合成夹具上，起点[0.05,1.2]，lambda取起点网络EE，调用make_sca_ordinary_inner(problem, rounds=2, maxiter=1)：

```json
{"p":[0.05,1.2],"success":true,"status":"stationary","gain":0.0,
 "solver":{"solver_ok":false,"solver_status":9,
 "solver_message":"Iteration limit reached","nit":1,"violation":0.0}}
```

这证明失败状态仍可冒充驻点。正常预算39项测试通过不能覆盖此故障。建议加入这个回归测试：迭代上限截断且未移动时，不得标为已收敛。若保留失败求解器返回的可行改善点，应单独标为inexact_improved，不得由此声称驻点或子问题最优。真正驻点需要成功求解或独立最优性检验。

另外，本次actions.json仅保存stage_a/stage_b的inner_status，递归搜索solver_ok字段结果为零。应将solver success/status/message/nit/violation和每次域重试记录穿过numerical/dinkelbach.py与allocation10/stage.py写入最终输出，不能只停留在内层返回字典。

## 功率下界解释的边界

五块的全RF、1mW是已观察事实；不能据此断言q=0时整个混合RF/VLC问题的最优解都在下界。固定其他干扰时，单RF链路R/p随自身功率下降而提高；这不等价于干扰耦合网络中每个功率坐标的最优值都必须为最小值。VLC信号项为p²，低功率处R/p可以随p上升。现有合成测试本身已展示VLC类内点最优。

修复状态问题后，固定候选匹配再作功率扰动/充分求解核对，才能区分真实边界解与过早停止。不能只为了让连续功率模块有动作而恢复用户已取消的QoS优化约束。

## 算法定位

尺度化线性功率下，二次变换构造凹下界的路线成立；SLSQP可作为这些子问题的数值求解器，仍属于SCA+Dinkelbach结构。局部SCA和有限匹配池不提供原混合非凸问题的全局最优保证。stationary目前只是实现中的停止标签，不能直接写成理论驻点证明。
