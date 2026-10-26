# 50/100车共享学习实现（未按当前模型验收）

`src/masac.py`：混合离散动作和连续功率、回放和更新；`models.py`：MLP/异构图编码器与actor/critic；`features.py`：状态特征；`credit.py`：合法反事实诊断。

这是已有开发代码，不能因文件存在、单测通过或reference回退执行成功就称“RL已有效”。尤其要检查奖励及预算是否仍为旧min-EE/GEE语义，必须与当前平均逐链路EE规范对齐后再训练。

50车的探针在 ../n50/diagnostics/。历史验收原文在 ../../research_history/documents/learning/；高回退、功率上漂、细差排序不足等负结果均保留。
