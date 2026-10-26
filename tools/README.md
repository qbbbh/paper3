# 仓库维护工具，不是研究算法

之前这里只有校验和大文件还原两个脚本，因为算法放在别处；这不是“代码只剩两个文件”。现在用途明确如下。

|脚本|用途|是否改变算法/仿真|
|---|---|---|
|restore_data.py|将data_parts还原成原始二进制，检查分段和整文件SHA-256|不改变数据、不运行仿真|
|build_runtime.py|按来源映射还原原目录/导入，验证原source_sha256|只复制字节，不修改科学源码|
|verify_snapshot.py|核对本次整理包的所有文件、源码与哈希|只检查|
|export_scene_csv.py|把三种规模的NPZ导出完整CSV，并逐字段回读比较|不生成新场景、不重跑SUMO|
|run.py|调用上述准备以及原始test/audit/allocation/foundation等入口|只有显式选择仿真任务时才运行算法|

运行算法需遵循RUNBOOK。`.runtime`是可重建的执行布局，不是第二份独立维护的算法。Chat审查科学实现应看channel_estimation与resource_allocation，不应只看本目录。
