# 共用运行与数据管理

`entrypoints/run_allocation.py` 是原始通用入口；`src/data.py` 是数据分区与哈希；`src/__init__.py` 是包文件。共同算法与学习算法的导入布局由根目录 tools/build_runtime.py 从来源映射还原，不改科学源码。
