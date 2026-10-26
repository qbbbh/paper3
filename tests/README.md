# 单测与独立审计

`source_snapshot/` 原样保存当前快照单测；`auditors/` 原样保存审计器；`fixtures/` 是固定数值夹具。通过 tools/run.py test 在还原布局下运行，避免移动源码引入假失败。

测试、源码哈希、CSV/NPZ复算分别回答不同问题。全绿不证明全局最优、覆盖率达标、MASAC学会、实时可用或论文贡献成立。
