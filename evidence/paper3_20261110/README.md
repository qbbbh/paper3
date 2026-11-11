# Paper3 20261110 数值证据

完整内容、统计口径和来源说明见 **[DATA_CATALOG](DATA_CATALOG.md)**。

仅用网页阅读时，先打开 **[逐方法小表索引](tables/by_method/README.md)**：三组各8个方法，共24份CSV，每份1000个计划周期，含失败。小表是原完整周期表的无损拆分，数值和列定义均未改变。

- [主方法结果报告](reports/10车与50车_功率改进全量核查_20261110.md)
- [同拓扑消融报告](reports/同拓扑消融_补算复用与全量结论.md)
- [外部审查问题](REVIEW_QUESTIONS.md)
- [字段与单位](tables/SCHEMA.json)、[配置与种子](tables/GROUP_CONFIGS.json)
- [验证记录](VALIDATION_REPORT.json)、[来源清单](SOURCE_MANIFEST.json)、[发布文件清单](PACKAGE_MANIFEST.json)

原完整周期表、全部逐链路表、原始对比结果与固定验证数组仍保留。完整恢复数据留在本地，未随本包上传；其用途与复算边界见DATA_CATALOG。

本包只保留当前只读核查入口`verify_package.py`。历史审计脚本和一次性导出工具已从当前目录移除，原来源与哈希保留在来源清单中；需要追溯实现时，可查[固定历史提交中的证据包](https://github.com/qbbbh/paper3/tree/6ae23c04c091edee70e228cae39d72efb2af04a8/evidence/paper3_20261110)。[改进前诊断与研究转变](baseline_records/pre_guard_cross_scale/README.md)仍完整保留。
