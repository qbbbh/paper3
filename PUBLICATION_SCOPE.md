# Paper3 审核材料与校验

- `releases/paper3_20261110` 提供现行实现、固定配置、测试、依赖和来源哈希。
- `data/` 直接提供经校验的交通和建筑输入。10 车与 50 车用于正式结果，100 车输入用于几何验证。
- `evidence/paper3_20261110` 提供完整周期表、逐链路表、旧对照结果、失败与后备记录、预定原始样例和独立审计。
- `manuscripts/paper3_20261111` 提供当前中文稿、论证和作者写作经验。
- [研究思路演变](RESEARCH_EVOLUTION.md)记录研究选择及历史依据。[固定历史提交](https://github.com/qbbbh/paper3/tree/6ae23c04c091edee70e228cae39d72efb2af04a8)提供此前公开的完整源码和文档。
- [Paper5 独立仓库](https://github.com/qbbbh/paper5-revision)提供该项目的现行规格。

完整历史 replay 需要本地原运行文件和对应恢复点。公开样例支持速率、EE 和服务条件的复算。正式结果的来源哈希与当前分发文件的校验值各自记录在 provenance 和根清单中。

```bash
python tools/verify_snapshot.py
python releases/paper3_20261110/tools/check_release.py
python evidence/paper3_20261110/verify_package.py
python tools/verify_review_publication.py
```
