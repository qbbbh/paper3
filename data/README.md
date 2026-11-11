# 当前 Paper3 输入数据

`n10/`、`n50/`、`n100/` 各保留四个原始输入文件：`mobility.npz`、`buildings.npz`、`scenario.json`、`audit.json`。12 个文件共 7,112,872 字节，均与原始场景逐字节一致；旧目录映射、字节数和 SHA-256 见 [INPUT_MANIFEST.json](INPUT_MANIFEST.json)。

- `n10`：用于 10A/10B，两组共享交通轨迹，信道种子不同。
- `n50`：用于 50B。
- `n100`：仅用于已有几何兼容性测试，没有本轮 100 车算法性能结论。

这些文件直接由 Git 保存。新克隆不需要运行历史 `restore_data.py`，也不依赖 `data_parts` 或本机 E 盘。程序仍使用原 `Scene` 及物理模型，只把目录定位改为本目录。

从仓库根目录执行 `python releases/paper3_20261110/tools/check_release.py`，即可核验全部 12 个输入、三个固定配置和现行分发源码。原始运行检查点和执行细数组不在这里，不能用交通文件代替保存状态复算所需的数据。
