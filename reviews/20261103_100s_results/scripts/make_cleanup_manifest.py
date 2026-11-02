"""生成 代码/ 目录清理清单（只列不删）。"""
from __future__ import annotations

import os
import sys
from collections import OrderedDict

ROOT = r"代码"


def size_of(paths):
    total = 0
    for p in paths:
        try:
            total += os.path.getsize(p)
        except OSError:
            pass
    return total


def main():
    groups = OrderedDict([
        ("A1_continuous_power_audit.json", []),
        ("A2_fine_samples.npz", []),
        ("A3___pycache__", []),
        ("A4_.venv", []),
        ("B1_zip", []),
    ])
    for dirpath, dirnames, filenames in os.walk(ROOT):
        low_dir = dirpath.lower()
        if ".venv" in low_dir:
            for f in filenames:
                groups["A4_.venv"].append(os.path.join(dirpath, f))
            continue
        if "__pycache__" in low_dir:
            for f in filenames:
                groups["A3___pycache__"].append(os.path.join(dirpath, f))
            continue
        for f in filenames:
            p = os.path.join(dirpath, f)
            low = f.lower()
            if low == "continuous_power_audit.json":
                groups["A1_continuous_power_audit.json"].append(p)
            elif low == "fine_samples.npz":
                groups["A2_fine_samples.npz"].append(p)
            elif low.endswith(".zip"):
                groups["B1_zip"].append(p)

    total_bytes = sum(size_of(v) for v in groups.values())
    lines = [
        "# 代码目录清理清单（2026-11-03）",
        "",
        "**本文件只列清单，未删除任何文件。**",
        "",
        "## 汇总",
        "",
        "| 组 | 内容 | 文件数 | 可释放 |",
        "|---|---|---:|---:|",
    ]
    for key, values in groups.items():
        lines.append("| %s | %s | %d | %.2f GB |"
                     % (key, key.split("_", 1)[1], len(values), size_of(values) / 1e9))
    lines += [
        "",
        "**合计可释放约 %.2f GB**（当前 `代码/` 共 14.25 GB）" % (total_bytes / 1e9),
        "",
        "## 保留理由（不要删）",
        "",
        "| 文件 | 用途 |",
        "|---|---|",
        "| `results.csv` | 论文全部数值的唯一来源 |",
        "| `summary.json` | 每次 run 的配置与汇总 |",
        "| `weak_link.json` | 候选修复层诊断（审计用） |",
        "| 场景 `*.npz` | SUMO 轨迹与建筑，重跑必需 |",
        "| `*.py` / `*.ps1` | 源码与启动脚本 |",
        "| `*.md` | 设计与审计记录 |",
        "",
        "## 需要你确认的两点",
        "",
        "1. **`实验归档_20261031g_主算法v4` 内的 audit/npz 也在 A 组清单里。**",
        "   删掉能省约 0.5 GB，但会改动归档 run 的目录内容（数值不受影响）。",
        "   若坚持“归档不可动”，请在该组路径中跳过 `实验归档_20261031g_主算法v4` 开头的行。",
        "2. **B1 组的 zip 都已解压到相邻目录**"
        "（如 `返回_交接_同学_50_100车开发_v2_20261023(1).zip` 与同名目录并存）。",
        "   删除 zip 前请确认目录内容完整。",
        "",
        "## 逐条路径",
        "",
    ]
    for key, values in groups.items():
        if not values:
            continue
        lines.append("### %s（%d 个，%.2f GB）" % (key, len(values), size_of(values) / 1e9))
        lines.append("")
        lines.extend(values)
        lines.append("")

    out_md = os.path.join(ROOT, "清理清单_20261103.md")
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    out_txt = os.path.join(ROOT, "清理清单_20261103_paths.txt")
    with open(out_txt, "w", encoding="utf-8") as fh:
        for key in ("A1_continuous_power_audit.json", "A2_fine_samples.npz",
                    "A3___pycache__", "A4_.venv", "B1_zip"):
            for p in groups[key]:
                fh.write(os.path.abspath(p) + "\n")

    print("wrote", out_md)
    print("wrote", out_txt)
    for key, values in groups.items():
        print("%-32s %5d files %7.2f GB" % (key, len(values), size_of(values) / 1e9))
    print("TOTAL %.2f GB" % (total_bytes / 1e9))
    return 0


if __name__ == "__main__":
    sys.exit(main())
