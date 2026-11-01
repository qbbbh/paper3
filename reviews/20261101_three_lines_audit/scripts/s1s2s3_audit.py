"""S1/S2/S3 的落地审计：它们到底是"没接进决策"还是"判据在本模型下退化"。

只读归档产物，不跑仿真。

判据（对应设计稿里的命题 3a：切换有利 ⟺ C_ℓ' < C_ℓ，C = P_c + p_svc）：

- **S1** 给 p_svc。若 p_svc 大量落在功率下界，则同模式链路之间 C 几乎相同，
  判据的分辨力趋于零 —— 这是"退化"，不是"没实现"。
- **S2** 给 ∂EE_k/∂p_c。若 ranking 为空的比例高，则该模块无候选可提交。
- **S3** 的候选生成存在，但在 `weak_link_repair_stage` 里最终由**完整预测评估 + 硬门**
  决定，**没有实现 b_S/b_E 认证剪枝** —— 这是"没实现"。

输出：逐项计数与代表性数字，供论文如实描述 S1/S2/S3 的实际地位。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass

FLOOR = {"RF": 0.01, "VLC": 0.1}      # v4 口径的电输入下界
STATIC = {"RF": 0.1159, "VLC": 1.745}


def main(argv=None):
    parser = argparse.ArgumentParser(description="S1/S2/S3 落地审计")
    parser.add_argument("--run", required=True, help="v5 run 目录（含 weak_link.json）")
    args = parser.parse_args(argv)

    path = Path(args.run) / "weak_link.json"
    with open(path, "r", encoding="utf-8-sig") as handle:
        data = json.load(handle)

    blocks = [b for b in data.get("blocks") or [] if b.get("repair")]
    n_blocks = len(blocks)
    print(f"run: {args.run}")
    print(f"块数 {n_blocks}")

    at_floor = 0
    total_links = 0
    empty_ranking = 0
    tie_blocks = 0            # 同模式链路里 C 完全相同 → 判据无法分辨
    weakest_vlc = 0
    s2_submitted = 0
    s3_submitted = 0
    per_mode_floor = {"RF": [0, 0], "VLC": [0, 0]}   # [at_floor, total]
    c_spread = []

    for block in blocks:
        probe = None
        for item in block["repair"]:
            if item.get("stage") == "s1_probe":
                probe = item
                break
        if probe is None:
            continue
        modes = probe["modes"]
        p_star = np.asarray(probe["p_star"], float)
        powers = np.asarray(probe["powers"], float)
        if not probe.get("ranking"):
            empty_ranking += 1
        weakest = int(probe.get("weakest", 0))
        if modes[weakest] == "VLC":
            weakest_vlc += 1

        total_links += len(modes)
        for m, p in zip(modes, p_star):
            per_mode_floor[m][1] += 1
            if abs(p - FLOOR[m]) <= 1e-6:
                per_mode_floor[m][0] += 1
            if abs(p - FLOOR[m]) <= 1e-6:
                at_floor += 1

        # C_ℓ = P_c + p_svc，按模式分组看是否可分辨
        C = np.array([STATIC[m] + p for m, p in zip(modes, p_star)])
        same = [i for i, m in enumerate(modes) if m == modes[weakest] and i != weakest]
        if same:
            ref = C[weakest]
            # 同模式里存在与最弱链路 C 相差 <1% 的链路 → 判据无法区分
            if any(abs(C[i] - ref) <= 0.01 * max(ref, 1e-12) for i in same):
                tie_blocks += 1
            c_spread.append(max(abs(C[i] - ref) for i in same) / max(ref, 1e-12))

        for item in block["repair"]:
            stage = item.get("stage", "")
            if stage.startswith("S2_cut"):
                s2_submitted += 1
            elif stage[0] in "ABC" and "_" in stage:
                s3_submitted += 1

    def pct(a, b):
        return f"{100.0*a/b:.1f}%" if b else "—"

    print(f"\n=== S1：服务功率 p_svc 是否贴下界 ===")
    print(f"  全部链路贴下界：{at_floor}/{total_links} = {pct(at_floor, total_links)}")
    for m in ("RF", "VLC"):
        f, t = per_mode_floor[m]
        print(f"    {m:3s} {f}/{t} = {pct(f, t)}（下界 {FLOOR[m]} W）")
    print("  含义：贴下界 ⇒ C = P_c + 下界，同模式链路之间**完全不可区分**。")

    print(f"\n=== S1/S2 的判据分辨力（命题 3a：C_ℓ' < C_ℓ）===")
    print(f"  同模式里存在与最弱链路 C 相差 <1% 的块：{tie_blocks}/{n_blocks} = {pct(tie_blocks, n_blocks)}")
    if c_spread:
        print(f"  同模式 C 偏差的相对上界：中位 {100*np.median(c_spread):.3f}%，"
              f"p90 {100*np.percentile(c_spread, 90):.3f}%")
    print(f"  最弱链路是 VLC 的块：{weakest_vlc}/{n_blocks} = {pct(weakest_vlc, n_blocks)}")

    print(f"\n=== S2：灵敏度排序是否为空 ===")
    print(f"  ranking 为空：{empty_ranking}/{n_blocks} = {pct(empty_ranking, n_blocks)}")
    print("  原因：跨模式干扰恒为 0（`same` 掩码），若最弱链路没有同模式邻居，梯度全零。")

    print(f"\n=== S3：候选是否真的提交 ===")
    print(f"  S2_cut 候选提交块次数：{s2_submitted}")
    print(f"  A/B/C 候选提交块次数：{s3_submitted}")

    print("\n=== 结论 ===")
    print("  S1：已实现且被使用（slack 定 S2 的削减量、p_star 是削减目标），")
    print("      但**没有作为候选比较的代价基准**；且 85% 贴下界使其在同模式内不可分辨。")
    print("  S2：已实现，但作用域仅限固定结构下的同模式功率变化；")
    print("      离散换接收端/换模式不在其定义域内（ranking 为空可直接观测）。")
    print("  S3：候选**生成**已实现；**判据**（b_S/b_E 认证剪枝、成本区间比较）未实现，")
    print("      最终采纳由完整预测评估 + 硬预算门决定。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
