"""Read-only, exact per-block comparison of two original numerical methods."""
from pathlib import Path
import hashlib
import json
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np

HERE = Path(__file__).resolve().parent
STORAGE = HERE.parent.parent
SOURCES = {
    "10A": STORAGE / "对比算法安全运行_20261106/runs/n10_A_100s_1000blocks",
    "10B": STORAGE / "对比算法安全运行_20261106/runs/n10_B_100s_1000blocks",
    "50B": STORAGE / "长跑保存修复_20261105/runs/R2_n50_d010_full100s_1000blocks",
}
FIELDS = ("tx", "rx", "mode", "power_w", "tx_port", "rx_port")
SCALARS = ("eta_ref", "delta", "qos_metric", "channel_seed", "forecast_seed", "chance_fallback")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect(group, source):
    started = time.perf_counter()
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    by_key = {(r["block"], r["method"]): r for r in summary["blocks"]}
    record = dict(group=group, source=str(source), summary_sha256=sha(source / "summary.json"),
                  source_sha256=summary["source_sha256"], config_sha256=summary["config_sha256"],
                  planned_blocks=summary["config"]["blocks"],
                  both_success_blocks=[], both_failure_blocks=[], only_maxmin_failure_blocks=[],
                  only_same_topo_failure_blocks=[], exact_action_equal_blocks=[],
                  exact_forecast_equal_blocks=[], exact_guard_inputs_equal_blocks=[],
                  different_guard_inputs_blocks=[], action_equal_forecast_different_blocks=[],
                  differences=[], failure_records=[], checked_inputs=[])
    for k in range(summary["config"]["start_block"], summary["config"]["start_block"]+summary["config"]["blocks"]):
        folder = source / "block_checkpoints" / f"block_{k:06d}"
        commit = json.loads((folder / "commit.json").read_text(encoding="utf-8"))
        raw_json = (folder / "records.json").read_bytes()
        if hashlib.sha256(raw_json).hexdigest() != commit["sha256"]["records.json"]:
            raise ValueError(f"record checksum mismatch: {group} {k}")
        payload = json.loads(raw_json)
        if payload["block"] != k or payload["source_sha256"] != summary["source_sha256"]:
            raise ValueError(f"record identity mismatch: {group} {k}")
        methods = {r["method"]: r for r in payload["block_records"]}
        a, b = methods["maxmin"], methods["same_topo_maxmin"]
        for method, r in (("maxmin", a), ("same_topo_maxmin", b)):
            if r["success"] != by_key[k, method]["success"]:
                raise ValueError(f"formal summary and checkpoint disagree: {group} {k} {method}")
        checks = dict(block=k, records_sha256=commit["sha256"]["records.json"])
        if not a["success"] or not b["success"]:
            category = ("both_failure_blocks" if not a["success"] and not b["success"] else
                        "only_maxmin_failure_blocks" if not a["success"] else "only_same_topo_failure_blocks")
            record[category].append(k)
            record["failure_records"].append(dict(block=k, maxmin_success=a["success"],
                same_topo_success=b["success"], maxmin_reason=a.get("failure_reason"),
                same_topo_reason=b.get("failure_reason")))
            record["checked_inputs"].append(checks)
            continue
        record["both_success_blocks"].append(k)
        npz_path = folder / "fine_samples.npz"
        npz_hash = sha(npz_path)
        if npz_hash != commit["sha256"]["fine_samples.npz"]:
            raise ValueError(f"array checksum mismatch: {group} {k}")
        checks["fine_samples_sha256"] = npz_hash
        record["checked_inputs"].append(checks)
        different_fields = []
        maximum_errors = {}
        with np.load(npz_path, allow_pickle=False) as raw:
            for field in FIELDS:
                left, right = raw[f"maxmin_b{k}_{field}"], raw[f"same_topo_maxmin_b{k}_{field}"]
                if not np.array_equal(left, right):
                    different_fields.append(field)
                    if left.shape == right.shape and left.dtype.kind in "fiu":
                        maximum_errors[field] = float(np.max(np.abs(left-right)))
            left, right = raw[f"maxmin_b{k}_forecast_samples_bps"], raw[f"same_topo_maxmin_b{k}_forecast_samples_bps"]
            forecast_equal = bool(np.array_equal(left, right))
            if not forecast_equal and left.shape == right.shape:
                maximum_errors["forecast_samples_bps"] = float(np.max(np.abs(left-right)))
        action_equal = not different_fields
        scalar_differences = [field for field in SCALARS if a.get(field) != b.get(field)]
        if action_equal:
            record["exact_action_equal_blocks"].append(k)
        if forecast_equal:
            record["exact_forecast_equal_blocks"].append(k)
        if action_equal and not forecast_equal:
            record["action_equal_forecast_different_blocks"].append(k)
        if action_equal and forecast_equal and not scalar_differences:
            record["exact_guard_inputs_equal_blocks"].append(k)
        else:
            record["different_guard_inputs_blocks"].append(k)
            record["differences"].append(dict(block=k, action_different_fields=different_fields,
                forecast_equal=forecast_equal, scalar_differences=scalar_differences,
                maximum_absolute_errors=maximum_errors))
        if k % 100 == 99:
            print(f"{group}: {k+1} blocks checked", flush=True)
    record["counts"] = {name[:-7]: len(value) for name, value in record.items() if name.endswith("_blocks") and isinstance(value, list)}
    record["all_common_success_inputs_equal"] = not record["different_guard_inputs_blocks"]
    record["elapsed_s"] = time.perf_counter()-started
    return record


def main():
    output = dict(scope="Original outputs only; no optimization or new simulation",
                  comparison="np.array_equal with original link ordering; no numerical tolerance",
                  action_fields=list(FIELDS), forecast_field="forecast_samples_bps",
                  scalar_fields=list(SCALARS), groups={})
    for group, source in SOURCES.items():
        output["groups"][group] = inspect(group, source)
    target = HERE / "原主方法与同拓扑动作等同性核查.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 原主方法与同拓扑动作等同性核查", "", "仅读取原始 block_checkpoints 与正式 summary，未运行优化。",
             "逐元素严格比较车辆连接、通信模式、功率、所选发射/接收端口和完整预测速率样本；不以总体均值判断相同。检查原始记录及数组的提交哈希，并核对双方正式成功/失败标记。",
             "", "| 组别 | 共同成功 | 双方失败 | 仅一方失败 | 完整动作相同 | 预测样本相同 | guard完整输入相同 | 不同输入需另算 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group, r in output["groups"].items():
        c = r["counts"]
        lines.append(f"| {group} | {c['both_success']} | {c['both_failure']} | {c['only_maxmin_failure']+c['only_same_topo_failure']} | {c['exact_action_equal']} | {c['exact_forecast_equal']} | {c['exact_guard_inputs_equal']} | {c['different_guard_inputs']} |")
    lines.extend(["", "同一周期共享物理配置、预测器、后验均值、最低速率条件及平均EE预算；这里进一步核对 eta_ref、delta、qos_metric、channel_seed、forecast_seed 和初始后备标记。完整输入相同的周期，使用同版本、同参数、确定性 guarded 函数可复用输出；不同输入的周期不能复用主方法输出。双方失败仍保留原失败，不能当作成功等同。", ""])
    for group, r in output["groups"].items():
        lines.extend([f"## {group}", "", f"不同输入周期（{len(r['different_guard_inputs_blocks'])}个）：", "",
                      ", ".join(map(str, r["different_guard_inputs_blocks"])) or "无", "",
                      "双方失败周期：" + (", ".join(map(str, r["both_failure_blocks"])) or "无"), "",
                      "动作相同但预测样本不同周期：" + (", ".join(map(str, r["action_equal_forecast_different_blocks"])) or "无"), ""])
    lines.extend(["具体差异字段、最大绝对差值、正式失败原因和逐块输入哈希详见同名JSON。", ""])
    target.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({g:r['counts'] for g,r in output['groups'].items()}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
