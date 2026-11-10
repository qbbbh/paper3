"""Losslessly split exported period tables for browser-only reviewers.

Reads and writes this evidence package only. No numerical calculation,
simulation, optimization, or original-source modification is performed.
"""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def main():
    dest=ROOT/"tables/by_method";dest.mkdir(parents=True,exist_ok=True)
    items=[];source_items=[]
    for group in ("10A","10B","50B"):
        original=ROOT/"tables"/(group+"_periods.csv")
        with original.open(encoding="utf-8",newline="") as f:
            reader=csv.reader(f);header=next(reader);rows=list(reader)
        mindex=header.index("method");bindex=header.index("block")
        assert len(rows)==8000
        methods=sorted({row[mindex] for row in rows});assert len(methods)==8
        source_items.append(dict(path=original.relative_to(ROOT).as_posix(),sha256=digest(original),row_count=len(rows)))
        recovered=[]
        for method in methods:
            selected=[row for row in rows if row[mindex]==method]
            assert len(selected)==1000 and [int(row[bindex]) for row in selected]==list(range(1000))
            output=dest/f"{group}_{method}_periods.csv"
            with output.open("w",encoding="utf-8",newline="") as f:
                writer=csv.writer(f);writer.writerow(header);writer.writerows(selected)
            with output.open(encoding="utf-8",newline="") as f:
                reader=csv.reader(f);assert next(reader)==header;check=list(reader)
            assert check==selected
            recovered.extend(check)
            items.append(dict(group=group,method=method,path=output.relative_to(ROOT).as_posix(),
                rows=1000,bytes=output.stat().st_size,sha256=digest(output),
                source_path=original.relative_to(ROOT).as_posix(),source_sha256=digest(original),
                transform="Exact CSV header and string-cell rows filtered by method; original order retained; no numeric parsing or rewriting"))
        # Every source row appears once; reconstruct the original row order.
        recovered_by_key={(row[bindex],row[mindex]):row for row in recovered}
        assert len(recovered_by_key)==len(rows)
        assert [recovered_by_key[row[bindex],row[mindex]] for row in rows]==rows
    index=dict(selection="All methods, all groups, all 1000 planned periods; failures retained",
               file_count=24,row_count=24000,source_files=source_items,files=items,
               validation="All split headers and all string cells equal the original table; disjoint complete row partition")
    write(dest/"INDEX.json",index)
    labels=dict(reference="初始分配",main_old="旧主方法",main_new="新版主方法",same_topo_old="旧同拓扑",
                same_topo_new="新版同拓扑",no_refine="无连续精修消融",aboagye_matching_qt="Aboagye对比",zappone_mmee_fp_fixed="Zappone对比")
    lines=["# 逐方法周期小表", "", "每份CSV含同一方法的全部1000个计划周期，保留失败；字段与原完整表完全相同。每份约0.4–0.5MB，方便仅有网页访问能力的审核者读取。", "",
           "[证据入口](../../README.md) · [字段与单位](../SCHEMA.json) · [拆分来源与哈希](INDEX.json)", "",
           "| 方法 | 10A | 10B | 50B |", "|---|---|---|---|"]
    for method in ("reference","main_old","main_new","same_topo_old","same_topo_new","no_refine","aboagye_matching_qt","zappone_mmee_fp_fixed"):
        links=[f"[{group}]({group}_{method}_periods.csv)" for group in ("10A","10B","50B")]
        lines.append("| "+labels[method]+" (`"+method+"`) | "+" | ".join(links)+" |")
    lines += ["", "失败周期没有可执行动作，EE字段留空，服务达标为False。相对初始分配的损失或预测预算检查在初始分配无有效结果时也留空，不应按零补齐。",
              "", "原完整周期表仍保留：[10A](../10A_periods.csv)、[10B](../10B_periods.csv)、[50B](../50B_periods.csv)。小表没有筛选有利周期，也没有重新统计或修改数值。"]
    (dest/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    manifest_path=ROOT/"SOURCE_MANIFEST.json";manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["transformations"]=[x for x in manifest.get("transformations",[]) if x.get("type")!="browser_period_partitions"]
    manifest["transformations"].append(dict(type="browser_period_partitions",source_files=source_items,
         index_path="tables/by_method/INDEX.json",index_sha256=digest(dest/"INDEX.json"),exports=items,
         validation="24 split tables, 24000 rows: exact header and row-cell equality, complete nonoverlapping partition"))
    write(manifest_path,manifest)
    report_path=ROOT/"VALIDATION_REPORT.json";report=json.loads(report_path.read_text(encoding="utf-8"))
    report["browser_period_partitions"]=dict(status="passed",source_table_count=3,split_file_count=24,
         rows_checked=24000,headers_and_string_cells_equal=True,complete_nonoverlapping=True,
         source_table_sha256={r["path"]:r["sha256"] for r in source_items},
         index_path="tables/by_method/INDEX.json",index_sha256=digest(dest/"INDEX.json"),
         numerical_recalculation="Not repeated: original full tables and numerical evidence unchanged")
    write(report_path,report)
    print(json.dumps(dict(split_files=24,rows_checked=24000,total_split_bytes=sum(x["bytes"] for x in items),
         max_split_bytes=max(x["bytes"] for x in items)),ensure_ascii=False))


if __name__=="__main__":main()
