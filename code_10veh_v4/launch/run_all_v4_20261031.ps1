# v4（2026-10-30 口径）主算法 + 消融 + 对照的批量驱动。
#
# 口径：RF 0.01–20 W 电输入（eta_PA=0.1 乘在信号分子）、VLC 0.1–30 W（sA 组），
# EE = 逐链路 r_k/(P_c+p_k) 的算术平均，REFERENCE_VERSION 为 v4。
#
# 顺序（先主算法，后消融/对照）：
#   1) e5_refine  4 组：主算法（maxmin + continuous + chance/mean + refine）
#   2) scenario_b 3 组：独立信道实现（channel seed 2026102801）
#   3) e4         6 组：不加 refine 的消融网格
#   4) delta_scan 2 组：delta=0.20 / 1.00
#
# 每组 100 块 = 10 s、100 次资源分配。已存在 summary.json 的目录会跳过，
# 所以 Tag 必须换新的，否则会静默复用旧口径的结果。
param(
    [string]$Python = "D:\anaconda\envs\pymarl\python.exe",
    [int]$Blocks    = 100,
    [int]$Draws     = 32,
    [string]$Tag    = "20261031"
)
$ErrorActionPreference = "Continue"
$logDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$master = Join-Path $logDir "run_all_v4_$Tag.log"

function Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Write-Output $line
    Add-Content -Path $master -Value $line -Encoding utf8
}

$stages = @(
    @{ name = "e5_refine";   script = "run_e5_refine_20261028.ps1" },
    @{ name = "scenario_b";  script = "run_scenario_b_20261028.ps1" },
    @{ name = "e4";          script = "run_e4_20261027.ps1" },
    @{ name = "delta_scan";  script = "run_delta_scan_20261028.ps1" }
)

Log "START v4 batch Tag=$Tag Blocks=$Blocks Draws=$Draws"
foreach ($s in $stages) {
    $path = Join-Path $PSScriptRoot $s.script
    Log "STAGE $($s.name) begin ($($s.script))"
    try {
        & $path -Python $Python -Blocks $Blocks -Draws $Draws -Tag $Tag 2>&1 |
            ForEach-Object { "$_" } | Tee-Object -FilePath (Join-Path $logDir "stage_$($s.name)_$Tag.log") -Append
        Log "STAGE $($s.name) end rc=$LASTEXITCODE"
    } catch {
        # 子脚本用 throw 报失败；这里记下来继续下一阶段，不静默跳过。
        Log "STAGE $($s.name) FAILED: $($_.Exception.Message)"
    }
}
Log "ALL DONE"
