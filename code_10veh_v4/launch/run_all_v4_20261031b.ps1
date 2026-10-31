# v4 全量重跑（单一 source_sha256）+ 对比算法。
#
# 为什么重跑：上一批（Tag 20261031）跑完后，`code/` 里又加了随机下界
# （`random_allocate` / `random_legal_action`）和审计里对 random 的豁免，
# 源码哈希从 10c083ab… 变成新值。物理没有改动，但为了让**所有对比数字**
# 出自同一棵源码树，这里用新 Tag 把 15 组主体 + 3 组对比一起重跑。
# 同种子同物理应当复现上一批的数值，这本身也是一条稳定性证据。
#
# 顺序：主算法 → 场景 B → 消融 → δ 扫描 → 对比（随机下界 / 完美 CSI 上界）
param(
    [string]$Python = "D:\anaconda\envs\pymarl\python.exe",
    [int]$Blocks    = 100,
    [int]$Draws     = 32,
    [string]$Tag    = "20261031b"
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
    @{ name = "delta_scan";  script = "run_delta_scan_20261028.ps1" },
    @{ name = "comparators"; script = "run_comparators_v4_20261031.ps1" }
)

Log "START v4 rerun+comparators Tag=$Tag Blocks=$Blocks Draws=$Draws"
foreach ($s in $stages) {
    $path = Join-Path $PSScriptRoot $s.script
    Log "STAGE $($s.name) begin ($($s.script))"
    try {
        & $path -Python $Python -Blocks $Blocks -Draws $Draws -Tag $Tag 2>&1 |
            ForEach-Object { "$_" } | Tee-Object -FilePath (Join-Path $logDir "stage_$($s.name)_$Tag.log") -Append
        Log "STAGE $($s.name) end rc=$LASTEXITCODE"
    } catch {
        Log "STAGE $($s.name) FAILED: $($_.Exception.Message)"
    }
}
Log "ALL DONE"
