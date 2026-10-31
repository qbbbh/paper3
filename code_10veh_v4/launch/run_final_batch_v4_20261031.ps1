# 最终对比批次：主算法 / 文献复现 / 非因果 oracle，同一棵源码树、同一种子。
#
# 为什么重跑主算法：`code/` 里加了文献复现分支（`search="literature"`）与审计里
# 的预算豁免，`source_sha256` 变了。物理与主算法路径没动，但要让对比表里的每个
# 数字都出自同一棵源码树，必须用新 Tag 重跑；同种子应当逐位复现 20261031b。
#
# 文献复现的配置：关联只做一次（等效 SINR + 匈牙利），功率走分式规划
# （二次变换 + SCA，`--literature-power-solver dinkelbach`）。它满足与主算法
# 相同的 QoS，但不受 `delta` 预算门约束（审计里已豁免）——那条门是本文方法之间
# 的相对约束。同一 run 里也输出 reference，便于逐块配对比较。
param(
    [string]$Python = "D:\anaconda\envs\pymarl\python.exe",
    [int]$Blocks    = 100,
    [int]$Draws     = 32,
    [string]$Tag    = "20261031e",
    [string[]]$Only = @()
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "code")
New-Item -ItemType Directory -Force -Path "runs", "logs" | Out-Null

$common = @(
    "-B","-I","run_allocation.py","--n","10","--phase","evaluate","--split","development",
    "--blocks","$Blocks","--start-block","2","--draws","$Draws",
    "--channel-seed","2026102401","--forecast-seed","2026102501","--learning-seed","2026102601",
    "--csi","estimated","--forecast-information","trajectory",
    "--power-solver","continuous","--reference-search","joint","--joint-rounds","8",
    "--qos-metric","chance","--qos-chance-inflation","1.35",
    "--qos-fallback-mean","--qos-chance-rounds","2","--threads","2","--delta","0.1"
)

$jobs = @(
    @{ name="e5_D_chance_d010"; extra=@("--methods","reference","maxmin","--maxmin-power-refine") },
    @{ name="e10_literature";   extra=@("--methods","reference","literature",
                                        "--literature-power-solver","dinkelbach") },
    @{ name="e9_oracle_bound";  extra=@("--csi","oracle","--methods","reference","maxmin",
                                        "--maxmin-power-refine") },
    # 文献基线的敏感性：标称功率 ×2 与功率求解器 ×1。用来回答"是不是我们把基线配弱了"。
    @{ name="e10_lit_nom001";   extra=@("--methods","literature",
                                        "--literature-power-solver","dinkelbach",
                                        "--literature-nominal-w","0.01") },
    @{ name="e10_lit_nom300";   extra=@("--methods","literature",
                                        "--literature-power-solver","dinkelbach",
                                        "--literature-nominal-w","3.0") },
    @{ name="e10_lit_cont";     extra=@("--methods","literature",
                                        "--literature-power-solver","continuous",
                                        "--literature-nominal-w","0.3") }
)

$master = Join-Path $PSScriptRoot "logs\final_batch_$Tag.log"
function Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Write-Output $line
    Add-Content -Path $master -Value $line -Encoding utf8
}

Log "START Tag=$Tag Blocks=$Blocks Draws=$Draws"
foreach ($j in $jobs) {
    if ($Only.Count -gt 0 -and -not ($Only -contains $j.name)) { continue }
    $out = "runs/$($j.name)_$Tag"
    $log = "logs/$($j.name)_$Tag.log"
    if (Test-Path "$out/summary.json") { Log "skip $($j.name)"; continue }
    $sw = [Diagnostics.Stopwatch]::StartNew()
    Log "BEGIN $($j.name)"
    $eap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python @common @($j.extra) "--out" $out 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log
    $rc = $LASTEXITCODE
    $ErrorActionPreference = $eap
    $sw.Stop()
    Log ("END {0} rc={1} {2:N1} min" -f $j.name, $rc, $sw.Elapsed.TotalMinutes)
    if ($rc -ne 0) { Log "FAILED $($j.name)"; throw "$($j.name) failed" }
}
Log "ALL DONE"
