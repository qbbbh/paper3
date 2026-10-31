# v4 对比算法：均匀随机合法动作下界 + 完美 CSI 上界。
#
# 随机下界（`--methods reference random`）：每块一次随机合法配对 + 对数均匀功率，
#   不做任何优化，也不受预算/QoS 守卫，违约如实记录。这是"什么都不做"的对照。
# 完美 CSI 上界（`--csi ideal`）：块初用真值、协方差置零，未来演化模型不变，
#   所以它是"估计无误差"的上界，不是"信道不变"的界。
#
# 注意：`--csi ideal` 只支持 two_state 估计器；QoS 约束下只允许
# reference / maxmin / random 三个方法。
param(
    [string]$Python = "D:\anaconda\envs\pymarl\python.exe",
    [int]$Blocks    = 100,
    [int]$Draws     = 32,
    [int]$Seed      = 2026102401,
    [string]$Tag    = "20261031b"
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "code")
New-Item -ItemType Directory -Force -Path "runs", "logs" | Out-Null

$common = @(
    "-B","-I","run_allocation.py","--n","10","--phase","evaluate","--split","development",
    "--blocks","$Blocks","--start-block","2","--draws","$Draws",
    "--channel-seed","$Seed","--forecast-seed","2026102501","--learning-seed","2026102601",
    "--forecast-information","trajectory",
    "--power-solver","continuous","--reference-search","joint","--joint-rounds","8",
    "--qos-metric","chance","--qos-chance-inflation","1.35",
    "--qos-fallback-mean","--qos-chance-rounds","2","--threads","2","--delta","0.1"
)

$jobs = @(
    @{ name="e7_random_chance_d010"; extra=@("--csi","estimated","--methods","reference","random") },
    @{ name="e7_ideal_ref";          extra=@("--csi","ideal","--estimator","two_state","--methods","reference") },
    @{ name="e7_ideal_chance_d010";  extra=@("--csi","ideal","--estimator","two_state","--methods","reference","maxmin","--maxmin-power-refine") }
)

foreach ($j in $jobs) {
    $out = "runs/$($j.name)_$Tag"
    $log = "logs/$($j.name)_$Tag.log"
    if (Test-Path "$out/summary.json") { Write-Host "skip $($j.name)"; continue }
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $eap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python @common @($j.extra) "--out" $out 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log
    $rc = $LASTEXITCODE
    $ErrorActionPreference = $eap
    $sw.Stop()
    Write-Host ("== {0} rc={1} {2:N1} min" -f $j.name, $rc, $sw.Elapsed.TotalMinutes)
    if ($rc -ne 0) { throw "$($j.name) failed" }
}
Write-Host "COMPARATORS DONE"
