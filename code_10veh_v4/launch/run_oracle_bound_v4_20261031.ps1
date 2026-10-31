# 非因果 oracle 上界：块内真值全知（深拷贝真值场提前推进），reference + maxmin 各一次。
#
# 与 `--csi ideal` 的区别：ideal 只知道块初状态，未来演化仍有过程噪声；oracle 知道
# 整块的真值轨迹。候选匹配同时用真值信念与估计信念各生成一组，避免同一启发式在不同
# 起点上落到不同局部最优而把"上界"压低（实测块 3：单起点 427.7 M → 双起点 482.3 M）。
param(
    [string]$Python = "D:\anaconda\envs\pymarl\python.exe",
    [int]$Blocks    = 100,
    [int]$Draws     = 32,
    [string]$Tag    = "20261031d"
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "code")
New-Item -ItemType Directory -Force -Path "runs", "logs" | Out-Null

$common = @(
    "-B","-I","run_allocation.py","--n","10","--phase","evaluate","--split","development",
    "--blocks","$Blocks","--start-block","2","--draws","$Draws",
    "--channel-seed","2026102401","--forecast-seed","2026102501","--learning-seed","2026102601",
    "--forecast-information","trajectory",
    "--power-solver","continuous","--reference-search","joint","--joint-rounds","8",
    "--qos-metric","chance","--qos-chance-inflation","1.35",
    "--qos-fallback-mean","--qos-chance-rounds","2","--threads","2","--delta","0.1",
    "--csi","oracle","--estimator","two_state",
    "--methods","reference","maxmin","--maxmin-power-refine"
)

$out = "runs/e9_oracle_bound_$Tag"
$log = "logs/e9_oracle_bound_$Tag.log"
if (Test-Path "$out/summary.json") { Write-Host "skip"; exit 0 }
$sw = [Diagnostics.Stopwatch]::StartNew()
$eap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $Python @common "--out" $out 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log
$rc = $LASTEXITCODE
$ErrorActionPreference = $eap
$sw.Stop()
Write-Host ("== oracle rc={0} {1:N1} min" -f $rc, $sw.Elapsed.TotalMinutes)
if ($rc -ne 0) { throw "oracle failed" }
