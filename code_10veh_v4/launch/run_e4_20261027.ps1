# E4: four-group ablation + delta scan (10 vehicles, 100 blocks, 1 channel seed)
# Usage: .\run_e4_20261027.ps1 [-Blocks 100] [-Draws 32]
param(
    [string]$Python = "D:\anaconda\envs\pymarl\python.exe",
    [int]$Blocks    = 100,
    [int]$Draws     = 32,
    [int]$Seed      = 2026102401,
    [string]$Tag    = "20261027"
)
$ErrorActionPreference = "Stop"
$code = Join-Path $PSScriptRoot "code"
Set-Location $code
New-Item -ItemType Directory -Force -Path "runs" | Out-Null
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

$common = @(
    "-B","-I","run_allocation.py","--n","10","--phase","evaluate","--split","development",
    "--blocks","$Blocks","--start-block","2","--draws","$Draws",
    "--channel-seed","$Seed","--forecast-seed","2026102501","--learning-seed","2026102601",
    "--csi","estimated","--forecast-information","trajectory",
    "--power-solver","continuous","--reference-search","joint","--joint-rounds","8",
    "--qos-fallback-mean","--qos-chance-rounds","2","--threads","2"
)

$jobs = @(
    @{ name="e4_A_mean_ref";          extra=@("--methods","reference","--qos-metric","mean","--delta","0.1") },
    @{ name="e4_C_chance_ref";        extra=@("--methods","reference","--qos-metric","chance","--qos-chance-inflation","1.35","--delta","0.1") },
    @{ name="e4_B_mean_maxmin_d005";  extra=@("--methods","reference","maxmin","--qos-metric","mean","--delta","0.05") },
    @{ name="e4_B_mean_maxmin_d010";  extra=@("--methods","reference","maxmin","--qos-metric","mean","--delta","0.1") },
    @{ name="e4_D_chance_maxmin_d005";extra=@("--methods","reference","maxmin","--qos-metric","chance","--qos-chance-inflation","1.35","--delta","0.05") },
    @{ name="e4_D_chance_maxmin_d010";extra=@("--methods","reference","maxmin","--qos-metric","chance","--qos-chance-inflation","1.35","--delta","0.1") }
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
Write-Host "E4 DONE"
