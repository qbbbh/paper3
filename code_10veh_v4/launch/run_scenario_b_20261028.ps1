# Scenario B: same SUMO track, independent channel realisation (new channel seed).
# Three groups: chance reference only, delta=0.05, delta=0.10.
# Usage: .\run_scenario_b_20261028.ps1 [-Blocks 100] [-Tag 20261028]
param(
    [string]$Python = "D:\anaconda\envs\pymarl\python.exe",
    [int]$Blocks    = 100,
    [int]$Draws     = 32,
    [int]$Seed      = 2026102801,
    [string]$Tag    = "20261028"
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "code")
New-Item -ItemType Directory -Force -Path "runs" | Out-Null
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

$common = @(
    "-B","-I","run_allocation.py","--n","10","--phase","evaluate","--split","development",
    "--blocks","$Blocks","--start-block","2","--draws","$Draws",
    "--channel-seed","$Seed","--forecast-seed","2026102501","--learning-seed","2026102601",
    "--csi","estimated","--forecast-information","trajectory",
    "--power-solver","continuous","--reference-search","joint","--joint-rounds","8",
    "--qos-metric","chance","--qos-chance-inflation","1.35",
    "--qos-fallback-mean","--qos-chance-rounds","2","--threads","2"
)

$jobs = @(
    @{ name="e6_B_ref";   extra=@("--methods","reference","--delta","0.05") },
    @{ name="e6_B_d005";  extra=@("--methods","reference","maxmin","--maxmin-power-refine","--delta","0.05") },
    @{ name="e6_B_d010";  extra=@("--methods","reference","maxmin","--maxmin-power-refine","--delta","0.1") }
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
    if ($rc -ne 0) { Write-Host "FAILED $($j.name) rc=$rc"; exit 1 }
    Write-Host "$($j.name) done in $([math]::Round($sw.Elapsed.TotalMinutes,1)) min"
}
Write-Host "ALL DONE"
