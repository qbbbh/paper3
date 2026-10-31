# E5: maxmin with continuous power refinement (fixed matching), delta scan
# Usage: .\run_e5_refine_20261028.ps1 [-Blocks 100] [-Draws 32] [-Tag v1]
param(
    [string]$Python = "D:\anaconda\envs\pymarl\python.exe",
    [int]$Blocks    = 100,
    [int]$Draws     = 32,
    [int]$Seed      = 2026102401,
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
    "--qos-fallback-mean","--qos-chance-rounds","2","--threads","2",
    "--methods","reference","maxmin","--maxmin-power-refine"
)

$jobs = @(
    @{ name="e5_B_mean_d005";   extra=@("--qos-metric","mean","--delta","0.05") },
    @{ name="e5_B_mean_d010";   extra=@("--qos-metric","mean","--delta","0.1") },
    @{ name="e5_D_chance_d005"; extra=@("--qos-metric","chance","--qos-chance-inflation","1.35","--delta","0.05") },
    @{ name="e5_D_chance_d010"; extra=@("--qos-metric","chance","--qos-chance-inflation","1.35","--delta","0.1") }
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
Write-Host "E5 DONE"
