param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("design", "validation", "smoke")]
    [string]$Role,

    [Parameter(Mandatory=$true)]
    [string]$RunDir,

    [Parameter(Mandatory=$true)]
    [int]$ChannelSeed,

    [Parameter(Mandatory=$true)]
    [string]$OutDir,

    [string]$Phase0Path = "",

    [string]$Python = "D:\anaconda\envs\pymarl\python.exe",

    [switch]$GeneratePhase0,

    [int]$MaxBlocks = 0,

    [int]$MaxCandidatesPerBlock = 0,

    [int]$MaxEvalsPerArm = 600,

    [int]$Starts = 5,

    [int]$LegacyMaxiter = 240,

    [int]$LegacyRestarts = 2
)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Phase0Script = Join-Path $Here "phase0_recoverability.py"
$AuditScript = Join-Path $Here "guard_r_final_audit.py"

if (-not (Test-Path $Python)) {
    throw "Python executable not found: $Python"
}
if (-not (Test-Path $RunDir)) {
    throw "Run directory not found: $RunDir"
}

if ($GeneratePhase0) {
    $Phase0Out = Join-Path $OutDir "phase0"
    New-Item -ItemType Directory -Force -Path $Phase0Out | Out-Null
    & $Python -B -I $Phase0Script `
        --blocks 100 `
        --run $RunDir `
        --channel-seed $ChannelSeed `
        --out $Phase0Out
    if ($LASTEXITCODE -ne 0) {
        throw "phase0_recoverability.py failed with exit code $LASTEXITCODE"
    }
    $Phase0Path = Join-Path $Phase0Out "phase0.json"
}

if (-not $Phase0Path -or -not (Test-Path $Phase0Path)) {
    throw "Provide -Phase0Path, or use -GeneratePhase0."
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$AuditOut = Join-Path $OutDir "audit"
$argsList = @(
    "-B", "-I", $AuditScript,
    "--run", $RunDir,
    "--phase0", $Phase0Path,
    "--channel-seed", "$ChannelSeed",
    "--phase0-seed", "$ChannelSeed",
    "--role", $Role,
    "--out", $AuditOut,
    "--max-evals-per-arm", "$MaxEvalsPerArm",
    "--starts", "$Starts",
    "--legacy-maxiter", "$LegacyMaxiter",
    "--legacy-restarts", "$LegacyRestarts"
)
if ($MaxBlocks -gt 0) {
    $argsList += @("--max-blocks", "$MaxBlocks")
}
if ($MaxCandidatesPerBlock -gt 0) {
    $argsList += @("--max-candidates-per-block", "$MaxCandidatesPerBlock")
}

& $Python @argsList
if ($LASTEXITCODE -ne 0) {
    throw "guard_r_final_audit.py failed with exit code $LASTEXITCODE"
}
Write-Host "Guard-R audit finished: $AuditOut"
