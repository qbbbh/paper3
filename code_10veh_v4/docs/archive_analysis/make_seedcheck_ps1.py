"""生成种子敏感性检查脚本（PowerShell 5.1 需 UTF-8 BOM，这里直接写 BOM）。"""
from pathlib import Path

BODY = r'''# 理想 CSI 的种子敏感性检查（不动 code/，source_sha256 不变）。
#
# 问题：`e7_ideal_ref` 的均 EE（480.0 M）比估计 CSI 的 reference（508.4 M）低 5.5%，
# 所以 ideal CSI 在这套启发式下不是经验上界。本脚本固定真值（channel_seed 不变），
# 只换 forecast_seed（影响预测抽样 → 影响决策），对 estimated / ideal 两种 CSI
# 各跑 4 个种子，看 5.5% 的差是不是落在种子噪声里。
#
# 只跑 reference（不跑 maxmin），100 块，与主体同一套参数。
param(
    [string]$Python = "D:\anaconda\envs\pymarl\python.exe",
    [int]$Blocks    = 100,
    [int]$Draws     = 32,
    [string]$Tag    = "20261031c"
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "code")
New-Item -ItemType Directory -Force -Path "runs", "logs" | Out-Null

$common = @(
    "-B","-I","run_allocation.py","--n","10","--phase","evaluate","--split","development",
    "--blocks","$Blocks","--start-block","2","--draws","$Draws",
    "--channel-seed","2026102401","--learning-seed","2026102601",
    "--forecast-information","trajectory",
    "--power-solver","continuous","--reference-search","joint","--joint-rounds","8",
    "--qos-metric","chance","--qos-chance-inflation","1.35",
    "--qos-fallback-mean","--qos-chance-rounds","2","--threads","2","--delta","0.1",
    "--methods","reference"
)

$jobs = @()
foreach ($s in 1,2,3,4) {
    $jobs += @{ name="e8_seed$s`_est";  fs=(2026102500+$s); extra=@("--csi","estimated") }
    $jobs += @{ name="e8_seed$s`_ideal"; fs=(2026102500+$s); extra=@("--csi","ideal","--estimator","two_state") }
}

foreach ($j in $jobs) {
    $out = "runs/$($j.name)_$Tag"
    $log = "logs/$($j.name)_$Tag.log"
    if (Test-Path "$out/summary.json") { Write-Host "skip $($j.name)"; continue }
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $eap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python @common @($j.extra) "--forecast-seed" $j.fs "--out" $out 2>&1 |
        ForEach-Object { "$_" } | Tee-Object -FilePath $log
    $rc = $LASTEXITCODE
    $ErrorActionPreference = $eap
    $sw.Stop()
    Write-Host ("== {0} fs={1} rc={2} {3:N1} min" -f $j.name, $j.fs, $rc, $sw.Elapsed.TotalMinutes)
    if ($rc -ne 0) { throw "$($j.name) failed" }
}
Write-Host "SEEDCHECK DONE"
'''


def main():
    p = Path(__file__).resolve().parents[1] / "run_seedcheck_v4_20261031.ps1"
    p.write_text(BODY, encoding="utf-8-sig")
    print("wrote", p)


if __name__ == "__main__":
    main()
