<#
.SYNOPSIS
  Canonical Windows entry point for every stage: resolves lane + config,
  sets XDRIFT_* environment variables, then delegates to src/scripts/run_lane.py
  which writes the run manifest.

.DESCRIPTION
  This host runs Windows PowerShell 5.1 (powershell.exe), not PowerShell 7, so
  $PSScriptRoot is EMPTY inside param() default-value expressions. That is why
  -Root defaults to '' here and is resolved in the script body.

  The script only bootstraps: it finds the repository root, a Python
  interpreter (project .venv first) and git (for the manifest commit SHA), then
  hands over to the OS-neutral Python launcher. All resolution logic, validation
  and manifest writing lives in src/scripts/run_lane.py so Windows/Linux/Kaggle
  cannot drift apart.

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File src\scripts\run_lane.ps1 `
      -Lane legacy -Config configs\audit.yaml `
      -Entry src\scripts\run_data_audit.py -- --dataset NF-UNSW-NB15-v3

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File src\scripts\run_lane.ps1 -ListLanes
#>
[CmdletBinding()]
param(
    [string]$Lane = '',
    [string]$Config = 'configs/base.yaml',
    [string]$Entry = '',
    [string]$RunName = '',
    [string]$Python = '',
    [string]$Root = '',
    [switch]$DryRun,
    [switch]$ListLanes,
    [switch]$SkipCudaCheck,
    [string[]]$Set = @(),
    [string[]]$EnvVars = @(),
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$RemainingArgs = @()
)

$ErrorActionPreference = 'Stop'

# Windows PowerShell 5.1: $PSScriptRoot is not available in param() defaults.
if ([string]::IsNullOrEmpty($Root)) {
    $Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
}

# --- Locate git for the run manifest (Windows-specific detection stays here,   ---
# --- never in portable src/ code).                                            ---
if (-not $env:XDRIFT_GIT) {
    $gitCmd = Get-Command git -ErrorAction SilentlyContinue
    if ($gitCmd -and $gitCmd.Source) {
        $env:XDRIFT_GIT = $gitCmd.Source
    }
    else {
        $gitCandidates = @(
            (Join-Path $env:ProgramFiles 'Git\bin\git.exe'),
            (Join-Path $env:ProgramFiles 'Git\cmd\git.exe'),
            (Join-Path ${env:ProgramFiles(x86)} 'Git\bin\git.exe')
        )
        foreach ($candidate in $gitCandidates) {
            if (-not [string]::IsNullOrEmpty($candidate) -and (Test-Path $candidate)) {
                $env:XDRIFT_GIT = $candidate
                break
            }
        }
    }
}
if ($env:XDRIFT_GIT) { Write-Verbose "[run_lane.ps1] git: $env:XDRIFT_GIT" }

# --- Locate Python: -Python > XDRIFT_PYTHON > project .venv > PATH python.     ---
if ([string]::IsNullOrEmpty($Python)) { $Python = $env:XDRIFT_PYTHON }
if ([string]::IsNullOrEmpty($Python)) {
    $venvPython = Join-Path $Root '.venv\Scripts\python.exe'
    if (Test-Path $venvPython) { $Python = $venvPython } else { $Python = 'python' }
}
if ($Python -ne 'python' -and -not (Test-Path $Python)) {
    throw "[run_lane.ps1] Python interpreter not found: $Python (run platforms/setup/setup_windows_legacy.ps1 first)"
}

$launcher = Join-Path $Root 'src\scripts\run_lane.py'
if (-not (Test-Path $launcher)) {
    throw "[run_lane.ps1] launcher not found: $launcher"
}

$launcherArgs = @($launcher, '--root', $Root)
if (-not [string]::IsNullOrEmpty($Lane)) { $launcherArgs += @('--lane', $Lane) }
if (-not [string]::IsNullOrEmpty($Config)) { $launcherArgs += @('--config', $Config) }
if (-not [string]::IsNullOrEmpty($Entry)) { $launcherArgs += @('--entry', $Entry) }
if (-not [string]::IsNullOrEmpty($RunName)) { $launcherArgs += @('--run-name', $RunName) }
if (-not [string]::IsNullOrEmpty($Python) -and $Python -ne 'python') { $launcherArgs += @('--python', $Python) }
if ($ListLanes) { $launcherArgs += '--list-lanes' }
if ($DryRun) { $launcherArgs += '--dry-run' }
if ($SkipCudaCheck) { $launcherArgs += '--skip-cuda-check' }
foreach ($item in $Set) { $launcherArgs += @('--set', $item) }
foreach ($item in $EnvVars) { $launcherArgs += @('--env', $item) }
if ($RemainingArgs.Count -gt 0) {
    $launcherArgs += @('--')
    $launcherArgs += $RemainingArgs
}

& $Python @launcherArgs
exit $LASTEXITCODE
