<#
.SYNOPSIS
  Bootstrap the CANONICAL lane on the user's PC (RTX 5060 8 GB, Blackwell sm_120).

.DESCRIPTION
  Creates a project-local .venv, installs torch 2.7.1+cu128 (required for
  Blackwell / sm_120) and PyTorch Geometric 2.7.0 plus common dependencies,
  writes a frozen lock file and runs the environment check with CUDA required.
  Idempotent: safe to re-run.

  Exact steps for the user:
    1) copy/clone this repository to the RTX 5060 PC (git bundle or ZIP)
    2) open PowerShell in the repository root
    3) pwsh -File platforms/setup/setup_windows_rtx5060.ps1
    4) pwsh -File src/scripts/check_env.py alternatives are handled by the script

.EXAMPLE
  pwsh -File platforms/setup/setup_windows_rtx5060.ps1
#>
[CmdletBinding()]
param(
    [string]$Root = '',
    [string]$Python = 'python',
    [switch]$SkipCudaCheck
)

$ErrorActionPreference = 'Stop'
# Windows PowerShell 5.1 leaves $PSScriptRoot empty inside param defaults.
if ([string]::IsNullOrEmpty($Root)) {
    $Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
}
$venv = Join-Path $Root '.venv'
$pythonExe = Join-Path $venv 'Scripts\python.exe'
$requirements = Join-Path $Root 'platforms\requirements'

Write-Host "[setup] canonical lane (RTX 5060 / cu128)"
Write-Host "[setup] root   : $Root"
Write-Host "[setup] venv   : $venv"

if (-not (Test-Path $pythonExe)) {
    Write-Host '[setup] creating venv'
    & $Python -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw 'failed to create venv' }
}

Write-Host '[setup] upgrading pip'
& $pythonExe -m pip install --upgrade pip wheel
if ($LASTEXITCODE -ne 0) { throw 'failed to upgrade pip' }

Write-Host '[setup] installing torch cu128 + PyG 2.7 (large download, be patient)'
& $pythonExe -m pip install -r (Join-Path $requirements 'gpu-blackwell-cu128.txt')
if ($LASTEXITCODE -ne 0) { throw 'torch cu128 install failed; check the NVIDIA driver supports CUDA 12.8+' }

Write-Host '[setup] installing common dependencies'
& $pythonExe -m pip install -r (Join-Path $requirements 'common.txt')
if ($LASTEXITCODE -ne 0) { throw 'failed to install common dependencies' }

# Windows MSVC runtime fix (harmless when the redistributable is already installed):
# hosts without the Visual C++ Redistributable fail to load torch's c10.dll.
Write-Host '[setup] applying Windows MSVC runtime fix'
& $pythonExe -m pip install --quiet msvc-runtime
$dllNames = @('msvcp140.dll','vcruntime140.dll','vcruntime140_1.dll','concrt140.dll','msvcp140_atomic_wait.dll','msvcp140_1.dll','msvcp140_2.dll')
$scriptsDir = Join-Path $venv 'Scripts'
$torchLib = Join-Path $venv 'Lib\site-packages\torch\lib'
if (Test-Path $torchLib) {
    foreach ($dll in $dllNames) {
        $srcDll = Join-Path $scriptsDir $dll
        if (Test-Path $srcDll) { Copy-Item -Path $srcDll -Destination (Join-Path $torchLib $dll) -Force }
    }
    Write-Host '[setup] MSVC runtime DLLs copied beside torch/lib'
} else {
    Write-Warning '[setup] torch/lib not found; MSVC runtime fix copied only to Scripts/'
}

Write-Host '[setup] writing lock file'
& $pythonExe -m pip freeze | Out-File -Encoding utf8 (Join-Path $requirements 'lock-rtx5060.txt')

$checkArgs = @((Join-Path $Root 'src\scripts\check_env.py'))
if (-not $SkipCudaCheck) { $checkArgs += '--require-cuda' }
Write-Host '[setup] running environment check (CUDA required)'
& $pythonExe @checkArgs
if ($LASTEXITCODE -ne 0) {
    Write-Warning 'environment check reported failures (see JSON above)'
    exit 1
}
Write-Host '[setup] canonical lane ready'
