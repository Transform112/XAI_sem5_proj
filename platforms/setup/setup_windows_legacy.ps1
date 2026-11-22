<#
.SYNOPSIS
  Bootstrap the LEGACY DEV lane (current PC, GTX 1050 3 GB / CPU fallback).

.DESCRIPTION
  Creates a project-local .venv, installs torch 2.4.1+cu118 and PyTorch Geometric
  2.6.1 plus common dependencies, writes a frozen lock file and runs the
  environment check. Idempotent: safe to re-run.

.EXAMPLE
  pwsh -File platforms/setup/setup_windows_legacy.ps1
#>
[CmdletBinding()]
param(
    [string]$Root = '',
    [string]$Python = 'python',
    [switch]$RequireCuda
)

$ErrorActionPreference = 'Stop'
# Windows PowerShell 5.1 leaves $PSScriptRoot empty inside param defaults.
if ([string]::IsNullOrEmpty($Root)) {
    $Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
}
$venv = Join-Path $Root '.venv'
$pythonExe = Join-Path $venv 'Scripts\python.exe'
$requirements = Join-Path $Root 'platforms\requirements'

Write-Host "[setup] root      : $Root"
Write-Host "[setup] python    : $Python"
Write-Host "[setup] venv      : $venv"

if (-not (Test-Path $pythonExe)) {
    Write-Host '[setup] creating venv'
    & $Python -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw 'failed to create venv' }
}

Write-Host '[setup] upgrading pip'
& $pythonExe -m pip install --upgrade pip wheel
if ($LASTEXITCODE -ne 0) { throw 'failed to upgrade pip' }

Write-Host '[setup] installing torch cu118 (this is a large download)'
& $pythonExe -m pip install -r (Join-Path $requirements 'gpu-legacy-cu118.txt')
if ($LASTEXITCODE -ne 0) { throw 'failed to install torch/PyG' }

Write-Host '[setup] installing common dependencies'
& $pythonExe -m pip install -r (Join-Path $requirements 'common.txt')
if ($LASTEXITCODE -ne 0) { throw 'failed to install common dependencies' }

# Windows MSVC runtime fix: hosts without the Visual C++ Redistributable fail to load
# torch's c10.dll. `msvc-runtime` places the DLLs in Scripts/, and copying them next to
# torch/lib makes them resolvable as c10.dll dependencies (verified on a clean Win11 host).
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
& $pythonExe -m pip freeze | Out-File -Encoding utf8 (Join-Path $requirements 'lock-legacy.txt')

$checkArgs = @((Join-Path $Root 'src\scripts\check_env.py'))
if ($RequireCuda) { $checkArgs += '--require-cuda' }
Write-Host "[setup] running environment check"
& $pythonExe @checkArgs
if ($LASTEXITCODE -ne 0) {
    Write-Warning 'environment check reported failures (see JSON above)'
    exit 1
}
Write-Host '[setup] legacy lane ready'
