[CmdletBinding()]
param(
    [string]$RepoRoot = ''
)

# Windows PowerShell 5.1: resolve $PSScriptRoot in the body.
if ([string]::IsNullOrEmpty($RepoRoot)) { $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path }
$ErrorActionPreference = 'Stop'

$git = $null
$candidates = @('git', 'C:\Program Files\Git\bin\git.exe')
foreach ($c in $candidates) {
    $cmd = Get-Command $c -ErrorAction SilentlyContinue
    if ($null -ne $cmd) { $git = $cmd.Source; break }
    if (Test-Path $c) { $git = $c; break }
}
if ($null -eq $git) { throw 'git not found; install git or use an existing snapshot' }

$codeDir = Join-Path $PSScriptRoot 'code'
New-Item -ItemType Directory -Force -Path $codeDir | Out-Null

# Runtime-relevant tracked paths only. All code now lives under src/ (analysis,
# scripts, tests, kaggle_lane, verification). Data, runs and signed result evidence
# are seeded/sealed separately, and raw literature captures are archived (gitignored).
$paths = @(
    'src', 'configs', 'docs', 'platforms', 'paper_writing',
    'README.md', '.gitignore'
)

Push-Location $RepoRoot
try {
    $sha = (& $git rev-parse --short HEAD).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'not a git repository' }
    $dirty = (& $git status --porcelain | Measure-Object).Count

    $zip = Join-Path $codeDir ("xdrift_repo_{0}.zip" -f $sha)
    Write-Host "[snapshot] archiving HEAD ($sha) -> $zip (runtime paths only)"
    $archiveArgs = @('archive', '--format=zip', "-o", $zip, 'HEAD', '--') + $paths
    & $git @archiveArgs
    if ($LASTEXITCODE -ne 0) { throw 'git archive failed' }

    $lines = @(
        ('snapshot_sha = {0}' -f $sha),
        ('snapshot_zip = {0}' -f $zip),
        ('zip_sha256 = {0}' -f (Get-FileHash -Algorithm SHA256 -Path $zip).Hash.ToLower()),
        ('zip_bytes = {0}' -f (Get-Item $zip).Length),
        ('included_paths = {0}' -f ($paths -join ', ')),
        ('dirty_worktree_files_not_in_snapshot = {0}' -f $dirty),
        ('created_at_utc = {0}' -f (Get-Date).ToUniversalTime().ToString('o')),
        '',
        'Copy the whole src/kaggle_lane/remote_rtx5060/ folder (including code/*.zip) to the RTX 5060 PC.'
    )
    $lines | Out-File -FilePath (Join-Path $codeDir 'snapshot_info.txt') -Encoding utf8
    $lines | ForEach-Object { Write-Host "[snapshot] $_" }
} finally {
    Pop-Location
}
