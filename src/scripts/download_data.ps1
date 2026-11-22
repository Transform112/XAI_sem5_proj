<#
.SYNOPSIS
  Download and verify the whole NetFlow-v3 dataset family (plus optional
  non-NIDS candidate) into datasets/raw, with provenance records.

.DESCRIPTION
  Sources are the public Kaggle mirrors of the UQ NLM/NIDS NetFlow-v3 datasets.
  Every archive is checked against an expected byte size, hashed (SHA-256),
  extracted, and recorded in datasets/raw/provenance.json. The UQ RDM originals
  require authentication; if the user downloads them manually, place them under
  datasets/raw/original/ and rerun the verification step.

.EXAMPLE
  pwsh -File src/scripts/download_data.ps1
  pwsh -File src/scripts/download_data.ps1 -IncludeElliptic
#>
[CmdletBinding()]
param(
    [string]$Root = '',
    [switch]$IncludeElliptic,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
# NOTE: Windows PowerShell 5.1 leaves $PSScriptRoot empty inside param defaults,
# so resolve the default in the body (this script must run on PS 5.1 hosts too).
if ([string]::IsNullOrEmpty($Root)) {
    $Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
}
$rawDir = Join-Path $Root 'datasets\raw'
New-Item -ItemType Directory -Force -Path $rawDir | Out-Null
$provenancePath = Join-Path $rawDir 'provenance.json'
$provenance = @{}
if (Test-Path $provenancePath) {
    $existing = Get-Content $provenancePath -Raw | ConvertFrom-Json
    foreach ($prop in $existing.PSObject.Properties) { $provenance[$prop.Name] = $prop.Value }
}

# Expected zip sizes are NOT fixed: Kaggle reports uncompressed dataset totals while
# the download endpoint returns a compressed archive. Integrity is enforced by:
#   - MinZipBytes sanity floor,
#   - SHA-256 recorded in provenance,
#   - row/class identity verification in src/scripts/verify_datasets.py (gate G1a).
$datasets = @(
    [pscustomobject]@{
        Name = 'NF-UNSW-NB15-v3'
        Url = 'https://www.kaggle.com/api/v1/datasets/download/seyhed/nf-unsw-nb15-v3'
        MinZipBytes = 20000000
        KaggleTotalBytesUncompressed = 577360958
        Flows = 2365424
    },
    [pscustomobject]@{
        Name = 'NF-ToN-IoT-v3'
        Url = 'https://www.kaggle.com/api/v1/datasets/download/seyhed/nf-ton-iot-v3'
        MinZipBytes = 20000000
        KaggleTotalBytesUncompressed = 453905772
        Flows = 27520260
    },
    [pscustomobject]@{
        Name = 'NF-BoT-IoT-v3'
        Url = 'https://www.kaggle.com/api/v1/datasets/download/seyhed/nf-bot-iot-v3'
        MinZipBytes = 20000000
        KaggleTotalBytesUncompressed = 78813598
        Flows = 16933808          # paper arXiv:2503.04404v3 (D-013)
        WebsiteFlows = 16993808   # UQ website value, reported as discrepancy only
    },
    [pscustomobject]@{
        Name = 'NF-CSE-CIC-IDS2018-v3'
        Url = 'https://www.kaggle.com/api/v1/datasets/download/seyhed/nf-cicids2018-v3'
        MinZipBytes = 20000000
        KaggleTotalBytesUncompressed = 839735661
        Flows = 20115529
    }
)
if ($IncludeElliptic) {
    $datasets += [pscustomobject]@{
        Name = 'Elliptic'
        Url = 'https://www.kaggle.com/api/v1/datasets/download/ellipticco/elliptic-data-set'
        MinZipBytes = 10000000
        # Kaggle API metadata value (verified 2026-11-01); the previous constant
        # 306436764 was wrong (scout T10 verification).
        KaggleTotalBytesUncompressed = 697459499
        Flows = $null
        Licence = 'CC BY-NC-ND 4.0 (creator Claudio Bellei; Kaggle ref ellipticco/elliptic-data-set)'
        Origin = 'Weber et al. 2019, KDD''19 ADSF workshop, arXiv:1908.02591; attribute Elliptic (www.elliptic.co)'
    }
}

foreach ($ds in $datasets) {
    $zipPath = Join-Path $rawDir ("{0}.zip" -f $ds.Name)
    $extractDir = Join-Path $rawDir $ds.Name
    $needDownload = $Force -or -not (Test-Path $zipPath)
    if (-not $needDownload) {
        $size = (Get-Item $zipPath).Length
        if ($size -lt $ds.MinZipBytes) {
            Write-Warning "[data] $($ds.Name) zip too small ($size bytes); re-downloading"
            $needDownload = $true
        }
    }
    if ($needDownload) {
        Write-Host "[data] downloading $($ds.Name) (compressed archive; Kaggle reports $($ds.KaggleTotalBytesUncompressed) uncompressed bytes) ..."
        & curl.exe -L --retry 3 --retry-delay 5 --max-time 7200 --fail -o $zipPath $ds.Url
        if ($LASTEXITCODE -ne 0) { throw "[data] download failed for $($ds.Name)" }
    }
    $size = (Get-Item $zipPath).Length
    if ($size -lt $ds.MinZipBytes) {
        throw "[data] $($ds.Name) archive is implausibly small: $size bytes"
    }
    $sha = (Get-FileHash -Algorithm SHA256 -Path $zipPath).Hash.ToLower()
    if (-not (Test-Path $extractDir)) {
        Write-Host "[data] extracting $($ds.Name)"
        Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force
    }
    $files = Get-ChildItem -Path $extractDir -Recurse -File | Select-Object Name, Length
    $provenance[$ds.Name] = [pscustomobject]@{
        source = $ds.Url
        mirror = 'kaggle'
        # D-020 fix: the Kaggle metadata "total bytes" is NOT the uncompressed
        # CSV size for this family (for ToN/BoT/CIC it equals the zip size), so
        # it is stored under a neutral name and the archive size is explicit.
        kaggle_metadata_total_bytes = $ds.KaggleTotalBytesUncompressed
        zip_bytes = $size
        sha256 = $sha
        downloaded_at = (Get-Date).ToString('o')
        extracted_files = @($files | ForEach-Object { [pscustomobject]@{ name = $_.Name; bytes = $_.Length } })
        published_flow_count = $ds.Flows
        published_count_source = 'paper arXiv:2503.04404v3 Tables 2-3 (D-013 authoritative)'
        origin = 'University of Queensland NIDS NetFlow-v3 dataset family (Sarhan et al.)'
    }
    # Optional per-dataset provenance fields (discrepancy + licence tracking).
    if ($ds.PSObject.Properties['WebsiteFlows']) {
        $provenance[$ds.Name] | Add-Member -NotePropertyName website_flow_count -NotePropertyValue $ds.WebsiteFlows
    }
    if ($ds.PSObject.Properties['Licence']) {
        $provenance[$ds.Name] | Add-Member -NotePropertyName licence -NotePropertyValue $ds.Licence
    }
    if ($ds.PSObject.Properties['Origin']) {
        $provenance[$ds.Name] | Add-Member -NotePropertyName origin_detail -NotePropertyValue $ds.Origin
    }
    Write-Host ("[data] {0}: zip {1} bytes, sha256 {2}" -f $ds.Name, $size, $sha)
}

$provenance | ConvertTo-Json -Depth 8 | Out-File -Encoding utf8 $provenancePath
Write-Host "[data] provenance written to $provenancePath"
Write-Host '[data] NEXT: run the dataset identity/verification stage (src/scripts/verify_datasets.py)'
