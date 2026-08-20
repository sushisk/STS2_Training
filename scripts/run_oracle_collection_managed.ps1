param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $ScenarioPath,
    [string] $ScenarioListPath,
    [string] $Date = (Get-Date -Format 'yyyyMMdd'),
    [string] $TrainingRoot = 'C:\STS2_Training',
    [string] $Python = 'python',
    [string] $ServerHost = '127.0.0.1',
    [int] $Port = 8765,
    [int] $ConnectTimeout = 30,
    [int] $DecisionTimeout = 300,
    [int] $MaxDecisions = 200,
    [int] $OracleBeamWidth = 32,
    [int] $OracleTopK = 8,
    [int] $OracleDepth = 4,
    [int] $TargetBeamWidth = 8,
    [int] $OracleTimeout = 300
)

$ErrorActionPreference = 'Stop'
if ($ScenarioListPath) {
    $ScenarioPath = Get-Content -LiteralPath $ScenarioListPath | Where-Object { $_.Trim() }
}
if (-not $ScenarioPath -or $ScenarioPath.Count -eq 0) {
    throw 'Specify -ScenarioPath or -ScenarioListPath with at least one scenario JSON path.'
}
$oracleDir = Join-Path $TrainingRoot (Join-Path 'data\oracle' $Date)
$faultDir = Join-Path $TrainingRoot (Join-Path 'data\oracle_fault' $Date)
$stagingDir = Join-Path $TrainingRoot (Join-Path 'data\.oracle_staging' ([guid]::NewGuid().ToString('N')))
New-Item -ItemType Directory -Force -Path $oracleDir, $faultDir, $stagingDir | Out-Null

function Get-LastJsonRecord([string] $Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    $last = Get-Content -LiteralPath $Path | Where-Object { $_.Trim() } | Select-Object -Last 1
    if (-not $last) { return $null }
    try { return ($last | ConvertFrom-Json) } catch { return $null }
}

function Get-UniquePath([string] $Directory, [string] $Name) {
    $candidate = Join-Path $Directory $Name
    if (-not (Test-Path -LiteralPath $candidate)) { return $candidate }
    $stem = [IO.Path]::GetFileNameWithoutExtension($Name)
    $ext = [IO.Path]::GetExtension($Name)
    return (Join-Path $Directory ("{0}-{1}{2}" -f $stem, (Get-Date -Format 'HHmmssfff'), $ext))
}

$manifestPath = Join-Path $stagingDir 'run_manifest.jsonl'
foreach ($input in $ScenarioPath) {
    $resolved = (Resolve-Path -LiteralPath $input).Path
    $base = [IO.Path]::GetFileNameWithoutExtension($resolved)
    $stagedJsonl = Join-Path $stagingDir ("{0}.jsonl" -f $base)
    $stdout = Join-Path $stagingDir ("{0}.stdout.log" -f $base)
    $stderr = Join-Path $stagingDir ("{0}.stderr.log" -f $base)
    $started = Get-Date
    $cliArgs = @(
        '-m', 'sts2_training.runner.oracle_collection',
        '--host', $ServerHost, '--port', $Port,
        '--connect-timeout', $ConnectTimeout,
        '--decision-timeout', $DecisionTimeout,
        '--max-decisions', $MaxDecisions,
        '--search-mode', 'none',
        '--scenario', $resolved, '--output', $stagedJsonl,
        '--oracle-beam-width', $OracleBeamWidth,
        '--oracle-top-k', $OracleTopK,
        '--oracle-depth', $OracleDepth,
        '--target-beam-width', $TargetBeamWidth,
        '--oracle-timeout', $OracleTimeout
    )

    Write-Host ("START {0}" -f $resolved)
    & $Python @cliArgs 1> $stdout 2> $stderr
    $exitCode = $LASTEXITCODE
    $result = Get-LastJsonRecord $stagedJsonl
    $success = $null -ne $result -and $result.record_type -eq 'combat_oracle_episode_result' -and [bool]$result.completed -and $result.termination_reason -eq 'terminal'
    $destination = if ($success) { Get-UniquePath $oracleDir ([IO.Path]::GetFileName($stagedJsonl)) } else { Get-UniquePath $faultDir ([IO.Path]::GetFileName($stagedJsonl)) }
    if (Test-Path -LiteralPath $stagedJsonl) { Move-Item -LiteralPath $stagedJsonl -Destination $destination }
    $logDestination = Split-Path -Parent $destination
    Move-Item -LiteralPath $stdout -Destination (Join-Path $logDestination ([IO.Path]::GetFileName($stdout)))
    Move-Item -LiteralPath $stderr -Destination (Join-Path $logDestination ([IO.Path]::GetFileName($stderr)))
    $manifest = [ordered]@{
        scenario = $resolved
        output = $destination
        success = $success
        exit_code = $exitCode
        record_type = if ($null -ne $result) { $result.record_type } else { $null }
        completed = if ($null -ne $result) { $result.completed } else { $null }
        termination_reason = if ($null -ne $result) { $result.termination_reason } else { 'no_episode_result' }
        decisions_collected = if ($null -ne $result) { $result.decisions_collected } else { $null }
        elapsed_s = if ($null -ne $result) { $result.elapsed_s } else { $null }
        started_at = $started.ToString('o')
        finished_at = (Get-Date).ToString('o')
    }
    ($manifest | ConvertTo-Json -Compress) | Add-Content -LiteralPath $manifestPath
    Write-Host ("{0}: {1} -> {2}" -f $(if ($success) { 'SUCCESS' } else { 'FAULT' }), $base, $destination)
}

Move-Item -LiteralPath $manifestPath -Destination (Join-Path $oracleDir ("run_manifest-{0}.jsonl" -f (Get-Date -Format 'HHmmssfff')))
Remove-Item -LiteralPath $stagingDir -Recurse -Force
