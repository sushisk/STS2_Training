param(
    [string]$RlRoot = $env:STS2_RL_ROOT,
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$TrainingRepo = 'sushisk/STS2_Training'
$TrainingPr = 19
$RlRepo = 'sushisk/STS2_RL'
$RlPr = 8

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
    )
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath exited with code $LASTEXITCODE"
    }
}

function Invoke-GhJson {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    $json = & gh @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "gh $($Arguments -join ' ') failed with code $LASTEXITCODE"
    }
    return ($json | Out-String | ConvertFrom-Json)
}

function Assert-CleanGitTree {
    param([Parameter(Mandatory = $true)][string]$Path, [Parameter(Mandatory = $true)][string]$Label)
    $dirty = & git -C $Path status --porcelain
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to inspect $Label git tree at $Path"
    }
    if ($dirty) {
        throw "$Label working tree must be clean before exact-pair validation"
    }
}

Invoke-Checked gh auth status

$trainingRoot = (& git rev-parse --show-toplevel).Trim()
if ($LASTEXITCODE -ne 0 -or -not $trainingRoot) {
    throw 'Run this script from the STS2_Training checkout.'
}
$trainingRoot = (Resolve-Path $trainingRoot).Path

if (-not $RlRoot) {
    throw 'Set STS2_RL_ROOT or pass -RlRoot to the STS2_RL checkout.'
}
$rlRootResolved = (Resolve-Path $RlRoot).Path

Assert-CleanGitTree -Path $trainingRoot -Label 'Training'
Assert-CleanGitTree -Path $rlRootResolved -Label 'RL'

$trainingPrInfo = Invoke-GhJson api "repos/$TrainingRepo/pulls/$TrainingPr"
$rlPrInfo = Invoke-GhJson api "repos/$RlRepo/pulls/$RlPr"
$currentTrainingSha = [string]$trainingPrInfo.head.sha
$currentRlSha = [string]$rlPrInfo.head.sha

$localTrainingSha = (& git -C $trainingRoot rev-parse HEAD).Trim()
$localRlSha = (& git -C $rlRootResolved rev-parse HEAD).Trim()

Write-Host "Training current PR head: $currentTrainingSha"
Write-Host "Training local HEAD:      $localTrainingSha"
Write-Host "RL current PR head:       $currentRlSha"
Write-Host "RL local HEAD:            $localRlSha"

if ($localTrainingSha -ne $currentTrainingSha) {
    throw "Training checkout is not PR #$TrainingPr current head. Checkout $currentTrainingSha first."
}
if ($localRlSha -ne $currentRlSha) {
    throw "RL checkout is not PR #$RlPr current head. Checkout $currentRlSha first."
}

if (-not $SkipInstall) {
    Write-Host 'Installing Training test dependencies...'
    Invoke-Checked python -m pip install -e "$trainingRoot[test]" pythonnet
}

$env:STS2_RL_ROOT = $rlRootResolved
Write-Host 'Running real Emulator paired v0.7 integration locally...'
Invoke-Checked python -m pytest "$trainingRoot/tests/integration/test_paired_rl_v07.py" -q

$latestTrainingPrInfo = Invoke-GhJson api "repos/$TrainingRepo/pulls/$TrainingPr"
$latestRlPrInfo = Invoke-GhJson api "repos/$RlRepo/pulls/$RlPr"
$latestTrainingSha = [string]$latestTrainingPrInfo.head.sha
$latestRlSha = [string]$latestRlPrInfo.head.sha

if ($latestTrainingSha -ne $currentTrainingSha -or $latestRlSha -ne $currentRlSha) {
    throw "A PR head moved during the real-environment test. Training=$latestTrainingSha RL=$latestRlSha; discard this result and rerun the current pair."
}

Write-Host 'Real Emulator paired v0.7 integration passed for the current exact pair.'
Write-Host 'This environment-dependent validation is local-only; no GitHub commit status is published.'
