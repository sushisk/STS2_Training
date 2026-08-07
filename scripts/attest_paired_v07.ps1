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
$PairStatusContext = 'paired-v07-trusted-exact-pair'

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
        throw "$Label working tree must be clean before exact-pair attestation"
    }
}

function Rerun-FailedGate {
    param(
        [Parameter(Mandatory = $true)][string]$Repo,
        [Parameter(Mandatory = $true)][string]$Workflow,
        [Parameter(Mandatory = $true)][string]$Commit
    )

    $runs = Invoke-GhJson run list --repo $Repo --workflow $Workflow --commit $Commit --limit 20 --json databaseId,status,conclusion
    $run = @($runs) | Select-Object -First 1
    if ($null -eq $run) {
        Write-Warning "No workflow run found for $Repo / $Workflow at $Commit."
        return
    }

    if ($run.status -eq 'completed' -and $run.conclusion -eq 'failure') {
        Write-Host "Re-running failed gate: $Repo / $Workflow (run $($run.databaseId))"
        Invoke-Checked gh run rerun $run.databaseId --failed --repo $Repo
        return
    }

    Write-Host "Gate does not need a failed-job rerun: $Repo / $Workflow status=$($run.status) conclusion=$($run.conclusion)"
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
Write-Host 'Running real Emulator paired v0.7 integration...'

$testSucceeded = $false
try {
    Invoke-Checked python -m pytest "$trainingRoot/tests/integration/test_paired_rl_v07.py" -q
    $testSucceeded = $true
}
finally {
    $latestTrainingPrInfo = Invoke-GhJson api "repos/$TrainingRepo/pulls/$TrainingPr"
    $latestRlPrInfo = Invoke-GhJson api "repos/$RlRepo/pulls/$RlPr"
    $latestTrainingSha = [string]$latestTrainingPrInfo.head.sha
    $latestRlSha = [string]$latestRlPrInfo.head.sha

    if ($latestTrainingSha -ne $currentTrainingSha -or $latestRlSha -ne $currentRlSha) {
        throw "A PR head moved during the real-environment test. Training=$latestTrainingSha RL=$latestRlSha; refusing to attest stale pair."
    }

    $state = if ($testSucceeded) { 'success' } else { 'failure' }
    $verb = if ($testSucceeded) { 'Passed' } else { 'Failed' }
    $description = "$verb with RL $($currentRlSha.Substring(0, 12))"

    Write-Host "Publishing $PairStatusContext=$state for Training $currentTrainingSha"
    Invoke-Checked gh api --method POST "repos/$TrainingRepo/statuses/$currentTrainingSha" `
        -f "state=$state" `
        -f "context=$PairStatusContext" `
        -f "description=$description" `
        -f "target_url=https://github.com/$TrainingRepo/pull/$TrainingPr"
}

if (-not $testSucceeded) {
    throw 'Paired v0.7 integration failed; failure attestation was published and gates remain blocked.'
}

Rerun-FailedGate -Repo $TrainingRepo -Workflow 'paired-v07-exact-pair' -Commit $currentTrainingSha
Rerun-FailedGate -Repo $RlRepo -Workflow 'paired-v07-counterpart-gate' -Commit $currentRlSha

Write-Host 'Exact-pair attestation published. GitHub-hosted gates were re-run where needed.'
