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
$AttestationContext = 'paired-v07-exact-pair'
$TrustedAssociations = @('OWNER', 'MEMBER', 'COLLABORATOR')

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

function Assert-TrustedPullRequest {
    param(
        [Parameter(Mandatory = $true)]$PrInfo,
        [Parameter(Mandatory = $true)][string]$ExpectedRepo,
        [Parameter(Mandatory = $true)][string]$Label
    )
    $headRepo = [string]$PrInfo.head.repo.full_name
    if ($headRepo -ne $ExpectedRepo) {
        throw "$Label PR head must come from trusted repository $ExpectedRepo, got $headRepo"
    }
    $association = [string]$PrInfo.author_association
    if ($TrustedAssociations -notcontains $association) {
        throw "$Label PR author association $association is not trusted for execution on the Emulator host"
    }
}

function Invoke-UncredentialedTestBody {
    param(
        [Parameter(Mandatory = $true)][string]$TrainingRoot,
        [Parameter(Mandatory = $true)][string]$RlRootResolved,
        [Parameter(Mandatory = $true)][string]$GhDirectory,
        [Parameter(Mandatory = $true)][bool]$SkipDependencyInstall
    )

    $originalPath = $env:PATH
    $hadGhToken = Test-Path Env:GH_TOKEN
    $hadGithubToken = Test-Path Env:GITHUB_TOKEN
    $savedGhToken = if ($hadGhToken) { $env:GH_TOKEN } else { $null }
    $savedGithubToken = if ($hadGithubToken) { $env:GITHUB_TOKEN } else { $null }

    try {
        # PR-controlled Python/test code must not inherit the GitHub API tokens or the
        # GitHub CLI executable used by the trusted wrapper. This is not a sandbox, so
        # the wrapper also refuses fork/untrusted-author PRs above.
        Remove-Item Env:GH_TOKEN -ErrorAction SilentlyContinue
        Remove-Item Env:GITHUB_TOKEN -ErrorAction SilentlyContinue
        $env:PATH = (($originalPath -split ';') | Where-Object {
            $_ -and $_.TrimEnd('\') -ine $GhDirectory.TrimEnd('\')
        }) -join ';'
        $env:STS2_RL_ROOT = $RlRootResolved

        if (-not $SkipDependencyInstall) {
            Write-Host 'Installing Training test dependencies without GitHub credentials...'
            Invoke-Checked python -m pip install -e "$TrainingRoot[test]" pythonnet
        }

        Write-Host 'Running real Emulator paired v0.7 integration without GitHub credentials...'
        Invoke-Checked python -m pytest "$TrainingRoot/tests/integration/test_paired_rl_v07.py" -q
    }
    finally {
        $env:PATH = $originalPath
        if ($hadGhToken) { $env:GH_TOKEN = $savedGhToken } else { Remove-Item Env:GH_TOKEN -ErrorAction SilentlyContinue }
        if ($hadGithubToken) { $env:GITHUB_TOKEN = $savedGithubToken } else { Remove-Item Env:GITHUB_TOKEN -ErrorAction SilentlyContinue }
    }
}

Invoke-Checked gh auth status
$ghDirectory = Split-Path -Parent (Get-Command gh -ErrorAction Stop).Source

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
Assert-TrustedPullRequest -PrInfo $trainingPrInfo -ExpectedRepo $TrainingRepo -Label 'Training'
Assert-TrustedPullRequest -PrInfo $rlPrInfo -ExpectedRepo $RlRepo -Label 'RL'

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

Invoke-UncredentialedTestBody `
    -TrainingRoot $trainingRoot `
    -RlRootResolved $rlRootResolved `
    -GhDirectory $ghDirectory `
    -SkipDependencyInstall ([bool]$SkipInstall)

$latestTrainingPrInfo = Invoke-GhJson api "repos/$TrainingRepo/pulls/$TrainingPr"
$latestRlPrInfo = Invoke-GhJson api "repos/$RlRepo/pulls/$RlPr"
$latestTrainingSha = [string]$latestTrainingPrInfo.head.sha
$latestRlSha = [string]$latestRlPrInfo.head.sha

if ($latestTrainingSha -ne $currentTrainingSha -or $latestRlSha -ne $currentRlSha) {
    throw "A PR head moved during the real-environment test. Training=$latestTrainingSha RL=$latestRlSha; discard this result and rerun the current pair."
}

$description = "Training=$currentTrainingSha RL=$currentRlSha"
if ($description.Length -gt 140) {
    throw 'Exact-pair attestation description exceeds GitHub commit-status limit'
}

foreach ($target in @(
    @{ Repo = $TrainingRepo; Sha = $currentTrainingSha },
    @{ Repo = $RlRepo; Sha = $currentRlSha }
)) {
    Invoke-Checked gh api --method POST "repos/$($target.Repo)/statuses/$($target.Sha)" `
        -f 'state=success' `
        -f "context=$AttestationContext" `
        -f "description=$description"
}

Write-Host "Real Emulator paired v0.7 integration passed for exact pair: $description"
Write-Host "Published '$AttestationContext' success status to both exact commits."
