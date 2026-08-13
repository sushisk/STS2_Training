param(
    [string]$RlRoot = $env:STS2_RL_ROOT,
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$TrainingRepo = 'sushisk/STS2_Training'
$TrainingPr = 22
$RlRepo = 'sushisk/STS2_RL'
$RlPr = 9

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
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )
    $dirty = & git -C $Path status --porcelain
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to inspect $Label git tree at $Path"
    }
    if ($dirty) {
        throw "$Label working tree must be clean for paired validation"
    }
}

function Assert-ExactLocalHead {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$ExpectedSha,
        [Parameter(Mandatory = $true)][string]$Label
    )
    $actualSha = (& git -C $Path rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $actualSha) {
        throw "Unable to inspect $Label HEAD at $Path"
    }
    if ($actualSha -ne $ExpectedSha) {
        throw "$Label checkout HEAD changed during paired validation: expected=$ExpectedSha actual=$actualSha"
    }
}

function Assert-SameRepoHead {
    param(
        [Parameter(Mandatory = $true)]$PrInfo,
        [Parameter(Mandatory = $true)][string]$ExpectedRepo,
        [Parameter(Mandatory = $true)][string]$Label
    )
    $headRepo = [string]$PrInfo.head.repo.full_name
    if ($headRepo -ne $ExpectedRepo) {
        throw "$Label PR head must come from $ExpectedRepo for this manual Emulator validator, got $headRepo"
    }
}

function Invoke-AdvisoryTestBody {
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
        # Credential stripping is hygiene only. It is NOT a sandbox: PR-controlled
        # Python/test code still has host filesystem/process/network access unless the
        # caller supplies an OS-level isolated environment.
        Remove-Item Env:GH_TOKEN -ErrorAction SilentlyContinue
        Remove-Item Env:GITHUB_TOKEN -ErrorAction SilentlyContinue
        $env:PATH = (($originalPath -split ';') | Where-Object {
            $_ -and $_.TrimEnd('\') -ine $GhDirectory.TrimEnd('\')
        }) -join ';'
        $env:STS2_RL_ROOT = $RlRootResolved

        if (-not $SkipDependencyInstall) {
            Write-Host 'Installing Training test dependencies without GitHub API credentials...'
            Invoke-Checked python -m pip install -e "$TrainingRoot[test]" pythonnet
        }

        Write-Host 'Running advisory real-Emulator paired v0.7 integration...'
        # The whole directory, not a single file: every real-Emulator paired test
        # belongs here (currently the wire-protocol paired test plus the runner
        # package's end-to-end episode test), and a future addition should not
        # need this script edited to be picked up.
        Invoke-Checked python -m pytest "$TrainingRoot/tests/integration" -q
    }
    finally {
        $env:PATH = $originalPath
        if ($hadGhToken) { $env:GH_TOKEN = $savedGhToken } else { Remove-Item Env:GH_TOKEN -ErrorAction SilentlyContinue }
        if ($hadGithubToken) { $env:GITHUB_TOKEN = $savedGithubToken } else { Remove-Item Env:GITHUB_TOKEN -ErrorAction SilentlyContinue }
    }
}

Write-Warning 'This script is advisory/manual validation only. It does not publish GitHub commit statuses and must not be used as branch-protection proof of exact-pair compatibility.'
Write-Warning 'Run PR-controlled Emulator tests only on a disposable/isolated host if the result will influence release decisions.'

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
Assert-SameRepoHead -PrInfo $trainingPrInfo -ExpectedRepo $TrainingRepo -Label 'Training'
Assert-SameRepoHead -PrInfo $rlPrInfo -ExpectedRepo $RlRepo -Label 'RL'

$currentTrainingSha = [string]$trainingPrInfo.head.sha
$currentRlSha = [string]$rlPrInfo.head.sha
Assert-ExactLocalHead -Path $trainingRoot -ExpectedSha $currentTrainingSha -Label 'Training'
Assert-ExactLocalHead -Path $rlRootResolved -ExpectedSha $currentRlSha -Label 'RL'

Write-Host "Training current PR head: $currentTrainingSha"
Write-Host "RL current PR head:       $currentRlSha"

Invoke-AdvisoryTestBody `
    -TrainingRoot $trainingRoot `
    -RlRootResolved $rlRootResolved `
    -GhDirectory $ghDirectory `
    -SkipDependencyInstall ([bool]$SkipInstall)

# The test code had host access. Verify only the properties this helper can actually
# verify: both local worktrees stayed clean and at the exact SHAs, and neither PR head
# moved while the test ran. These checks do not turn the host into a security boundary.
Assert-ExactLocalHead -Path $trainingRoot -ExpectedSha $currentTrainingSha -Label 'Training'
Assert-ExactLocalHead -Path $rlRootResolved -ExpectedSha $currentRlSha -Label 'RL'
Assert-CleanGitTree -Path $trainingRoot -Label 'Training'
Assert-CleanGitTree -Path $rlRootResolved -Label 'RL'

$latestTrainingPrInfo = Invoke-GhJson api "repos/$TrainingRepo/pulls/$TrainingPr"
$latestRlPrInfo = Invoke-GhJson api "repos/$RlRepo/pulls/$RlPr"
$latestTrainingSha = [string]$latestTrainingPrInfo.head.sha
$latestRlSha = [string]$latestRlPrInfo.head.sha

if ($latestTrainingSha -ne $currentTrainingSha -or $latestRlSha -ne $currentRlSha) {
    throw "A PR head moved during the real-environment test. Training=$latestTrainingSha RL=$latestRlSha; discard this result and rerun the current pair."
}

Write-Host "Advisory real-Emulator paired v0.7 validation passed for Training=$currentTrainingSha RL=$currentRlSha"
Write-Host 'No GitHub status was published. A trusted exact-pair release gate requires the isolated orchestrator described in docs/dto_paired_release_gate_v0.7.md.'
