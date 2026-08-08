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
$TrustedStatusIssuer = 'sushisk'
$TrustedStatusIssuerId = 136587185
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
        throw "$Label working tree must be clean for exact-pair validation"
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
        throw "$Label checkout HEAD changed during exact-pair validation: expected=$ExpectedSha actual=$actualSha"
    }
}

function Assert-TrustedStatusIssuer {
    $viewer = Invoke-GhJson api user
    $login = [string]$viewer.login
    $id = [int64]$viewer.id
    if ($login -ne $TrustedStatusIssuer -or $id -ne $TrustedStatusIssuerId) {
        throw "GitHub status issuer must be $TrustedStatusIssuer ($TrustedStatusIssuerId), got $login ($id)"
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

function Get-CurrentHeadWorkflowRun {
    param(
        [Parameter(Mandatory = $true)][string]$Repo,
        [Parameter(Mandatory = $true)][string]$HeadSha,
        [Parameter(Mandatory = $true)][string]$WorkflowName
    )
    $deadline = (Get-Date).AddMinutes(2)
    do {
        $runs = Invoke-GhJson api "repos/$Repo/actions/runs?head_sha=$HeadSha&event=pull_request&per_page=100"
        $matches = @(
            $runs.workflow_runs |
                Where-Object { [string]$_.name -eq $WorkflowName -and [string]$_.head_sha -eq $HeadSha } |
                Sort-Object -Property created_at -Descending
        )
        if ($matches.Count -gt 0) {
            return $matches[0]
        }
        Start-Sleep -Seconds 2
    } while ((Get-Date) -lt $deadline)
    throw "No pull_request workflow run named '$WorkflowName' found for $Repo@$HeadSha"
}

function Wait-WorkflowRunCompleted {
    param(
        [Parameter(Mandatory = $true)][string]$Repo,
        [Parameter(Mandatory = $true)][int64]$RunId,
        [Parameter(Mandatory = $true)][datetime]$Deadline
    )
    do {
        $run = Invoke-GhJson api "repos/$Repo/actions/runs/$RunId"
        if ([string]$run.status -eq 'completed') {
            return $run
        }
        Start-Sleep -Seconds 3
    } while ((Get-Date) -lt $Deadline)
    throw "Timed out waiting for GitHub Actions run $Repo#$RunId"
}

function Assert-PairGateGreen {
    param(
        [Parameter(Mandatory = $true)][string]$Repo,
        [Parameter(Mandatory = $true)][string]$HeadSha,
        [Parameter(Mandatory = $true)][string]$WorkflowName
    )
    $run = Get-CurrentHeadWorkflowRun -Repo $Repo -HeadSha $HeadSha -WorkflowName $WorkflowName
    $run = Wait-WorkflowRunCompleted -Repo $Repo -RunId ([int64]$run.id) -Deadline ((Get-Date).AddMinutes(5))
    if ([string]$run.conclusion -ne 'success') {
        Write-Host "Re-running $WorkflowName for $Repo@$HeadSha after publishing attestation..."
        Invoke-Checked gh api --method POST "repos/$Repo/actions/runs/$($run.id)/rerun"
        $run = Wait-WorkflowRunCompleted -Repo $Repo -RunId ([int64]$run.id) -Deadline ((Get-Date).AddMinutes(5))
    }
    if ([string]$run.conclusion -ne 'success') {
        throw "$WorkflowName did not become green for $Repo@$HeadSha; conclusion=$($run.conclusion)"
    }
    Write-Host "Verified green GitHub check: $Repo / $WorkflowName / $HeadSha"
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
Assert-TrustedStatusIssuer
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
Assert-ExactLocalHead -Path $trainingRoot -ExpectedSha $currentTrainingSha -Label 'Training'
Assert-ExactLocalHead -Path $rlRootResolved -ExpectedSha $currentRlSha -Label 'RL'

Write-Host "Training current PR head: $currentTrainingSha"
Write-Host "RL current PR head:       $currentRlSha"

Invoke-UncredentialedTestBody `
    -TrainingRoot $trainingRoot `
    -RlRootResolved $rlRootResolved `
    -GhDirectory $ghDirectory `
    -SkipDependencyInstall ([bool]$SkipInstall)

# PR-controlled test code had filesystem access to both checkouts. Before granting a
# trusted status, prove that the exact commits we inspected are still checked out and
# that neither worktree was modified during the test run.
Assert-ExactLocalHead -Path $trainingRoot -ExpectedSha $currentTrainingSha -Label 'Training'
Assert-ExactLocalHead -Path $rlRootResolved -ExpectedSha $currentRlSha -Label 'RL'
Assert-CleanGitTree -Path $trainingRoot -Label 'Training'
Assert-CleanGitTree -Path $rlRootResolved -Label 'RL'
Assert-TrustedStatusIssuer

$latestTrainingPrInfo = Invoke-GhJson api "repos/$TrainingRepo/pulls/$TrainingPr"
$latestRlPrInfo = Invoke-GhJson api "repos/$RlRepo/pulls/$RlPr"
Assert-TrustedPullRequest -PrInfo $latestTrainingPrInfo -ExpectedRepo $TrainingRepo -Label 'Training'
Assert-TrustedPullRequest -PrInfo $latestRlPrInfo -ExpectedRepo $RlRepo -Label 'RL'
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
Write-Host "Published '$AttestationContext' success status as trusted issuer $TrustedStatusIssuer ($TrustedStatusIssuerId)."

Assert-PairGateGreen -Repo $TrainingRepo -HeadSha $currentTrainingSha -WorkflowName 'paired-v07-exact-pair'
Assert-PairGateGreen -Repo $RlRepo -HeadSha $currentRlSha -WorkflowName 'paired-v07-counterpart-gate'
Write-Host 'Both exact-pair GitHub checks are green.'
