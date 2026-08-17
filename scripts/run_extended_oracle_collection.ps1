$ErrorActionPreference = 'Continue'
$trainingRoot = 'C:\STS2_Training'
$outputDir = Join-Path $trainingRoot 'data\combat_oracle_extended_20260817'
$runLog = Join-Path $outputDir 'collection_runner.log'

Set-Location $trainingRoot
$files = Get-ChildItem (Join-Path $trainingRoot 'data\scenarios\godmode_harvested') -File -Filter '*.json' | Sort-Object Name
$selected = @()
foreach ($character in @('IRONCLAD','SILENT','DEFECT','REGENT','NECROBINDER')) {
    $selected += @($files | Where-Object { $_.Name.ToLowerInvariant().StartsWith($character.ToLowerInvariant() + '-') } | Select-Object -First 6)
}

for ($index = 14; $index -lt 30; $index++) {
    $number = $index + 1
    $scenario = $selected[$index]
    $output = Join-Path $outputDir (('{0:D2}-{1}.jsonl' -f $number, $scenario.BaseName))
    if (Test-Path -LiteralPath $output) {
        $output = Join-Path $outputDir (('{0:D2}-{1}.retry.jsonl' -f $number, $scenario.BaseName))
    }
    Add-Content -LiteralPath $runLog -Value ("[{0}] start {1} -> {2}" -f (Get-Date -Format o), $scenario.Name, $output)
    & python -m sts2_training.runner.oracle_collection `
        --host 127.0.0.1 --port 8765 `
        --connect-timeout 30 --decision-timeout 300 --max-decisions 200 `
        --search-mode none --scenario $scenario.FullName --output $output `
        --oracle-beam-width 32 --oracle-top-k 8 --oracle-depth 4 `
        --target-beam-width 8 --oracle-timeout 300 *>> $runLog
    Add-Content -LiteralPath $runLog -Value ("[{0}] exit={1} size={2}" -f (Get-Date -Format o), $LASTEXITCODE, (Get-Item -LiteralPath $output -ErrorAction SilentlyContinue).Length)
}
