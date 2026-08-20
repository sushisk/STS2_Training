param(
    [string] $SourceRoot = 'C:\STS2_Training\data\oracle',
    [string] $DestinationRoot = 'C:\STS2_Training\data\oracle_dataset'
)

$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $DestinationRoot | Out-Null
$aggregatePath = Join-Path $DestinationRoot 'oracle_dataset.jsonl'
$manifestPath = Join-Path $DestinationRoot 'manifest.jsonl'
Remove-Item -LiteralPath $aggregatePath, $manifestPath -Force -ErrorAction SilentlyContinue

$utf8 = [Text.UTF8Encoding]::new($false)
$writer = [IO.StreamWriter]::new($aggregatePath, $false, $utf8)
$manifestWriter = [IO.StreamWriter]::new($manifestPath, $false, $utf8)
$episodes = 0
$decisions = 0
$bytes = 0

try {
    foreach ($dateDir in Get-ChildItem $SourceRoot -Directory | Sort-Object Name) {
        $destinationDateDir = Join-Path $DestinationRoot $dateDir.Name
        New-Item -ItemType Directory -Force -Path $destinationDateDir | Out-Null

        foreach ($source in Get-ChildItem $dateDir.FullName -Filter *.jsonl -File | Sort-Object Name) {
            $last = Get-Content $source.FullName | Where-Object { $_.Trim() } | Select-Object -Last 1
            if (-not $last) { continue }
            try { $result = $last | ConvertFrom-Json } catch { continue }
            if ($result.record_type -ne 'combat_oracle_episode_result' -or
                $result.completed -ne $true -or
                $result.termination_reason -ne 'terminal') { continue }

            $destination = Join-Path $destinationDateDir $source.Name
            Copy-Item -LiteralPath $source.FullName -Destination $destination -Force

            $reader = [IO.StreamReader]::new($source.FullName)
            $lineCount = 0
            $decisionCount = 0
            try {
                while ($null -ne ($line = $reader.ReadLine())) {
                    if ($line.Length -eq 0) { continue }
                    $writer.WriteLine($line)
                    $lineCount++
                    if ($line.Contains('"record_type": "combat_oracle_decision"') -or
                        $line.Contains('"record_type":"combat_oracle_decision"')) {
                        $decisionCount++
                    }
                }
            }
            finally { $reader.Dispose() }

            $manifest = [ordered]@{
                date = $dateDir.Name
                source = $source.FullName
                copied = $destination
                bytes = $source.Length
                jsonl_lines = $lineCount
                decision_records = $decisionCount
            }
            $manifestWriter.WriteLine(($manifest | ConvertTo-Json -Compress))
            $episodes++
            $decisions += $decisionCount
            $bytes += $source.Length
        }
    }
}
finally {
    $writer.Dispose()
    $manifestWriter.Dispose()
}

Write-Output "episodes=$episodes decisions=$decisions bytes=$bytes aggregate=$aggregatePath manifest=$manifestPath"
