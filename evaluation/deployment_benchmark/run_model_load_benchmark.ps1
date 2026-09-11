param(
    [Parameter(Mandatory = $true)]
    [string]$Serial
)

$ErrorActionPreference = "Stop"

$adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"

$package = "com.example.edge_ai_smart_bank_transfers"
$activity = "$package/.MainActivity"

$benchmarkDir = $PSScriptRoot
$outCsv = Join-Path $benchmarkDir "model_load_benchmark.csv"

function ADB {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    & $adb -s $Serial @Arguments
}

function Parse-Number {
    param([string]$Value)

    return [double]::Parse(
        $Value.Replace(",", "."),
        [System.Globalization.CultureInfo]::InvariantCulture
    )
}

function Measure-Load {

    param(
        [string]$Condition,
        [int]$Index
    )

    Write-Host ""
    Write-Host "----------------------------------------------"
    Write-Host "$Condition measurement $Index"
    Write-Host "----------------------------------------------"

    ADB shell am force-stop $package | Out-Null
    Start-Sleep -Seconds 2

    ADB logcat -c | Out-Null

    $batteryRaw = ADB shell dumpsys battery

    $levelLine =
        $batteryRaw |
        Where-Object { $_ -match "^\s*level:" } |
        Select-Object -First 1

    $tempLine =
        $batteryRaw |
        Where-Object { $_ -match "^\s*temperature:" } |
        Select-Object -First 1

    $batteryPercent =
        [int](($levelLine -split ":", 2)[1].Trim())

    $batteryTemperature =
        [double](($tempLine -split ":", 2)[1].Trim()) / 10.0

    ADB shell am start -n $activity | Out-Null

    Write-Host "Waiting for model initialization..."

    Start-Sleep -Seconds 7

    $logs = ADB logcat -d -t 400

    $modelLine =
        $logs |
        Select-String "DEPLOYMENT_BENCHMARK_MODEL" |
        Select-Object -Last 1

    $contextLine =
        $logs |
        Select-String "DEPLOYMENT_BENCHMARK_CONTEXT" |
        Select-Object -Last 1

    if (-not $modelLine) {
        throw "MODEL line not found for $Condition measurement $Index."
    }

    if (-not $contextLine) {
        throw "CONTEXT line not found for $Condition measurement $Index."
    }

    if (
        $modelLine.Line -notmatch
        'model_load_ms=([0-9]+(?:[.,][0-9]+)?)'
    ) {
        throw "Unable to parse model load time."
    }

    $modelLoad = Parse-Number $Matches[1]

    if (
        $contextLine.Line -notmatch
        'context_creation_ms=([0-9]+(?:[.,][0-9]+)?)'
    ) {
        throw "Unable to parse context creation time."
    }

    $contextLoad = Parse-Number $Matches[1]

    Write-Host ("Model load:       {0:N3} ms" -f $modelLoad)
    Write-Host ("Context creation: {0:N3} ms" -f $contextLoad)

    return [pscustomobject]@{
        condition = $Condition
        measurement = $Index
        model_load_ms = $modelLoad
        context_creation_ms = $contextLoad
        battery_percent = $batteryPercent
        battery_temperature_c = $batteryTemperature
    }
}

$state = (ADB get-state).Trim()

if ($state -ne "device") {
    throw "ADB device unavailable: $Serial"
}

$results = @()

# The device must have been rebooted immediately before
# executing this script. The first application process start
# is therefore treated as the post-reboot condition.
$results += Measure-Load `
    -Condition "cold_post_reboot" `
    -Index 1

for ($i = 1; $i -le 5; $i++) {

    $results += Measure-Load `
        -Condition "warm_process_restart" `
        -Index $i
}

$results |
    Export-Csv `
        $outCsv `
        -NoTypeInformation `
        -Encoding UTF8

$cold = $results[0]

$warm = @(
    $results |
    Where-Object {
        $_.condition -eq "warm_process_restart"
    }
)

$values = @(
    $warm |
    ForEach-Object {
        [double]$_.model_load_ms
    }
)

$mean =
    ($values | Measure-Object -Average).Average

$sorted =
    @($values | Sort-Object)

$median =
    $sorted[2]

$min =
    ($values | Measure-Object -Minimum).Minimum

$max =
    ($values | Measure-Object -Maximum).Maximum

$reduction =
    (($cold.model_load_ms - $mean) /
     $cold.model_load_ms) * 100.0

Write-Host ""
Write-Host "=============================================="
Write-Host " MODEL LOAD BENCHMARK COMPLETE"
Write-Host "=============================================="
Write-Host ""

Write-Host (
    "Cold post-reboot: {0:N3} ms" `
    -f $cold.model_load_ms
)

Write-Host ""

foreach ($row in $warm) {

    Write-Host (
        "Warm run {0}: {1:N3} ms" `
        -f $row.measurement,
           $row.model_load_ms
    )
}

Write-Host ""
Write-Host ("Warm mean:   {0:N3} ms" -f $mean)
Write-Host ("Warm median: {0:N3} ms" -f $median)
Write-Host ("Warm min:    {0:N3} ms" -f $min)
Write-Host ("Warm max:    {0:N3} ms" -f $max)

Write-Host (
    "Reduction vs cold: {0:N2}%" `
    -f $reduction
)

Write-Host ""
Write-Host "Saved:"
Write-Host $outCsv
Write-Host ""
