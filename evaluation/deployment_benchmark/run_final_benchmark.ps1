$ErrorActionPreference = "Stop"

$adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"

# The script lives in:
# evaluation/deployment_benchmark/
$benchmarkDir = $PSScriptRoot
$repoRoot = Resolve-Path (Join-Path $benchmarkDir "..\..")
$mobileDir = Join-Path $repoRoot "mobile-app"

$package = "com.example.edge_ai_smart_bank_transfers"
$activity = "$package/.MainActivity"

$rawDir = Join-Path $benchmarkDir "raw"

New-Item -ItemType Directory -Force $rawDir | Out-Null

$runsFile = Join-Path $benchmarkDir "runs.csv"
$thermalFile = Join-Path $benchmarkDir "thermal_battery.csv"

$logFile = Join-Path $rawDir "benchmark_logcat.txt"
$procInitialFile = Join-Path $rawDir "proc_status_initial.txt"
$procFinalFile = Join-Path $rawDir "proc_status_final.txt"

$batteryInitialFile = Join-Path $rawDir "battery_benchmark_initial.txt"
$batteryFinalFile = Join-Path $rawDir "battery_benchmark_final.txt"

$thermalInitialFile = Join-Path $rawDir "thermal_benchmark_initial.txt"
$thermalFinalFile = Join-Path $rawDir "thermal_benchmark_final.txt"

Write-Host ""
Write-Host "=============================================="
Write-Host " FINAL DEPLOYMENT BENCHMARK"
Write-Host " 17 cycles x 6 prompts = 102 runs"
Write-Host "=============================================="
Write-Host ""

# ---------------------------------------------------------
# 1. Check device
# ---------------------------------------------------------

if (-not (Test-Path $adb)) {
    throw "adb.exe not found at $adb"
}

$state = (& $adb get-state 2>$null).Trim()

if ($state -ne "device") {
    throw "Samsung device not available through ADB."
}

Write-Host "ADB device detected."

# ---------------------------------------------------------
# 2. Build benchmark APK
# ---------------------------------------------------------

Push-Location $mobileDir

try {
    Write-Host ""
    Write-Host "Building benchmark APK..."

    flutter build apk `
        --debug `
        -t lib/benchmark_main.dart `
        --dart-define=BENCHMARK_CYCLES=17

    if ($LASTEXITCODE -ne 0) {
        throw "Flutter build failed."
    }

    $apk = Join-Path `
        $mobileDir `
        "build\app\outputs\flutter-apk\app-debug.apk"

    if (-not (Test-Path $apk)) {
        throw "APK not found: $apk"
    }

    Write-Host ""
    Write-Host "Installing APK without deleting app data/model..."

    & $adb install -r $apk | Out-Host

    if ($LASTEXITCODE -ne 0) {
        throw "APK installation failed."
    }
}
finally {
    Pop-Location
}

# ---------------------------------------------------------
# 3. Save initial battery / thermal state
# ---------------------------------------------------------

& $adb shell dumpsys battery |
    Set-Content $batteryInitialFile

& $adb shell dumpsys thermalservice |
    Set-Content $thermalInitialFile

# ---------------------------------------------------------
# 4. Clear old logs and start benchmark
# ---------------------------------------------------------

& $adb logcat -c

& $adb shell am force-stop $package

Start-Sleep -Seconds 2

Write-Host ""
Write-Host "Starting benchmark app..."

& $adb shell am start -n $activity | Out-Null

Start-Sleep -Seconds 2

$appPid = (& $adb shell pidof $package).Trim()

if ($appPid) {
    & $adb shell cat "/proc/$appPid/status" |
        Set-Content $procInitialFile
}

# ---------------------------------------------------------
# 5. Monitor battery / thermal state
# ---------------------------------------------------------

$thermalSamples = @()
$startTime = Get-Date
$sampleNumber = 0
$complete = $false

Write-Host ""
Write-Host "Benchmark running. Do not use the phone."
Write-Host ""

while (-not $complete) {

    $sampleNumber++

    $battery = & $adb shell dumpsys battery

    $levelLine =
        $battery |
        Where-Object { $_ -match "^\s*level:" } |
        Select-Object -First 1

    $temperatureLine =
        $battery |
        Where-Object { $_ -match "^\s*temperature:" } |
        Select-Object -First 1

    $statusLine =
        $battery |
        Where-Object { $_ -match "^\s*status:" } |
        Select-Object -First 1

    $batteryLevel = $null
    $batteryTemperature = $null
    $batteryStatus = $null

    if ($levelLine) {
        $batteryLevel =
            [int](($levelLine -split ":", 2)[1].Trim())
    }

    if ($temperatureLine) {
        $rawTemperature =
            [double](($temperatureLine -split ":", 2)[1].Trim())

        $batteryTemperature =
            [math]::Round($rawTemperature / 10.0, 1)
    }

    if ($statusLine) {
        $batteryStatus =
            (($statusLine -split ":", 2)[1].Trim())
    }

    $thermalText =
        (& $adb shell dumpsys thermalservice) -join "`n"

    $thermalStatus = $null

    $patterns = @(
        'Thermal Status:\s*(\d+)',
        'mStatus\s*[=:]\s*(\d+)',
        'status\s*[=:]\s*(\d+)'
    )

    foreach ($pattern in $patterns) {

        $match = [regex]::Match(
            $thermalText,
            $pattern,
            [System.Text.RegularExpressions.RegexOptions]::IgnoreCase
        )

        if ($match.Success) {
            $thermalStatus = [int]$match.Groups[1].Value
            break
        }
    }

    $elapsed =
        [math]::Round(
            ((Get-Date) - $startTime).TotalSeconds,
            3
        )

    $thermalSamples += [pscustomobject]@{
        sample                = $sampleNumber
        elapsed_s             = $elapsed
        timestamp             = (Get-Date).ToString("o")
        battery_percent       = $batteryLevel
        battery_temperature_c = $batteryTemperature
        battery_status_raw    = $batteryStatus
        thermal_status        = $thermalStatus
    }

    Write-Host (
        "Sample {0} | elapsed={1}s | battery={2}% | temp={3} C | thermal={4}" `
        -f $sampleNumber,
           $elapsed,
           $batteryLevel,
           $batteryTemperature,
           $thermalStatus
    )

    $recent =
        (& $adb logcat -d -t 120) -join "`n"

    if (
        $recent -match
        'DEPLOYMENT_BENCHMARK_COMPLETE runs=102'
    ) {
        $complete = $true
        break
    }

    if (
        $recent -match
        'DEPLOYMENT_BENCHMARK_ERROR'
    ) {
        throw "Benchmark application reported an error."
    }

    if (((Get-Date) - $startTime).TotalMinutes -gt 10) {
        throw "Benchmark timed out after 10 minutes."
    }

    Start-Sleep -Seconds 5
}

# ---------------------------------------------------------
# 6. Save final device state
# ---------------------------------------------------------

$thermalSamples |
    Export-Csv `
        $thermalFile `
        -NoTypeInformation `
        -Encoding UTF8

$appPid = (& $adb shell pidof $package).Trim()

if ($appPid) {
    & $adb shell cat "/proc/$appPid/status" |
        Set-Content $procFinalFile
}

& $adb shell dumpsys battery |
    Set-Content $batteryFinalFile

& $adb shell dumpsys thermalservice |
    Set-Content $thermalFinalFile

# ---------------------------------------------------------
# 7. Save complete logcat
# ---------------------------------------------------------

$logs = & $adb logcat -d

$logs |
    Set-Content `
        $logFile `
        -Encoding UTF8

# ---------------------------------------------------------
# 8. Extract native metrics
# ---------------------------------------------------------

$metricLines =
    $logs |
    Where-Object {
        $_ -match 'EdgeAI-Benchmark: METRICS'
    }

$tasks = @(
    "GENERATION",
    "GENERATION_CALENDAR",
    "COMPLETION",
    "COMPLETION",
    "NORMALIZATION",
    "NORMALIZATION"
)

$results = @()

foreach ($line in $metricLines) {

    $pattern =
        'prompt_tokens=(\d+)\s+' +
        'generated_tokens=(\d+)\s+' +
        'ttft_ms=([\d.]+)\s+' +
        'prompt_decode_ms=([\d.]+)\s+' +
        'generation_ms=([\d.]+)\s+' +
        'total_ms=([\d.]+)\s+' +
        'throughput_tok_s=([\d.]+)\s+' +
        'eog=(\w+)'

    $m = [regex]::Match($line, $pattern)

    if (-not $m.Success) {
        continue
    }

    $run = $results.Count + 1
    $cycle = [math]::Ceiling($run / 6.0)

    $promptIndex = ($run - 1) % 6
    $promptId = "P$($promptIndex + 1)"

    $results += [pscustomobject]@{
        run              = $run
        cycle            = $cycle
        prompt_id        = $promptId
        task             = $tasks[$promptIndex]
        prompt_tokens    = [int]$m.Groups[1].Value
        generated_tokens = [int]$m.Groups[2].Value
        ttft_ms          = [double]$m.Groups[3].Value
        prompt_decode_ms = [double]$m.Groups[4].Value
        generation_ms    = [double]$m.Groups[5].Value
        total_ms         = [double]$m.Groups[6].Value
        throughput_tok_s = [double]$m.Groups[7].Value
        eog              = $m.Groups[8].Value
    }
}

$results |
    Export-Csv `
        $runsFile `
        -NoTypeInformation `
        -Encoding UTF8

# ---------------------------------------------------------
# 9. Extract model/context initialization timings
# ---------------------------------------------------------

$modelLine =
    $logs |
    Where-Object {
        $_ -match 'DEPLOYMENT_BENCHMARK_MODEL'
    } |
    Select-Object -Last 1

$contextLine =
    $logs |
    Where-Object {
        $_ -match 'DEPLOYMENT_BENCHMARK_CONTEXT'
    } |
    Select-Object -Last 1

$modelLoadMs = $null
$contextCreationMs = $null

if ($modelLine -match 'model_load_ms=([\d.]+)') {
    $modelLoadMs = [double]$Matches[1]
}

if ($contextLine -match 'context_creation_ms=([\d.]+)') {
    $contextCreationMs = [double]$Matches[1]
}

$initialization = [ordered]@{
    model_load_ms        = $modelLoadMs
    context_creation_ms  = $contextCreationMs
}

$initialization |
    ConvertTo-Json |
    Set-Content `
        (Join-Path $benchmarkDir "initialization.json") `
        -Encoding UTF8

# ---------------------------------------------------------
# 10. Validation
# ---------------------------------------------------------

Write-Host ""
Write-Host "=============================================="
Write-Host " BENCHMARK COMPLETE"
Write-Host "=============================================="
Write-Host ""

Write-Host "Metric rows: $($results.Count)"
Write-Host "Thermal/battery samples: $($thermalSamples.Count)"
Write-Host "Model load: $modelLoadMs ms"
Write-Host "Context creation: $contextCreationMs ms"

$nonEog =
    $results |
    Where-Object { $_.eog -ne "yes" }

Write-Host "Runs without EOG: $($nonEog.Count)"
Write-Host ""

if ($results.Count -ne 102) {
    throw "Expected 102 runs but captured $($results.Count)."
}

if ($nonEog.Count -ne 0) {
    Write-Warning "$($nonEog.Count) runs did not end with EOG."
}

Write-Host "102 / 102 runs captured successfully."
Write-Host ""
Write-Host "Results:"
Write-Host "  $runsFile"
Write-Host "  $thermalFile"
Write-Host "  $logFile"
Write-Host ""
