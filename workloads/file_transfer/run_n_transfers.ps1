[CmdletBinding()]
param(
    [ValidateRange(1, 256)]
    [int]$N = 2,

    [ValidateNotNullOrEmpty()]
    [string]$Experiment = "file-transfer",

    [ValidateRange(0, 86400)]
    [int]$Stagger = 20,

    [ValidateRange(1, 86400)]
    [int]$Duration = 100,

    [ValidateNotNullOrEmpty()]
    [string]$CC = "unspecified",

    [ValidateNotNullOrEmpty()]
    [string]$BaseUrl = "http://100.90.202.72:8080/",

    [ValidateNotNullOrEmpty()]
    [string[]]$Files = @("file-100MiB.bin"),

    [ValidateNotNullOrEmpty()]
    [string]$OutputRoot = "data",

    [ValidateNotNullOrEmpty()]
    [string]$CurlPath = "curl.exe"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

if ($CC -notmatch '^[A-Za-z0-9._-]+$') {
    throw "CC must contain only letters, numbers, dots, underscores, or hyphens."
}

$baseUri = $null
if (-not [Uri]::TryCreate($BaseUrl, [UriKind]::Absolute, [ref]$baseUri) -or
    $baseUri.Scheme -notin @("http", "https")) {
    throw "BaseUrl must be an absolute HTTP or HTTPS URL."
}

$cleanFiles = @()
foreach ($fileName in $Files) {
    $relativeName = $fileName.Trim().Replace("\", "/").TrimStart("/")
    if ([string]::IsNullOrWhiteSpace($relativeName)) {
        throw "Files cannot contain an empty file name."
    }
    if ($relativeName -split "/" -contains "..") {
        throw "File paths cannot contain '..' segments: $fileName"
    }
    $cleanFiles += $relativeName
}

$curlCommand = Get-Command $CurlPath -CommandType Application -ErrorAction Stop
$curlExe = $curlCommand.Source
$curlVersion = (& $curlExe --version | Select-Object -First 1)
if ($LASTEXITCODE -ne 0 -or $curlVersion -notmatch '^curl (\d+)\.(\d+)\.(\d+)') {
    throw "Could not determine the curl version from '$curlExe'."
}
$curlMajor = [int]$Matches[1]
$curlMinor = [int]$Matches[2]
if ($curlMajor -lt 7 -or ($curlMajor -eq 7 -and $curlMinor -lt 75)) {
    throw "curl 7.75 or newer is required for JSON transfer metrics and exit codes. Found: $curlVersion"
}

if ([IO.Path]::IsPathRooted($OutputRoot)) {
    $outputRootPath = [IO.Path]::GetFullPath($OutputRoot)
} else {
    $outputRootPath = [IO.Path]::GetFullPath((Join-Path $scriptRoot $OutputRoot))
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss-fff"
$suffix = [guid]::NewGuid().ToString("N").Substring(0, 8)
$runId = "n${N}-transfer-${stamp}-${suffix}"
$experimentDir = Join-Path (Join-Path $outputRootPath $CC) $runId
$logsDir = Join-Path $experimentDir "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

$runStarted = [DateTimeOffset]::UtcNow
$runClock = [Diagnostics.Stopwatch]::StartNew()
$connectTimeout = [math]::Min(10, $Duration)
$metadataPath = Join-Path $experimentDir "metadata.json"
$summaryPath = Join-Path $experimentDir "summary.csv"

$metadata = [ordered]@{
    schema_version = 1
    experiment_id = $runId
    experiment = $Experiment
    cc = $CC
    n = $N
    stagger_seconds = $Stagger
    timeout_seconds = $Duration
    base_url = $baseUri.AbsoluteUri
    files = @($cleanFiles)
    started_utc = $runStarted.ToString("o")
    client_host = [Environment]::MachineName
    curl = $curlVersion
}
$metadata | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $metadataPath -Encoding UTF8

Write-Host "Starting $N file transfers (experiment: $Experiment, CC label: $CC)"
Write-Host "Each transfer has a $Duration s timeout; starts are staggered by $Stagger s."
Write-Host "Server: $($baseUri.AbsoluteUri)"
Write-Host "Data directory: $experimentDir"

$flows = @()
for ($i = 1; $i -le $N; $i++) {
    $fileName = $cleanFiles[($i - 1) % $cleanFiles.Count]
    $encodedSegments = @($fileName -split "/" | ForEach-Object {
        [Uri]::EscapeDataString($_)
    })
    $encodedName = $encodedSegments -join "/"
    $fileUri = [Uri]::new(
        [Uri]::new($baseUri.AbsoluteUri.TrimEnd("/") + "/"),
        $encodedName
    )
    $uriBuilder = [UriBuilder]::new($fileUri)
    $uriBuilder.Query = "olympus_run=$([Uri]::EscapeDataString($runId))&flow=$i"
    $url = $uriBuilder.Uri.AbsoluteUri

    $metricsPath = Join-Path $logsDir "transfer-${i}.curl.json"
    $stderrPath = Join-Path $logsDir "transfer-${i}.err.log"
    $plannedOffset = ($i - 1) * $Stagger
    $startedUtc = [DateTimeOffset]::UtcNow
    $actualOffset = $runClock.Elapsed.TotalSeconds

    $curlArgs = @(
        "--silent",
        "--show-error",
        "--location",
        "--fail",
        "--noproxy", "*",
        "--output", "NUL",
        "--connect-timeout", "$connectTimeout",
        "--max-time", "$Duration",
        "--write-out", "%{json}",
        $url
    )

    $proc = Start-Process -FilePath $curlExe `
        -ArgumentList $curlArgs `
        -RedirectStandardOutput $metricsPath `
        -RedirectStandardError $stderrPath `
        -NoNewWindow `
        -PassThru

    $flows += [pscustomobject]@{
        FlowId = $i
        FileName = $fileName
        Url = $url
        PlannedOffset = $plannedOffset
        ActualOffset = $actualOffset
        StartedUtc = $startedUtc
        MetricsPath = $metricsPath
        StderrPath = $stderrPath
        Process = $proc
    }

    Write-Host ("  Transfer {0} started (PID {1}, file {2})" -f $i, $proc.Id, $fileName)
    if ($i -lt $N -and $Stagger -gt 0) {
        Write-Host "  Waiting $Stagger s before the next transfer..."
        Start-Sleep -Seconds $Stagger
    }
}

$rows = @()
foreach ($flow in $flows) {
    $flow.Process.WaitForExit()
    $flow.Process.Refresh()
    $rawProcessExitCode = $flow.Process.ExitCode
    $collectedUtc = [DateTimeOffset]::UtcNow

    $curlMetrics = $null
    $metricsError = ""
    try {
        $rawMetrics = (Get-Content -LiteralPath $flow.MetricsPath -Raw).Trim()
        if ([string]::IsNullOrWhiteSpace($rawMetrics)) {
            throw "curl did not emit metrics"
        }
        $curlMetrics = $rawMetrics | ConvertFrom-Json
    } catch {
        $metricsError = $_.Exception.Message
    }

    $stderrText = ""
    if (Test-Path -LiteralPath $flow.StderrPath -PathType Leaf) {
        $stderrRaw = Get-Content -LiteralPath $flow.StderrPath -Raw
        if ($null -ne $stderrRaw) {
            $stderrText = $stderrRaw.Trim()
        }
    }

    $processExitCode = if ($null -ne $rawProcessExitCode) {
        [int]$rawProcessExitCode
    } elseif ($null -ne $curlMetrics) {
        [int]$curlMetrics.exitcode
    } else {
        -1
    }
    $httpCode = if ($null -ne $curlMetrics) { [int]$curlMetrics.http_code } else { 0 }
    $observedSeconds = if ($null -ne $curlMetrics) {
        [double]$curlMetrics.time_total
    } else {
        ($collectedUtc - $flow.StartedUtc).TotalSeconds
    }
    $endedUtc = if ($null -ne $curlMetrics) {
        $flow.StartedUtc.AddSeconds($observedSeconds)
    } else {
        $collectedUtc
    }
    $completed = (
        $processExitCode -eq 0 -and
        $httpCode -ge 200 -and
        $httpCode -lt 400
    )
    $status = if ($completed) {
        "completed"
    } elseif ($processExitCode -eq 28) {
        "timed_out"
    } elseif ($processExitCode -eq 22) {
        "http_error"
    } else {
        "failed"
    }

    $errorParts = @($stderrText, $metricsError) | Where-Object {
        -not [string]::IsNullOrWhiteSpace($_)
    }
    $errorText = $errorParts -join " | "
    $fctSeconds = if ($completed) { $observedSeconds } else { $null }
    $sizeBytes = if ($null -ne $curlMetrics) { [double]$curlMetrics.size_download } else { 0 }
    $speedBytesPerSecond = if ($null -ne $curlMetrics) {
        [double]$curlMetrics.speed_download
    } else {
        0
    }

    $rows += [pscustomobject][ordered]@{
        experiment_id = $runId
        experiment = $Experiment
        cc = $CC
        n = $N
        flow_id = $flow.FlowId
        file_name = $flow.FileName
        base_url = $baseUri.AbsoluteUri
        url = $flow.Url
        stagger_seconds = $Stagger
        planned_start_offset_seconds = $flow.PlannedOffset
        actual_start_offset_seconds = [math]::Round($flow.ActualOffset, 6)
        timeout_seconds = $Duration
        started_utc = $flow.StartedUtc.ToString("o")
        ended_utc = $endedUtc.ToString("o")
        status = $status
        completed = $completed
        http_code = $httpCode
        curl_exit_code = $processExitCode
        fct_seconds = $fctSeconds
        observed_seconds = [math]::Round($observedSeconds, 6)
        time_to_first_byte_seconds = if ($null -ne $curlMetrics) {
            [double]$curlMetrics.time_starttransfer
        } else {
            $null
        }
        size_bytes = $sizeBytes
        mean_goodput_mbps = [math]::Round(($speedBytesPerSecond * 8 / 1000000), 6)
        error = $errorText
    }

    $colour = if ($completed) { "Green" } else { "Yellow" }
    Write-Host (
        "  Transfer {0}: {1}, HTTP {2}, observed {3:N3} s, {4:N2} Mbit/s" -f
        $flow.FlowId,
        $status,
        $httpCode,
        $observedSeconds,
        ($speedBytesPerSecond * 8 / 1000000)
    ) -ForegroundColor $colour
    $flow.Process.Dispose()
}

$rows |
    Sort-Object flow_id |
    Export-Csv -LiteralPath $summaryPath -NoTypeInformation -Encoding UTF8

$completedCount = @($rows | Where-Object completed).Count
$metadata["completed_utc"] = [DateTimeOffset]::UtcNow.ToString("o")
$metadata["completed_transfers"] = $completedCount
$metadata["incomplete_transfers"] = $N - $completedCount
$metadata | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $metadataPath -Encoding UTF8

Write-Host ""
Write-Host "Summary: $summaryPath"
if ($completedCount -lt $N) {
    Write-Warning "$($N - $completedCount) of $N transfers did not complete. They remain in the data with a blank FCT."
}
Write-Host "After repeated runs, average them with: python .\analyze_results.py"
