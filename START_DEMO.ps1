# Opens the built demo (dist\index.html) full screen in Chrome or Edge.
# Works offline: the page is one self-contained file, no server and no network.
#   powershell -ExecutionPolicy Bypass -File START_DEMO.ps1           # launch
#   powershell -ExecutionPolicy Bypass -File START_DEMO.ps1 -DryRun   # only report what would run
param([switch]$DryRun)

$ErrorActionPreference = 'Stop'
$html = Join-Path $PSScriptRoot 'dist\index.html'
if (-not (Test-Path $html)) {
    Write-Host 'dist\index.html not found. Build it once with:  npm ci   then   npm run build' -ForegroundColor Red
    exit 1
}
$url = ([System.Uri](Resolve-Path $html).Path).AbsoluteUri

$candidates = @(
    (Join-Path $env:ProgramFiles 'Google\Chrome\Application\chrome.exe'),
    (Join-Path ${env:ProgramFiles(x86)} 'Google\Chrome\Application\chrome.exe'),
    (Join-Path $env:LOCALAPPDATA 'Google\Chrome\Application\chrome.exe'),
    (Join-Path ${env:ProgramFiles(x86)} 'Microsoft\Edge\Application\msedge.exe'),
    (Join-Path $env:ProgramFiles 'Microsoft\Edge\Application\msedge.exe')
)
$browser = $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1

# A separate browser profile makes the full-screen flags apply even when the
# browser is already open with other windows.
$profileDir = Join-Path $env:LOCALAPPDATA 'NonparametricDemoBrowserProfile'
$browserArgs = @("--user-data-dir=`"$profileDir`"", '--no-first-run', '--no-default-browser-check', '--start-fullscreen', "--app=$url")

if ($DryRun) {
    Write-Host "Page:    $url"
    Write-Host "Browser: $(if ($browser) { $browser } else { 'default browser (Chrome/Edge not found)' })"
    Write-Host "Args:    $($browserArgs -join ' ')"
    exit 0
}

if ($browser) {
    Start-Process -FilePath $browser -ArgumentList $browserArgs
} else {
    Start-Process $url
}
