$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$AppUrl = "http://localhost:5000"
$DockerDesktopCandidates = @()
if ($env:ProgramFiles) {
    $DockerDesktopCandidates += (Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe")
}
if (${env:ProgramFiles(x86)}) {
    $DockerDesktopCandidates += (Join-Path ${env:ProgramFiles(x86)} "Docker\Docker\Docker Desktop.exe")
}
if ($env:LOCALAPPDATA) {
    $DockerDesktopCandidates += (Join-Path $env:LOCALAPPDATA "Docker\Docker Desktop.exe")
}
$DockerDesktopPaths = $DockerDesktopCandidates | Where-Object { $_ -and (Test-Path $_) }

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Fail-Friendly {
    param([string]$Message)
    Write-Host ""
    Write-Host "CHYBA: $Message" -ForegroundColor Red
    Write-Host ""
    Write-Host "Okno se zavre za 20 sekund." -ForegroundColor Yellow
    Start-Sleep -Seconds 20
    exit 1
}

function Test-DockerReady {
    try {
        docker info *> $null
        return ($LASTEXITCODE -eq 0)
    }
    catch {
        return $false
    }
}

Set-Location $ProjectDir

Write-Host "Radha Subtitle Tool - spusteni aplikace" -ForegroundColor Green
Write-Host "Slozka projektu: $ProjectDir"

Write-Step "Kontroluji Docker"
$DockerCommand = Get-Command docker -ErrorAction SilentlyContinue
if (-not $DockerCommand) {
    Fail-Friendly "Prikaz docker nebyl nalezen. Nainstalujte Docker Desktop pro Windows a zkuste to znovu."
}

if (-not (Test-DockerReady)) {
    Write-Step "Docker zatim neni dostupny, zkousim spustit Docker Desktop"

    if (-not $DockerDesktopPaths -or $DockerDesktopPaths.Count -eq 0) {
        Fail-Friendly "Docker Desktop neni nainstalovany nebo nebyl nalezen v obvyklem umisteni. Nainstalujte Docker Desktop pro Windows."
    }

    $DockerDesktopExe = $DockerDesktopPaths[0]
    try {
        Start-Process -FilePath $DockerDesktopExe -WindowStyle Hidden
    }
    catch {
        Fail-Friendly "Docker Desktop se nepodarilo spustit: $($_.Exception.Message)"
    }
}

Write-Step "Cekam, az bude Docker dostupny"
$DockerReady = $false
for ($i = 1; $i -le 60; $i++) {
    if (Test-DockerReady) {
        $DockerReady = $true
        break
    }
    Write-Host "Docker startuje... pokus $i/60"
    Start-Sleep -Seconds 3
}

if (-not $DockerReady) {
    Fail-Friendly "Docker Desktop se nespustil vcas. Zkontrolujte, ze Docker Desktop bezi a mate k nemu pristup."
}

Write-Step "Spoustim aplikaci pres docker compose"
try {
    docker compose up -d --build
    if ($LASTEXITCODE -ne 0) {
        Fail-Friendly "docker compose up -d --build skoncil chybou."
    }
}
catch {
    Fail-Friendly "Aplikaci se nepodarilo spustit pres Docker Compose: $($_.Exception.Message)"
}

Write-Step "Oteviram webove rozhrani"
try {
    Start-Process $AppUrl
}
catch {
    Write-Host "Prohlizec se nepodarilo otevrit automaticky. Otevrete rucne: $AppUrl" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Hotovo. Web bezi na: $AppUrl" -ForegroundColor Green
Write-Host "Toto okno se zavre za 5 sekund."
Start-Sleep -Seconds 5
