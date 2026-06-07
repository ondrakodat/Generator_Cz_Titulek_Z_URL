$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ShortcutPath = Join-Path ([Environment]::GetFolderPath("Desktop")) "Radha Subtitle Tool.lnk"
$TargetPath = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$StartScript = Join-Path $ProjectDir "start_app.ps1"
$IconPath = Join-Path $ProjectDir "assets\icon.ico"

if (-not (Test-Path $StartScript)) {
    Write-Error "Soubor start_app.ps1 nebyl nalezen: $StartScript"
    exit 1
}

$Shell = New-Object -ComObject WScript.Shell
$Shortcut = $Shell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $TargetPath
$Shortcut.Arguments = "-ExecutionPolicy Bypass -NoProfile -File `"$StartScript`""
$Shortcut.WorkingDirectory = $ProjectDir
$Shortcut.Description = "Radha Subtitle Tool"

if (Test-Path $IconPath) {
    $Shortcut.IconLocation = $IconPath
}

$Shortcut.Save()
Write-Host "Zastupce byl vytvoren na plose: $ShortcutPath"
