@echo off
setlocal

cd /d "%~dp0"
set "APP_URL=http://127.0.0.1:5000"
if not exist "temp\pip_tmp" mkdir "temp\pip_tmp"
set "TEMP=%CD%\temp\pip_tmp"
set "TMP=%TEMP%"

netstat -ano | findstr ":5000" | findstr "LISTENING" >nul
if %ERRORLEVEL%==0 (
    echo Radha Subtitle Tool uz bezi na portu 5000.
    start "" "%APP_URL%"
    exit /b 0
)

where python >nul 2>nul
if errorlevel 1 (
    echo Python nebyl nalezen. Nainstalujte Python 3.11+ a zaskrtnete Add python.exe to PATH.
    pause
    exit /b 1
)

if exist ".venv\Scripts\python.exe" (
    if not exist ".venv\Scripts\pip.exe" (
        echo Virtualni prostredi neni kompletni, vytvarim ho znovu...
        rmdir /s /q ".venv"
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo Vytvarim virtualni prostredi .venv...
    python -m venv .venv
    if errorlevel 1 (
        echo Virtualni prostredi se nepodarilo vytvorit.
        pause
        exit /b 1
    )
)

if not exist ".venv\Scripts\pip.exe" (
    echo Doplnuji pip do virtualniho prostredi...
    ".venv\Scripts\python.exe" -m ensurepip --upgrade --default-pip
    if errorlevel 1 (
        echo Pip se nepodarilo doplnit. Zkontrolujte opravneni ke slozce TEMP nebo spustte PowerShell jako bezny uzivatel.
        pause
        exit /b 1
    )
)

echo Instaluji nebo aktualizuji zavislosti...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Instalace zavislosti selhala.
    pause
    exit /b 1
)

where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo FFmpeg nebyl nalezen v PATH. Nainstalujte ho napr. pres: winget install Gyan.FFmpeg
    pause
    exit /b 1
)

echo Spoustim Radha Subtitle Tool...
start "Radha Subtitle Tool Server" cmd /k ".venv\Scripts\python.exe web_app.py"
timeout /t 3 /nobreak >nul
start "" "%APP_URL%"

endlocal
