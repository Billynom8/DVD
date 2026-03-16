@echo off
setlocal enabledelayedexpansion

REM --- ANCHORING ---
REM Force the script to run from its own directory [Critical for 'Run as Admin']
cd /d "%~dp0"

REM DVD Universal Smart Installer (v1.0 - Adapted for DVD)

REM --- LOG INITIALIZATION ---
set "LOGFILE=%~dp0install_log.txt"

REM --- HEALTH CHECK [Pre-flight Permissions] ---
echo [0/7] Verifying Environment...

REM Check if running from a ZIP
echo "%~dp0" | findstr /i "Temp" >nul
if !errorlevel! equ 0 (
    echo [WARNING] It looks like you are running this from a temporary folder or a ZIP.
    echo Please EXTRACT the folder to your Desktop or a permanent location first.
)

REM Check for Admin [Required for persistent pathing/winget]
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Note: Not running as Administrator. Some features [winget] might prompt for permission.
)

REM Clear the log and start the session
echo [%date% %time%] --- NEW INSTALLATION/UPDATE SESSION --- > "%LOGFILE%"

call :log "[1/7] Checking for Git..."
where git >nul 2>&1
if %errorlevel% neq 0 (
    call :log "[INFO] Git not found. Attempting install via Winget..."
    winget install --id Git.Git -e --source winget >> "%LOGFILE%" 2>&1
    if %errorlevel% neq 0 (
        call :log "[ERROR] Git install failed. Please install manually."
        pause && exit /b 1
    )
    set "PATH=%PATH%;C:\Program Files\Git\cmd"
)

call :log "[2/7] Checking for UV..."
where uv >nul 2>&1
if %errorlevel% neq 0 (
    call :log "[INFO] uv not found. Installing..."
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex" >> "%LOGFILE%" 2>&1
    set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
)

call :log "[3/7] Analyzing Path..."

REM 1. Move out of _install if we are in it
for %%I in ("%CD%") do set "CUR_NAME=%%~nxI"
if /i "!CUR_NAME!"=="_install" (
    call :log "[INFO] Moving out of _install folder..."
    cd ..
)

REM 2. Loop to escape nested DVD folders that lack pyproject.toml
:path_loop
for %%I in ("%CD%") do set "CUR_NAME=%%~nxI"
set "PREFIX=!CUR_NAME:~0,3!"
if exist "pyproject.toml" (
    call :log "[INFO] Project folder verified [pyproject.toml found]."
    set "ALREADY_HOME=true"
    goto :found_home
) else (
    if /i "!PREFIX!"=="DVD" (
        REM If we are in DVD but no pyproject.toml, we might be nested inside another DVD folder
        call :log "[INFO] In a DVD folder without pyproject.toml, moving up one level..."
        cd ..
        goto :path_loop
    )
    set "ALREADY_HOME=false"
)

:found_home

call :log "[4/7] Repository Selection..."

REM Define URL
set "DVD_URL=https://github.com/Billynom8/DVD.git"

if "!ALREADY_HOME!"=="false" (
    REM Check for a subfolder that might be the project
    set "FOUND_SUB="
    for /d %%D in (DVD*) do (
        if exist "%%D\pyproject.toml" set "FOUND_SUB=%%D"
    )

    if defined FOUND_SUB (
        call :log "[INFO] Found project in subfolder: !FOUND_SUB!"
        cd "!FOUND_SUB!"
        set "ALREADY_HOME=true"
    ) else (
        REM If we are already in a folder named DVD, we should probably not clone into it creating DVD/DVD
        for %%I in ("%CD%") do set "CUR_NAME=%%~nxI"
        set "PREFIX=!CUR_NAME:~0,3!"
        
        if /i "!PREFIX!"=="DVD" (
            echo [WARNING] You are in a folder named '!CUR_NAME!' but it is not a valid DVD repository.
            set /p confirm_clone="Clone fresh repository into a subfolder? [Y/N]: "
            if /i "!confirm_clone!" neq "Y" (
                call :log "[ERROR] Installation cancelled by user to avoid nested folders."
                pause && exit /b 1
            )
        )

        call :log "[INFO] Cloning fresh repository from !DVD_URL!..."
        git clone --recurse-submodules !DVD_URL!
        if !errorlevel! neq 0 (
            call :log "[ERROR] Git clone failed. Check your internet connection."
            pause && exit /b 1
        )
        cd DVD
        set "ALREADY_HOME=true"
        set "JUST_CLONED=true"
    )
)

REM Standardize Remotes
where git >nul 2>&1
if %errorlevel% equ 0 (
    if exist ".git" (
        call :log "[INFO] Standardizing remotes [origin=EnVision-Research/DVD]..."
        git remote set-url origin !DVD_URL! 2>nul || git remote add origin !DVD_URL! 2>nul
    )
)

REM Optional Update Pull [Only if we didn't just clone it]
if "!ALREADY_HOME!"=="true" if "!JUST_CLONED!" neq "true" (
    echo.
    set /p do_pull="Detected existing repo. Pull latest code from origin? [Y/N] [Default=N]: "
    if /i "!do_pull!"=="Y" (
        call :log "[INFO] Pulling updates from origin..."
        
        REM Detect branch [main or master]
        git fetch origin main --quiet 2>nul
        set "BR=master"
        if !errorlevel! equ 0 set "BR=main"
        
        git pull origin !BR! >> "%LOGFILE%" 2>&1
        if !errorlevel! neq 0 (
            echo.
            echo [WARNING] Git pull failed. This usually means you have local changes.
            set /p force_pull="Would you like to DISCARD your local changes and force update? [Y/N]: "
            if /i "!force_pull!"=="Y" (
                call :log "[INFO] Forcing update [reset --hard]..."
                git reset --hard origin/!BR! >> "%LOGFILE%" 2>&1
                git pull origin !BR! >> "%LOGFILE%" 2>&1
            ) else (
                call :log "[SKIP] User declined force update. Manual resolution required."
            )
        )
    )
)

call :log "[5/7] Setting up Environment..."
call :log "[INFO] Pinning Python 3.12 and syncing..."
uv python pin 3.12 >> "%LOGFILE%" 2>&1
uv sync >> "%LOGFILE%" 2>&1

REM --- [6/7] WEIGHTS SECTION ---
echo.
echo =========================================================
echo MODEL WEIGHTS DOWNLOAD
echo =========================================================
set /p get_weights="Would you like to download DVD weights now? [Y/N]: "

if /i "!get_weights!"=="Y" (
    call :log "[INFO] Starting weight downloads..."
    if not exist "ckpt" mkdir ckpt

    set /p hf_login="Do you need to log in to Hugging Face? [Y/N]: "
    if /i "!hf_login!"=="Y" (
        echo [PROMPT] Please paste your Hugging Face Access Token below:
        uv run hf auth login
    )

    call :log "[INFO] Downloading DVD v1.0 Models from FayeHongfeiZhang/DVD..."
    uv run hf download FayeHongfeiZhang/DVD --revision main --local-dir ckpt
    
    call :log "[SUCCESS] Weights downloaded to /ckpt."
) else (
    call :log "[SKIP] User skipped weights."
)

call :log "[7/7] Finalizing..."
echo.
echo INSTALLATION SUCCESSFUL
echo Log location: %LOGFILE%
call :log "[FINISH] Session complete."

pause
exit /b

REM --- THE LOGGING FUNCTION ---
:log
echo %~1
echo [%time%] %~1 >> "%LOGFILE%"
exit /b
