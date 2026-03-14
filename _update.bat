@echo off
setlocal enabledelayedexpansion

echo.
echo === Git Pull ===
git pull
set GIT_EXIT=%ERRORLEVEL%

if %GIT_EXIT% neq 0 (
    echo.
    echo Git pull failed or has conflicts.
    echo.
    set /p STASH_CHOICE="Stash local changes and overwrite? (y/n): "
    if /i "!STASH_CHOICE!"=="y" (
        echo.
        echo === Stashing local changes ===
        git stash
        echo === Pulling again ===
        git pull
    ) else (
        echo Skipping pull. Running uv sync with local changes...
    )
)

echo.
echo === UV Sync ===
uv sync

echo.
echo Done!
pause
