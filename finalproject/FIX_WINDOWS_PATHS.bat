@echo off
REM Enable Windows Long Path Support for Python and Pip

echo ======================================
echo Fixing Windows Long Path Issue
echo ======================================
echo.

REM Check if running as Administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: This script requires Administrator privileges!
    echo Please run Command Prompt as Administrator and try again.
    pause
    exit /b 1
)

echo Attempting to enable Long Path support in Windows Registry...
echo.

REM Enable Long Path support via Registry
reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1 /f

if %errorLevel% equ 0 (
    echo ✓ Long Path support ENABLED successfully!
    echo.
    echo NOTE: You may need to RESTART your computer for changes to take effect.
    echo.
) else (
    echo ✗ Failed to enable Long Path support
    echo Please try running this script as Administrator
)

echo.
pause