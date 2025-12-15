@echo off
setlocal
chcp 65001 >nul
title control_agent (NIE ZAMYKA SIE SAM)

REM dopasuj do swojej sciezki:
cd /d "E:\BOT ANK\bot\moje_AI\yolov8\FULL BOT"

set "PYTHON=python"
set "CONTROL_AGENT=scripts\control_agent\control_agent.py"
set "CONFIG=scripts\control_agent\train.json"
set "PORT=8765"

echo ============================================================
echo [control_agent] CWD: %CD%
echo [control_agent] Uzywany python: %PYTHON%
echo [control_agent] Skrypt:        %CONTROL_AGENT%
echo [control_agent] Config:        %CONFIG%
echo [control_agent] Port:          %PORT%
echo ============================================================
echo.
echo [control_agent] CMD:
echo   %PYTHON% %CONTROL_AGENT% --config "%CONFIG%" --port %PORT%
echo.

%PYTHON% %CONTROL_AGENT% --config "%CONFIG%" --port %PORT%
set "ERR=%ERRORLEVEL%"

echo.
echo [control_agent] Proces Pythona zakonczony, ERRORLEVEL=%ERR%
echo [control_agent] Okno NIE zamknie sie samo.
echo          Wcisnij dowolny klawisz, aby zamknac to okno.
echo.
pause
endlocal
exit /b 0
