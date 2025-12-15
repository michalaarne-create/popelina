@echo off
setlocal
chcp 65001 >nul
title kbd_scroll_agent (NIE ZAMYKA SIE SAM)

cd /d "E:\BOT ANK\bot\moje_AI\yolov8\FULL BOT"

set "PYTHON=python"
set "KBD_AGENT=scripts\control_agent\kbd_scroll_agent.py"
set "PORT=8766"

echo ============================================================
echo [kbd_agent] CWD: %CD%
echo [kbd_agent] Uzywany python: %PYTHON%
echo [kbd_agent] Skrypt:        %KBD_AGENT%
echo [kbd_agent] Port:          %PORT%
echo ============================================================
echo.
echo [kbd_agent] CMD:
echo   %PYTHON% %KBD_AGENT% --port %PORT%
echo.

%PYTHON% %KBD_AGENT% --port %PORT%
set "ERR=%ERRORLEVEL%"

echo.
echo [kbd_agent] Proces Pythona zakonczony, ERRORLEVEL=%ERR%
echo [kbd_agent] Okno NIE zamknie sie samo.
echo          Wcisnij dowolny klawisz, aby zamknac to okno.
echo.
pause
endlocal
exit /b 0
