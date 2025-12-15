@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul
title Dual ChatGPT Orchestrator (VM, classic input)

REM === KONFIG (AUTOMATYCZNE SCIEZKI WZGL. TEGO .BAT) ===
REM katalog z tym plikiem .bat
set "SCRIPT_DIR=%~dp0"
REM WORK_DIR = katalog nadrzedny wzgledem chat_gpt (czyli FULL BOT)
pushd "%SCRIPT_DIR%.." >nul
set "WORK_DIR=%CD%"
popd >nul
REM BASE_DIR = katalog nadrzedny wzgledem FULL BOT (czyli moje_AI)
pushd "%WORK_DIR%.." >nul
set "BASE_DIR=%CD%"
popd >nul
set "REC_OUTPUT_DIR=%WORK_DIR%\dom_live"

REM skrypty (względem WORK_DIR)
set "CONTROL_AGENT=%WORK_DIR%\scripts\control_agent\control_agent.py"
set "KBD_AGENT=%WORK_DIR%\scripts\control_agent\kbd_scroll_agent.py"
set "ORCHESTRATOR=%WORK_DIR%\scripts\dual_chatgpt_consensus\dual_chatgpt_consensus.py"
set "AI_REC=%WORK_DIR%\dom_renderer\ai_recorder_live.py"
set "AI_REC_BAT=%WORK_DIR%\chat_gpt\run_ai_recorder.bat"


REM recorder potrzebuje dom_renderer w PYTHONPATH
set "PYTHONPATH=%WORK_DIR%;%BASE_DIR%"

REM domyślny profil przeglądarki (podaj własne wartości, jeśli chcesz)
set "RECORDER_USER_DATA_DIR=C:\Users\user\AppData\Local\Google\Chrome\User Data"
set "RECORDER_PROFILE_DIR=Profile 5"
set "RECORDER_BROWSER_EXE=%ProgramFiles%\Google\Chrome\Application\chrome.exe"
if not exist "%RECORDER_BROWSER_EXE%" set "RECORDER_BROWSER_EXE=%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"

REM porty / parametry
set "CONTROL_AGENT_PORT=8765"
set "KBD_AGENT_PORT=8766"
set "MAX_ROUNDS=8"
set "REPLY_WAIT=14"

REM >>> TUTAJ USTALAMY PYTHONA W VM <<<
set "PYTHON=python"

echo [VARS]
echo   BASE_DIR       = %BASE_DIR%
echo   WORK_DIR       = %WORK_DIR%
echo   REC_OUTPUT_DIR = %REC_OUTPUT_DIR%
echo   CONTROL_AGENT  = %CONTROL_AGENT%
echo   KBD_AGENT      = %KBD_AGENT%
echo   ORCHESTRATOR   = %ORCHESTRATOR%
echo   AI_REC         = %AI_REC%
echo   PYTHONPATH     = %PYTHONPATH%
echo   chrome_exe     = %RECORDER_BROWSER_EXE%
echo   profile_dir    = %RECORDER_PROFILE_DIR%
echo   user_data_dir  = %RECORDER_USER_DATA_DIR%
echo   PYTHON         = %PYTHON%
echo.

where %PYTHON%
if errorlevel 1 (
    echo [FATAL] Nie znalazlem "%PYTHON%" w PATH wewnatrz VM. Ustaw pelna sciezke do python.exe
    goto HOLD
) 

echo.
pause

echo [STEP] Po pauzie wchodze w czesc startujaca agenty i recorder...
echo.

REM === przejdz do katalogu roboczego ===
cd /d "%WORK_DIR%"  || (
    echo [FATAL] Nie moge cd do "%WORK_DIR%"
    goto HOLD
)

REM utworz folder na snapshoty
if not exist "%REC_OUTPUT_DIR%" mkdir "%REC_OUTPUT_DIR%"

REM usun ewentualny STOP
if exist "%REC_OUTPUT_DIR%\STOP" del /q "%REC_OUTPUT_DIR%\STOP"

REM usun STARE current_*.* 
REM (pozostawiam current_*.* w katalogu – nie czyścimy ich przy starcie)

REM === kontrola plikow .py ===
if not exist "%CONTROL_AGENT%" (
    echo [FATAL] Brak pliku: %CONTROL_AGENT%
    goto HOLD
)

if not exist "%ORCHESTRATOR%" (
    echo [FATAL] Brak pliku: %ORCHESTRATOR%
    goto HOLD
)

if not exist "%AI_REC%" (
    echo [FATAL] Brak pliku ai_recorder_live.py pod: %AI_REC%
    goto HOLD
)
if not exist "%AI_REC_BAT%" (
    echo [FATAL] Brak pliku run_ai_recorder.bat pod: %AI_REC_BAT%
    goto HOLD
)

REM === start control_agent (MINIMIZED, osobne okno, cmd /K – NIE zamknie sie samo) ===
echo [RUN] control_agent...
echo [CMD] "%PYTHON%" "%CONTROL_AGENT%" --config scripts\control_agent\train.json --port %CONTROL_AGENT_PORT%
start "control_agent" /MIN /D "%WORK_DIR%" cmd /K "%PYTHON%" "%CONTROL_AGENT%" --config scripts\control_agent\train.json --port %CONTROL_AGENT_PORT%

REM === start kbd_scroll_agent (MINIMIZED, osobne okno) ===
if exist "%KBD_AGENT%" (
    echo [RUN] kbd_scroll_agent...
    echo [CMD] "%PYTHON%" "%KBD_AGENT%" --port %KBD_AGENT_PORT%
    start "kbd_scroll_agent" /MIN /D "%WORK_DIR%" cmd /K "%PYTHON%" "%KBD_AGENT%" --port %KBD_AGENT_PORT%
) else (
    echo [WARN] Brak pliku: %KBD_AGENT%  (pomijam agenta klawiatury)
)

REM === start AI Recorder Live (NORMALNE okno, debug) ===
echo [RUN] AI Recorder Live (DEBUG, widoczne okno)...
echo [INFO] W nowo otwartym oknie "ai_recorder_live" zobaczysz ewentualne bledy (Playwright, CDP, brak modulu itp).

echo [CMD] start "ai_recorder_live" /D "%WORK_DIR%" cmd /K ""%AI_REC_BAT%""
start "ai_recorder_live" /D "%WORK_DIR%" cmd /K ""%AI_REC_BAT%""

echo.
echo [INFO] Poczekaj az otworzy sie przegladarka z okna recorder'a.
echo [INFO] Zaloguj sie i otworz 2 zakladki ChatGPT w TYM oknie przegladarki.
echo [INFO] Potem wroc do TEJ konsoli – skrypt bedzie sprawdzal pliki current_*.json.
echo.

REM === PROSTE czekanie na pliki recorder'a ===
set WAIT_SEC=120

:wait_loop
if exist "%REC_OUTPUT_DIR%\current_page.json" if exist "%REC_OUTPUT_DIR%\current_clickables.json" goto files_ready

set /a WAIT_SEC-=1
if %WAIT_SEC% LEQ 0 goto wait_timeout

timeout /t 1 /nobreak >nul
goto wait_loop

:wait_timeout
echo [ERROR] Po 120 sekundach dalej brak plikow current_page.json / current_clickables.json w:
echo         %REC_OUTPUT_DIR%
echo [TIP] Sprawdz okno "ai_recorder_live" (czy nie ma bledu Playwright/CDP).
echo [TIP] Sprawdz tez log: %REC_OUTPUT_DIR%\rec.log
echo.
dir "%REC_OUTPUT_DIR%"
goto HOLD

:files_ready
echo [OK] Pliki recorder'a wykryte:
dir "%REC_OUTPUT_DIR%\current_*.*"
echo.

REM === Orchestrator w OSOBNYM oknie (tu wpiszesz pierwszy prompt) ===
echo [RUN] Orchestrator - osobne okno (ta konsola moze byc zamknieta bez ubijania innych okien).
echo.
echo [CMD] start "orchestrator" /D "%WORK_DIR%" cmd /K ""%PYTHON%" "%ORCHESTRATOR%" --rec-dir "%REC_OUTPUT_DIR%" --port-mouse %CONTROL_AGENT_PORT% --port-kbd %KBD_AGENT_PORT% --max-rounds %MAX_ROUNDS% --reply-wait %REPLY_WAIT%""
echo.
start "orchestrator" /D "%WORK_DIR%" cmd /K ""%PYTHON%" "%ORCHESTRATOR%" --rec-dir "%REC_OUTPUT_DIR%" --port-mouse %CONTROL_AGENT_PORT% --port-kbd %KBD_AGENT_PORT% --max-rounds %MAX_ROUNDS% --reply-wait %REPLY_WAIT%""

echo.
echo [INFO] Orchestrator wystartowal w osobnym oknie.
echo [INFO] Ta konsola zostanie na HOLD; po wcisnieciu klawisza ubije wszystkie okna z tego runa.
echo.
pause
goto CLEANUP

:HOLD
echo.
echo [HOLD] Skrypt .bat NIE wyjdzie sam. To okno tez zostanie otwarte.
echo       Wcisnij klawisz albo CTRL+C, zeby zamknac.
pause

:CLEANUP
echo [CLEANUP] Ubicie okien: control_agent, kbd_scroll_agent, ai_recorder_live, orchestrator
for %%T in ("control_agent" "kbd_scroll_agent" "ai_recorder_live" "orchestrator") do (
    taskkill /FI "WINDOWTITLE eq %%~T" /T /F >nul 2>&1
)
echo [CLEANUP] Gotowe. Zamykam.

:END
endlocal
exit /b 0
