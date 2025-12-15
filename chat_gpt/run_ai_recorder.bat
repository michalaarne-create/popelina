@echo off
setlocal
chcp 65001 >nul
title ai_recorder_live (NIE ZAMYKA SIE SAM)

REM autodetekcja sciezek wzgledem tego pliku .bat
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%.." >nul
set "WORK_DIR=%CD%"
popd >nul

set "PYTHON=python"
set "AI_REC=%WORK_DIR%\dom_renderer\ai_recorder_live.py"
set "REC_OUTPUT_DIR=%WORK_DIR%\dom_live"
set "START_URL=https://chatgpt.com"
set "EXTRA_URL=%START_URL%"
REM domyslna konfiguracja profilu (uzyj istniejacego profilu uzytkownika) - mozna nadpisac z zewnatrz
if not defined RECORDER_USER_DATA_DIR set "RECORDER_USER_DATA_DIR=C:\Users\user\AppData\Local\Google\Chrome\User Data"
if not defined RECORDER_PROFILE_DIR set "RECORDER_PROFILE_DIR=Profile 5"
if not defined RECORDER_BROWSER_EXE set "RECORDER_BROWSER_EXE=%ProgramFiles%\Google\Chrome\Application\chrome.exe"
if not exist "%RECORDER_BROWSER_EXE%" set "RECORDER_BROWSER_EXE=%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"
REM jesli wskazany profil nie istnieje, spadamy do prywatnego profilu w dom_live
if not exist "%RECORDER_USER_DATA_DIR%" (
    echo [WARN] Podany RECORDER_USER_DATA_DIR nie istnieje, uzywam lokalnego profilu w dom_live.
    set "RECORDER_USER_DATA_DIR=%REC_OUTPUT_DIR%\\_chrome_profile"
    set "RECORDER_PROFILE_DIR=Default"
)

echo ============================================================
echo [ai_recorder] CWD: %WORK_DIR%
echo [ai_recorder] Uzywany python: %PYTHON%
echo [ai_recorder] Skrypt:        %AI_REC%
echo [ai_recorder] Output dir:    %REC_OUTPUT_DIR%
echo [ai_recorder] chrome_exe:    %RECORDER_BROWSER_EXE%
echo [ai_recorder] profile_dir:   %RECORDER_PROFILE_DIR%
echo [ai_recorder] user_data_dir: %RECORDER_USER_DATA_DIR%
echo ============================================================
echo.
echo [ai_recorder] CMD:
echo   "%PYTHON%" "%AI_REC%" --output-dir "%REC_OUTPUT_DIR%" --chrome-exe "%RECORDER_BROWSER_EXE%" --user-data-dir "%RECORDER_USER_DATA_DIR%" --profile-directory "%RECORDER_PROFILE_DIR%" --url "%START_URL%" --extra-url "%EXTRA_URL%" --connect-existing --fps 2 --dom-only --log-file "%REC_OUTPUT_DIR%\rec.log" --verbose
echo.

"%PYTHON%" "%AI_REC%" --output-dir "%REC_OUTPUT_DIR%" --chrome-exe "%RECORDER_BROWSER_EXE%" --user-data-dir "%RECORDER_USER_DATA_DIR%" --profile-directory "%RECORDER_PROFILE_DIR%" --url "%START_URL%" --extra-url "%EXTRA_URL%" --connect-existing --fps 2 --dom-only --log-file "%REC_OUTPUT_DIR%\rec.log" --verbose
set "ERR=%ERRORLEVEL%"

echo.
echo [ai_recorder] Proces Pythona zakonczony, ERRORLEVEL=%ERR%
echo [ai_recorder] Okno NIE zamknie sie samo.
echo          Wcisnij dowolny klawisz, aby zamknac to okno.
echo.
pause
endlocal
exit /b 0
