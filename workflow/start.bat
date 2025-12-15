@echo off
REM 1. Wejście do głównego katalogu (jak u Ciebie)
cd /d "E:\BOT ANK\bot\moje_AI"

REM 2. Jeśli potrzebujesz tego scripts/activate (np. do PATH/CUDA), to go odpal
if exist scripts\activate.bat (
    call scripts\activate.bat
) else (
    if exist scripts\activate (
        call scripts\activate
    )
)

REM 3. Przejście do folderu z botem
cd /d "E:\BOT ANK\bot\moje_AI\yolov8\FULL BOT"

REM 4. Odpalenie appa z Twojego konkretnego Pythona z env ai_cuda
"E:\miniconda3\envs\ai_cuda\python.exe" workflow\app.py

echo.
echo [KONIEC] Nacisnij dowolny klawisz...
pause >nul
