@echo off
REM === 1. Wejdź do folderu z env ===
cd /d "E:\BOT ANK\bot\moje_AI"

REM === 2. Aktywuj środowisko (Twój skrypt) ===
call scripts\activate.bat

REM === 3. Wejdź do folderu z projektem ===
cd /d "E:\BOT ANK\bot\moje_AI\yolov8\FULL BOT"

REM === 4. Aktywuj środowisko Conda ===
call "C:\Program Files\miniconda3\condabin\conda.bat" activate ai_cuda

REM === 5. Uruchom aplikację ===
python workflow\app.py

REM === 6. Zostaw otwarte okno ===
pause
