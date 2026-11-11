@echo off
REM UrzaLLM embedding: sets up a local Python environment and runs embed.py.
REM Usage: double-click, or run "run.bat" in this folder. Extra arguments go to embed.py.
cd /d "%~dp0"

where py >nul 2>nul && (set PY=py -3) || (set PY=python)

if not exist .venv (
    echo Creating virtual environment...
    %PY% -m venv .venv || goto :error
)
call .venv\Scripts\activate.bat || goto :error

python -c "import torch" 2>nul || (
    echo Installing PyTorch. For an NVIDIA GPU, see README.md first.
    python -m pip install --upgrade pip || goto :error
    python -m pip install torch || goto :error
)
python -m pip install -r requirements.txt || goto :error

python embed.py %*
if errorlevel 1 goto :error
pause
exit /b 0

:error
echo.
echo Something went wrong. Please send the messages above back.
pause
exit /b 1
