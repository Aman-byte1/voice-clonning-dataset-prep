@echo off
REM ============================================================
REM  Voice Cloning Dataset Pipeline - Run Script (Windows)
REM ============================================================

REM Set HuggingFace token
set HF_TOKEN=your_token_here

echo.
echo ============================================================
echo  Voice Cloning Dataset Pipeline
echo ============================================================
echo.

REM Step 1: Install dependencies
echo [1/3] Installing dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to install dependencies
    exit /b 1
)

echo.
echo [2/3] Generating audio dataset...
python generate_dataset.py --config config.yaml
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to generate dataset
    exit /b 1
)

echo.
echo [3/3] Pushing dataset to HuggingFace...
python push_to_hub.py --config config.yaml
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to push dataset
    exit /b 1
)

echo.
echo ============================================================
echo  Pipeline completed successfully!
echo ============================================================
