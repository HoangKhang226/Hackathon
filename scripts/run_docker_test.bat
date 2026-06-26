@echo off
cd /d "%~dp0.."
echo ========================================
echo BUILD DOCKER IMAGE
echo ========================================
set HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
docker build --build-arg HF_TOKEN=%HF_TOKEN% -t hackathon_agent:v1 .
if %errorlevel% neq 0 (
    echo [ERROR] Docker build failed!
    pause
    exit /b %errorlevel%
)

echo.
echo ========================================
echo RUN DOCKER CONTAINER
echo ========================================
if not exist "data" mkdir data
if not exist "output" mkdir output

docker run --rm ^
  --gpus all ^
  -v "%cd%\data:/data" ^
  -v "%cd%\output:/output" ^
  hackathon_agent:v1

echo.
echo ========================================
echo TEST FINISHED. CHECK OUTPUT FOLDER.
echo ========================================
pause
