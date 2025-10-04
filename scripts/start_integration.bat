@echo off
REM Start Integration Services Script for Windows
REM This script starts all required services for the multilingual integration

echo Starting Multilingual Integration Services...
echo =============================================

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Docker is not running. Please start Docker first.
    exit /b 1
)

REM Check if Docker Compose is available
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: docker-compose is not installed. Please install it first.
    exit /b 1
)

REM Create necessary directories
echo Creating necessary directories...
if not exist "audio_cache" mkdir audio_cache
if not exist "nlp_cache" mkdir nlp_cache
if not exist "kb_data" mkdir kb_data
if not exist "logs" mkdir logs
if not exist "monitoring\prometheus" mkdir monitoring\prometheus
if not exist "monitoring\grafana\dashboards" mkdir monitoring\grafana\dashboards
if not exist "monitoring\grafana\datasources" mkdir monitoring\grafana\datasources
if not exist "nginx\ssl" mkdir nginx\ssl

echo Directories created successfully.

REM Start services with Docker Compose
echo Starting services with Docker Compose...
docker-compose -f docker-compose.integration.yml up -d

REM Wait for services to be ready
echo Waiting for services to be ready...
timeout /t 30 /nobreak >nul

REM Check service health
echo Checking service health...

REM Check multilingual API
echo Checking multilingual-api on port 8000...
curl -f http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ multilingual-api is healthy
) else (
    echo ✗ multilingual-api is not responding
)

REM Check Vaani TTS
echo Checking vaani-tts on port 8001...
curl -f http://localhost:8001/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ vaani-tts is healthy
) else (
    echo ✗ vaani-tts is not responding
)

REM Check Indigenous NLP
echo Checking indigenous-nlp on port 8002...
curl -f http://localhost:8002/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ indigenous-nlp is healthy
) else (
    echo ✗ indigenous-nlp is not responding
)

REM Check Knowledge Base
echo Checking knowledge-base on port 8003...
curl -f http://localhost:8003/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ knowledge-base is healthy
) else (
    echo ✗ knowledge-base is not responding
)

echo.
echo Integration services started successfully!
echo.
echo Service URLs:
echo =============
echo Multilingual API: http://localhost:8000
echo Vaani TTS: http://localhost:8001
echo Indigenous NLP: http://localhost:8002
echo Knowledge Base: http://localhost:8003
echo Redis: localhost:6379
echo PostgreSQL: localhost:5432
echo Nginx Load Balancer: http://localhost
echo Grafana Dashboard: http://localhost:3000
echo Prometheus: http://localhost:9090
echo.
echo API Documentation:
echo ==================
echo Multilingual API Docs: http://localhost:8000/docs
echo.
echo To test the integration:
echo =======================
echo python examples/complete_integration_example.py
echo.
echo To stop services:
echo =================
echo docker-compose -f docker-compose.integration.yml down
echo.
echo To view logs:
echo =============
echo docker-compose -f docker-compose.integration.yml logs -f

pause
