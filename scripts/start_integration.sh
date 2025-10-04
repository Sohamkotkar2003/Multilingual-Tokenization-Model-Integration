#!/bin/bash

# Start Integration Services Script
# This script starts all required services for the multilingual integration

set -e

echo "Starting Multilingual Integration Services..."
echo "============================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "Error: docker-compose is not installed. Please install it first."
    exit 1
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p audio_cache
mkdir -p nlp_cache
mkdir -p kb_data
mkdir -p logs
mkdir -p monitoring/prometheus
mkdir -p monitoring/grafana/dashboards
mkdir -p monitoring/grafana/datasources
mkdir -p nginx/ssl

# Set permissions
chmod 755 audio_cache
chmod 755 nlp_cache
chmod 755 kb_data
chmod 755 logs

echo "Directories created successfully."

# Start services with Docker Compose
echo "Starting services with Docker Compose..."
docker-compose -f docker-compose.integration.yml up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 30

# Check service health
echo "Checking service health..."

services=(
    "multilingual-api:8000"
    "vaani-tts:8001"
    "indigenous-nlp:8002"
    "knowledge-base:8003"
    "redis:6379"
    "postgres:5432"
)

for service in "${services[@]}"; do
    IFS=':' read -r name port <<< "$service"
    echo "Checking $name on port $port..."
    
    if curl -f "http://localhost:$port/health" > /dev/null 2>&1; then
        echo "✓ $name is healthy"
    else
        echo "✗ $name is not responding"
    fi
done

echo ""
echo "Integration services started successfully!"
echo ""
echo "Service URLs:"
echo "============="
echo "Multilingual API: http://localhost:8000"
echo "Vaani TTS: http://localhost:8001"
echo "Indigenous NLP: http://localhost:8002"
echo "Knowledge Base: http://localhost:8003"
echo "Redis: localhost:6379"
echo "PostgreSQL: localhost:5432"
echo "Nginx Load Balancer: http://localhost"
echo "Grafana Dashboard: http://localhost:3000"
echo "Prometheus: http://localhost:9090"
echo ""
echo "API Documentation:"
echo "=================="
echo "Multilingual API Docs: http://localhost:8000/docs"
echo ""
echo "To test the integration:"
echo "======================="
echo "python examples/complete_integration_example.py"
echo ""
echo "To stop services:"
echo "================="
echo "docker-compose -f docker-compose.integration.yml down"
echo ""
echo "To view logs:"
echo "============="
echo "docker-compose -f docker-compose.integration.yml logs -f"
