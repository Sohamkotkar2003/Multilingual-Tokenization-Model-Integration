# BHIV System Summary for Vinayak Tiwari

## Overview

This document provides a summary of the BHIV (Brain-Computer Interface for Human Intelligence Visualization) system, focusing on the Core dataset connector, its linkage with the MCP streaming flow, and the integration of the RL endpoint.

## System Components

### 1. Core System Components

#### MCP Bridge (`mcp_bridge.py`)
- Central API that routes tasks to appropriate agents
- Listens on port 8002
- Handles both JSON and file-based task requests
- Integrates with reinforcement learning components

#### Agent Registry (`agents/agent_registry.py`)
- Manages agent configurations and routing
- Supports both HTTP API and Python module agents
- Dynamic agent selection based on input type

#### Simple API (`simple_api.py`)
- Three specialized endpoints:
  - `/ask-vedas`: Spiritual wisdom
  - `/edumentor`: Educational content
  - `/wellness`: Health advice
- Listens on port 8000

#### Web Interface (`integration/web_interface.py`)
- Bootstrap UI for file uploads and processing
- Authentication system (admin/secret, user/secret)
- Dashboard with analytics and NLO statistics
- Listens on port 8003

### 2. Agent Bucket System

#### Text Processing Agents
- **Text Agent**: General text processing using Groq API
- **Stream Transformer Agent**: General purpose processing for multiple input types

#### File Processing Agents
- **Archive Agent**: PDF document processing
- **Image Agent**: Image analysis and description
- **Audio Agent**: Audio transcription and processing

#### Specialized Agents (HTTP API)
- **Vedas Agent**: Spiritual wisdom from Vedic texts
- **EduMentor Agent**: Educational content
- **Wellness Agent**: Wellness advice

### 3. Reward System

#### Reinforcement Learning Components
- **Model Selector**: UCB-based model selection
- **Agent Selector**: Performance-based agent routing
- **Reward Functions**: Quality-based reward calculation
- **Replay Buffer**: Historical performance storage
- **RL Context**: Centralized logging and action tracking

## Core Dataset Connector and MCP Streaming Flow

### Data Flow Architecture
```
Input Sources → MCP Bridge → Agent Selection → Processing → 
Reward Calculation → Storage → NLO Generation
```

### Connection Details

1. **Input Sources**:
   - Web Interface (file uploads)
   - CLI Runner (command-line processing)
   - Direct API calls

2. **MCP Bridge Processing**:
   - Routes tasks based on input type
   - Integrates with Agent Registry for agent selection
   - Processes results through reward functions
   - Logs to MongoDB and agent memory

3. **Agent Processing**:
   - Text/JSON inputs → HTTP API agents
   - File inputs → Python module agents
   - Results → NLO generation via Nipun Adapter

### RL Endpoint Integration

Soham's RL endpoint connects through:
- Model selection and agent routing using UCB algorithms
- Reward calculation based on output quality
- Performance logging for continuous improvement

## System Linkage

### Core System
- MCP Bridge as central routing hub
- Agent Registry for configuration management
- Nipun Adapter for NLO generation

### Bucket System
- Agent Bucket for JSON structure
- Individual specialized agents for processing

### Reward System
- Reward Functions for quality assessment
- Replay Buffer for historical data
- RL Context for centralized logging
- Model/Agent Selectors for performance-based routing

## Current System Status

- **MCP Bridge**: Running on port 8002 (degraded due to agent registry issue)
- **Simple API**: Running on port 8000 (healthy)
- **Web Interface**: Running on port 8003 (healthy)

## Access Information

### Web Interface
- URL: http://localhost:8003
- Credentials: 
  - Admin: admin/secret
  - User: user/secret

### API Endpoints
- MCP Bridge: http://localhost:8002
- Simple API: http://localhost:8000
- Documentation: Available at `/docs` endpoint for each service

### File Locations
- Main Repository: `E:\BHIV-Fouth-Installment-main\BHIV-Fouth-Installment-main`
- Configuration: `config/` directory
- Logs: `logs/` directory
- Vector Stores: `vector_stores/` directory (if exists)

## Next Steps

1. Resolve the agent registry issue in the MCP Bridge
2. Configure API keys for the Simple API models
3. Ensure MongoDB connectivity for persistent storage
4. Verify RL endpoint integration with Soham's system
5. Test end-to-end data flow with sample inputs

This summary provides the essential information needed to understand and work with the BHIV system components.