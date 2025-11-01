# BHIV Core Dataset Connector and MCP Streaming Flow

## Overview

This document explains how the BHIV Core dataset connector links with the MCP (Multi-Modal Connector Pipeline) streaming flow. It also details how Soham's RL endpoint connects into this stream for continuous learning and improvement.

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │   CLI Runner    │    │   Simple API    │
│   (Port 8003)   │    │  (Enhanced)     │    │   (Port 8000)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   MCP Bridge    │
                    │   (Port 8002)   │
                    └─────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Agent Registry  │    │ Nipun Adapter   │    │   MongoDB       │
│ (Dynamic Config)│    │ (NLO Generator) │    │ (NLO Storage)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
    ┌────────────────────────────┼────────────────────────────┐
    │                            │                            │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Text Agent    │    │  Archive Agent  │    │  Image Agent    │    │  Audio Agent    │
│   (Enhanced)    │    │   (Enhanced)    │    │   (BLIP Model)  │    │ (Wav2Vec2 Model)│
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │                       │
         └───────────────────────┼───────────────────────┼───────────────────────┘
                                 │                       │
                    ┌─────────────────┐         ┌─────────────────┐
                    │ Reinforcement   │         │ RL Retraining   │
                    │ Learning System │         │    System       │
                    │ (UCB Enhanced)  │         │  (Automated)    │
                    └─────────────────┘         └─────────────────┘
```

## Core Dataset Connector

The Core dataset connector is implemented through the MCP Bridge, which serves as the central routing mechanism for all data processing requests. It connects different input sources with appropriate processing agents based on the data type.

### Data Entry Points

1. **Web Interface** (`integration/web_interface.py`)
   - File upload interface with authentication
   - Real-time processing status updates
   - Dashboard with analytics and NLO statistics

2. **CLI Runner** (`cli_runner.py`)
   - Command-line interface for batch processing
   - Support for multiple file formats
   - Multiple output formats (JSON, CSV, text)

3. **Simple API** (`simple_api.py`)
   - Three specialized endpoints:
     - `/ask-vedas` - Spiritual wisdom
     - `/edumentor` - Educational content
     - `/wellness` - Health advice

### MCP Bridge Functionality

The MCP Bridge (`mcp_bridge.py`) is the core component that orchestrates the data flow:

1. **Task Routing**:
   - Receives tasks via `/handle_task` or `/handle_task_with_file` endpoints
   - Uses Agent Registry to find appropriate agent based on input type
   - Routes to specific agents (Python modules or HTTP APIs)

2. **Agent Processing**:
   - Text/JSON inputs go to HTTP API agents (vedas_agent, edumentor_agent, wellness_agent)
   - File inputs go to Python module agents (archive_agent, image_agent, audio_agent)
   - Results are processed through NLO (Named Learning Object) generation

3. **Data Flow**:
   ```
   Input Data → MCP Bridge → Agent Selection → Processing → 
   Reward Calculation → Replay Buffer → MongoDB Storage → NLO Generation
   ```

## RL Endpoint Integration

Soham's RL endpoint connects into the stream through the reinforcement learning layer:

### Components

1. **Model Selector** (`reinforcement/model_selector.py`)
   - UCB-based model selection with dynamic exploration rates
   - Task-specific weights for different models
   - Performance tracking and history management

2. **Agent Selector** (`reinforcement/agent_selector.py`)
   - Intelligent agent routing based on historical performance
   - Task complexity-based exploration rates
   - Performance tracking and history management

3. **Reward Functions** (`reinforcement/reward_functions.py`)
   - Sophisticated reward calculation based on output quality
   - Response time and success metrics
   - Integration with RL context for logging

4. **Replay Buffer** (`reinforcement/replay_buffer.py`)
   - Stores past runs for RL training
   - Persistent storage in `logs/learning_log.json`
   - Integration with MongoDB for analytics

### RL Data Flow

```
Task Request → RL Context Logger → Model/Agent Selection → 
Processing → Reward Calculation → Replay Buffer → 
Learning Dashboard Analytics
```

### Connection Points

1. **Model Selection**: The RL layer intercepts model selection decisions
2. **Agent Routing**: RL-based agent routing for optimal performance
3. **Reward Tracking**: Automatic reward calculation from output quality
4. **Performance Logging**: Comprehensive logging for continuous improvement

## Core, Bucket, and Reward Systems Linkage

### Core System
- **MCP Bridge**: Central routing and processing hub
- **Agent Registry**: Configuration management for all agents
- **Nipun Adapter**: NLO generation and MongoDB integration

### Bucket System
- **Agent Bucket** (`agent_bucket.py`): JSON structure for agents
- **Individual Agents**: Specialized processing modules
  - Text Agent: General text processing
  - Archive Agent: PDF document processing
  - Image Agent: Image analysis and description
  - Audio Agent: Audio transcription and processing

### Reward System
- **Reward Functions**: Quality-based reward calculation
- **Replay Buffer**: Historical performance storage
- **RL Context**: Centralized logging and action tracking
- **Model/Agent Selectors**: Performance-based selection algorithms

## System Status

As of the latest check, all components are running:

- **MCP Bridge**: Running on port 8002 (degraded due to agent registry issue)
- **Simple API**: Running on port 8000 (healthy)
- **Web Interface**: Running on port 8003 (healthy)

## Next Steps

1. Resolve the agent registry issue in the MCP Bridge
2. Configure API keys for the Simple API models
3. Ensure MongoDB connectivity for persistent storage
4. Verify RL endpoint integration with Soham's system
5. Test end-to-end data flow with sample inputs

This documentation ensures that Vinayak Tiwari has clear understanding of the linkage between the Core, Bucket, and Reward systems in the BHIV Task Bank.