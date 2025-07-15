# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a cryptocurrency trading analysis agent that compares real trading data with simulation data to identify price discrepancies. The agent uses GPT-4o through LangChain to provide intelligent analysis of trading inconsistencies.

## Commands

### Setup and Dependencies
```bash
# Create virtual environment
python -m venv env

# Activate virtual environment
source env/bin/activate  # Linux/macOS
# or
env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
# Create .env file with:
# OPENAI_API_KEY=your_api_key_here
# OPENAI_MODEL=gpt-4o  # Optional, defaults to gpt-4o
```

### Running the Application
```bash
# IMPORTANT: Always activate virtual environment first
source env/bin/activate

# 🎯 RECOMMENDED: Deep Trading Agent with YES/NO validation
python run_deep_agent.py
# OR
python run_agent.py  # Now points to deep agent

# ⚠️ DEPRECATED: Old vague agents (avoid these)
# python trading_agent.py  # Old enhanced agent (vague responses)
# python agent.py          # Legacy agent (vague responses)
```

## Architecture

Clean, focused architecture with YES/NO validation capabilities:

### Core Components

1. **deep_trading_agent.py**: Main agent with precise validation:
   - `ValidateSimulationAccuracy`: YES/NO verdict across multiple dates
   - `AnalyzeSpecificDate`: Deep analysis with YES/NO verdict for specific dates
   - `GetAvailableDates`: Lists available simulation dates
   - `AnalyzeAllDiscrepancies`: Comprehensive multi-date analysis  
   - `ExplainMethodology`: Details analysis techniques

2. **deep_analysis_engine.py**: Advanced analysis engine:
   - `DeepAnalysisEngine` class: Advanced trade matching with time-window tolerance
   - `validate_simulation_accuracy()`: YES/NO validation with 80% threshold
   - `analyze_execution_quality()`: Price improvement and timing analysis
   - `analyze_market_conditions()`: Spread, liquidity, and volatility analysis
   - `diagnose_accuracy_issues()`: Crash detection, latency analysis, volatility correlation
   - `generate_insights()`: AI-powered actionable recommendations
   - Proper handling of different data formats (CSV vs pickle, timestamps vs datetime)

3. **data_manager.py**: Data management with intelligent caching:
   - `DataManager` class: Centralized data handling with caching
   - `get_real_trade_chunks()`: Cached real data loading and chunking
   - `get_simulation_data_for_dates()`: Smart simulation data loading
   - Automatic cache invalidation and memory management
   - Comprehensive error handling and logging

4. **run_agent.py**: Main entry point (now points to deep agent)
5. **run_deep_agent.py**: Direct deep agent runner

## Data Structure

- **Real Data**: `data/agent_live_data.csv` - Contains actual trading records
- **Simulation Data**: `data/HL_btcusdt_BTC/*.pickle` - Daily simulation files (June 2025)
- **Metadata**: `data/HL_btcusdt_BTC/metadata.json` - Configuration for data processing

The simulation data uses a master/slave setup (FUTURE/SPOT) for BTC/USDT trading pairs.

## Key Implementation Details

### NEW Deep Analysis Features (RECOMMENDED)
- **Advanced Trade Matching**: 5-second time window with price/quantity proximity scoring
- **Execution Quality Metrics**: Price improvement, timing latency, and market impact analysis
- **Market Condition Analysis**: Real-time spread, liquidity depth, and volatility assessment
- **Behavioral Insights**: Pattern detection for trading performance optimization
- **Multi-Format Support**: Handles CSV (real trades) and pickle (simulation) data seamlessly
- **Timestamp Normalization**: Converts datetime strings to Unix timestamps for accurate matching
- **Smart Matching Algorithm**: Prevents double-matching with dynamic exclusion
- **Actionable Recommendations**: AI-generated insights with specific improvement suggestions

### Performance Optimizations
- **Intelligent Caching**: Data is cached in memory to avoid reloading on every request
- **Lazy Loading**: Simulation data is only loaded when needed for specific dates
- **Chunked Processing**: Real data is processed in daily chunks for memory efficiency
- **Cache Invalidation**: Smart cache management prevents stale data issues

### Analysis Features  
- **Configurable Thresholds**: Default $2 discrepancy threshold (easily adjustable)
- **Temporal Matching**: 2-second tolerance for timestamp-based merging (legacy), 5-second for deep analysis
- **Statistical Analysis**: Comprehensive statistics including averages, max/min, percentages
- **AI Insights**: GPT-4o powered analysis of discrepancy patterns and causes
- **Multiple Output Formats**: Structured results with different detail levels
- **Smart Tool Selection**: Agent intelligently chooses optimal tools to avoid unnecessary data loading
- **Real-Trade Analysis**: Dedicated tools for PnL, statistics, and trade analysis without simulation data
- **Real-Time Progress**: Friendly progress indicators show exactly which files are loading
- **Anti-Loop Protection**: Built-in safeguards prevent infinite loops and timeout issues

### Error Handling
- **Graceful Degradation**: System continues working even if some data is unavailable
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **Input Validation**: Robust validation of data formats and structures
- **Exception Recovery**: Proper error handling with informative messages

### Data Compatibility
- **Multiple Formats**: Handles CSV, pickle, dict, and list data formats
- **Flexible Schemas**: Adapts to different column names and data structures
- **Encoding Support**: Proper UTF-8 encoding to handle special characters
- **Metadata Preservation**: Maintains simulation metadata for analysis context

## Optimal Usage Patterns

### Memory-Efficient Queries (Use These First)
- **PnL Analysis**: "Calculate my total PnL" → Uses `CalculatePnL` (cached data only)
- **Trade Statistics**: "Show me real trade statistics" → Uses `GetRealTradeStatistics` (cached data only)  
- **Trade Analysis**: "Analyze my real trading performance" → Uses `AnalyzeRealTrades` (cached data only)
- **Data Overview**: "Show me available dates" → Uses `GetAvailableDates` (instant)
- **Technical Details**: "Explain validation process" → Uses `ExplainValidationProcess` (instant)

### Targeted Analysis (Loads Specific Data)
- **Single Date**: "Analyze discrepancies for 10-06-2025" → Uses `AnalyzeSpecificDate` (loads 1 simulation file)
- **Data Preview**: "Preview simulation data for 15-06-2025" → Uses `PreviewData` (loads 1 simulation file)

### Heavy Operations (Use Sparingly)
- **Full Analysis**: "Analyze all discrepancies" → Uses `AnalyzeDiscrepancies` (loads all 19 simulation files)

### Performance Tips
- Start with real-trade analysis for quick insights
- Use specific dates when comparing real vs simulation
- Cache is persistent across sessions - data loads only once
- Use `ClearCache` if you need to force data reload