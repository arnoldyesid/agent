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

# Run the clean, user-friendly agent (recommended)
python run_agent.py

# Or run the enhanced agent directly
python trading_agent.py

# Or run the legacy agent
python agent.py

# Test data loading
python test_loader.py
```

## Architecture

The codebase has been completely refactored with a clean, modular architecture:

### New Enhanced Architecture (trading_agent.py)

1. **trading_agent.py**: Main enhanced agent with 13+ intelligent tools:
   - `AnalyzeDiscrepancies`: Comprehensive analysis with caching
   - `GetAvailableDates`: Lists available trading dates
   - `AnalyzeSpecificDate`: Targeted analysis for specific dates
   - `ListDataFiles`: File system exploration
   - `PreviewData`: Data preview and inspection
   - `ClearCache`: Cache management
   - `GetDataStatistics`: System statistics and health
   - `AnalyzePositionSizing`: Position sizing patterns and optimization
   - `AnalyzeExecutionQuality`: Trade timing and execution analysis
   - `AnalyzeMarketContext`: Market conditions and trend analysis
   - `AnalyzePerformanceAttribution`: Factor analysis of performance drivers
   - `AnalyzeTradeSequences`: Behavioral patterns and streak analysis
   - `GetComprehensiveAnalysis`: Deep multi-dimensional analysis

2. **data_manager.py**: Advanced data management with intelligent caching:
   - `DataManager` class: Centralized data handling with caching
   - `get_real_trade_chunks()`: Cached real data loading and chunking
   - `get_simulation_data_for_dates()`: Smart simulation data loading
   - Automatic cache invalidation and memory management
   - Comprehensive error handling and logging

3. **analysis_engine.py**: Sophisticated analysis engine:
   - `AnalysisEngine` class: Advanced discrepancy analysis
   - Configurable thresholds and merge tolerances  
   - Statistical analysis and pattern detection
   - AI-powered insights with comprehensive reporting
   - Robust error handling and status reporting

4. **enhanced_analysis.py**: Deep trading analysis capabilities:
   - `EnhancedTradeAnalyzer` class: Advanced behavioral and performance analysis
   - Position sizing optimization and pattern analysis
   - Execution quality assessment and timing optimization
   - Market context analysis and trend correlation
   - Performance attribution and factor analysis
   - Trading sequence analysis and behavioral insights

### Legacy Architecture (for compatibility)

1. **agent.py**: Original implementation (maintained for compatibility)
2. **data_loader.py**: Original data loading (fixed encoding issues)
3. **discrepancy.py**: Original analysis (fixed deprecation warnings)

## Data Structure

- **Real Data**: `data/agent_live_data.csv` - Contains actual trading records
- **Simulation Data**: `data/HL_btcusdt_BTC/*.pickle` - Daily simulation files (June 2025)
- **Metadata**: `data/HL_btcusdt_BTC/metadata.json` - Configuration for data processing

The simulation data uses a master/slave setup (FUTURE/SPOT) for BTC/USDT trading pairs.

## Key Implementation Details

### Performance Optimizations
- **Intelligent Caching**: Data is cached in memory to avoid reloading on every request
- **Lazy Loading**: Simulation data is only loaded when needed for specific dates
- **Chunked Processing**: Real data is processed in daily chunks for memory efficiency
- **Cache Invalidation**: Smart cache management prevents stale data issues

### Analysis Features  
- **Configurable Thresholds**: Default $2 discrepancy threshold (easily adjustable)
- **Temporal Matching**: 2-second tolerance for timestamp-based merging
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