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

# ⚡ RECOMMENDED: Lightning-Fast Agent (No RAM issues, instant responses)
# Step 1: Preprocess data ONCE (may take 5-15 minutes)
python preprocess_data.py

# Step 2: Run fast agent (instant responses forever!)
python run_fast_agent.py

# 🐌 ALTERNATIVE: Slower agents (may timeout or consume lots of RAM)
python run_deep_agent.py  # Deep agent (slower, may get stuck)
python run_agent.py       # Points to deep agent

# ⚠️ DEPRECATED: Old vague agents (avoid these)
# python trading_agent.py  # Old enhanced agent (vague responses)
# python agent.py          # Legacy agent (vague responses)
```

## Architecture

Clean, focused architecture with YES/NO validation capabilities:

### ⚡ Fast Architecture (RECOMMENDED)

1. **preprocess_data.py**: One-time data preprocessing:
   - `DataPreprocessor` class: Analyzes all simulation data upfront
   - Saves validation results to JSON files
   - Creates quick lookup tables for instant access
   - Eliminates RAM consumption during agent use

2. **fast_trading_agent.py**: Lightning-fast agent with instant responses:
   - `GetOverallValidation`: Instant YES/NO verdict from cached results
   - `AnalyzeSpecificDate`: Detailed analysis from pre-computed data
   - `GetProblematicDates`: Instant identification of accuracy issues
   - `GetAvailableDates`: Quick list of all analyzed dates
   - No heavy data processing - just smart interpretation

3. **run_fast_agent.py**: Fast agent runner

### 🐌 Traditional Architecture (Slower, may timeout)

4. **deep_trading_agent.py**: Real-time analysis agent:
   - `ValidateSimulationAccuracy`: YES/NO verdict (slow, processes data live)
   - `AnalyzeSpecificDate`: Deep analysis (may timeout on large datasets)
   - Heavy RAM usage and processing time

5. **deep_analysis_engine.py**: Analysis engine:
   - Real-time data processing and matching
   - May consume significant RAM and time

6. **data_manager.py**: Data management with caching
7. **run_agent.py**: Points to deep agent
8. **run_deep_agent.py**: Direct deep agent runner

## Data Structure

- **Real Data**: `data/agent_live_data.csv` - Contains actual trading records
- **Simulation Data**: `data/HL_btcusdt_BTC/*.pickle` - Daily simulation files (June 2025)
- **Metadata**: `data/HL_btcusdt_BTC/metadata.json` - Configuration for data processing

The simulation data uses a master/slave setup (FUTURE/SPOT) for BTC/USDT trading pairs.

## ⚡ Recommended Workflow (Lightning-Fast)

### Step 1: One-Time Preprocessing (5-15 minutes)
```bash
source env/bin/activate
python preprocess_data.py
```
**What this does:**
- Analyzes ALL simulation files upfront
- Calculates validation results for every date
- Saves results to `analysis_results/` directory
- Creates quick lookup tables for instant access

### Step 2: Lightning-Fast Agent (Instant responses)
```bash
python run_fast_agent.py
```
**Benefits:**
- ⚡ Instant responses (no waiting)
- 🧠 No RAM consumption during use
- 🎯 Same accurate YES/NO validation 
- 🔍 Deep investigation capabilities
- 📊 All the same metrics and insights

### Example Fast Agent Session:
```
⚡ Fast Agent: Is my simulation accurate?
✅ OVERALL SIMULATION ACCURACY: YES
• Success Rate: 85.7%
• Total Accurate Matches: 2,341 / 2,678
• Real Trade EC: $1,674,832.45

⚡ Fast Agent: Show problematic dates
⚠️ Found 3 problematic dates:
• 15-06-2025: ❌ NO (45.2% accuracy)
• 18-06-2025: ⚠️ LOW ACCURACY (62.1%)

⚡ Fast Agent: Analyze discrepancies for 15-06-2025  
❌ 15-06-2025 ANALYSIS: NO
• Reason: Poor accuracy (45.2%) - significant timing discrepancies detected
• Diagnostic: High average latency (8.2s) - possible network/execution delays
```

## Key Implementation Details

### ⚡ Fast Architecture Features (RECOMMENDED)
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