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

# ⚡ STREAMLINED: Lightning-Fast Agent (Only option - optimized and focused)
# Step 1: Preprocess data ONCE (may take 5-15 minutes)
python improved_preprocess_data.py

# Step 2: Run fast agent (instant responses with deep analysis!)
python run_fast_agent.py
```

## Architecture

Streamlined, focused architecture optimized for performance and deep analysis:

### ⚡ Optimized Fast Architecture (Only Option)

1. **improved_preprocess_data.py**: Enhanced one-time data preprocessing:
   - `ImprovedDataPreprocessor` class: Advanced analysis with granular accuracy reporting
   - Supports YES/NO/PARTIAL verdicts for nuanced assessment
   - Better temporal overlap detection and diagnostics
   - Improved matching criteria with multiple tolerance levels
   - Comprehensive business intelligence insights
   - Saves detailed validation results to JSON files

2. **fast_trading_agent.py**: Lightning-fast agent with deep analysis capabilities:
   - `GetOverallValidation`: Instant comprehensive verdict from cached results
   - `AnalyzeSpecificDate`: Detailed analysis with business intelligence
   - `GetProblematicDates`: Smart identification of accuracy issues with root cause analysis
   - `GetAvailableDates`: Quick list of all analyzed dates with performance metrics
   - Enhanced pattern analysis and trade matching insights
   - Business recommendations and actionable next steps

3. **run_fast_agent.py**: Optimized agent runner

## Data Structure

- **Real Data**: `data/agent_live_data.csv` - Contains actual trading records
- **Simulation Data**: `data/HL_btcusdt_BTC/*.pickle` - Daily simulation files (June 2025)
- **Metadata**: `data/HL_btcusdt_BTC/metadata.json` - Configuration for data processing

The simulation data uses a master/slave setup (FUTURE/SPOT) for BTC/USDT trading pairs.

## ⚡ Streamlined Workflow (Optimized Performance)

### Step 1: Enhanced One-Time Preprocessing (5-15 minutes)
```bash
source env/bin/activate
python improved_preprocess_data.py
```
**What this does:**
- Performs advanced analysis with granular accuracy reporting
- Supports YES/NO/PARTIAL verdicts for nuanced assessment
- Detects temporal overlaps and provides comprehensive diagnostics
- Calculates validation results with multiple tolerance levels
- Generates business intelligence insights and recommendations
- Saves detailed results to `analysis_results/` directory

### Step 2: Enhanced Fast Agent (Instant deep responses)
```bash
python run_fast_agent.py
```
**Enhanced Benefits:**
- ⚡ Instant responses with deep analysis
- 🧠 Zero RAM consumption during use
- 🎯 Granular accuracy assessment (not just binary)
- 🔍 Advanced pattern recognition and diagnostics
- 💼 Business intelligence and actionable recommendations
- 📊 Comprehensive trade matching insights

### Example Enhanced Fast Agent Session:
```
⚡ Fast Agent: Is my simulation accurate?
❌ OVERALL SIMULATION ACCURACY: NO (61.3%)
• Grade: C (Fair) - Major Improvements Required
• Successful Dates: 19/31 (61.3%)
• Average Execution Accuracy: 61.3%
• Key Issue: 12 dates with complete system failures

⚡ Fast Agent: Analyze discrepancies for 22-06-2025
✅ 22-06-2025 DEEP ANALYSIS: YES (100.0%)
• Trade Pattern Analysis: 1,913 perfect matches
• Timing Patterns: Average latency -1.0s (excellent)
• Business Intelligence: EXCELLENT PERFORMANCE
• Recommendation: Consider scaling up on similar conditions
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