# 🚀 Enhanced Trading Analysis Agent

A sophisticated cryptocurrency trading analysis agent that compares real trading data with simulation data to identify price discrepancies and provide deep insights into trading performance. Powered by AI for intelligent analysis of trading patterns, behavior, and optimization opportunities.

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Analysis Capabilities](#-analysis-capabilities)
- [Data Structure](#-data-structure)
- [Architecture](#-architecture)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

## ✨ Features

### 🎯 **Core Analysis**
- **Real vs Simulation Comparison**: Compare actual trading results with simulated outcomes
- **Discrepancy Detection**: Identify significant price differences with configurable thresholds
- **PnL Analysis**: Comprehensive profit/loss calculations and breakdowns
- **Performance Metrics**: Advanced risk metrics, Sharpe ratios, and drawdown analysis

### 🔍 **Deep Analysis Capabilities**
- **Position Sizing Analysis**: Optimize position sizes and analyze sizing patterns
- **Execution Quality Assessment**: Trade timing, clustering, and frequency analysis
- **Market Context Analysis**: Performance correlation with market conditions
- **Performance Attribution**: Factor analysis of what drives trading performance
- **Trade Sequence Analysis**: Behavioral patterns, streaks, and consistency metrics
- **Comprehensive Analysis**: Multi-dimensional deep trading insights

### 🚀 **Performance Features**
- **Intelligent Caching**: Fast data loading with automatic cache management
- **Chunked Processing**: Memory-efficient handling of large datasets
- **Progress Tracking**: Real-time progress indicators for long operations
- **Error Recovery**: Robust error handling with graceful degradation

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone <repository-url>
cd Agent

# 2. Create virtual environment
python -m venv env
source env/bin/activate  # Linux/macOS
# or
env\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your OpenAI API key

# 5. Run the agent
python run_agent.py
```

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- OpenAI API key
- Git (for cloning)

### Detailed Installation Steps

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd Agent
   ```

2. **Create Virtual Environment**
   ```bash
   # Create virtual environment
   python -m venv env
   
   # Activate virtual environment
   source env/bin/activate  # Linux/macOS
   env\Scripts\activate     # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   python test_enhanced.py
   ```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```bash
# Required: OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Model Configuration (default: gpt-4o)
OPENAI_MODEL=gpt-4o

# Optional: Analysis Configuration
DISCREPANCY_THRESHOLD=2.0
MERGE_TOLERANCE_SECONDS=2
```

### Configuration Options

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | *Required* | Your API key |
| `OPENAI_MODEL` | OpenAI model to use | `gpt-4o` | `gpt-4o`, `gpt-4`, `gpt-3.5-turbo` |
| `DISCREPANCY_THRESHOLD` | Price difference threshold ($) | `2.0` | Any positive number |
| `MERGE_TOLERANCE_SECONDS` | Time matching tolerance | `2` | Any positive integer |

## 🎮 Usage

### 1. Interactive Agent Mode (Recommended)

```bash
# Activate environment
source env/bin/activate

# Run the enhanced agent
python run_agent.py
```

**Example Queries:**
```
"Calculate my total PnL and show me the breakdown"
"Give me a comprehensive analysis of my trading performance"
"Analyze my position sizing patterns"
"Show me execution quality metrics"
"What factors contribute most to my performance?"
"Compare real vs simulation data for June 10th"
"Analyze winning and losing streaks"
```

### 2. Simple Interface Mode

If the AI agent fails to start, the system automatically falls back to a simple command interface:

```bash
Commands available:
- pnl: Calculate profit/loss
- stats: Show trade statistics
- dates: List available dates
- validation: Explain validation process
- exit: Quit the application
```

### 3. Direct Component Testing

```bash
# Test data loading
python test_loader.py

# Test enhanced analysis
python test_enhanced.py

# Test complete system
python test_final.py

# Legacy agent (compatibility)
python agent.py
```

## 🔍 Analysis Capabilities

### 📊 **Basic Analysis**

| Tool | Description | Use Case |
|------|-------------|----------|
| **PnL Calculation** | Total profit/loss with fees | "Calculate my PnL" |
| **Trade Statistics** | Volume, counts, averages | "Show me trade stats" |
| **Available Dates** | List of trading dates | "What dates are available?" |
| **Real Trade Analysis** | Performance overview | "Analyze my real trades" |

### 🎯 **Advanced Analysis**

| Tool | Description | Use Case |
|------|-------------|----------|
| **Position Sizing** | Size patterns and optimization | "Analyze my position sizes" |
| **Execution Quality** | Timing and frequency analysis | "How's my execution quality?" |
| **Market Context** | Performance vs market conditions | "Analyze market context" |
| **Performance Attribution** | Factor analysis of performance | "What drives my performance?" |
| **Trade Sequences** | Behavioral patterns and streaks | "Show me my trading patterns" |
| **Risk Metrics** | Drawdown, Sharpe ratio, volatility | "Calculate risk metrics" |

### 🔄 **Comparison Analysis**

| Tool | Description | Use Case |
|------|-------------|----------|
| **Discrepancy Analysis** | Real vs simulation comparison | "Compare all real vs sim data" |
| **Specific Date Analysis** | Targeted date comparison | "Analyze June 15th discrepancies" |
| **Time Range Comparison** | Performance across periods | "Compare weekly performance" |

### 🔍 **Comprehensive Analysis**

Get everything in one report:
```
"Give me a comprehensive analysis of my trading"
```

This provides:
- Position sizing analysis
- Execution quality assessment
- Market context correlation
- Performance attribution
- Trade sequence patterns
- All in one detailed report

## 📁 Data Structure

### Real Trading Data
- **File**: `data/agent_live_data.csv`
- **Contains**: Actual trading records with timestamps, prices, quantities, PnL
- **Format**: CSV with standardized columns

### Simulation Data
- **Directory**: `data/HL_btcusdt_BTC/`
- **Files**: Daily pickle files (e.g., `parrot_HL_btcusdt_BTC_10-06-2025.pickle`)
- **Contains**: Simulated trading data for comparison
- **Coverage**: June 2025 trading days

### Required Columns

| Column | Description | Type |
|--------|-------------|------|
| `timestamp` | Trade execution time | DateTime |
| `price` | Trade price | Float |
| `quantity` | Trade quantity | Float |
| `side` | Buy (B) or Sell (A) | String |
| `realized_pnl` | Profit/loss from trade | Float |
| `total_fee` | Trading fees | Float |

## 🏗️ Architecture

### Core Components

```
Agent/
├── trading_agent.py      # Main enhanced agent (13+ tools)
├── enhanced_analysis.py  # Deep analysis capabilities
├── data_manager.py       # Intelligent data management + caching
├── analysis_engine.py    # Discrepancy analysis engine
├── run_agent.py         # User-friendly entry point
└── legacy files         # Original implementation (compatibility)
```

### Key Features

- **Modular Design**: Clean separation of concerns
- **Intelligent Caching**: Avoid reloading data unnecessarily
- **Memory Efficiency**: Chunked processing for large datasets
- **Error Recovery**: Graceful handling of failures
- **Progress Tracking**: Real-time feedback on operations
- **AI Integration**: GPT-4o powered insights and analysis

## 🐛 Troubleshooting

### Common Issues

**1. Agent Won't Start**
```bash
# Check API key
echo $OPENAI_API_KEY

# Verify environment
source env/bin/activate
python -c "import openai; print('OK')"

# Use fallback mode
# The system automatically falls back to simple interface
```

**2. Data Loading Errors**
```bash
# Test data loading
python test_loader.py

# Check data files
ls data/
ls data/HL_btcusdt_BTC/
```

**3. Memory Issues**
```bash
# Clear cache
python -c "
from data_manager import clear_data_cache
clear_data_cache()
print('Cache cleared')
"
```

**4. Performance Issues**
- Use specific date analysis instead of full analysis
- Ask about real trades only (faster)
- Clear cache if data seems stale

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Support

1. Check the [Troubleshooting](#-troubleshooting) section
2. Run test scripts to identify issues
3. Check log files for error details
4. Verify environment configuration

## 🔧 Development

### Running Tests

```bash
# Test enhanced analysis
python test_enhanced.py

# Test agent functionality
python test_final.py

# Test data loading
python test_loader.py
```

### Adding New Analysis

1. Add method to `EnhancedTradeAnalyzer` in `enhanced_analysis.py`
2. Add tool to `TradingAgent._create_tools()` in `trading_agent.py`
3. Add wrapper method to `TradingAgent`
4. Test with sample data

### Performance Optimization

- Use caching for repeated data access
- Process data in chunks for memory efficiency
- Implement lazy loading for large datasets
- Add progress indicators for long operations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

### Code Style

- Follow existing code patterns
- Add comprehensive docstrings
- Include error handling
- Update tests and documentation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for GPT-4o API
- LangChain for agent framework
- Pandas for data processing
- The trading community for inspiration

---

## 🎯 Quick Reference

### Most Useful Commands

```python
# Fast queries (cached data only)
"Calculate my PnL"
"Show me trade statistics"
"Analyze my real trading performance"

# Targeted analysis (loads specific data)
"Analyze discrepancies for June 10th"
"Preview simulation data for June 15th"

# Deep analysis (comprehensive)
"Give me a comprehensive analysis"
"Analyze my position sizing patterns"
"Show me execution quality metrics"

# Heavy operations (use sparingly)
"Analyze all discrepancies across all dates"
```

### Performance Tips

- ✅ Start with real-trade analysis for quick insights
- ✅ Use specific dates when comparing real vs simulation
- ✅ Cache persists across sessions - data loads only once
- ✅ Use `ClearCache` if you need to force data reload
- ⚠️ "Analyze all discrepancies" loads all 19 simulation files

---

**Happy Trading Analysis! 🚀📊**