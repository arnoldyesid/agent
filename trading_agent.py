#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading Analysis Agent

Main agent application with improved architecture, caching, and error handling.
"""

import os
import pandas as pd
import signal
import threading
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
import logging

from data_manager import DataManager, get_available_dates, clear_data_cache
from analysis_engine import AnalysisEngine
from enhanced_analysis import EnhancedTradeAnalyzer

# Configure logging for cleaner user experience
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors to user
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # Default to gpt-4o

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env file")


class TradingAgent:
    """Enhanced trading analysis agent with caching and improved tools."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0,
            api_key=OPENAI_API_KEY
        )
        
        self.data_manager = DataManager()
        self.analysis_engine = AnalysisEngine(self.llm)
        self.enhanced_analyzer = EnhancedTradeAnalyzer()
        
        # Initialize agent with tools and better error handling
        self.agent = initialize_agent(
            tools=self._create_tools(),
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,  # Reduced verbosity for cleaner output
            handle_parsing_errors=True,
            max_iterations=3,  # Prevent infinite loops - reduced from 5
            early_stopping_method="force"  # Force stop on parsing errors
        )
        
        logger.info("Trading agent initialized successfully")
    
    def _create_tools(self) -> list:
        """Create tools for the agent."""
        return [
            Tool(
                name="AnalyzeRealTrades",
                func=self._analyze_real_trades,
                description="Analyze ONLY real trading data - PnL, profits, losses, trade statistics. Use this for questions about real trade performance, profits, PnL calculations. Does NOT load simulation data."
            ),
            Tool(
                name="CalculatePnL",
                func=self._calculate_pnl,
                description="Calculate total PnL (profit and loss) from real trades. Use this for questions about total profits, losses, or benefits from trading."
            ),
            Tool(
                name="GetRealTradeStatistics",
                func=self._get_real_trade_stats,
                description="Get detailed statistics about real trades only - volume, counts, averages, etc. Does NOT load simulation data."
            ),
            Tool(
                name="AnalyzeDiscrepancies",
                func=self._analyze_discrepancies,
                description="Compare real vs simulated data for ALL dates. WARNING: Loads ALL simulation files. Only use when specifically asked to compare real vs simulation data across all dates."
            ),
            Tool(
                name="GetAvailableDates", 
                func=self._get_available_dates,
                description="Get available trading dates from the real data."
            ),
            Tool(
                name="AnalyzeSpecificDate",
                func=self._analyze_specific_date,
                description="Compare real vs simulation data for ONE specific date. Input should be date in format 'dd-mm-yyyy'. Only loads simulation data for that date."
            ),
            Tool(
                name="ListDataFiles",
                func=self._list_data_files,
                description="List available data files in the data directory."
            ),
            Tool(
                name="PreviewData",
                func=self._preview_data,
                description="Preview real or simulation data. Input: 'real' for real data, 'sim' for simulation preview, or specific date like '10-06-2025' for that date's simulation data."
            ),
            Tool(
                name="ClearCache",
                func=self._clear_cache,
                description="Clear data cache to force reload of all data."
            ),
            Tool(
                name="GetDataStatistics",
                func=self._get_data_statistics,
                description="Get basic statistics about the loaded data."
            ),
            Tool(
                name="ExplainValidationProcess",
                func=self._explain_validation_process,
                description="Explain the technical validation process, methodology, or algorithm used to detect and validate discrepancies between real and simulated trading data. Use this for questions about 'validation', 'process', 'methodology', 'how it works', or 'validation process'."
            ),
            Tool(
                name="AnalyzeDailyPerformance",
                func=self._analyze_daily_performance,
                description="Analyze daily trading performance including best/worst days, daily PnL breakdown, and volatility analysis. Use for questions about daily performance, best/worst trading days, daily profits/losses."
            ),
            Tool(
                name="AnalyzeTradingPatterns",
                func=self._analyze_trading_patterns,
                description="Analyze trading patterns including time-of-day analysis, consecutive streaks, and behavioral patterns. Use for questions about trading habits, time patterns, streaks, or behavioral analysis."
            ),
            Tool(
                name="AnalyzeRiskMetrics",
                func=self._analyze_risk_metrics,
                description="Calculate advanced risk metrics including maximum drawdown, Sharpe ratio, volatility metrics, and risk-adjusted returns. Use for questions about risk analysis, drawdowns, volatility, or performance ratios."
            ),
            Tool(
                name="CompareTimeRanges",
                func=self._compare_time_ranges,
                description="Compare trading performance across different time periods (weeks, months, specific date ranges). Use for questions about performance comparison, trends over time, or period analysis."
            ),
            Tool(
                name="AnalyzePositionSizing",
                func=self._analyze_position_sizing,
                description="Analyze position sizing patterns, optimal position sizes, and position size impact on performance. Use for questions about trade sizes, position management, or sizing strategies."
            ),
            Tool(
                name="AnalyzeExecutionQuality", 
                func=self._analyze_execution_quality,
                description="Analyze trade execution quality, timing patterns, and trading frequency. Use for questions about execution timing, trade clustering, or optimal trading hours."
            ),
            Tool(
                name="AnalyzeMarketContext",
                func=self._analyze_market_context,
                description="Analyze trades in market context including price trends and market conditions. Use for questions about market timing, trend analysis, or performance in different market conditions."
            ),
            Tool(
                name="AnalyzePerformanceAttribution",
                func=self._analyze_performance_attribution,
                description="Analyze what factors contribute most to trading performance (size, timing, direction). Use for questions about performance drivers or factor analysis."
            ),
            Tool(
                name="AnalyzeTradeSequences",
                func=self._analyze_trade_sequences,
                description="Analyze trading sequences, streaks, and behavioral patterns. Use for questions about winning/losing streaks, trading consistency, or behavioral analysis."
            ),
            Tool(
                name="GetComprehensiveAnalysis",
                func=self._get_comprehensive_analysis,
                description="Get a comprehensive deep analysis combining all enhanced analysis methods. Use when asked for complete, detailed, or comprehensive trading analysis."
            )
        ]
    
    def _analyze_discrepancies(self, _: str) -> str:
        """Analyze all available discrepancies with caching."""
        try:
            print("\n🔍 Starting comprehensive analysis of ALL dates...")
            print("📊 This will load simulation data for all 19 dates (may take a moment)")
            print("💡 Tip: For faster analysis, try asking about a specific date instead!")
            print("⏳ You'll see progress as files load... (Press Ctrl+C to cancel)\n")
            
            logger.info("Starting comprehensive discrepancy analysis")
            
            # Get real data chunks (cached)
            chunks = self.data_manager.get_real_trade_chunks()
            
            if not chunks:
                return "No real trading data available for analysis."
            
            all_results = []
            total_discrepancies = 0
            
            print(f"🔄 Processing {len(chunks)} daily chunks...")
            print("📅 Progress: ", end="", flush=True)
            
            for i, real_chunk in enumerate(chunks):
                
                # Get dates for this chunk
                date_strs = set(real_chunk["timestamp"].dt.strftime("%d-%m-%Y").unique())
                date_str = list(date_strs)[0]
                
                # Load simulation data (cached)
                try:
                    sim_df = self.data_manager.get_simulation_data_for_dates(date_strs)
                except ValueError as e:
                    print(f"❌ No data for {date_str}")
                    continue
                
                # Analyze discrepancies
                results = self.analysis_engine.analyze_chunk_discrepancy(real_chunk, sim_df)
                
                if results["status"] == "discrepancies_found":
                    total_discrepancies += len(results["discrepancies"])
                    formatted_result = self.analysis_engine.format_results(results)
                    all_results.append(f"📅 {date_str}: {formatted_result}")
                    print("⚠️", end=" ", flush=True)  # Discrepancies found
                elif results["status"] == "no_discrepancies":
                    all_results.append(f"📅 {date_str}: No significant discrepancies")
                    print("✅", end=" ", flush=True)  # Clean
                else:
                    all_results.append(f"📅 {date_str}: {results['message']}")
                    print("ℹ️", end=" ", flush=True)  # Info
            
            print(f"\n🎉 Analysis complete! Found {total_discrepancies} total discrepancies across {len(chunks)} chunks\n")
            
            if not all_results:
                return "No data could be processed for analysis."
            
            summary = f"📊 Comprehensive Analysis Results:\nFound {total_discrepancies} total discrepancies across {len(chunks)} chunks\n\n"
            return summary + "\n\n".join(all_results[:5])  # Limit output
            
        except Exception as e:
            logger.error(f"Error in discrepancy analysis: {e}")
            return f"Analysis failed: {str(e)}"
    
    def _analyze_specific_date(self, date_str: str) -> str:
        """Analyze discrepancies for a specific date."""
        try:
            date_str = date_str.strip()
            logger.info(f"Analyzing discrepancies for date: {date_str}")
            
            # Get real data for the date
            chunks = self.data_manager.get_real_trade_chunks()
            matching_chunk = None
            
            for chunk in chunks:
                chunk_dates = set(chunk["timestamp"].dt.strftime("%d-%m-%Y").unique())
                if date_str in chunk_dates:
                    matching_chunk = chunk[chunk["timestamp"].dt.strftime("%d-%m-%Y") == date_str]
                    break
            
            if matching_chunk is None or matching_chunk.empty:
                return f"No real trading data found for date: {date_str}"
            
            # Get simulation data
            sim_df = self.data_manager.get_simulation_data_for_dates({date_str})
            
            # Analyze
            results = self.analysis_engine.analyze_chunk_discrepancy(matching_chunk, sim_df)
            return self.analysis_engine.format_results(results)
            
        except Exception as e:
            logger.error(f"Error analyzing date {date_str}: {e}")
            return f"Analysis failed for {date_str}: {str(e)}"
    
    def _get_available_dates(self, _: str) -> str:
        """Get available trading dates."""
        try:
            dates = get_available_dates()
            sorted_dates = sorted(list(dates))
            return f"Available trading dates ({len(dates)} total): {', '.join(sorted_dates)}"
        except Exception as e:
            return f"Error getting available dates: {str(e)}"
    
    def _list_data_files(self, _: str) -> str:
        """List available data files."""
        try:
            data_dir = "data"
            if not os.path.exists(data_dir):
                return "Data directory not found."
            
            files = os.listdir(data_dir)
            
            # Separate real and simulation files
            real_files = [f for f in files if f.endswith('.csv')]
            sim_dir = os.path.join(data_dir, "HL_btcusdt_BTC")
            
            result = ["Available data files:"]
            
            if real_files:
                result.append(f"\nReal data files: {', '.join(real_files)}")
            
            if os.path.exists(sim_dir):
                sim_files = [f for f in os.listdir(sim_dir) if f.endswith('.pickle')]
                result.append(f"\nSimulation files: {len(sim_files)} pickle files in HL_btcusdt_BTC/")
            
            return "\n".join(result)
            
        except Exception as e:
            return f"Error listing files: {str(e)}"
    
    def _preview_data(self, data_type: str) -> str:
        """Preview real or simulation data."""
        try:
            data_type = data_type.strip()
            
            if data_type.lower() == "real":
                chunks = self.data_manager.get_real_trade_chunks()
                if not chunks:
                    return "No real data available."
                
                first_chunk = chunks[0]
                return f"Real data preview (first chunk):\nShape: {first_chunk.shape}\nColumns: {list(first_chunk.columns)}\n\nSample data:\n{first_chunk.head(3).to_string(index=False)}"
            
            elif data_type.lower() in ["sim", "simulation"]:
                # Get first available date for preview
                available_dates = self.data_manager.get_available_dates()
                if not available_dates:
                    return "No simulation dates available."
                
                first_date = sorted(list(available_dates))[0]
                sim_df = self.data_manager.get_simulation_data_for_dates({first_date})
                return f"Simulation data preview for {first_date}:\nShape: {sim_df.shape}\nColumns: {list(sim_df.columns)}\n\nSample data:\n{sim_df.head(3).to_string(index=False)}"
            
            else:
                # Check if it's a valid date format (dd-mm-yyyy)
                if len(data_type) == 10 and data_type.count('-') == 2:
                    try:
                        sim_df = self.data_manager.get_simulation_data_for_dates({data_type})
                        return f"Simulation data preview for {data_type}:\nShape: {sim_df.shape}\nColumns: {list(sim_df.columns)}\n\nSample data:\n{sim_df.head(3).to_string(index=False)}"
                    except ValueError:
                        return f"No simulation data found for date: {data_type}. Available dates: {', '.join(sorted(self.data_manager.get_available_dates()))}"
                else:
                    return f"Invalid input '{data_type}'. Use 'real' for real data, 'sim' for simulation data, or a specific date in format 'dd-mm-yyyy'."
                
        except Exception as e:
            return f"Error previewing data: {str(e)}"
    
    def _clear_cache(self, _: str) -> str:
        """Clear data cache."""
        try:
            clear_data_cache()
            return "Data cache cleared successfully. Next data access will reload from files."
        except Exception as e:
            return f"Error clearing cache: {str(e)}"
    
    def _get_data_statistics(self, _: str) -> str:
        """Get basic statistics about the data."""
        try:
            # Get real data stats
            chunks = self.data_manager.get_real_trade_chunks()
            total_real_records = sum(len(chunk) for chunk in chunks)
            available_dates = get_available_dates()
            
            # Get simulation data stats 
            sim_dir = os.path.join("data", "HL_btcusdt_BTC")
            sim_files = len([f for f in os.listdir(sim_dir) if f.endswith('.pickle')]) if os.path.exists(sim_dir) else 0
            
            return f"""Data Statistics:
- Real trading records: {total_real_records:,}
- Daily chunks: {len(chunks)}
- Available dates: {len(available_dates)}
- Simulation files: {sim_files}
- Date range: {min(available_dates) if available_dates else 'N/A'} to {max(available_dates) if available_dates else 'N/A'}
- Cache status: {"Active" if self.data_manager._real_data_cache is not None else "Empty"}"""
            
        except Exception as e:
            return f"Error getting statistics: {str(e)}"
    
    def _analyze_real_trades(self, _: str) -> str:
        """Analyze real trading data only - no simulation data loaded."""
        try:
            logger.info("Analyzing real trades only (no simulation data)")
            
            # Get real data chunks (cached)
            chunks = self.data_manager.get_real_trade_chunks()
            
            if not chunks:
                return "No real trading data available for analysis."
            
            # Combine all chunks
            all_real_data = pd.concat(chunks, ignore_index=True)
            
            # Calculate key metrics
            total_trades = len(all_real_data)
            total_volume = all_real_data['amount'].sum() if 'amount' in all_real_data.columns else 0
            avg_price = all_real_data['price'].mean() if 'price' in all_real_data.columns else 0
            
            # PnL analysis
            total_pnl = all_real_data['realized_pnl'].sum() if 'realized_pnl' in all_real_data.columns else 0
            total_fees = all_real_data['total_fee'].sum() if 'total_fee' in all_real_data.columns else 0
            
            # Side analysis
            buy_trades = len(all_real_data[all_real_data['side'] == 'B']) if 'side' in all_real_data.columns else 0
            sell_trades = len(all_real_data[all_real_data['side'] == 'A']) if 'side' in all_real_data.columns else 0
            
            # Date range
            if 'timestamp' in all_real_data.columns:
                start_date = all_real_data['timestamp'].min().strftime('%Y-%m-%d')
                end_date = all_real_data['timestamp'].max().strftime('%Y-%m-%d')
                date_info = f"Period: {start_date} to {end_date}"
            else:
                date_info = "Date information not available"
            
            result = f"""Real Trading Analysis:

{date_info}
- Total trades: {total_trades:,}
- Buy trades: {buy_trades:,}
- Sell trades: {sell_trades:,}
- Total volume: {total_volume:,.2f}
- Average price: ${avg_price:,.2f}
- Total PnL: ${total_pnl:,.2f}
- Total fees: ${total_fees:,.2f}
- Net profit: ${total_pnl - total_fees:,.2f}"""
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing real trades: {e}")
            return f"Error analyzing real trades: {str(e)}"
    
    def _calculate_pnl(self, _: str) -> str:
        """Calculate PnL from real trades only."""
        try:
            logger.info("Calculating PnL from real trades")
            
            # Get real data chunks (cached)
            chunks = self.data_manager.get_real_trade_chunks()
            
            if not chunks:
                return "No real trading data available for PnL calculation."
            
            # Combine all chunks
            all_real_data = pd.concat(chunks, ignore_index=True)
            
            if 'realized_pnl' not in all_real_data.columns:
                return "No 'realized_pnl' column found in real trading data."
            
            # Calculate PnL metrics
            total_pnl = all_real_data['realized_pnl'].sum()
            positive_pnl = all_real_data[all_real_data['realized_pnl'] > 0]['realized_pnl'].sum()
            negative_pnl = all_real_data[all_real_data['realized_pnl'] < 0]['realized_pnl'].sum()
            profitable_trades = len(all_real_data[all_real_data['realized_pnl'] > 0])
            losing_trades = len(all_real_data[all_real_data['realized_pnl'] < 0])
            neutral_trades = len(all_real_data[all_real_data['realized_pnl'] == 0])
            
            # Calculate fees if available
            total_fees = 0
            if 'total_fee' in all_real_data.columns:
                total_fees = all_real_data['total_fee'].sum()
            
            # Net profit
            net_profit = total_pnl - total_fees
            
            # Win rate
            total_trading_trades = profitable_trades + losing_trades
            win_rate = (profitable_trades / total_trading_trades * 100) if total_trading_trades > 0 else 0
            
            result = f"""PnL Analysis:

Total Realized PnL: ${total_pnl:,.2f}
- Profits: ${positive_pnl:,.2f} ({profitable_trades} trades)
- Losses: ${negative_pnl:,.2f} ({losing_trades} trades)
- Neutral: {neutral_trades} trades

Total Fees: ${total_fees:,.2f}
Net Profit: ${net_profit:,.2f}

Win Rate: {win_rate:.1f}% ({profitable_trades}/{total_trading_trades} profitable trades)"""
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating PnL: {e}")
            return f"Error calculating PnL: {str(e)}"
    
    def _get_real_trade_stats(self, _: str) -> str:
        """Get detailed statistics about real trades only."""
        try:
            logger.info("Getting real trade statistics")
            
            # Get real data chunks (cached)
            chunks = self.data_manager.get_real_trade_chunks()
            
            if not chunks:
                return "No real trading data available for statistics."
            
            # Combine all chunks
            all_real_data = pd.concat(chunks, ignore_index=True)
            
            stats = []
            
            # Basic counts
            stats.append(f"Total Trades: {len(all_real_data):,}")
            
            # Price statistics
            if 'price' in all_real_data.columns:
                price_stats = all_real_data['price'].describe()
                stats.append(f"\nPrice Statistics:")
                stats.append(f"- Average: ${price_stats['mean']:,.2f}")
                stats.append(f"- Min: ${price_stats['min']:,.2f}")
                stats.append(f"- Max: ${price_stats['max']:,.2f}")
                stats.append(f"- Std Dev: ${price_stats['std']:,.2f}")
            
            # Volume statistics
            if 'quantity' in all_real_data.columns:
                qty_stats = all_real_data['quantity'].describe()
                stats.append(f"\nQuantity Statistics:")
                stats.append(f"- Total: {all_real_data['quantity'].sum():,.4f}")
                stats.append(f"- Average: {qty_stats['mean']:,.4f}")
                stats.append(f"- Min: {qty_stats['min']:,.4f}")
                stats.append(f"- Max: {qty_stats['max']:,.4f}")
            
            # Side distribution
            if 'side' in all_real_data.columns:
                side_counts = all_real_data['side'].value_counts()
                stats.append(f"\nTrade Sides:")
                for side, count in side_counts.items():
                    side_name = "Buy" if side == "B" else "Sell" if side == "A" else side
                    stats.append(f"- {side_name}: {count:,} trades")
            
            # Status distribution
            if 'status' in all_real_data.columns:
                status_counts = all_real_data['status'].value_counts()
                stats.append(f"\nTrade Status:")
                for status, count in status_counts.items():
                    stats.append(f"- {status}: {count:,} trades")
            
            # Date range
            if 'timestamp' in all_real_data.columns:
                dates = pd.to_datetime(all_real_data['timestamp'])
                stats.append(f"\nTime Period:")
                stats.append(f"- From: {dates.min().strftime('%Y-%m-%d %H:%M:%S')}")
                stats.append(f"- To: {dates.max().strftime('%Y-%m-%d %H:%M:%S')}")
                stats.append(f"- Duration: {(dates.max() - dates.min()).days} days")
            
            return "\n".join(stats)
            
        except Exception as e:
            logger.error(f"Error getting real trade statistics: {e}")
            return f"Error getting real trade statistics: {str(e)}"
    
    def _try_fallback_response(self, user_input: str) -> str:
        """Try to handle common queries directly when agent fails."""
        try:
            user_input_lower = user_input.lower()
            
            # PnL related queries
            if any(word in user_input_lower for word in ['pnl', 'profit', 'loss', 'benefit', 'money', 'earn']):
                return self._calculate_pnl("")
            
            # Statistics queries  
            if any(word in user_input_lower for word in ['statistic', 'stats', 'summary', 'overview']):
                return self._get_real_trade_stats("")
            
            # Available dates
            if any(word in user_input_lower for word in ['date', 'available', 'when', 'time']):
                return self._get_available_dates("")
            
            # Data preview
            if any(word in user_input_lower for word in ['preview', 'show', 'look', 'see']):
                if 'real' in user_input_lower:
                    return self._preview_data("real")
                elif 'sim' in user_input_lower:
                    return self._preview_data("sim")
            
            # Validation process questions
            if any(word in user_input_lower for word in ['validation', 'process', 'validate', 'methodology', 'method', 'algorithm', 'detect']):
                return self._explain_validation_process("")
            
            # Help
            if any(word in user_input_lower for word in ['help', 'what', 'how', 'can']):
                return """I can help you analyze your trading data! Try asking:
• "Calculate my PnL" - Get profit/loss analysis
• "Show me trade statistics" - Detailed trade stats
• "What dates are available?" - See available data
• "Preview real data" - Look at real trade data
• "Explain validation process" - Technical methodology details
• "Analyze discrepancies for [date]" - Compare real vs simulation for specific date"""
            
            return None
            
        except Exception as e:
            logger.warning(f"Fallback response error: {e}")
            return None
    
    def _run_with_timeout(self, user_input: str, timeout: int = 30) -> str:
        """Run agent with timeout to prevent infinite loops."""
        result = {"response": None, "error": None}
        
        def target():
            try:
                result["response"] = self.agent.run(user_input)
            except Exception as e:
                result["error"] = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            # Timeout occurred
            raise TimeoutError(f"Agent took longer than {timeout} seconds to respond")
        
        if result["error"]:
            raise result["error"]
        
        return result["response"]
    
    def _explain_validation_process(self, _: str) -> str:
        """Explain the technical validation process for discrepancies."""
        return """🔍 **Discrepancy Validation Process**

## **Data Preparation Phase**
1. **Real Data Loading**: 
   - Load actual trading records from `agent_live_data.csv`
   - Parse timestamps and ensure data integrity
   - Group trades by daily chunks for efficient processing

2. **Simulation Data Loading**:
   - Load corresponding simulation data from pickle files
   - Handle multiple data formats (dict, DataFrame, list)
   - Extract metadata (broker, pair_symbol, master/slave setup)

## **Temporal Alignment Process**
3. **Timestamp Synchronization**:
   - Use `pd.merge_asof()` for timestamp-based matching
   - Apply 2-second tolerance window for trade matching
   - Direction: "nearest" - finds closest simulation trade to each real trade

4. **Data Merging**:
   - Real trades: renamed to `price_real`
   - Simulation trades: renamed to `price_sim` 
   - Merge on timestamp with tolerance for market timing differences

## **Discrepancy Detection**
5. **Price Difference Calculation**:
   ```python
   price_diff = abs(price_real - price_sim)
   ```
   - Calculate absolute difference between real and simulated prices
   - Apply configurable threshold (default: $2.00)

6. **Statistical Analysis**:
   - Count total comparisons made
   - Calculate percentage differences: `(price_diff / price_real) * 100`
   - Identify profitable vs losing trades in discrepancies
   - Generate win rate and distribution statistics

## **AI-Powered Analysis**
7. **Pattern Recognition**:
   - Feed discrepancy samples to GPT-4o
   - Analyze potential causes (latency, slippage, market conditions)
   - Provide recommendations for simulation improvements

## **Validation Criteria**
8. **Success Metrics**:
   - **Threshold Compliance**: <$2 difference considered acceptable
   - **Temporal Matching**: Successful timestamp alignment within 2s
   - **Volume Coverage**: Percentage of real trades successfully matched
   - **Statistical Significance**: Distribution of discrepancies

## **Technical Implementation**
- **Framework**: Pandas for data manipulation, LangChain for AI analysis
- **Caching**: Persistent storage to avoid reloading large datasets
- **Error Handling**: Graceful handling of missing data or format issues
- **Performance**: Chunked processing for memory efficiency

## **Output Interpretation**
- **"No discrepancies"**: All price differences < $2 threshold
- **"Discrepancies found"**: Specific trades with >$2 difference + AI analysis
- **Statistics**: Count, averages, max/min differences, percentage analysis

This process ensures robust validation between real market execution and simulation accuracy."""
    
    def _analyze_daily_performance(self, _: str) -> str:
        """Analyze daily trading performance with detailed breakdown."""
        try:
            chunks = self.data_manager.get_real_trade_chunks()
            
            if not chunks:
                return "No trading data available for daily performance analysis."
            
            daily_metrics = {}
            
            for chunk in chunks:
                if chunk.empty:
                    continue
                    
                date_str = chunk['timestamp'].dt.date.iloc[0].strftime('%Y-%m-%d')
                
                # Daily PnL
                daily_pnl = chunk['realized_pnl'].sum() if 'realized_pnl' in chunk.columns else 0
                
                # Daily volume and trade count
                trade_count = len(chunk)
                total_volume = chunk['amount'].sum() if 'amount' in chunk.columns else 0
                
                # Price volatility
                price_volatility = chunk['price'].std() if 'price' in chunk.columns and len(chunk) > 1 else 0
                
                # Win rate for the day
                if 'realized_pnl' in chunk.columns:
                    profitable_trades = len(chunk[chunk['realized_pnl'] > 0])
                    total_trades = len(chunk[chunk['realized_pnl'] != 0])
                    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
                else:
                    win_rate = 0
                
                daily_metrics[date_str] = {
                    'pnl': daily_pnl,
                    'trades': trade_count,
                    'volume': total_volume,
                    'volatility': price_volatility,
                    'win_rate': win_rate
                }
            
            if not daily_metrics:
                return "No daily metrics could be calculated."
            
            # Find best and worst days
            best_day = max(daily_metrics, key=lambda x: daily_metrics[x]['pnl'])
            worst_day = min(daily_metrics, key=lambda x: daily_metrics[x]['pnl'])
            most_volatile_day = max(daily_metrics, key=lambda x: daily_metrics[x]['volatility'])
            most_active_day = max(daily_metrics, key=lambda x: daily_metrics[x]['trades'])
            
            # Calculate averages
            avg_daily_pnl = sum(d['pnl'] for d in daily_metrics.values()) / len(daily_metrics)
            avg_daily_trades = sum(d['trades'] for d in daily_metrics.values()) / len(daily_metrics)
            
            result = f"""📊 **Daily Performance Analysis**

## **Best & Worst Performance**
🏆 **Best Day**: {best_day}
   - PnL: ${daily_metrics[best_day]['pnl']:,.2f}
   - Trades: {daily_metrics[best_day]['trades']}
   - Win Rate: {daily_metrics[best_day]['win_rate']:.1f}%

📉 **Worst Day**: {worst_day}
   - PnL: ${daily_metrics[worst_day]['pnl']:,.2f}
   - Trades: {daily_metrics[worst_day]['trades']}
   - Win Rate: {daily_metrics[worst_day]['win_rate']:.1f}%

## **Volatility & Activity**
🌊 **Most Volatile Day**: {most_volatile_day}
   - Price Std Dev: ${daily_metrics[most_volatile_day]['volatility']:,.2f}
   - Daily PnL: ${daily_metrics[most_volatile_day]['pnl']:,.2f}

📈 **Most Active Day**: {most_active_day}
   - Trades: {daily_metrics[most_active_day]['trades']}
   - Volume: ${daily_metrics[most_active_day]['volume']:,.2f}

## **Daily Averages**
📊 **Average Daily PnL**: ${avg_daily_pnl:,.2f}
📊 **Average Daily Trades**: {avg_daily_trades:.1f}
📊 **Trading Days Analyzed**: {len(daily_metrics)}

## **Performance Distribution**
🟢 **Profitable Days**: {len([d for d in daily_metrics.values() if d['pnl'] > 0])}
🔴 **Loss Days**: {len([d for d in daily_metrics.values() if d['pnl'] < 0])}
⚪ **Breakeven Days**: {len([d for d in daily_metrics.values() if d['pnl'] == 0])}"""

            return result
            
        except Exception as e:
            return f"Error analyzing daily performance: {str(e)}"
    
    def _analyze_trading_patterns(self, _: str) -> str:
        """Analyze trading patterns and behavioral metrics."""
        try:
            chunks = self.data_manager.get_real_trade_chunks()
            
            if not chunks:
                return "No trading data available for pattern analysis."
            
            all_data = pd.concat(chunks, ignore_index=True)
            all_data['timestamp'] = pd.to_datetime(all_data['timestamp'])
            all_data = all_data.sort_values('timestamp')
            
            # Time of day analysis
            all_data['hour'] = all_data['timestamp'].dt.hour
            hourly_counts = all_data['hour'].value_counts().sort_index()
            peak_hour = hourly_counts.idxmax()
            peak_count = hourly_counts.max()
            
            # Day of week analysis
            all_data['day_of_week'] = all_data['timestamp'].dt.day_name()
            daily_counts = all_data['day_of_week'].value_counts()
            most_active_day = daily_counts.idxmax()
            
            # Consecutive streak analysis
            if 'realized_pnl' in all_data.columns:
                # Winning streaks
                wins = all_data['realized_pnl'] > 0
                win_streaks = []
                current_streak = 0
                
                for is_win in wins:
                    if is_win:
                        current_streak += 1
                    else:
                        if current_streak > 0:
                            win_streaks.append(current_streak)
                        current_streak = 0
                if current_streak > 0:
                    win_streaks.append(current_streak)
                
                # Losing streaks
                losses = all_data['realized_pnl'] < 0
                loss_streaks = []
                current_streak = 0
                
                for is_loss in losses:
                    if is_loss:
                        current_streak += 1
                    else:
                        if current_streak > 0:
                            loss_streaks.append(current_streak)
                        current_streak = 0
                if current_streak > 0:
                    loss_streaks.append(current_streak)
                
                max_win_streak = max(win_streaks) if win_streaks else 0
                max_loss_streak = max(loss_streaks) if loss_streaks else 0
                avg_win_streak = sum(win_streaks) / len(win_streaks) if win_streaks else 0
                avg_loss_streak = sum(loss_streaks) / len(loss_streaks) if loss_streaks else 0
            else:
                max_win_streak = max_loss_streak = avg_win_streak = avg_loss_streak = 0
            
            # Trade size patterns
            if 'quantity' in all_data.columns:
                avg_trade_size = all_data['quantity'].mean()
                max_trade_size = all_data['quantity'].max()
                min_trade_size = all_data['quantity'].min()
            else:
                avg_trade_size = max_trade_size = min_trade_size = 0
            
            # Side distribution
            if 'side' in all_data.columns:
                side_counts = all_data['side'].value_counts()
                buy_count = side_counts.get('B', 0)
                sell_count = side_counts.get('A', 0)
                total_directional = buy_count + sell_count
                buy_percentage = (buy_count / total_directional * 100) if total_directional > 0 else 0
            else:
                buy_count = sell_count = buy_percentage = 0
            
            result = f"""🔍 **Trading Pattern Analysis**

## **Time Patterns**
⏰ **Peak Trading Hour**: {peak_hour}:00 ({peak_count} trades)
📅 **Most Active Day**: {most_active_day} ({daily_counts[most_active_day]} trades)

## **Streak Analysis**
🏆 **Longest Winning Streak**: {max_win_streak} trades
📉 **Longest Losing Streak**: {max_loss_streak} trades
📊 **Average Winning Streak**: {avg_win_streak:.1f} trades
📊 **Average Losing Streak**: {avg_loss_streak:.1f} trades

## **Trade Behavior**
📏 **Average Trade Size**: {avg_trade_size:.4f}
📈 **Largest Trade**: {max_trade_size:.4f}
📉 **Smallest Trade**: {min_trade_size:.4f}

## **Directional Bias**
🟢 **Buy Trades**: {buy_count} ({buy_percentage:.1f}%)
🔴 **Sell Trades**: {sell_count} ({100-buy_percentage:.1f}%)

## **Activity Distribution**
📊 **Total Trades Analyzed**: {len(all_data)}
📊 **Unique Trading Days**: {all_data['timestamp'].dt.date.nunique()}
📊 **Average Trades/Day**: {len(all_data) / all_data['timestamp'].dt.date.nunique():.1f}"""

            return result
            
        except Exception as e:
            return f"Error analyzing trading patterns: {str(e)}"
    
    def _analyze_risk_metrics(self, _: str) -> str:
        """Calculate advanced risk metrics and ratios."""
        try:
            chunks = self.data_manager.get_real_trade_chunks()
            
            if not chunks:
                return "No trading data available for risk analysis."
            
            all_data = pd.concat(chunks, ignore_index=True)
            
            if 'realized_pnl' not in all_data.columns:
                return "No PnL data available for risk metrics calculation."
            
            # Basic metrics
            total_pnl = all_data['realized_pnl'].sum()
            pnl_std = all_data['realized_pnl'].std()
            trade_count = len(all_data[all_data['realized_pnl'] != 0])
            
            # Calculate daily PnL for drawdown analysis
            all_data['date'] = pd.to_datetime(all_data['timestamp']).dt.date
            daily_pnl = all_data.groupby('date')['realized_pnl'].sum()
            
            # Maximum Drawdown
            cumulative_pnl = daily_pnl.cumsum()
            running_max = cumulative_pnl.expanding().max()
            drawdown = cumulative_pnl - running_max
            max_drawdown = drawdown.min()
            
            # Win Rate
            profitable_trades = len(all_data[all_data['realized_pnl'] > 0])
            losing_trades = len(all_data[all_data['realized_pnl'] < 0])
            total_directional_trades = profitable_trades + losing_trades
            win_rate = (profitable_trades / total_directional_trades * 100) if total_directional_trades > 0 else 0
            
            # Average win/loss
            wins = all_data[all_data['realized_pnl'] > 0]['realized_pnl']
            losses = all_data[all_data['realized_pnl'] < 0]['realized_pnl']
            avg_win = wins.mean() if len(wins) > 0 else 0
            avg_loss = losses.mean() if len(losses) > 0 else 0
            
            # Profit Factor
            total_wins = wins.sum() if len(wins) > 0 else 0
            total_losses = abs(losses.sum()) if len(losses) > 0 else 0
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            # Sharpe-like ratio (assuming daily returns)
            if len(daily_pnl) > 1 and daily_pnl.std() > 0:
                avg_daily_return = daily_pnl.mean()
                daily_volatility = daily_pnl.std()
                sharpe_like = avg_daily_return / daily_volatility
            else:
                sharpe_like = 0
            
            # Risk-Reward Ratio
            risk_reward = abs(avg_win / avg_loss) if avg_loss < 0 else 0
            
            # Volatility metrics
            if 'price' in all_data.columns:
                price_volatility = all_data['price'].std()
                price_range = all_data['price'].max() - all_data['price'].min()
            else:
                price_volatility = price_range = 0
            
            result = f"""⚠️ **Risk Metrics Analysis**

## **Core Risk Metrics**
📉 **Maximum Drawdown**: ${max_drawdown:,.2f}
📊 **PnL Volatility**: ${pnl_std:,.2f}
📈 **Total PnL**: ${total_pnl:,.2f}
⚖️ **Sharpe-like Ratio**: {sharpe_like:.3f}

## **Win/Loss Analysis**
🎯 **Win Rate**: {win_rate:.1f}%
💰 **Average Win**: ${avg_win:,.2f}
💸 **Average Loss**: ${avg_loss:,.2f}
⚖️ **Risk-Reward Ratio**: {risk_reward:.2f}
📊 **Profit Factor**: {profit_factor:.2f}

## **Trade Distribution**
✅ **Winning Trades**: {profitable_trades}
❌ **Losing Trades**: {losing_trades}
🎲 **Total Analyzed**: {trade_count}

## **Market Exposure**
🌊 **Price Volatility**: ${price_volatility:,.2f}
📏 **Price Range**: ${price_range:,.2f}
📅 **Trading Days**: {len(daily_pnl)}
📊 **Avg Daily PnL**: ${daily_pnl.mean():,.2f}

## **Risk Assessment**
{"🟢 **LOW RISK**" if max_drawdown > -100 and win_rate > 50 else "🟡 **MEDIUM RISK**" if max_drawdown > -500 else "🔴 **HIGH RISK**"}
- Max drawdown: ${max_drawdown:,.2f}
- Win rate: {win_rate:.1f}%
- Profit factor: {profit_factor:.2f}"""

            return result
            
        except Exception as e:
            return f"Error calculating risk metrics: {str(e)}"
    
    def _compare_time_ranges(self, date_range: str) -> str:
        """Compare trading performance across different time periods."""
        try:
            chunks = self.data_manager.get_real_trade_chunks()
            
            if not chunks:
                return "No trading data available for time range comparison."
            
            all_data = pd.concat(chunks, ignore_index=True)
            all_data['timestamp'] = pd.to_datetime(all_data['timestamp'])
            all_data = all_data.sort_values('timestamp')
            
            # Split data into weeks
            all_data['week'] = all_data['timestamp'].dt.isocalendar().week
            all_data['year_week'] = all_data['timestamp'].dt.strftime('%Y-W%V')
            
            weekly_metrics = {}
            
            for week, week_data in all_data.groupby('year_week'):
                if 'realized_pnl' in week_data.columns:
                    weekly_pnl = week_data['realized_pnl'].sum()
                    trade_count = len(week_data)
                    profitable_trades = len(week_data[week_data['realized_pnl'] > 0])
                    total_trades = len(week_data[week_data['realized_pnl'] != 0])
                    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
                    
                    weekly_metrics[week] = {
                        'pnl': weekly_pnl,
                        'trades': trade_count,
                        'win_rate': win_rate,
                        'start_date': week_data['timestamp'].min().strftime('%Y-%m-%d'),
                        'end_date': week_data['timestamp'].max().strftime('%Y-%m-%d')
                    }
            
            if not weekly_metrics:
                return "No weekly metrics could be calculated."
            
            # Find best and worst weeks
            best_week = max(weekly_metrics, key=lambda x: weekly_metrics[x]['pnl'])
            worst_week = min(weekly_metrics, key=lambda x: weekly_metrics[x]['pnl'])
            
            # Calculate trends
            weeks_sorted = sorted(weekly_metrics.items(), key=lambda x: x[1]['start_date'])
            first_half_pnl = sum(w[1]['pnl'] for w in weeks_sorted[:len(weeks_sorted)//2])
            second_half_pnl = sum(w[1]['pnl'] for w in weeks_sorted[len(weeks_sorted)//2:])
            
            trend = "📈 IMPROVING" if second_half_pnl > first_half_pnl else "📉 DECLINING" if second_half_pnl < first_half_pnl else "➡️ STABLE"
            
            result = f"""📅 **Time Range Comparison Analysis**

## **Weekly Performance Overview**
📊 **Total Weeks Analyzed**: {len(weekly_metrics)}
📈 **Performance Trend**: {trend}

## **Best vs Worst Week**
🏆 **Best Week** ({best_week}):
   - Period: {weekly_metrics[best_week]['start_date']} to {weekly_metrics[best_week]['end_date']}
   - PnL: ${weekly_metrics[best_week]['pnl']:,.2f}
   - Trades: {weekly_metrics[best_week]['trades']}
   - Win Rate: {weekly_metrics[best_week]['win_rate']:.1f}%

📉 **Worst Week** ({worst_week}):
   - Period: {weekly_metrics[worst_week]['start_date']} to {weekly_metrics[worst_week]['end_date']}
   - PnL: ${weekly_metrics[worst_week]['pnl']:,.2f}
   - Trades: {weekly_metrics[worst_week]['trades']}
   - Win Rate: {weekly_metrics[worst_week]['win_rate']:.1f}%

## **Period Comparison**
📊 **First Half Performance**: ${first_half_pnl:,.2f}
📊 **Second Half Performance**: ${second_half_pnl:,.2f}
📊 **Improvement**: ${second_half_pnl - first_half_pnl:,.2f}

## **Consistency Metrics**
📊 **Profitable Weeks**: {len([w for w in weekly_metrics.values() if w['pnl'] > 0])}
📊 **Loss Weeks**: {len([w for w in weekly_metrics.values() if w['pnl'] < 0])}
📊 **Average Weekly PnL**: ${sum(w['pnl'] for w in weekly_metrics.values()) / len(weekly_metrics):,.2f}
📊 **Weekly PnL Std Dev**: ${pd.Series([w['pnl'] for w in weekly_metrics.values()]).std():,.2f}"""

            return result
            
        except Exception as e:
            return f"Error comparing time ranges: {str(e)}"
    
    def _analyze_position_sizing(self, _: str) -> str:
        """Analyze position sizing patterns using enhanced analyzer."""
        try:
            logger.info("Analyzing position sizing patterns")
            
            # Get real data chunks (cached)
            chunks = self.data_manager.get_real_trade_chunks()
            
            if not chunks:
                return "No real trading data available for position sizing analysis."
            
            # Combine all chunks
            all_real_data = pd.concat(chunks, ignore_index=True)
            
            return self.enhanced_analyzer.analyze_position_sizing(all_real_data)
            
        except Exception as e:
            return f"Error analyzing position sizing: {str(e)}"
    
    def _analyze_execution_quality(self, _: str) -> str:
        """Analyze execution quality using enhanced analyzer."""
        try:
            logger.info("Analyzing execution quality")
            
            # Get real data chunks (cached)
            chunks = self.data_manager.get_real_trade_chunks()
            
            if not chunks:
                return "No real trading data available for execution quality analysis."
            
            # Combine all chunks
            all_real_data = pd.concat(chunks, ignore_index=True)
            
            return self.enhanced_analyzer.analyze_execution_quality(all_real_data)
            
        except Exception as e:
            return f"Error analyzing execution quality: {str(e)}"
    
    def _analyze_market_context(self, _: str) -> str:
        """Analyze market context using enhanced analyzer."""
        try:
            logger.info("Analyzing market context")
            
            # Get real data chunks (cached)
            chunks = self.data_manager.get_real_trade_chunks()
            
            if not chunks:
                return "No real trading data available for market context analysis."
            
            # Combine all chunks
            all_real_data = pd.concat(chunks, ignore_index=True)
            
            return self.enhanced_analyzer.analyze_market_context(all_real_data)
            
        except Exception as e:
            return f"Error analyzing market context: {str(e)}"
    
    def _analyze_performance_attribution(self, _: str) -> str:
        """Analyze performance attribution using enhanced analyzer."""
        try:
            logger.info("Analyzing performance attribution")
            
            # Get real data chunks (cached)
            chunks = self.data_manager.get_real_trade_chunks()
            
            if not chunks:
                return "No real trading data available for performance attribution analysis."
            
            # Combine all chunks
            all_real_data = pd.concat(chunks, ignore_index=True)
            
            return self.enhanced_analyzer.analyze_performance_attribution(all_real_data)
            
        except Exception as e:
            return f"Error analyzing performance attribution: {str(e)}"
    
    def _analyze_trade_sequences(self, _: str) -> str:
        """Analyze trade sequences using enhanced analyzer."""
        try:
            logger.info("Analyzing trade sequences")
            
            # Get real data chunks (cached)
            chunks = self.data_manager.get_real_trade_chunks()
            
            if not chunks:
                return "No real trading data available for trade sequence analysis."
            
            # Combine all chunks
            all_real_data = pd.concat(chunks, ignore_index=True)
            
            return self.enhanced_analyzer.analyze_trade_sequences(all_real_data)
            
        except Exception as e:
            return f"Error analyzing trade sequences: {str(e)}"
    
    def _get_comprehensive_analysis(self, _: str) -> str:
        """Get comprehensive deep analysis using enhanced analyzer."""
        try:
            logger.info("Running comprehensive analysis")
            
            # Get real data chunks (cached)
            chunks = self.data_manager.get_real_trade_chunks()
            
            if not chunks:
                return "No real trading data available for comprehensive analysis."
            
            # Combine all chunks
            all_real_data = pd.concat(chunks, ignore_index=True)
            
            return self.enhanced_analyzer.get_comprehensive_analysis(all_real_data)
            
        except Exception as e:
            return f"Error in comprehensive analysis: {str(e)}"
    
    def run_interactive(self):
        """Run the agent in interactive mode."""
        print("🚀 Trading Analysis Agent Ready!")
        print("💡 Try asking: 'Calculate my PnL' or 'Show me trade statistics'")
        print("⚡ For fast queries, ask about real trades only")
        print("🛑 Type 'exit' or 'quit' to stop, Ctrl+C to interrupt long operations\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in {"exit", "quit", "q"}:
                    print("👋 Goodbye! Thanks for using the trading agent!")
                    break
                
                if not user_input:
                    continue
                
                # Handle special cases that might cause loops
                if len(user_input) < 3:
                    print("\n💡 Please ask a more specific question about your trading data.\n")
                    continue
                
                # Handle non-question statements that might confuse the agent
                if not any(char in user_input for char in ['?', 'what', 'how', 'why', 'when', 'where', 'show', 'analyze', 'calculate', 'get', 'explain']):
                    if any(word in user_input.lower() for word in ['enhanced', 'capabilities', 'analysis', 'trading']):
                        print("\n🎉 Yes! This agent has enhanced deep analysis capabilities!")
                        print("💡 Try asking: 'Give me a comprehensive analysis' or 'Show me trade statistics'\n")
                        continue
                    else:
                        print("\n💡 Please ask a question about your trading data.")
                        print("🔍 For example: 'Calculate my PnL' or 'Analyze my trades'\n")
                        continue
                
                # Process the request with timeout protection
                try:
                    response = self._run_with_timeout(user_input, timeout=30)
                    print(f"\nAgent: {response}\n")
                except TimeoutError:
                    print(f"\n⏰ That request is taking too long. Let me try a direct approach...")
                    fallback_response = self._try_fallback_response(user_input)
                    if fallback_response:
                        print(f"\nAgent: {fallback_response}\n")
                    else:
                        print(f"\n💡 Try asking something simpler like: 'Calculate my PnL' or 'Show me trade statistics'\n")
                
                except Exception as agent_error:
                    # Try direct fallback for common queries
                    fallback_response = self._try_fallback_response(user_input)
                    if fallback_response:
                        print(f"\nAgent: {fallback_response}\n")
                    else:
                        print(f"\n❌ I had trouble processing that request.")
                        print(f"💡 Try asking something like: 'Calculate my PnL' or 'Show me trade statistics'\n")
                        logger.warning(f"Agent error: {agent_error}")
                
            except KeyboardInterrupt:
                print("\n\n⏹️ Operation interrupted by user")
                print("💡 Try asking about specific dates or real trades for faster analysis")
                print("🔄 You can continue with another question or type 'exit' to quit\n")
                continue
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                print(f"\n❌ Error: {str(e)}")
                print("💡 Try rephrasing your question or ask for help\n")


def main():
    """Main entry point."""
    try:
        agent = TradingAgent()
        agent.run_interactive()
    except Exception as e:
        logger.error(f"Failed to start agent: {e}")
        print(f"Failed to start agent: {e}")


if __name__ == "__main__":
    main()
