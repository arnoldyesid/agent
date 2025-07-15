#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Trading Analysis Agent

Clean, focused agent with accurate discrepancy analysis and deep insights.
"""

import os
import pickle
import signal
import threading
from datetime import datetime
from typing import Dict, List, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
import logging

from deep_analysis_engine import DeepAnalysisEngine

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env file")


class DeepTradingAgent:
    """Advanced trading analysis agent with deep insights and accurate matching."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0,
            api_key=OPENAI_API_KEY
        )
        
        self.analysis_engine = DeepAnalysisEngine()
        self.agent = None
        self.shutdown_event = threading.Event()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self._initialize_tools()
        self._initialize_agent()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print("\n🛑 Shutting down agent gracefully...")
        self.shutdown_event.set()
    
    def _get_available_dates(self, query: str = "") -> str:
        """Get list of available simulation dates"""
        try:
            import glob
            pattern = "data/HL_btcusdt_BTC/parrot_HL_btcusdt_BTC_*.pickle"
            files = glob.glob(pattern)
            
            dates = []
            for file in files:
                # Extract date from filename
                basename = os.path.basename(file)
                date_part = basename.replace("parrot_HL_btcusdt_BTC_", "").replace(".pickle", "")
                dates.append(date_part)
            
            dates.sort()
            
            return f"📅 Available simulation dates ({len(dates)} total):\n" + "\n".join(f"  • {date}" for date in dates)
            
        except Exception as e:
            return f"❌ Error getting available dates: {str(e)}"
    
    def _analyze_specific_date(self, date_str: str) -> str:
        """Analyze discrepancies for a specific date"""
        try:
            # Load simulation data for specific date
            sim_file = f"data/HL_btcusdt_BTC/parrot_HL_btcusdt_BTC_{date_str}.pickle"
            
            if not os.path.exists(sim_file):
                return f"❌ Simulation file not found for date {date_str}"
            
            print(f"📊 Loading simulation data for {date_str}...")
            with open(sim_file, 'rb') as f:
                sim_data = pickle.load(f)
            
            # Run deep analysis with validation
            print("🔍 Performing accuracy validation and deep analysis...")
            results = self.analysis_engine.analyze_discrepancies(
                "data/agent_live_data.csv", 
                sim_data,
                date_str
            )
            
            if 'error' in results:
                return f"❌ Analysis failed: {results['error']}"
            
            # Get validation results
            validation = results.get('validation_result', {})
            metrics = results.get('metrics', {})
            diagnostics = results.get('diagnostics', [])
            
            # Format accuracy verdict
            verdict = validation.get('accuracy_verdict', 'UNKNOWN')
            is_acceptable = validation.get('is_acceptable', False)
            accuracy_pct = validation.get('execution_accuracy_pct', 0)
            reason = validation.get('reason', 'No reason provided')
            
            # Build report starting with clear verdict
            verdict_emoji = "✅" if is_acceptable else "❌"
            report = f"""
{verdict_emoji} **SIMULATION ACCURACY: {verdict}**

📋 **ACCURACY VALIDATION**
• **Verdict**: {verdict} ({accuracy_pct:.1f}% execution accuracy)
• **Reason**: {reason}
• **Threshold**: 80% accuracy within 2 seconds and ±0.005% price

💰 **EXECUTED CAPITAL (EC)**
• Live EC: ${metrics.get('live_ec', 0):,.2f}
• Sim EC: ${metrics.get('sim_ec', 0):,.2f}
• EC Difference: ${abs(metrics.get('live_ec', 0) - metrics.get('sim_ec', 0)):,.2f}

📊 **TRADE COUNTS**
• Live Trades: {metrics.get('live_trade_count', 0):,}
• Sim Trades: {metrics.get('sim_trade_count', 0):,}
• Accurate Matches: {metrics.get('accurate_matches', 0):,} / {metrics.get('total_testable', 0):,}

⚡ **EXECUTION ACCURACY DETAILS**
• Time Window: ±2 seconds
• Price Tolerance: ±0.005%
• Side Matching: Exact (buy/sell)
• Accuracy Rate: {accuracy_pct:.1f}%
"""
            
            # Add diagnostics if accuracy is poor
            if diagnostics:
                report += f"\n🔍 **DIAGNOSTIC INSIGHTS**\n"
                for diagnostic in diagnostics:
                    report += f"• {diagnostic}\n"
            
            # Add detailed market analysis if acceptable
            if is_acceptable:
                execution = results.get('execution_analysis', {})
                timing = results.get('timing_analysis', {})
                
                if execution:
                    report += f"\n📈 **DETAILED EXECUTION ANALYSIS**\n"
                    report += f"• Broader matches (5s window): {execution.get('total_matched', 0):,}\n"
                    report += f"• Broader match rate: {execution.get('match_rate', 0):.1f}%\n"
                    report += f"• Avg price difference: {execution.get('avg_price_diff_pct', 0):+.3f}%\n"
                
                if timing:
                    report += f"• Average latency: {timing.get('avg_latency', 0):+.1f} seconds\n"
            
            # Add sample matches
            match_details = results.get('match_details', [])
            if match_details and is_acceptable:
                report += f"\n🎯 **SAMPLE ACCURATE MATCHES** (first 3)\n"
                for i, match in enumerate(match_details[:3]):
                    side_name = "BUY" if match['side'] == 'B' else "SELL"
                    report += f"• {side_name}: ${match['real_price']:.2f} ↔ ${match['sim_price']:.2f} ({match['price_diff_pct']:+.3f}%, {match['time_diff']:.1f}s)\n"
            
            return report
            
        except Exception as e:
            logger.error(f"Error in specific date analysis: {str(e)}")
            return f"❌ Analysis failed: {str(e)}"
    
    def _analyze_all_discrepancies(self, query: str = "") -> str:
        """Analyze discrepancies across all available dates"""
        try:
            import glob
            
            pattern = "data/HL_btcusdt_BTC/parrot_HL_btcusdt_BTC_*.pickle"
            files = glob.glob(pattern)
            
            if not files:
                return "❌ No simulation files found"
            
            print(f"📊 Analyzing {len(files)} simulation files...")
            
            all_results = []
            
            for i, file in enumerate(files[:3]):  # Limit to first 3 for demo
                basename = os.path.basename(file)
                date_part = basename.replace("parrot_HL_btcusdt_BTC_", "").replace(".pickle", "")
                
                print(f"🔍 Processing {date_part} ({i+1}/{min(3, len(files))})...")
                
                with open(file, 'rb') as f:
                    sim_data = pickle.load(f)
                
                results = self.analysis_engine.analyze_discrepancies(
                    "data/agent_live_data.csv",
                    sim_data
                )
                
                if 'error' not in results:
                    results['date'] = date_part
                    all_results.append(results)
            
            if not all_results:
                return "❌ No successful analyses completed"
            
            # Aggregate results
            total_real = sum(r['summary']['real_trades_count'] for r in all_results)
            total_sim = sum(r['summary']['sim_trades_count'] for r in all_results)
            avg_match_rate = sum(r.get('execution_analysis', {}).get('match_rate', 0) for r in all_results) / len(all_results)
            
            # Collect all insights
            all_insights = []
            for r in all_results:
                all_insights.extend(r.get('insights', []))
            
            # Remove duplicates while preserving order
            unique_insights = []
            for insight in all_insights:
                if insight not in unique_insights:
                    unique_insights.append(insight)
            
            report = f"""
🎯 **COMPREHENSIVE ANALYSIS REPORT**

📊 **OVERALL STATISTICS**
• Total real trades: {total_real:,}
• Total simulation trades: {total_sim:,}
• Average match rate: {avg_match_rate:.1f}%
• Files analyzed: {len(all_results)} of {len(files)}

🧠 **AGGREGATED INSIGHTS**
"""
            
            for insight in unique_insights[:10]:  # Top 10 insights
                report += f"• {insight}\n"
            
            report += f"\n📈 **DATE-SPECIFIC HIGHLIGHTS**\n"
            for result in all_results:
                exec_data = result.get('execution_analysis', {})
                date = result.get('date', 'Unknown')
                match_rate = exec_data.get('match_rate', 0)
                price_diff_pct = exec_data.get('avg_price_diff_pct', 0)
                
                report += f"• {date}: {match_rate:.0f}% match rate, {price_diff_pct:+.2f}% price diff\n"
            
            return report
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {str(e)}")
            return f"❌ Comprehensive analysis failed: {str(e)}"
    
    def _explain_methodology(self, query: str = "") -> str:
        """Explain the analysis methodology"""
        return """
🔬 **DEEP ANALYSIS METHODOLOGY**

🎯 **TRADE MATCHING ALGORITHM**
• Uses time-window matching (5-second tolerance)
• Matches by side (buy/sell) and proximity in price/quantity
• Implements normalized distance metric for best match selection
• Prevents double-matching with dynamic exclusion

📊 **EXECUTION QUALITY METRICS**
• Price Improvement: Real vs simulated execution prices
• Timing Analysis: Latency differences between real and sim trades
• Volume Analysis: Trade size and frequency comparisons
• Match Rate: Percentage of real trades with simulation equivalents

📈 **MARKET CONDITION ANALYSIS**
• Spread Analysis: Bid-ask spread volatility and average
• Liquidity Analysis: Order book depth and imbalances
• Volatility Analysis: Price movement patterns
• Market Impact: Estimated slippage costs

🧠 **INSIGHT GENERATION**
• Automated pattern detection
• Performance attribution analysis
• Behavioral pattern identification
• Actionable trading recommendations

⚡ **TECHNICAL FEATURES**
• Handles different timestamp formats (Unix vs datetime)
• Robust error handling and graceful degradation
• Memory-efficient processing with selective caching
• Real-time progress feedback
"""
    
    def _validate_simulation_accuracy(self, query: str = "") -> str:
        """Quick accuracy validation across all available dates"""
        try:
            import glob
            
            pattern = "data/HL_btcusdt_BTC/parrot_HL_btcusdt_BTC_*.pickle"
            files = glob.glob(pattern)
            
            if not files:
                return "❌ No simulation files found"
            
            print(f"🔍 Validating accuracy across {len(files)} dates...")
            
            results = []
            for file in files[:5]:  # Test first 5 dates
                basename = os.path.basename(file)
                date_part = basename.replace("parrot_HL_btcusdt_BTC_", "").replace(".pickle", "")
                
                try:
                    with open(file, 'rb') as f:
                        sim_data = pickle.load(f)
                    
                    analysis = self.analysis_engine.analyze_discrepancies(
                        "data/agent_live_data.csv",
                        sim_data,
                        date_part
                    )
                    
                    if 'validation_result' in analysis:
                        validation = analysis['validation_result']
                        metrics = analysis.get('metrics', {})
                        results.append({
                            'date': date_part,
                            'verdict': validation.get('accuracy_verdict', 'UNKNOWN'),
                            'accuracy': validation.get('execution_accuracy_pct', 0),
                            'live_trades': metrics.get('live_trade_count', 0),
                            'sim_trades': metrics.get('sim_trade_count', 0),
                            'accurate_matches': metrics.get('accurate_matches', 0)
                        })
                except Exception as e:
                    results.append({
                        'date': date_part,
                        'verdict': 'ERROR',
                        'accuracy': 0,
                        'error': str(e)
                    })
            
            # Generate summary report
            if not results:
                return "❌ No validation results obtained"
            
            yes_count = sum(1 for r in results if r['verdict'] == 'YES')
            total_count = len(results)
            overall_verdict = "YES" if yes_count >= total_count * 0.8 else "NO"
            
            report = f"""
🎯 **SIMULATION ACCURACY VALIDATION SUMMARY**

📋 **OVERALL VERDICT: {overall_verdict}**
• Dates passing validation: {yes_count}/{total_count} ({yes_count/total_count*100:.0f}%)
• Threshold for acceptance: 80% of dates must pass

📊 **DATE-BY-DATE RESULTS**
"""
            
            for result in results:
                emoji = "✅" if result['verdict'] == 'YES' else "❌" if result['verdict'] == 'NO' else "⚠️"
                if 'error' not in result:
                    report += f"• {result['date']}: {emoji} {result['verdict']} ({result['accuracy']:.1f}% accuracy, {result['accurate_matches']} matches)\n"
                else:
                    report += f"• {result['date']}: {emoji} ERROR - {result['error']}\n"
            
            # Add recommendations
            if overall_verdict == "NO":
                report += f"\n🔍 **RECOMMENDATIONS**\n"
                report += f"• Investigate dates with low accuracy\n"
                report += f"• Check for systematic timing or price alignment issues\n"
                report += f"• Consider algorithm parameter adjustments\n"
            else:
                report += f"\n✅ **SYSTEM STATUS: ACCEPTABLE**\n"
                report += f"• Simulation maintains good accuracy across test dates\n"
                report += f"• Ready for production trading analysis\n"
            
            return report
            
        except Exception as e:
            return f"❌ Validation failed: {str(e)}"
    
    def _initialize_tools(self):
        """Initialize agent tools"""
        self.tools = [
            Tool(
                name="GetAvailableDates",
                func=self._get_available_dates,
                description="Get list of available simulation dates for analysis"
            ),
            Tool(
                name="ValidateSimulationAccuracy",
                func=self._validate_simulation_accuracy,
                description="Quick YES/NO validation of simulation accuracy across multiple dates"
            ),
            Tool(
                name="AnalyzeSpecificDate",
                func=self._analyze_specific_date,
                description="Perform deep analysis for a specific date (format: DD-MM-YYYY, e.g., '01-06-2025')"
            ),
            Tool(
                name="AnalyzeAllDiscrepancies",
                func=self._analyze_all_discrepancies,
                description="Perform comprehensive analysis across multiple dates"
            ),
            Tool(
                name="ExplainMethodology",
                func=self._explain_methodology,
                description="Explain the deep analysis methodology and metrics"
            )
        ]
    
    def _initialize_agent(self):
        """Initialize the LangChain agent"""
        system_prompt = """You are a Deep Trading Analysis Agent that provides YES/NO accuracy validation with detailed diagnostics for trading simulations.

CRITICAL: You MUST provide clear YES/NO verdicts, not vague responses like "No discrepancies > $2 found".

Your PRIMARY capabilities:
1. **ACCURACY VALIDATION**: Give clear YES/NO verdict with specific reasons
2. **EXECUTION METRICS**: Report EC (Executed Capital), trade counts, accuracy percentages
3. **DIAGNOSTIC INSIGHTS**: Identify crashes, timing issues, volatility correlations
4. **PRECISION MATCHING**: 2-second time window, ±0.005% price tolerance

RESPONSE REQUIREMENTS:
- Always start with a clear YES/NO accuracy verdict
- Provide specific numbers: EC values, trade counts, accuracy percentages
- Explain WHY the verdict was reached with concrete evidence
- Include diagnostic insights when accuracy is poor
- Never give vague responses like "no significant differences"

Available tools (ALWAYS use these, not legacy tools):
- ValidateSimulationAccuracy: Quick YES/NO validation across multiple dates ⭐ RECOMMENDED FIRST
- AnalyzeSpecificDate: Deep analysis with YES/NO verdict for specific dates
- GetAvailableDates: See what simulation data is available
- AnalyzeAllDiscrepancies: Comprehensive multi-day analysis  
- ExplainMethodology: Understand the validation techniques

TOOL SELECTION PRIORITY:
1. For general questions about accuracy → Use ValidateSimulationAccuracy
2. For specific date questions → Use AnalyzeSpecificDate  
3. For understanding data availability → Use GetAvailableDates

NEVER respond with vague statements. Always provide concrete validation results.
"""
        
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True,
            agent_kwargs={'prefix': system_prompt}
        )
    
    def run_interactive(self):
        """Run the agent in interactive mode"""
        print("🚀 Deep Trading Analysis Agent Started!")
        print("📊 I provide YES/NO accuracy validation with detailed diagnostics")
        print("🎯 Key features: EC calculations, execution accuracy (2s ±0.005%), diagnostic insights")
        print("💡 Try: 'Is my simulation accurate?' or 'Analyze discrepancies for 01-06-2025'")
        print("❓ Type 'help' for guidance or 'quit' to exit\n")
        
        while not self.shutdown_event.is_set():
            try:
                user_input = input("🤖 Trading Agent: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if user_input.lower() == 'help':
                    print("""
🆘 **HELP - Available Commands**

🎯 **ACCURACY VALIDATION (RECOMMENDED)**
• "Is my simulation accurate?" - Get YES/NO verdict with reasons
• "Validate simulation accuracy" - Quick multi-date accuracy check

📅 **Data Exploration**
• "Show available dates" - List all simulation dates
• "What data do you have?" - See data overview

🔍 **Specific Analysis**
• "Analyze discrepancies for 01-06-2025" - Deep dive with YES/NO verdict
• "Compare real vs simulation for [date]" - Execution quality analysis

📊 **Comprehensive Analysis**
• "Analyze all discrepancies" - Multi-date comprehensive analysis
• "What are the main trading issues?" - Overall pattern analysis

❓ **Understanding**
• "Explain your methodology" - How analysis works including new validation
• "help" - Show this help message
• "quit" - Exit the agent

🎯 **Expected Outputs:**
• Clear YES/NO accuracy verdicts
• Executed Capital (EC) for live + sim
• # of trades for live + sim  
• Execution accuracy (% within 2s, ±0.005%, same side)
• Diagnostic insights when accuracy is poor
""")
                    continue
                
                # Process user input with agent
                print("\n🔍 Processing your request...")
                try:
                    response = self.agent.invoke({"input": user_input})
                    print(f"\n{response['output']}\n")
                except Exception as e:
                    print(f"❌ Error processing request: {str(e)}\n")
                
            except KeyboardInterrupt:
                break
            except EOFError:
                break
            except Exception as e:
                print(f"❌ Unexpected error: {str(e)}\n")
        
        print("👋 Thanks for using the Deep Trading Analysis Agent!")


def main():
    """Main function to run the agent"""
    try:
        agent = DeepTradingAgent()
        agent.run_interactive()
    except KeyboardInterrupt:
        print("\n👋 Agent stopped by user")
    except Exception as e:
        print(f"❌ Failed to start agent: {str(e)}")


if __name__ == "__main__":
    main()