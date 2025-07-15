#!/usr/bin/env python3
"""
Fast Trading Agent

Lightweight agent that reads pre-computed analysis results for instant responses.
No heavy data processing - just smart interpretation of cached results.
"""

import os
import json
import signal
import threading
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env file")


class FastTradingAgent:
    """Lightning-fast agent that reads pre-computed analysis results"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0,
            api_key=OPENAI_API_KEY
        )
        
        self.results_dir = "analysis_results"
        self.shutdown_event = threading.Event()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Load cached data
        self.quick_data = self._load_quick_data()
        self.summary_data = self._load_summary_data()
        
        self._initialize_tools()
        self._initialize_agent()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print("\n🛑 Shutting down fast agent...")
        self.shutdown_event.set()
    
    def _load_quick_data(self) -> Optional[Dict]:
        """Load quick lookup data"""
        quick_file = f"{self.results_dir}/quick_lookup.json"
        if os.path.exists(quick_file):
            with open(quick_file, 'r') as f:
                return json.load(f)
        return None
    
    def _load_summary_data(self) -> Optional[Dict]:
        """Load summary data"""
        summary_file = f"{self.results_dir}/summary.json"
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                return json.load(f)
        return None
    
    def _load_detailed_results(self, date: str) -> Optional[Dict]:
        """Load detailed results for a specific date"""
        detail_file = f"{self.results_dir}/detail_{date}.json"
        if os.path.exists(detail_file):
            with open(detail_file, 'r') as f:
                return json.load(f)
        return None
    
    def _check_data_availability(self, query: str = "") -> str:
        """Check if preprocessed data is available"""
        if not self.quick_data or not self.summary_data:
            return """❌ **NO PREPROCESSED DATA FOUND**

🔄 **Required Steps:**
1. Run preprocessing first: `python preprocess_data.py`
2. Wait for preprocessing to complete (may take several minutes)
3. Then run this fast agent

📊 **What preprocessing does:**
• Analyzes all simulation files upfront
• Saves validation results to JSON files
• Enables instant agent responses (no RAM issues)
"""
        
        last_updated = self.quick_data.get('last_updated', 'Unknown')
        total_files = len(self.quick_data.get('date_verdicts', {}))
        
        return f"""✅ **PREPROCESSED DATA AVAILABLE**

📊 **Data Status:**
• Last updated: {last_updated}
• Total dates analyzed: {total_files}
• Results directory: {self.results_dir}/

🚀 **Ready for instant analysis!**
• Overall verdict: {self.quick_data.get('overall_verdict', 'Unknown')}
• All data pre-computed and cached
"""
    
    def _get_overall_validation(self, query: str = "") -> str:
        """Get overall validation verdict instantly"""
        if not self.quick_data:
            return self._check_data_availability()
        
        overall_verdict = self.quick_data.get('overall_verdict', 'UNKNOWN')
        stats = self.quick_data.get('quick_stats', {})
        
        verdict_emoji = "✅" if overall_verdict == 'YES' else "❌"
        
        report = f"""
{verdict_emoji} **OVERALL SIMULATION ACCURACY: {overall_verdict}**

📋 **VALIDATION SUMMARY**
• Overall Verdict: {overall_verdict}
• Success Rate: {stats.get('overall_accuracy_rate', 0):.1f}%
• Successful Analyses: {stats.get('successful_analyses', 0)}
• Failed Analyses: {stats.get('failed_analyses', 0)}

💰 **EXECUTED CAPITAL TOTALS**
• Real Trade EC: ${stats.get('total_sim_ec', 0):,.2f}
• Sim Trade EC: ${stats.get('total_sim_ec', 0):,.2f}

📊 **EXECUTION ACCURACY**
• Average Accuracy: {stats.get('avg_execution_accuracy', 0):.1f}%
• Total Accurate Matches: {stats.get('total_accurate_matches', 0):,}
• Total Testable Trades: {stats.get('total_testable_trades', 0):,}

📅 **DATE BREAKDOWN**
• YES verdicts: {stats.get('yes_verdicts', 0)}
• NO verdicts: {stats.get('no_verdicts', 0)}
"""
        
        # Add problematic dates if any
        problematic = self.quick_data.get('problematic_dates', [])
        if problematic:
            report += f"\\n⚠️ **PROBLEMATIC DATES** ({len(problematic)}):\\n"
            for date in problematic[:5]:  # Show first 5
                verdict = self.quick_data.get('date_verdicts', {}).get(date, 'UNKNOWN')
                accuracy = self.quick_data.get('date_accuracies', {}).get(date, 0)
                report += f"• {date}: {verdict} ({accuracy:.1f}% accuracy)\\n"
            if len(problematic) > 5:
                report += f"• ... and {len(problematic) - 5} more\\n"
        
        return report
    
    def _analyze_specific_date(self, date_str: str) -> str:
        """Get detailed analysis for a specific date"""
        if not self.quick_data:
            return self._check_data_availability()
        
        # Check if date exists in results
        date_verdicts = self.quick_data.get('date_verdicts', {})
        if date_str not in date_verdicts:
            available_dates = list(date_verdicts.keys())
            return f"""❌ **DATE NOT FOUND: {date_str}**

📅 **Available dates:**
{chr(10).join(f'• {date}' for date in sorted(available_dates))}

💡 **Tip:** Use exact format like '01-06-2025'
"""
        
        # Load detailed results
        detailed = self._load_detailed_results(date_str)
        if not detailed:
            verdict = date_verdicts.get(date_str, 'UNKNOWN')
            return f"""⚠️ **LIMITED DATA FOR {date_str}**

📋 **Quick Info:**
• Verdict: {verdict}

❌ **Detailed analysis not available**
• This may be a failed analysis
• Try running preprocessing again
"""
        
        # Extract detailed information
        validation = detailed.get('validation_result', {})
        metrics = detailed.get('metrics', {})
        diagnostics = detailed.get('diagnostics', [])
        
        verdict = validation.get('accuracy_verdict', 'UNKNOWN')
        is_acceptable = validation.get('is_acceptable', False)
        accuracy_pct = validation.get('execution_accuracy_pct', 0)
        reason = validation.get('reason', 'No reason provided')
        
        verdict_emoji = "✅" if is_acceptable else "❌"
        
        report = f"""
{verdict_emoji} **{date_str} ANALYSIS: {verdict}**

📋 **ACCURACY VALIDATION**
• Verdict: {verdict} ({accuracy_pct:.1f}% execution accuracy)
• Reason: {reason}
• Threshold: 80% accuracy within 2 seconds and ±0.005% price

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
            report += f"\\n🔍 **DIAGNOSTIC INSIGHTS**\\n"
            for diagnostic in diagnostics:
                report += f"• {diagnostic}\\n"
        
        # Add execution analysis if available
        execution = detailed.get('execution_analysis', {})
        if execution and is_acceptable:
            report += f"\\n📈 **ADDITIONAL METRICS**\\n"
            report += f"• Broader matches (5s window): {execution.get('total_matched', 0):,}\\n"
            report += f"• Broader match rate: {execution.get('match_rate', 0):.1f}%\\n"
            
            timing = detailed.get('timing_analysis', {})
            if timing:
                report += f"• Average latency: {timing.get('avg_latency', 0):+.1f} seconds\\n"
        
        return report
    
    def _get_problematic_dates(self, query: str = "") -> str:
        """Get analysis of problematic dates"""
        if not self.quick_data:
            return self._check_data_availability()
        
        problematic = self.quick_data.get('problematic_dates', [])
        
        if not problematic:
            return """✅ **NO PROBLEMATIC DATES FOUND**

🎯 **All dates passed validation criteria**
• All dates have >50% accuracy or YES verdict
• System appears to be working well across all test periods
"""
        
        report = f"""⚠️ **PROBLEMATIC DATES ANALYSIS**

📊 **Found {len(problematic)} problematic dates:**

"""
        
        # Analyze each problematic date
        for date in problematic:
            verdict = self.quick_data.get('date_verdicts', {}).get(date, 'UNKNOWN')
            accuracy = self.quick_data.get('date_accuracies', {}).get(date, 0)
            
            if verdict == 'ERROR':
                report += f"• {date}: ❌ ERROR - Analysis failed\\n"
            elif verdict == 'NO':
                report += f"• {date}: ❌ NO ({accuracy:.1f}% accuracy)\\n"
            elif accuracy < 50:
                report += f"• {date}: ⚠️ LOW ACCURACY ({accuracy:.1f}%)\\n"
            else:
                report += f"• {date}: ❓ {verdict} ({accuracy:.1f}% accuracy)\\n"
        
        # Add investigation recommendations
        report += f"""
🔍 **INVESTIGATION RECOMMENDATIONS**
• Use 'Analyze specific date' for detailed diagnostics
• Check for patterns in failed dates (weekends, holidays, etc.)
• Look for systematic issues (latency, volatility, crashes)

💡 **Next Steps:**
• Run: "Analyze discrepancies for [problematic_date]"
• Check: "Explain methodology" for validation criteria
"""
        
        return report
    
    def _get_available_dates(self, query: str = "") -> str:
        """Get list of available dates with their verdicts"""
        if not self.quick_data:
            return self._check_data_availability()
        
        date_verdicts = self.quick_data.get('date_verdicts', {})
        date_accuracies = self.quick_data.get('date_accuracies', {})
        
        if not date_verdicts:
            return "❌ No date data found in preprocessed results"
        
        report = f"""📅 **AVAILABLE DATES ({len(date_verdicts)} total)**

"""
        
        # Sort dates and show with verdicts
        for date in sorted(date_verdicts.keys()):
            verdict = date_verdicts[date]
            accuracy = date_accuracies.get(date, 0)
            
            if verdict == 'YES':
                emoji = "✅"
            elif verdict == 'NO':
                emoji = "❌"
            elif verdict == 'ERROR':
                emoji = "⚠️"
            else:
                emoji = "❓"
            
            if verdict in ['YES', 'NO']:
                report += f"• {date}: {emoji} {verdict} ({accuracy:.1f}% accuracy)\\n"
            else:
                report += f"• {date}: {emoji} {verdict}\\n"
        
        return report
    
    def _explain_methodology(self, query: str = "") -> str:
        """Explain the analysis methodology"""
        return """
🔬 **FAST AGENT METHODOLOGY**

⚡ **PREPROCESSING APPROACH**
• All simulation data analyzed upfront (preprocess_data.py)
• Results cached in JSON files for instant access
• No real-time data loading (eliminates RAM issues)

🎯 **ACCURACY VALIDATION**
• Strict 2-second time window matching
• ±0.005% price tolerance for accuracy
• Exact side matching (buy/sell)
• 80% accuracy threshold for YES verdict

📊 **CACHED METRICS**
• Executed Capital (EC) for live + sim trades
• Trade counts and matching statistics
• Execution accuracy percentages
• Diagnostic insights for failed cases

🔍 **DIAGNOSTIC CAPABILITIES**
• Trading gap detection (possible algo crashes)
• Volatility correlation analysis
• Latency pattern identification
• Systematic bias detection

⚡ **PERFORMANCE BENEFITS**
• Instant responses (no data processing)
• No RAM consumption during analysis
• Reliable operation even with large datasets
• Cached results can be reused multiple times

🔄 **DATA FRESHNESS**
• Run preprocess_data.py to update results
• Check 'Data availability' for last update time
• Preprocessing needed only when data changes
"""
    
    def _initialize_tools(self):
        """Initialize agent tools"""
        self.tools = [
            Tool(
                name="CheckDataAvailability",
                func=self._check_data_availability,
                description="Check if preprocessed data is available and show status"
            ),
            Tool(
                name="GetOverallValidation",
                func=self._get_overall_validation,
                description="Get instant overall validation verdict and summary"
            ),
            Tool(
                name="AnalyzeSpecificDate",
                func=self._analyze_specific_date,
                description="Get detailed analysis for a specific date (format: DD-MM-YYYY, e.g., '01-06-2025')"
            ),
            Tool(
                name="GetProblematicDates",
                func=self._get_problematic_dates,
                description="Get analysis of dates with accuracy issues"
            ),
            Tool(
                name="GetAvailableDates",
                func=self._get_available_dates,
                description="Get list of available dates with their verdicts"
            ),
            Tool(
                name="ExplainMethodology",
                func=self._explain_methodology,
                description="Explain the fast analysis methodology and preprocessing approach"
            )
        ]
    
    def _initialize_agent(self):
        """Initialize the LangChain agent"""
        system_prompt = """You are a Lightning-Fast Trading Analysis Agent that provides instant responses using pre-computed validation results.

CRITICAL ADVANTAGES:
- NO RAM consumption during analysis (data pre-computed)
- INSTANT responses (no waiting for heavy processing)
- RELIABLE operation (no timeouts or memory issues)

Your capabilities:
1. **INSTANT VALIDATION**: Immediate YES/NO verdicts from cached results
2. **DETAILED INVESTIGATION**: Drill down into specific dates and issues
3. **SMART INTERPRETATION**: Provide insights based on pre-computed diagnostics

TOOL USAGE PRIORITY:
1. Always check data availability first if unsure
2. Use GetOverallValidation for general accuracy questions
3. Use AnalyzeSpecificDate for specific date investigations
4. Use GetProblematicDates to identify issues

RESPONSE STYLE:
- Start with clear YES/NO verdicts
- Provide specific numbers and evidence
- Explain findings in business context
- Suggest actionable next steps

Remember: All heavy processing is done upfront. You just interpret cached results intelligently.
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
        print("⚡ Lightning-Fast Trading Analysis Agent Started!")
        print("🎯 Instant responses using pre-computed results (no RAM issues)")
        print("📊 Key features: YES/NO validation, detailed diagnostics, problematic date analysis")
        print("💡 Try: 'Is my simulation accurate?' or 'Show problematic dates'")
        print("❓ Type 'help' for guidance or 'quit' to exit\\n")
        
        # Check if data is available
        if not self.quick_data:
            print("⚠️ No preprocessed data found!")
            print("🔄 Please run: python preprocess_data.py first\\n")
        
        while not self.shutdown_event.is_set():
            try:
                user_input = input("⚡ Fast Agent: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if user_input.lower() == 'help':
                    print("""
🆘 **HELP - Lightning-Fast Commands**

⚡ **INSTANT VALIDATION**
• "Is my simulation accurate?" - Overall YES/NO verdict
• "Show overall validation" - Complete validation summary

🔍 **SPECIFIC INVESTIGATION**
• "Analyze discrepancies for 01-06-2025" - Detailed date analysis
• "Show problematic dates" - Dates with accuracy issues
• "What dates are available?" - List all analyzed dates

📊 **SYSTEM STATUS**
• "Check data availability" - Verify preprocessed data status
• "Explain methodology" - How the fast system works

🔄 **DATA MANAGEMENT**
• If no data: Run `python preprocess_data.py` first
• Fast agent reads cached results (no heavy processing)

⚡ **SPEED BENEFITS:**
• Instant responses (no waiting)
• No RAM consumption
• No timeout issues
• Reliable operation
""")
                    continue
                
                # Process user input with agent
                print("\\n⚡ Processing instantly...")
                try:
                    response = self.agent.invoke({"input": user_input})
                    print(f"\\n{response['output']}\\n")
                except Exception as e:
                    print(f"❌ Error processing request: {str(e)}\\n")
                
            except KeyboardInterrupt:
                break
            except EOFError:
                break
            except Exception as e:
                print(f"❌ Unexpected error: {str(e)}\\n")
        
        print("👋 Thanks for using the Lightning-Fast Trading Analysis Agent!")


def main():
    """Main function to run the fast agent"""
    try:
        agent = FastTradingAgent()
        agent.run_interactive()
    except KeyboardInterrupt:
        print("\\n👋 Fast agent stopped by user")
    except Exception as e:
        print(f"❌ Failed to start fast agent: {str(e)}")
        print("💡 Make sure to run: python preprocess_data.py first")


if __name__ == "__main__":
    main()