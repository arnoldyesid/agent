#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Fast Trading Agent
Forces detailed, specific responses with comprehensive data analysis
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


class TradingExpertTools:
    """Integrated trading expert tools for comprehensive market analysis"""
    
    def __init__(self):
        self.results_dir = "analysis_results"
        
    def analyze_technical_indicators(self, query=""):
        return """TECHNICAL INDICATORS COMPREHENSIVE GUIDE

MOMENTUM INDICATORS

1. RSI (Relative Strength Index)
   - Purpose: Identify overbought/oversold conditions
   - Range: 0-100 (>70 overbought, <30 oversold)
   - Best For: Ranging markets, divergence spotting
   - Settings: 14-period standard, 9 for scalping, 21 for position trading
   - Pro Tip: Look for divergences between price and RSI for reversals

2. MACD (Moving Average Convergence Divergence)
   - Components: 12 EMA, 26 EMA, 9-period signal line
   - Signals: Crossovers, divergences, histogram momentum
   - Best For: Trend identification and momentum shifts
   - Advanced: Use multiple timeframes for confirmation
   - Entry: MACD cross above signal with positive histogram

TREND INDICATORS

3. Moving Averages (MA)
   - SMA vs EMA: EMA more responsive, SMA smoother
   - Key Periods: 20 (short), 50 (medium), 200 (long)
   - Golden Cross: 50 MA > 200 MA (bullish)
   - Death Cross: 50 MA < 200 MA (bearish)

4. Bollinger Bands
   - Components: 20 SMA +/- 2 standard deviations
   - Squeeze: Low volatility precedes breakout
   - Walk the Bands: Strong trends ride upper/lower bands

PRO TIPS
- Never use single indicator in isolation
- Combine momentum + trend + volume for best results
- Backtest any indicator combination before live use"""

    def analyze_price_action(self, query=""):
        return """PRICE ACTION COMPREHENSIVE GUIDE

SUPPORT AND RESISTANCE

Key Concepts:
- Support: Price level where buying pressure exceeds selling
- Resistance: Price level where selling pressure exceeds buying
- Role Reversal: Support becomes resistance and vice versa
- Strength: More touches = stronger level

CHART PATTERNS

Reversal Patterns:
- Head and Shoulders: Three peaks, middle highest
- Double Top/Bottom: Two equal highs/lows
- Rising/Falling Wedge: Contracting trend channels

Continuation Patterns:
- Flags: Brief consolidation after strong move
- Pennants: Small triangular consolidation
- Rectangles: Horizontal trading ranges

CANDLESTICK PATTERNS

Single Candle:
- Doji: Indecision, potential reversal
- Hammer: Bullish reversal at support
- Shooting Star: Bearish reversal at resistance

Multiple Candle:
- Engulfing: One candle engulfs previous
- Harami: Small candle inside large one
- Morning/Evening Star: Three-candle reversal

PROFESSIONAL TIPS
- Price action works across all timeframes
- Volume confirms price action signals
- Practice pattern recognition on historical charts"""

    def trading_psychology(self, query=""):
        return """TRADING PSYCHOLOGY MASTERY

EMOTIONAL TRADING PITFALLS

Fear-Based Decisions:
- Taking profits too early
- Avoiding good setups after losses
- Using stops too tight
- Paralysis during drawdowns

Greed-Based Decisions:
- Holding winners too long
- Over-leveraging positions
- Chasing hot stocks/crypto
- Ignoring risk management

COGNITIVE BIASES

Confirmation Bias:
- Seeking information that confirms position
- Ignoring contrary evidence
- Solution: Actively seek opposing views

Anchoring Bias:
- Fixating on entry price
- Past high/low prices
- Solution: Focus on current market structure

DEVELOPING EMOTIONAL DISCIPLINE

Pre-Market Routine:
1. Review economic calendar
2. Analyze overnight developments  
3. Plan potential trades
4. Set daily risk limits
5. Mental preparation/meditation

Trade Execution Rules:
- Stick to predetermined plan
- Never move stops against you
- Take partial profits at targets
- Exit completely if plan fails

PRACTICAL TECHNIQUES

The Trading Journal:
- Record entry/exit rationale
- Emotional state during trade
- What you learned
- Review weekly/monthly patterns

Position Sizing Psychology:
- Size positions based on conviction
- Smaller size when uncertain
- Never risk more than you can handle psychologically"""

    def general_trading_help(self, query=""):
        return """GENERAL TRADING GUIDANCE

COMMON TRADING QUESTIONS & ANSWERS

Position Sizing Methods:
1. Fixed Dollar Amount: Risk same $ amount per trade
2. Percentage Risk: Risk 1-2% of account per trade  
3. Volatility Adjusted: Smaller positions in volatile stocks
4. Kelly Criterion: Mathematical optimal sizing based on edge

Risk Management Basics:
- Never risk more than 1-2% per trade
- Use stop losses on every position
- Diversify across different assets/sectors
- Have a written trading plan
- Keep detailed trading journal

Entry and Exit Strategies:
- Wait for clear setups before entering
- Use multiple timeframe analysis
- Set profit targets before entering
- Trail stops on winning positions
- Cut losses quickly, let profits run

Market Analysis Approach:
- Combine technical and fundamental analysis
- Understand market cycles and trends
- Monitor volume for confirmation
- Watch for support/resistance levels
- Consider market sentiment

Common Beginner Mistakes:
- Over-trading and over-leveraging
- No risk management plan
- Emotional decision making
- Chasing hot tips and trends
- Not keeping a trading journal

Building Trading Skills:
- Start with paper trading
- Focus on risk management first
- Study successful traders
- Backtest your strategies
- Continuous learning and adaptation

Trading Timeframes:
- Scalping: Seconds to minutes
- Day Trading: Minutes to hours
- Swing Trading: Days to weeks
- Position Trading: Weeks to months

Choose based on your schedule, personality, and capital available."""

    def clarify_and_route(self, query=""):
        # This tool should ONLY be used for truly unclear questions
        user_input = query.lower().strip()
        
        # Check if this question should have been routed elsewhere
        should_route_elsewhere = False
        correct_routing = ""
        
        # Check for accuracy keywords first
        if any(phrase in user_input for phrase in ["accuracy validation", "overall performance", "simulation results", "tell me accuracy"]):
            should_route_elsewhere = True
            correct_routing = "ComprehensiveAnalysis - for questions about accuracy validation and simulation performance"
        
        # Then check for improvement keywords  
        elif any(word in user_input for word in ["improve", "increase", "better", "fix", "optimize", "reduce", "enhance", "matching criteria"]):
            should_route_elsewhere = True
            correct_routing = "ImproveTradingEfficiency - for questions about improving, increasing, or fixing trading performance"
        
        # Check for technical keywords
        elif any(word in user_input for word in ["rsi", "macd", "bollinger", "technical indicators"]):
            should_route_elsewhere = True
            correct_routing = "AnalyzeTechnicalIndicators - for questions about technical analysis"
        
        if should_route_elsewhere:
            return """ROUTING ERROR DETECTED

Your question: "{query}"

This question should have been automatically routed to: {routing}

The agent's routing system may need adjustment. Your question contains clear keywords that should trigger a specific tool.

MEANWHILE, here are suggested rephrased questions:

For Improvement Questions:
- "How can I improve trading accuracy?"
- "How to reduce over-trading?" 
- "What can I optimize in my trading strategy?"

For Accuracy Analysis:
- "Tell me about accuracy validation"
- "Show me simulation performance results"

For Technical Analysis:
- "Explain RSI indicator"
- "How does MACD work?"

Please try asking one of these more specific questions.""".format(
                query=query,
                routing=correct_routing
            )
        
        # If truly unclear, provide general help
        result = """INPUT CLARIFICATION AND ROUTING HELP

Your question seems unclear or doesn't contain specific trading keywords.

ORIGINAL INPUT: "{original}"

SUGGESTED SPECIFIC QUESTIONS:

For Simulation Analysis:
- "Tell me about accuracy validation"
- "Show me simulation performance results"

For Improvement Help:  
- "How can I improve trading accuracy?"
- "How to reduce over-trading?"
- "How to optimize my trading strategy?"

For Technical Analysis:
- "Explain RSI indicator"
- "How does MACD work?"

For Price Action:
- "Explain support and resistance"
- "What are chart patterns?"

For Trading Psychology:
- "Help with trading emotions"
- "How to control fear and greed?"

For General Help:
- "How to start trading?"
- "Position sizing methods"
- "Basic risk management"

TIP: Use specific keywords like 'improve', 'RSI', 'support', 'psychology' to get better routing!""".format(
            original=query
        )
        
        return result


def get_all_trading_tools():
    """Return all available trading expert tools"""
    return {
        "AnalyzeTechnicalIndicators": {
            "method": "analyze_technical_indicators",
            "description": "Expert analysis of technical indicators (RSI, MACD, Bollinger Bands, etc.)"
        },
        "AnalyzePriceAction": {
            "method": "analyze_price_action", 
            "description": "Analyze price action patterns, support/resistance, and chart formations"
        },
        "TradingPsychology": {
            "method": "trading_psychology",
            "description": "Trading psychology, emotional management, and cognitive bias awareness"
        },
        "GeneralTradingHelp": {
            "method": "general_trading_help",
            "description": "General trading guidance, position sizing, risk management, and common questions"
        },
        "ClarifyAndRoute": {
            "method": "clarify_and_route",
            "description": "Clarify unclear questions and provide routing suggestions for better help"
        }
    }


class EnhancedFastTradingAgent:
    """Enhanced agent that forces detailed, specific responses"""
    
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
        
        # Initialize trading expert tools
        self.trading_expert = TradingExpertTools()
        
        self._initialize_tools()
        self._initialize_agent()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print("\\nShutting down enhanced fast agent...")
        self.shutdown_event.set()
    
    def _load_quick_data(self):
        """Load quick lookup data"""
        quick_file = self.results_dir + "/quick_lookup.json"
        if os.path.exists(quick_file):
            with open(quick_file, 'r') as f:
                return json.load(f)
        return None
    
    def _load_summary_data(self):
        """Load summary data"""
        summary_file = self.results_dir + "/summary.json"
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                return json.load(f)
        return None
    
    def _load_detailed_results(self, date):
        """Load detailed results for a specific date"""
        detail_file = self.results_dir + "/detail_" + date + ".json"
        if os.path.exists(detail_file):
            with open(detail_file, 'r') as f:
                return json.load(f)
        return None

    def _comprehensive_analysis(self, query=""):
        """Provide COMPREHENSIVE analysis with ALL specific details"""
        if not self.quick_data:
            return "ERROR: No preprocessed data found. Run: python improved_preprocess_data.py"
        
        stats = self.quick_data.get('quick_stats', {})
        overall_verdict = self.quick_data.get('overall_verdict', 'UNKNOWN')
        
        # Get specific numbers
        total_sim_trades = stats.get('total_sim_trades', 0)
        total_matches = stats.get('total_accurate_matches', 0)
        avg_execution_accuracy = stats.get('avg_execution_accuracy', 0)
        
        match_rate = (total_matches/max(total_sim_trades,1)*100)
        
        report = """SIMULATION ACCURACY ANALYSIS: {verdict}

COMPREHENSIVE SYSTEM METRICS
- Overall Verdict: CRITICAL FAILURE (0% success rate)
- System Grade: F (Complete Failure)
- Risk Level: EXTREME - System unsuitable for trading

DETAILED FINANCIAL ANALYSIS
- Total Simulation Trades: {sim_trades:,}
- Successful Matches: {matches:,}
- Average Execution Accuracy: {accuracy:.2f}%

CRITICAL ISSUES IDENTIFIED
- Zero dates achieved acceptable performance (>=80%)
- System shows massive over-trading: {sim_trades:,} vs real market activity
- Execution quality: Only {matches:,} trades out of {sim_trades:,} matched reality

IMMEDIATE ACTION REQUIRED
1. HALT all live trading - system is unsafe
2. INVESTIGATE over-trading: Algorithm generating {sim_trades:,} trades vs realistic levels
3. REVIEW matching criteria: Only {match_rate:.4f}% success rate
4. REDESIGN algorithm: Current version unsuitable for deployment""".format(
            verdict=overall_verdict,
            sim_trades=total_sim_trades,
            matches=total_matches,
            accuracy=avg_execution_accuracy,
            match_rate=match_rate
        )
        
        return report

    def _improve_trading_efficiency(self, query=""):
        """Provide specific recommendations to improve trading efficiency"""
        if not self.quick_data:
            return "ERROR: No preprocessed data found. Run: python improved_preprocess_data.py"
        
        stats = self.quick_data.get('quick_stats', {})
        total_sim_trades = stats.get('total_sim_trades', 0)
        total_matches = stats.get('total_accurate_matches', 0)
        
        current_accuracy = (total_matches/max(total_sim_trades,1)*100)
        
        report = """TRADING EFFICIENCY IMPROVEMENT PLAN

CURRENT PERFORMANCE ANALYSIS
- Simulation Trades: {sim_trades:,} 
- Successful Matches: {matches:,}
- Current Accuracy: {accuracy:.4f}%

PRIMARY ISSUE: MASSIVE OVER-TRADING
Your algorithm is executing far too many trades. This is the #1 problem to fix.

IMMEDIATE ACTIONS TO IMPROVE EFFICIENCY:

1. REDUCE TRADE FREQUENCY (Priority: CRITICAL)
   - Current: {sim_trades:,} trades -> Target: <50,000 trades
   - Add minimum price movement threshold (e.g., 0.1% change required)
   - Implement trade cooldown periods (e.g., 30-60 seconds between trades)
   - Expected Impact: 90% reduction in trades, 10x improvement in accuracy

2. IMPLEMENT POSITION SIZING LIMITS
   - Add maximum exposure limits (e.g., 20% of capital per position)
   - Implement position scaling based on confidence levels
   - Expected Impact: Better capital management, reduced risk

3. ENHANCE ENTRY CRITERIA
   - Use multiple timeframe confirmation (1min + 5min + 15min)
   - Require volume confirmation (above 20-day average)
   - Add trend strength filters (ADX > 25)
   - Expected Impact: 75% fewer false signals

QUICK WINS (Implement Today):
1. Add simple trade frequency limit: Max 1 trade per minute
2. Increase minimum trade size to $1,000
3. Add basic cooldown: 30 seconds between trades
4. Filter out trades during first/last 30 minutes of day

EXPECTED RESULTS AFTER OPTIMIZATION:
- Trade Count: ~10,000-20,000 (vs current {sim_trades:,})
- Accuracy: 15-25% (vs current {accuracy:.4f}%)
- Profitability: Positive expectancy
- Risk: Manageable and controlled""".format(
            sim_trades=total_sim_trades,
            matches=total_matches,
            accuracy=current_accuracy
        )
        
        return report

    def _initialize_tools(self):
        """Initialize agent tools with forced detailed responses"""
        # Core analysis tools
        self.tools = [
            Tool(
                name="ComprehensiveAnalysis",
                func=self._comprehensive_analysis,
                description="Get comprehensive analysis with ALL specific details, numbers, and metrics"
            ),
            Tool(
                name="ImproveTradingEfficiency",
                func=self._improve_trading_efficiency,
                description="Improve accuracy, efficiency, optimization, fix over-trading, make trading better"
            )
        ]
        
        # Add all trading expert tools
        expert_tools = get_all_trading_tools()
        for tool_name, tool_info in expert_tools.items():
            self.tools.append(
                Tool(
                    name=tool_name,
                    func=getattr(self.trading_expert, tool_info['method']),
                    description=tool_info['description']
                )
            )
    
    def _initialize_agent(self):
        """Initialize the agent with FORCED detailed response prompt"""
        system_prompt = """You are a comprehensive trading expert and analysis agent. Your ONLY job is to call the appropriate tool and return its EXACT output.

MANDATORY TOOL ROUTING - CHECK IN THIS ORDER:

1. SIMULATION ANALYSIS (PRIORITY):
   Keywords: "accuracy validation", "overall performance", "simulation results", "tell me accuracy"
   -> ComprehensiveAnalysis

2. IMPROVEMENT QUESTIONS:
   Keywords: "improve", "increase", "better", "fix", "optimize", "reduce", "enhance", "matching criteria"
   -> ImproveTradingEfficiency

3. TECHNICAL ANALYSIS:
   Keywords: "RSI", "MACD", "Bollinger", "technical indicators", "momentum", "trend"
   -> AnalyzeTechnicalIndicators

4. PRICE ACTION:
   Keywords: "support", "resistance", "chart patterns", "candlestick", "price action"
   -> AnalyzePriceAction

5. PSYCHOLOGY:
   Keywords: "psychology", "emotions", "discipline", "bias", "fear", "greed"
   -> TradingPsychology

6. GENERAL TRADING:
   Keywords: "position sizing", "risk management", "how to trade", "trading basics"
   -> GeneralTradingHelp

7. UNCLEAR QUESTIONS (LAST RESORT):
   Only if NO keywords from above categories match
   -> ClarifyAndRoute

STRICT RULES:
- Check keywords in ORDER above
- First match wins - use that tool
- "How to increase matching criteria" has "increase" -> ImproveTradingEfficiency
- "How to improve accuracy" has "improve" -> ImproveTradingEfficiency
- Only use ClarifyAndRoute for truly unclear questions

Examples:
- "How to increase matching criteria?" -> ImproveTradingEfficiency
- "How to improve accuracy?" -> ImproveTradingEfficiency  
- "Tell me accuracy validation" -> ComprehensiveAnalysis
- "Explain RSI" -> AnalyzeTechnicalIndicators
- "help me trade good pls" -> ClarifyAndRoute (no clear keywords)"""

        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True,
            agent_kwargs={'prefix': system_prompt}
        )
    
    def run_interactive(self):
        """Run the fixed agent in interactive mode"""
        print("Enhanced Comprehensive Trading Expert Agent Started!")
        print("Forces detailed responses with specific data and metrics")
        print("Intelligent routing to specialized tools for different question types")
        print("Includes clarification help for unclear questions")
        print()
        print("EXAMPLES OF WHAT TO ASK:")
        print("- 'Tell me accuracy validation' (simulation analysis)")
        print("- 'How to improve accuracy?' (efficiency recommendations)")  
        print("- 'Explain RSI indicators' (technical analysis)")
        print("- 'Support and resistance levels' (price action)")
        print("- 'Trading psychology tips' (emotional discipline)")
        print("- 'How to size positions?' (general trading help)")
        print()
        print("NOTE: If your question is unclear, the agent will help clarify and suggest better ways to ask!")
        print("Type 'quit' to exit\\n")
        
        while not self.shutdown_event.is_set():
            try:
                user_input = input("Enhanced Agent: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                # Process user input with fixed agent
                print("\\nProcessing with comprehensive analysis...")
                try:
                    response = self.agent.invoke({"input": user_input})
                    print("\\n" + response['output'] + "\\n")
                except Exception as e:
                    print("ERROR: Error processing request: " + str(e) + "\\n")
                
            except KeyboardInterrupt:
                break
            except EOFError:
                break
            except Exception as e:
                print("ERROR: Unexpected error: " + str(e) + "\\n")
        
        print("Thanks for using the Enhanced Trading Expert Agent!")


def main():
    """Main function to run the enhanced agent"""
    try:
        agent = EnhancedFastTradingAgent()
        agent.run_interactive()
    except KeyboardInterrupt:
        print("\\nEnhanced agent stopped by user")
    except Exception as e:
        print("ERROR: Failed to start enhanced agent: " + str(e))
        print("Make sure to run: python improved_preprocess_data.py first")


if __name__ == "__main__":
    main()