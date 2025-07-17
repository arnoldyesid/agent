#!/usr/bin/env python3
"""
Enhanced Fast Trading Agent

Provides deep, data-driven responses with comprehensive business intelligence.
All responses include specific metrics, financial analysis, and actionable recommendations.
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


class EnhancedFastTradingAgent:
    """Enhanced agent with deep, data-driven responses and business intelligence"""
    
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
        print("\n🛑 Shutting down enhanced fast agent...")
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
1. Run enhanced preprocessing: `python improved_preprocess_data.py`
2. Wait for preprocessing to complete (may take several minutes)
3. Then run this enhanced fast agent

📊 **What enhanced preprocessing provides:**
• Advanced analysis with granular accuracy reporting
• Business intelligence insights and recommendations
• Comprehensive diagnostics and financial analysis
• Enables instant deep responses (no RAM issues)
"""
        
        last_updated = self.quick_data.get('last_updated', 'Unknown')
        total_files = len(self.quick_data.get('date_verdicts', {}))
        
        return f"""✅ **ENHANCED PREPROCESSED DATA AVAILABLE**

📊 **Data Status:**
• Last updated: {last_updated}
• Total dates analyzed: {total_files}
• Analysis type: Advanced with business intelligence
• Results directory: {self.results_dir}/

🚀 **Ready for deep analysis!**
• Overall verdict: {self.quick_data.get('overall_verdict', 'Unknown')}
• Enhanced metrics and business insights available
• Financial analysis and risk assessment ready
"""

    def _get_comprehensive_validation(self, query: str = "") -> str:
        """Get comprehensive validation with detailed business intelligence"""
        if not self.quick_data:
            return self._check_data_availability()
        
        overall_verdict = self.quick_data.get('overall_verdict', 'UNKNOWN')
        stats = self.quick_data.get('quick_stats', {})
        
        verdict_emoji = "✅" if overall_verdict == 'YES' else "⚠️" if overall_verdict == 'PARTIAL' else "❌"
        
        # Performance grading
        success_rate = stats.get('overall_accuracy_rate', 0)
        if success_rate >= 90:
            grade = "A+ (Exceptional)"
            risk_level = "MINIMAL"
        elif success_rate >= 80:
            grade = "A (Excellent)"
            risk_level = "LOW"
        elif success_rate >= 70:
            grade = "B (Good)"
            risk_level = "MODERATE"
        elif success_rate >= 60:
            grade = "C (Fair)"
            risk_level = "HIGH"
        elif success_rate >= 50:
            grade = "D (Poor)"
            risk_level = "VERY HIGH"
        else:
            grade = "F (Critical Failure)"
            risk_level = "EXTREME"
        
        # Calculate financial metrics
        total_sim_ec = stats.get('total_sim_ec', 0)
        total_sim_trades = stats.get('total_sim_trades', 0)
        total_matches = stats.get('total_accurate_matches', 0)
        avg_trade_size = total_sim_ec / max(total_sim_trades, 1)
        
        # System reliability
        successful_days = stats.get('successful_analyses', 0)
        failed_days = stats.get('failed_analyses', 0)
        total_days = successful_days + failed_days
        system_uptime = (successful_days / max(total_days, 1)) * 100
        
        report = f"""{verdict_emoji} **COMPREHENSIVE TRADING SYSTEM ANALYSIS: {overall_verdict}**

🎯 **EXECUTIVE DASHBOARD**
• System Grade: {grade}
• Risk Level: {risk_level}
• Success Rate: {success_rate:.1f}% ({stats.get('yes_verdicts', 0)}/{successful_days} days)
• System Uptime: {system_uptime:.1f}% ({successful_days}/{total_days} days operational)

💰 **FINANCIAL PERFORMANCE ANALYSIS**
• Total Executed Capital: ${total_sim_ec:,.2f}
• Daily Average EC: ${total_sim_ec/max(successful_days, 1):,.2f}
• Total Trades Processed: {total_sim_trades:,}
• Average Trade Size: ${avg_trade_size:,.2f}
• Successful Executions: {total_matches:,} trades
• Execution Success Rate: {(total_matches/max(stats.get('total_testable_trades', 1), 1)*100):.1f}%

📊 **OPERATIONAL METRICS**
• Perfect Performance Days: {stats.get('yes_verdicts', 0)} days
• Partial Success Days: {stats.get('partial_verdicts', 0) if 'partial_verdicts' in stats else 0} days  
• Complete Failure Days: {stats.get('no_verdicts', 0)} days
• Average Execution Accuracy: {stats.get('avg_execution_accuracy', 0):.2f}%
• System Consistency: {(stats.get('yes_verdicts', 0)/max(successful_days, 1)*100):.1f}%"""
        
        # Risk Assessment
        if success_rate < 50:
            report += f"""

🚨 **CRITICAL RISK ALERT - IMMEDIATE ACTION REQUIRED**
• Failure Rate: {100-success_rate:.1f}% - UNACCEPTABLE FOR LIVE TRADING
• Financial Risk: EXTREME - Potential for significant losses
• System Status: CRITICAL FAILURE
• Capital at Risk: HIGH PROBABILITY OF LOSS

🛑 **URGENT RECOMMENDATIONS:**
1. SUSPEND all live trading immediately
2. Conduct comprehensive system audit
3. Review algorithm logic and parameters
4. Investigate connectivity and data feed issues
5. Implement fail-safe mechanisms
"""
        elif success_rate < 75:
            report += f"""

⚠️ **MODERATE RISK WARNING - CAUTION ADVISED**
• Risk Level: {risk_level} - System needs improvement
• Failure Rate: {100-success_rate:.1f}% - Above acceptable threshold
• Capital Protection: Reduce position sizes recommended

🔧 **IMPROVEMENT RECOMMENDATIONS:**
1. Reduce trading position sizes by 50%
2. Implement additional monitoring systems
3. Focus on problematic date patterns
4. Consider algorithm parameter optimization
"""
        else:
            report += f"""

✅ **LOW RISK ASSESSMENT - STRONG PERFORMANCE**
• System showing reliable performance
• Failure rate within acceptable limits
• Capital protection: Strong execution accuracy

🚀 **OPTIMIZATION OPPORTUNITIES:**
1. Consider scaling up successful strategies
2. Analyze top-performing dates for patterns
3. Maintain current risk management protocols
"""
        
        # Detailed failure analysis
        problematic = self.quick_data.get('problematic_dates', [])
        if problematic:
            report += f"""

🔍 **FAILURE PATTERN ANALYSIS** ({len(problematic)} problematic dates):"""
            
            # Categorize failures by time period
            early_failures = [d for d in problematic if d.startswith(('01-06', '02-06', '03-06', '04-06', '05-06'))]
            mid_failures = [d for d in problematic if d.startswith(('15-06', '16-06', '17-06', '18-06', '19-06'))]
            late_failures = [d for d in problematic if d.startswith(('25-06', '26-06', '27-06', '28-06', '29-06', '30-06'))]
            
            if early_failures:
                report += f"""
• Early Period Issues: {len(early_failures)} days - Likely initialization problems"""
            if mid_failures:
                report += f"""
• Mid Period Issues: {len(mid_failures)} days - System instability"""
            if late_failures:
                report += f"""
• Late Period Issues: {len(late_failures)} days - Possible system degradation"""
            
            # Show critical failures with specific dates
            report += f"""

📉 **WORST PERFORMING DATES:**"""
            for i, date in enumerate(problematic[:5]):
                verdict = self.quick_data.get('date_verdicts', {}).get(date, 'UNKNOWN')
                accuracy = self.quick_data.get('date_accuracies', {}).get(date, 0)
                report += f"""
{i+1}. {date}: {verdict} ({accuracy:.1f}% accuracy) - REQUIRES INVESTIGATION"""
        
        # Best performing dates
        excellent_dates = self.quick_data.get('excellent_dates', [])
        if excellent_dates:
            report += f"""

🏆 **TOP PERFORMING DATES** ({len(excellent_dates)} excellent days):"""
            for i, date in enumerate(excellent_dates[:5]):
                accuracy = self.quick_data.get('date_accuracies', {}).get(date, 0)
                report += f"""
{i+1}. {date}: EXCELLENT ({accuracy:.1f}% accuracy) - PATTERN FOR REPLICATION"""
        
        # Strategic action plan
        report += f"""

💡 **STRATEGIC ACTION PLAN**"""
        
        if success_rate >= 85:
            report += f"""
1. 🎯 **SCALE STRATEGY**: High-confidence system ready for increased allocation
2. 📊 **PATTERN ANALYSIS**: Study top performers for optimization
3. 🔄 **MAINTAIN MONITORING**: Continue current surveillance protocols
4. 📈 **GROWTH OPPORTUNITY**: Consider 20-30% position increase"""
        elif success_rate >= 70:
            report += f"""
1. 🔧 **STABILITY FOCUS**: Prioritize consistency over growth
2. 📉 **FAILURE REMEDIATION**: Address {stats.get('no_verdicts', 0)} failed dates
3. 📊 **MONITORING INCREASE**: Implement enhanced surveillance
4. ⚖️ **RISK MANAGEMENT**: Maintain conservative position sizing"""
        else:
            report += f"""
1. 🛠️ **CRITICAL OVERHAUL**: System requires fundamental improvements
2. 🔍 **ROOT CAUSE ANALYSIS**: Investigate {len(problematic)} failure dates
3. 🚫 **TRADING HALT**: Consider suspension until reliability improves
4. 💻 **SYSTEM REDESIGN**: Algorithm and infrastructure review needed"""
        
        return report

    def _analyze_specific_date_enhanced(self, date_str: str) -> str:
        """Enhanced specific date analysis with comprehensive business intelligence"""
        if not self.quick_data:
            return self._check_data_availability()
        
        # Check if date exists
        date_verdicts = self.quick_data.get('date_verdicts', {})
        if date_str not in date_verdicts:
            available_dates = list(date_verdicts.keys())
            return f"""❌ **DATE NOT FOUND: {date_str}**

📅 **Available dates:**
{chr(10).join(f'• {date}' for date in sorted(available_dates))}

💡 **Tip:** Use exact format like '22-06-2025'"""
        
        # Load detailed results
        detailed = self._load_detailed_results(date_str)
        if not detailed:
            verdict = date_verdicts.get(date_str, 'UNKNOWN')
            return f"""⚠️ **LIMITED DATA FOR {date_str}**

📋 **Quick Info:**
• Verdict: {verdict}

❌ **Detailed analysis not available**
• Run enhanced preprocessing to generate full analysis"""
        
        # Extract comprehensive information
        validation = detailed.get('validation_result', {})
        metrics = detailed.get('metrics', {})
        diagnostics = detailed.get('diagnostics', [])
        match_details = detailed.get('match_details', [])
        temporal = detailed.get('temporal_analysis', {})
        
        verdict = validation.get('accuracy_verdict', 'UNKNOWN')
        is_acceptable = validation.get('is_acceptable', False)
        accuracy_pct = validation.get('execution_accuracy_pct', 0)
        reason = validation.get('reason', 'No reason provided')
        
        verdict_emoji = "✅" if verdict == 'YES' else "⚠️" if verdict == 'PARTIAL' else "❌"
        
        # Financial calculations
        live_ec = metrics.get('live_ec', 0)
        sim_ec = metrics.get('sim_ec', 0)
        live_trades = metrics.get('live_trade_count', 0)
        sim_trades = metrics.get('sim_trade_count', 0)
        accurate_matches = metrics.get('accurate_matches', 0)
        total_testable = metrics.get('total_testable', 0)
        
        avg_live_trade = live_ec / max(live_trades, 1)
        avg_sim_trade = sim_ec / max(sim_trades, 1)
        match_rate = (accurate_matches / max(total_testable, 1)) * 100
        
        report = f"""{verdict_emoji} **COMPREHENSIVE DATE ANALYSIS: {date_str}**

🎯 **PERFORMANCE VERDICT: {verdict}**
• Execution Accuracy: {accuracy_pct:.1f}%
• Business Impact: {('POSITIVE' if is_acceptable else 'NEGATIVE')}
• Risk Level: {('LOW' if accuracy_pct >= 80 else 'MODERATE' if accuracy_pct >= 50 else 'HIGH')}
• Reason: {reason}

💰 **FINANCIAL ANALYSIS**
• Live Executed Capital: ${live_ec:,.2f}
• Simulation EC: ${sim_ec:,.2f}
• EC Variance: ${abs(live_ec - sim_ec):,.2f} ({abs((sim_ec/max(live_ec, 1) - 1) * 100):.1f}%)
• Live Avg Trade Size: ${avg_live_trade:,.2f}
• Sim Avg Trade Size: ${avg_sim_trade:,.2f}

📊 **EXECUTION METRICS**
• Total Live Trades: {live_trades:,}
• Total Sim Trades: {sim_trades:,}
• Trade Volume Ratio: {(sim_trades/max(live_trades, 1)):.1f}:1
• Successful Matches: {accurate_matches:,} of {total_testable:,} testable
• Match Success Rate: {match_rate:.1f}%
• Execution Efficiency: {('HIGH' if match_rate >= 90 else 'MODERATE' if match_rate >= 70 else 'LOW')}"""
        
        # Temporal analysis
        if temporal:
            report += f"""

⏰ **TEMPORAL ANALYSIS**
• Market Period: {temporal.get('real_period', 'N/A')}
• Simulation Period: {temporal.get('sim_period', 'N/A')}
• Overlap Duration: {temporal.get('overlap_duration_hours', 0):.1f} hours
• Data Quality: {('EXCELLENT' if temporal.get('overlap_duration_hours', 0) > 12 else 'GOOD' if temporal.get('overlap_duration_hours', 0) > 6 else 'LIMITED')}"""
        
        # Advanced pattern analysis
        if match_details and len(match_details) > 0:
            # Analyze timing patterns
            time_diffs = [abs(float(match.get('time_diff', 0))) for match in match_details]
            price_diffs = [float(match.get('price_diff_pct', 0)) for match in match_details]
            
            avg_time_diff = sum(time_diffs) / len(time_diffs) if time_diffs else 0
            avg_price_diff = sum(price_diffs) / len(price_diffs) if price_diffs else 0
            max_time_diff = max(time_diffs) if time_diffs else 0
            max_price_diff = max(price_diffs) if price_diffs else 0
            
            # Count execution patterns
            instant_executions = sum(1 for td in time_diffs if td == 0)
            fast_executions = sum(1 for td in time_diffs if td <= 1)
            precise_pricing = sum(1 for pd in price_diffs if pd <= 0.001)
            
            report += f"""

🎯 **EXECUTION PATTERN ANALYSIS** (Sample: {len(match_details)} trades)
• Average Timing Deviation: {avg_time_diff:.2f} seconds
• Average Price Deviation: {avg_price_diff:.4f}%
• Maximum Time Lag: {max_time_diff:.0f} seconds
• Maximum Price Variance: {max_price_diff:.4f}%
• Instant Executions: {instant_executions} ({(instant_executions/len(match_details)*100):.1f}%)
• Fast Executions (≤1s): {fast_executions} ({(fast_executions/len(match_details)*100):.1f}%)
• Precise Pricing (≤0.001%): {precise_pricing} ({(precise_pricing/len(match_details)*100):.1f}%)"""
        
        # Diagnostic insights
        if diagnostics:
            report += f"""

🔍 **DIAGNOSTIC INSIGHTS**"""
            for diagnostic in diagnostics:
                report += f"""
• {diagnostic}"""
        
        # Business intelligence and recommendations
        if is_acceptable:
            if accuracy_pct == 100:
                report += f"""

💼 **BUSINESS INTELLIGENCE: EXCEPTIONAL PERFORMANCE**
• Status: GOLD STANDARD DAY - Perfect execution achieved
• Confidence Level: MAXIMUM - All criteria exceeded
• Scaling Opportunity: HIGH - Ideal conditions for position increases
• Pattern Value: CRITICAL - Study for replication strategies

🚀 **STRATEGIC RECOMMENDATIONS:**
1. 📈 SCALE UP: Consider 25-50% position increase on similar conditions
2. 🔍 PATTERN STUDY: Analyze market conditions for replication
3. 📊 BENCHMARK: Use as performance standard for future dates
4. 💡 OPTIMIZATION: Investigate what made this day perfect"""
            else:
                report += f"""

💼 **BUSINESS INTELLIGENCE: STRONG PERFORMANCE**
• Status: HIGH QUALITY EXECUTION - Meets all standards
• Confidence Level: HIGH - Reliable system performance
• Risk Assessment: LOW - Safe for normal operations
• Improvement Potential: MODERATE - Minor optimizations possible

✅ **OPERATIONAL RECOMMENDATIONS:**
1. ✅ MAINTAIN: Continue current strategy and parameters
2. 📊 MONITOR: Track consistency across similar market conditions
3. 🔧 FINE-TUNE: Minor parameter adjustments for optimization
4. 📈 CONFIDENCE: High reliability for standard position sizing"""
        else:
            report += f"""

💼 **BUSINESS INTELLIGENCE: CRITICAL FAILURE**
• Status: SYSTEM BREAKDOWN - Complete execution failure
• Confidence Level: ZERO - Unreliable for trading
• Risk Assessment: EXTREME - High probability of losses
• Investigation Priority: URGENT - Immediate analysis required

🚨 **CRITICAL ACTION ITEMS:**
1. 🛑 HALT TRADING: Suspend operations on similar conditions
2. 🔍 ROOT CAUSE: Investigate algorithm and connectivity failures
3. 📋 AUDIT: Review system logs and error reports
4. 🔧 FIX: Address identified issues before resuming
5. 🧪 TEST: Validate fixes through simulation before live trading"""
        
        return report

    def _get_top_performing_dates(self, query: str = "") -> str:
        """Get analysis of best performing dates with success patterns"""
        if not self.quick_data:
            return self._check_data_availability()
        
        date_verdicts = self.quick_data.get('date_verdicts', {})
        date_accuracies = self.quick_data.get('date_accuracies', {})
        
        # Get all successful dates and sort by accuracy
        successful_dates = [
            (date, accuracy) for date, accuracy in date_accuracies.items()
            if date_verdicts.get(date) == 'YES' and accuracy > 0
        ]
        
        if not successful_dates:
            return """❌ **NO HIGH-PERFORMING DATES FOUND**

🔍 **Analysis:**
• No dates achieved acceptable performance criteria
• System appears to have systematic issues
• Comprehensive review and optimization needed"""
        
        successful_dates.sort(key=lambda x: x[1], reverse=True)
        
        report = f"""🏆 **TOP PERFORMING DATES ANALYSIS**

📊 **PERFORMANCE SUMMARY**
• Total Successful Dates: {len(successful_dates)}
• Success Rate: {(len(successful_dates) / len(date_verdicts) * 100):.1f}%
• Average Success Accuracy: {(sum(acc for _, acc in successful_dates) / len(successful_dates)):.1f}%

🥇 **TOP 10 PERFORMERS:**"""
        
        for i, (date, accuracy) in enumerate(successful_dates[:10]):
            # Load detailed data for financial metrics
            detailed = self._load_detailed_results(date)
            if detailed:
                metrics = detailed.get('metrics', {})
                matches = metrics.get('accurate_matches', 0)
                sim_trades = metrics.get('sim_trade_count', 0)
                sim_ec = metrics.get('sim_ec', 0)
                
                report += f"""
{i+1}. {date}: {accuracy:.1f}% accuracy
   • Trades Matched: {matches:,}
   • Total Sim Trades: {sim_trades:,}
   • Executed Capital: ${sim_ec:,.2f}"""
            else:
                report += f"""
{i+1}. {date}: {accuracy:.1f}% accuracy"""
        
        # Pattern analysis
        if len(successful_dates) >= 3:
            # Analyze patterns by day of week, time periods, etc.
            early_month = sum(1 for date, _ in successful_dates if date.startswith(('01-', '02-', '03-', '04-', '05-')))
            mid_month = sum(1 for date, _ in successful_dates if date.startswith(('10-', '11-', '12-', '13-', '14-', '15-', '16-', '17-', '18-', '19-', '20-')))
            late_month = sum(1 for date, _ in successful_dates if date.startswith(('25-', '26-', '27-', '28-', '29-', '30-')))
            
            report += f"""

🔍 **SUCCESS PATTERN ANALYSIS**
• Early Month Success: {early_month} dates
• Mid Month Success: {mid_month} dates  
• Late Month Success: {late_month} dates
• Peak Performance Period: {('Mid-month' if mid_month >= max(early_month, late_month) else 'Early month' if early_month >= late_month else 'Late month')}"""
        
        # Strategic insights
        avg_top_5 = sum(acc for _, acc in successful_dates[:5]) / min(5, len(successful_dates))
        
        report += f"""

💡 **STRATEGIC INSIGHTS**
• Top 5 Average: {avg_top_5:.1f}% accuracy
• Consistency Factor: {('HIGH' if len(successful_dates) >= 15 else 'MODERATE' if len(successful_dates) >= 10 else 'LOW')}
• Replication Potential: {('EXCELLENT' if avg_top_5 >= 95 else 'GOOD' if avg_top_5 >= 90 else 'MODERATE')}

🚀 **OPTIMIZATION RECOMMENDATIONS**
1. 📊 PATTERN STUDY: Analyze top performers for common market conditions
2. 🎯 PARAMETER TUNING: Optimize settings based on successful date characteristics
3. 📈 SCALING STRATEGY: Use top-performing patterns for position sizing
4. 🔄 MONITORING: Track when current conditions match historical successes"""
        
        return report

    def _get_problematic_dates_enhanced(self, query: str = "") -> str:
        """Enhanced analysis of problematic dates with root cause analysis"""
        if not self.quick_data:
            return self._check_data_availability()
        
        problematic = self.quick_data.get('problematic_dates', [])
        
        if not problematic:
            return """✅ **NO PROBLEMATIC DATES FOUND**

🎯 **Excellent News:**
• All dates passed validation criteria
• System demonstrates strong reliability
• No critical failures detected
• Ready for confident live trading"""
        
        date_verdicts = self.quick_data.get('date_verdicts', {})
        date_accuracies = self.quick_data.get('date_accuracies', {})
        
        # Categorize problems
        critical_failures = []  # 0% accuracy
        poor_performance = []   # >0% but <50%
        
        for date in problematic:
            accuracy = date_accuracies.get(date, 0)
            if accuracy == 0:
                critical_failures.append(date)
            else:
                poor_performance.append(date)
        
        report = f"""🚨 **COMPREHENSIVE FAILURE ANALYSIS**

📊 **PROBLEM SUMMARY**
• Total Problematic Dates: {len(problematic)}
• Critical Failures (0%): {len(critical_failures)}
• Poor Performance (<50%): {len(poor_performance)}
• Failure Rate: {(len(problematic) / len(date_verdicts) * 100):.1f}%
• System Reliability: {((len(date_verdicts) - len(problematic)) / len(date_verdicts) * 100):.1f}%"""
        
        # Critical failures analysis
        if critical_failures:
            report += f"""

💥 **CRITICAL FAILURES** ({len(critical_failures)} dates):"""
            
            # Pattern analysis
            early_failures = [d for d in critical_failures if d.startswith(('01-06', '02-06', '03-06', '04-06', '05-06'))]
            late_failures = [d for d in critical_failures if d.startswith(('25-06', '26-06', '27-06', '28-06', '29-06', '30-06'))]
            scattered_failures = [d for d in critical_failures if d not in early_failures and d not in late_failures]
            
            if early_failures:
                report += f"""
• Early Period Cluster: {len(early_failures)} dates - INITIALIZATION ISSUES
  Dates: {', '.join(early_failures)}"""
            
            if late_failures:
                report += f"""
• Late Period Cluster: {len(late_failures)} dates - SYSTEM DEGRADATION
  Dates: {', '.join(late_failures)}"""
                
            if scattered_failures:
                report += f"""
• Scattered Failures: {len(scattered_failures)} dates - RANDOM ISSUES
  Dates: {', '.join(scattered_failures)}"""
            
            # Detailed failure investigation
            report += f"""

🔍 **FAILURE INVESTIGATION:**"""
            for i, date in enumerate(critical_failures[:5]):
                detailed = self._load_detailed_results(date)
                if detailed:
                    diagnostics = detailed.get('diagnostics', [])
                    metrics = detailed.get('metrics', {})
                    sim_trades = metrics.get('sim_trade_count', 0)
                    
                    report += f"""
{i+1}. {date}: COMPLETE SYSTEM FAILURE
   • Sim Trades: {sim_trades:,} (algorithm running)
   • Testable Matches: 0 (no temporal overlap or matching criteria)"""
                    
                    if diagnostics:
                        report += f"""
   • Key Issues: {'; '.join(diagnostics[:2])}"""
                else:
                    report += f"""
{i+1}. {date}: COMPLETE SYSTEM FAILURE (no detailed data)"""
        
        # Poor performance analysis
        if poor_performance:
            report += f"""

⚠️ **POOR PERFORMANCE DATES** ({len(poor_performance)} dates):"""
            for date in poor_performance:
                accuracy = date_accuracies.get(date, 0)
                report += f"""
• {date}: {accuracy:.1f}% accuracy - BELOW THRESHOLD"""
        
        # Root cause analysis
        report += f"""

🔬 **ROOT CAUSE ANALYSIS**"""
        
        if len(early_failures) >= 3:
            report += f"""
• INITIALIZATION PROBLEM: {len(early_failures)} early failures suggest algorithm startup issues
  - Possible causes: Configuration errors, data feed delays, system warm-up issues
  - Solution: Improve initialization sequence and validation"""
        
        if len(late_failures) >= 3:
            report += f"""
• DEGRADATION PATTERN: {len(late_failures)} late failures suggest system deterioration
  - Possible causes: Memory leaks, performance degradation, connectivity issues
  - Solution: System maintenance, resource monitoring, restart procedures"""
        
        if len(scattered_failures) >= 3:
            report += f"""
• RANDOM FAILURES: {len(scattered_failures)} scattered failures suggest intermittent issues
  - Possible causes: Network instability, market data issues, external dependencies
  - Solution: Enhanced error handling, redundancy, monitoring systems"""
        
        # Financial impact assessment
        total_potential_ec = 0
        for date in problematic:
            detailed = self._load_detailed_results(date)
            if detailed:
                metrics = detailed.get('metrics', {})
                total_potential_ec += metrics.get('sim_ec', 0)
        
        report += f"""

💰 **FINANCIAL IMPACT ASSESSMENT**
• Potential Lost Execution Value: ${total_potential_ec:,.2f}
• Average Daily Impact: ${total_potential_ec/max(len(problematic), 1):,.2f}
• Risk Level: {('EXTREME' if len(critical_failures) >= 10 else 'HIGH' if len(critical_failures) >= 5 else 'MODERATE')}

🛠️ **URGENT ACTION PLAN**"""
        
        if len(critical_failures) >= 10:
            report += f"""
1. 🚨 EMERGENCY HALT: System too unreliable for live trading
2. 🔧 COMPLETE OVERHAUL: Algorithm and infrastructure review required
3. 📊 DATA AUDIT: Validate all data feeds and connections
4. 🧪 EXTENSIVE TESTING: Full system validation before restart"""
        elif len(critical_failures) >= 5:
            report += f"""
1. ⚠️ CAUTIOUS OPERATION: Reduce position sizes by 75%
2. 🔍 TARGETED FIXES: Address specific failure patterns identified
3. 📈 ENHANCED MONITORING: Real-time failure detection systems
4. 🔄 FREQUENT VALIDATION: Daily system health checks"""
        else:
            report += f"""
1. 📊 FOCUSED IMPROVEMENT: Address specific failure dates
2. 🛡️ RISK MITIGATION: Implement better error handling
3. 📈 MONITORING UPGRADE: Enhanced surveillance systems
4. 🔧 PREVENTIVE MAINTENANCE: Regular system optimization"""
        
        return report

    def _initialize_tools(self):
        """Initialize enhanced agent tools"""
        self.tools = [
            Tool(
                name="CheckDataAvailability",
                func=self._check_data_availability,
                description="Check if enhanced preprocessed data is available and show comprehensive status"
            ),
            Tool(
                name="GetComprehensiveValidation",
                func=self._get_comprehensive_validation,
                description="Get comprehensive validation with detailed business intelligence and financial analysis"
            ),
            Tool(
                name="AnalyzeSpecificDateEnhanced",
                func=self._analyze_specific_date_enhanced,
                description="Get enhanced detailed analysis for a specific date with business intelligence (format: DD-MM-YYYY)"
            ),
            Tool(
                name="GetProblematicDatesEnhanced",
                func=self._get_problematic_dates_enhanced,
                description="Get enhanced analysis of problematic dates with root cause analysis and action plans"
            ),
            Tool(
                name="GetTopPerformingDates",
                func=self._get_top_performing_dates,
                description="Get analysis of best performing dates with success patterns and optimization insights"
            ),
            Tool(
                name="GetAvailableDates",
                func=self._get_available_dates,
                description="Get list of available dates with their verdicts and performance metrics"
            )
        ]
    
    def _get_available_dates(self, query: str = "") -> str:
        """Get list of available dates with enhanced metrics"""
        if not self.quick_data:
            return self._check_data_availability()
        
        date_verdicts = self.quick_data.get('date_verdicts', {})
        date_accuracies = self.quick_data.get('date_accuracies', {})
        
        if not date_verdicts:
            return "❌ No date data found in preprocessed results"
        
        # Categorize dates
        excellent = []
        good = []
        poor = []
        failed = []
        
        for date in sorted(date_verdicts.keys()):
            verdict = date_verdicts[date]
            accuracy = date_accuracies.get(date, 0)
            
            if verdict == 'YES' and accuracy >= 95:
                excellent.append((date, accuracy))
            elif verdict == 'YES':
                good.append((date, accuracy))
            elif verdict == 'PARTIAL':
                poor.append((date, accuracy))
            else:
                failed.append((date, accuracy))
        
        report = f"""📅 **COMPREHENSIVE DATE PERFORMANCE OVERVIEW** ({len(date_verdicts)} total dates)

🏆 **EXCELLENT PERFORMANCE** ({len(excellent)} dates):"""
        for date, accuracy in excellent:
            report += f"""
• {date}: 🟢 {accuracy:.1f}% - GOLD STANDARD"""
        
        if good:
            report += f"""

✅ **GOOD PERFORMANCE** ({len(good)} dates):"""
            for date, accuracy in good[:10]:  # Show first 10
                report += f"""
• {date}: 🟡 {accuracy:.1f}% - ACCEPTABLE"""
            if len(good) > 10:
                report += f"""
• ... and {len(good) - 10} more good performance dates"""
        
        if poor:
            report += f"""

⚠️ **POOR PERFORMANCE** ({len(poor)} dates):"""
            for date, accuracy in poor:
                report += f"""
• {date}: 🟠 {accuracy:.1f}% - NEEDS IMPROVEMENT"""
        
        if failed:
            report += f"""

❌ **FAILED DATES** ({len(failed)} dates):"""
            for date, accuracy in failed[:5]:  # Show first 5 failures
                report += f"""
• {date}: 🔴 {accuracy:.1f}% - CRITICAL ISSUE"""
            if len(failed) > 5:
                report += f"""
• ... and {len(failed) - 5} more failed dates"""
        
        # Summary statistics
        total_success = len(excellent) + len(good)
        success_rate = (total_success / len(date_verdicts)) * 100
        
        report += f"""

📊 **PERFORMANCE SUMMARY**
• Success Rate: {success_rate:.1f}% ({total_success}/{len(date_verdicts)} dates)
• Excellence Rate: {(len(excellent) / len(date_verdicts) * 100):.1f}%
• Failure Rate: {(len(failed) / len(date_verdicts) * 100):.1f}%
• System Grade: {('A+' if success_rate >= 90 else 'A' if success_rate >= 80 else 'B' if success_rate >= 70 else 'C' if success_rate >= 60 else 'D' if success_rate >= 50 else 'F')}"""
        
        return report
    
    def _initialize_agent(self):
        """Initialize the enhanced LangChain agent"""
        system_prompt = """You are an Enhanced Lightning-Fast Trading Analysis Agent that provides comprehensive, data-driven responses with deep business intelligence.

CRITICAL CAPABILITIES:
- COMPREHENSIVE ANALYSIS: Provide detailed financial metrics, business intelligence, and strategic insights
- DATA-DRIVEN RESPONSES: Always include specific numbers, percentages, and concrete evidence
- BUSINESS INTELLIGENCE: Offer actionable recommendations and risk assessments
- FINANCIAL ANALYSIS: Include executed capital, trade metrics, and performance grading

Your enhanced capabilities:
1. **COMPREHENSIVE VALIDATION**: Detailed system analysis with financial metrics and risk assessment
2. **ENHANCED DATE INVESTIGATION**: Deep dive into specific dates with pattern analysis and business recommendations
3. **INTELLIGENT PATTERN RECOGNITION**: Identify success/failure patterns with actionable insights
4. **FINANCIAL INTELLIGENCE**: Provide executed capital analysis, risk assessment, and strategic recommendations

TOOL USAGE PRIORITY:
1. Use GetComprehensiveValidation for overall system analysis questions
2. Use AnalyzeSpecificDateEnhanced for detailed date investigations  
3. Use GetTopPerformingDates for best performer analysis
4. Use GetProblematicDatesEnhanced for failure analysis with root causes
5. Use GetAvailableDates for date listings with performance metrics

RESPONSE STANDARDS:
- Start with clear verdict and business impact
- Include specific financial metrics and percentages
- Provide risk assessment and confidence levels
- Offer concrete, actionable recommendations
- Explain business implications and strategic value
- Use performance grading (A+ to F) and risk levels (LOW to EXTREME)

NEVER provide vague responses like "consider investigating" - always give specific, actionable, data-driven insights with concrete numbers and business intelligence.
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
        """Run the enhanced agent in interactive mode"""
        print("⚡ Enhanced Lightning-Fast Trading Analysis Agent Started!")
        print("🎯 Comprehensive responses with deep business intelligence")
        print("📊 Financial analysis, risk assessment, and strategic recommendations")
        print("💡 Try: 'Is my simulation accurate?' or 'Analyze discrepancies for 22-06-2025'")
        print("❓ Type 'help' for guidance or 'quit' to exit\\n")
        
        # Check if data is available
        if not self.quick_data:
            print("⚠️ No preprocessed data found!")
            print("🔄 Please run: python improved_preprocess_data.py first\\n")
        
        while not self.shutdown_event.is_set():
            try:
                user_input = input("⚡ Enhanced Agent: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if user_input.lower() == 'help':
                    print("""
🆘 **ENHANCED AGENT CAPABILITIES**

⚡ **COMPREHENSIVE ANALYSIS**
• "Is my simulation accurate?" - Complete system analysis with business intelligence
• "Show overall performance" - Detailed financial and operational metrics

🔍 **DEEP DATE INVESTIGATION**
• "Analyze discrepancies for 22-06-2025" - Enhanced date analysis with patterns
• "Detailed analysis for [date]" - Comprehensive business intelligence for specific dates

📊 **PERFORMANCE INSIGHTS**
• "Show best performing dates" - Top performers with success patterns
• "Show problematic dates" - Failure analysis with root causes
• "What dates are available?" - Complete date listing with performance grades

🎯 **BUSINESS INTELLIGENCE FEATURES:**
• Financial impact analysis and risk assessment
• Performance grading (A+ to F) and confidence levels
• Strategic recommendations and action plans
• Pattern recognition and optimization insights
• Root cause analysis for failures
• Scaling and optimization opportunities

⚡ **RESPONSE QUALITY:**
• Specific metrics and concrete numbers
• Business implications and strategic value
• Actionable recommendations with priority levels
• Risk assessment and financial impact analysis
""")
                    continue
                
                # Process user input with enhanced agent
                print("\\n⚡ Processing with deep analysis...")
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
        
        print("👋 Thanks for using the Enhanced Lightning-Fast Trading Analysis Agent!")


def main():
    """Main function to run the enhanced fast agent"""
    try:
        agent = EnhancedFastTradingAgent()
        agent.run_interactive()
    except KeyboardInterrupt:
        print("\\n👋 Enhanced fast agent stopped by user")
    except Exception as e:
        print(f"❌ Failed to start enhanced fast agent: {str(e)}")
        print("💡 Make sure to run: python improved_preprocess_data.py first")


if __name__ == "__main__":
    main()