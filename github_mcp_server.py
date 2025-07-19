#!/usr/bin/env python3
"""
Trading Analysis MCP Server - GitHub Version
MCP server for trading analysis from GitHub
Compatible with existing file structure
"""

import json
import os
import sys
from typing import Dict, Any, Optional
import asyncio

# MCP imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    print("ERROR: Installing MCP...")
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'mcp'])
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent

class TradingAnalysisServer:
    def __init__(self):
        self.results_dir = "analysis_results"
        self.server = Server("trading-analysis-github")
        self.setup_tools()
    
    def setup_tools(self):
        """Setup MCP tools"""
        
        @self.server.list_tools()
        async def list_tools():
            return [
                Tool(
                    name="get_overall_analysis",
                    description="📊 Get comprehensive overall trading performance analysis",
                    inputSchema={"type": "object", "properties": {}}
                ),
                Tool(
                    name="get_date_analysis", 
                    description="📅 Get detailed analysis for a specific date (format: DD-MM-YYYY)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "date": {
                                "type": "string",
                                "description": "Date in DD-MM-YYYY format (e.g., 22-06-2025)"
                            }
                        },
                        "required": ["date"]
                    }
                ),
                Tool(
                    name="get_problematic_dates",
                    description="⚠️ Get the 10 worst performing dates",
                    inputSchema={"type": "object", "properties": {}}
                ),
                Tool(
                    name="get_trading_recommendations",
                    description="💡 Get specific recommendations to improve performance",
                    inputSchema={"type": "object", "properties": {}}
                ),
                Tool(
                    name="get_all_dates",
                    description="📋 View all analyzed dates with their accuracy scores",
                    inputSchema={"type": "object", "properties": {}}
                ),
                Tool(
                    name="get_summary_stats",
                    description="📈 Get summary statistics of the trading system",
                    inputSchema={"type": "object", "properties": {}}
                ),
                Tool(
                    name="analyze_spike_addition",
                    description="🎯 Analyze impact of SpikeAddition parameter",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "threshold": {
                                "type": "number",
                                "description": "SpikeAddition threshold value in dollars (default: 25)",
                                "default": 25
                            }
                        }
                    }
                ),
                Tool(
                    name="get_system_status",
                    description="🔧 Check server status and available files",
                    inputSchema={"type": "object", "properties": {}}
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict):
            try:
                if name == "get_overall_analysis":
                    result = self.get_overall_analysis()
                elif name == "get_date_analysis":
                    date = arguments.get("date")
                    result = self.get_date_analysis(date)
                elif name == "get_problematic_dates":
                    result = self.get_problematic_dates()
                elif name == "get_trading_recommendations":
                    result = self.get_trading_recommendations()
                elif name == "get_all_dates":
                    result = self.get_all_dates()
                elif name == "get_summary_stats":
                    result = self.get_summary_stats()
                elif name == "analyze_spike_addition":
                    threshold = arguments.get("threshold", 25)
                    result = self.analyze_spike_addition(threshold)
                elif name == "get_system_status":
                    result = self.get_system_status()
                else:
                    result = {"error": f"❌ Unknown tool: {name}"}
                
                return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]
                
            except Exception as e:
                error_result = {
                    "error": f"❌ Error executing {name}: {str(e)}",
                    "current_directory": os.getcwd(),
                    "available_files": self._list_json_files()
                }
                return [TextContent(type="text", text=json.dumps(error_result, indent=2, ensure_ascii=False))]

    def get_system_status(self):
        """Check system status"""
        key_files = {
            "quick_lookup.json": os.path.exists(f"{self.results_dir}/quick_lookup.json"),
            "summary.json": os.path.exists(f"{self.results_dir}/summary.json")
        }
        
        detail_files = [f for f in os.listdir(self.results_dir) if f.startswith('detail_') and f.endswith('.json')]
        
        return {
            "🟢 status": "System running correctly",
            "📁 results_directory": self.results_dir,
            "📍 current_directory": os.getcwd(),
            "📊 key_files": key_files,
            "📅 detail_files_found": len(detail_files),
            "📋 sample_detail_files": detail_files[:5],
            "✅ system_ready": all(key_files.values()) and len(detail_files) > 0
        }

    def get_overall_analysis(self):
        """Complete overall performance analysis"""
        data = self._load_quick_lookup()
        if not data:
            return self._error_no_data()
        
        stats = data.get('quick_stats', {})
        verdict = data.get('overall_verdict', 'UNKNOWN')
        
        total_sim = stats.get('total_sim_trades', 0)
        total_matches = stats.get('total_accurate_matches', 0)
        accuracy = stats.get('avg_execution_accuracy', 0)
        
        return {
            "🏆 overall_verdict": verdict,
            "📊 key_metrics": {
                "total_simulation_trades": f"{total_sim:,}",
                "successful_matches": f"{total_matches:,}",
                "average_accuracy_pct": f"{accuracy:.3f}%",
                "analyzed_dates": stats.get('successful_analyses', 0),
                "failed_analyses": stats.get('failed_analyses', 0)
            },
            "💰 financial_impact": {
                "total_simulated_capital": f"${stats.get('total_sim_ec', 0):,.2f}",
                "testable_trades": f"{stats.get('total_testable_trades', 0):,}",
                "match_rate": f"{(total_matches / max(total_sim, 1) * 100):.3f}%"
            },
            "📈 performance_breakdown": {
                "YES_verdicts": stats.get('yes_verdicts', 0),
                "PARTIAL_verdicts": stats.get('partial_verdicts', 0), 
                "NO_verdicts": stats.get('no_verdicts', 0),
                "overall_success_rate": f"{stats.get('overall_accuracy_rate', 0):.2f}%"
            },
            "⚠️ critical_issues": self._get_critical_issues(stats, verdict),
            "🎯 next_steps": [
                "🛑 HALT live trading immediately",
                "📉 Reduce trade frequency by 90%",
                "⚙️ Optimize matching criteria",
                "🛡️ Implement proper risk management"
            ],
            "📅 last_updated": data.get('last_updated', 'Unknown')
        }

    def get_date_analysis(self, date):
        """Detailed analysis for specific date"""
        if not date:
            return {"error": "❌ Please provide a date in DD-MM-YYYY format"}
        
        detail_data = self._load_detailed_results(date)
        if not detail_data:
            available_dates = self._get_available_dates()
            return {
                "error": f"❌ No data found for {date}",
                "📅 available_dates": available_dates[:10],
                "💡 suggestion": "Use 'get_all_dates' tool to see all available dates"
            }
        
        validation = detail_data.get('validation_result', {})
        metrics = detail_data.get('metrics', {})
        temporal = detail_data.get('temporal_analysis', {})
        
        live_trades = metrics.get('live_trade_count', 0)
        sim_trades = metrics.get('sim_trade_count', 0)
        accuracy = validation.get('execution_accuracy_pct', 0)
        
        return {
            "📅 date": date,
            "🎯 execution_verdict": validation.get('verdict', 'UNKNOWN'),
            "📊 accuracy": f"{accuracy:.3f}%",
            "🎓 grade": validation.get('grade', 'N/A'),
            "💹 trade_metrics": {
                "live_trades": f"{live_trades:,}",
                "simulation_trades": f"{sim_trades:,}",
                "accurate_matches": f"{metrics.get('accurate_matches', 0):,}",
                "sim_vs_live_ratio": f"{sim_trades/max(live_trades, 1):.1f}x",
                "overtrading_status": "🔴 SEVERE" if sim_trades > live_trades * 10 else "🟡 MODERATE" if sim_trades > live_trades * 3 else "🟢 NONE"
            },
            "💰 capital_analysis": {
                "live_executed_capital": f"${metrics.get('live_ec', 0):,.2f}",
                "simulation_executed_capital": f"${metrics.get('sim_ec', 0):,.2f}",
                "volume_ratio": f"{metrics.get('sim_ec', 0) / max(metrics.get('live_ec', 1), 1):.2f}x"
            },
            "⏰ temporal_coverage": {
                "real_trading_period": temporal.get('real_period', 'N/A'),
                "simulation_period": temporal.get('sim_period', 'N/A'),
                "overlap_hours": temporal.get('overlap_duration_hours', 0)
            },
            "📋 specific_recommendations": self._get_date_recommendations(accuracy, sim_trades, live_trades)
        }

    def get_problematic_dates(self):
        """Worst performing dates"""
        data = self._load_quick_lookup()
        if not data:
            return self._error_no_data()
        
        date_accuracies = data.get('date_accuracies', {})
        sorted_dates = sorted(date_accuracies.items(), key=lambda x: x[1])
        
        return {
            "⚠️ worst_performing_dates": [
                {
                    "date": date,
                    "accuracy_pct": f"{accuracy:.3f}%",
                    "status": "🔴 CRITICAL" if accuracy < 0.1 else "🟡 POOR" if accuracy < 0.2 else "🟠 WEAK"
                }
                for date, accuracy in sorted_dates[:10]
            ],
            "📊 summary": {
                "total_problematic_dates": len([a for a in date_accuracies.values() if a < 1.0]),
                "worst_accuracy": f"{min(date_accuracies.values()):.3f}%" if date_accuracies else "0%",
                "best_accuracy": f"{max(date_accuracies.values()):.3f}%" if date_accuracies else "0%",
                "average_accuracy": f"{sum(date_accuracies.values()) / len(date_accuracies):.3f}%" if date_accuracies else "0%"
            },
            "🔍 common_issues": [
                "Excessive overtrading (too many simulation trades)",
                "Poor timing alignment between live and sim",
                "Market condition mismatches",
                "Algorithm parameters need tuning"
            ]
        }

    def get_trading_recommendations(self):
        """Trading improvement recommendations"""
        data = self._load_quick_lookup()
        stats = data.get('quick_stats', {}) if data else {}
        
        total_sim_trades = stats.get('total_sim_trades', 0)
        total_matches = stats.get('total_accurate_matches', 0)
        accuracy = stats.get('avg_execution_accuracy', 0)
        
        return {
            "📊 current_performance": {
                "simulation_trades": f"{total_sim_trades:,}",
                "successful_matches": f"{total_matches:,}",
                "accuracy_pct": f"{accuracy:.3f}%"
            },
            "🎯 priority_actions": [
                {
                    "priority": 1,
                    "category": "🛑 Reduce Overtrading",
                    "current": f"{total_sim_trades:,} simulation trades",
                    "target": "<50,000 trades (90% reduction needed)",
                    "actions": [
                        "Increase minimum price movement threshold to 0.1%",
                        "Add 60-second cooldown between trades",
                        "Require multiple timeframe confirmation",
                        "Implement position limits (max 10 concurrent)"
                    ]
                },
                {
                    "priority": 2,
                    "category": "⚙️ Optimize Matching Criteria", 
                    "current": f"{accuracy:.2f}% accuracy",
                    "target": ">10% accuracy",
                    "actions": [
                        "Adjust SpikeAddition threshold (test $10, $15, $20)",
                        "Extend time window to 10 seconds",
                        "Add volume-weighted matching",
                        "Implement dynamic thresholds"
                    ]
                },
                {
                    "priority": 3,
                    "category": "📈 Enhance Signal Quality",
                    "actions": [
                        "Add trend strength filters (ADX > 25)",
                        "Require volume confirmation (>20-day avg)",
                        "Implement volatility adjustments",
                        "Add market session filters"
                    ]
                }
            ],
            "⚡ quick_wins": [
                "Trade frequency limit: Max 1 trade/minute",
                "Minimum trade size: $1,000",
                "Session filters: Avoid first/last 30min",
                "Basic stops: 2% max loss per trade"
            ],
            "📈 expected_improvements": {
                "trade_count_reduction": "90%",
                "accuracy_target": "10-25%",
                "capital_efficiency": "10x improvement",
                "risk_management": "Controlled exposure"
            },
            "🎯 success_metrics": [
                "Daily trade count <1,000",
                "Win rate >40%",
                "Average trade size >$5,000",
                "Maximum drawdown <5%"
            ]
        }

    def get_all_dates(self):
        """All analyzed dates with performance"""
        data = self._load_quick_lookup()
        if not data:
            return self._error_no_data()
        
        date_accuracies = data.get('date_accuracies', {})
        date_verdicts = data.get('date_verdicts', {})
        sorted_dates = sorted(date_accuracies.items(), key=lambda x: x[1], reverse=True)
        
        analyzed_dates = []
        for date, accuracy in sorted_dates:
            verdict = date_verdicts.get(date, 'UNKNOWN')
            analyzed_dates.append({
                "date": date,
                "accuracy_pct": f"{accuracy:.3f}%",
                "verdict": verdict,
                "status": "🟢 SUCCESS" if verdict == "YES" else "🟡 PARTIAL" if verdict == "PARTIAL" else "🔴 FAILED"
            })
        
        return {
            "📋 all_analyzed_dates": analyzed_dates,
            "📊 summary": {
                "total_dates": len(sorted_dates),
                "dates_above_1_pct": len([a for a in date_accuracies.values() if a > 1]),
                "dates_above_10_pct": len([a for a in date_accuracies.values() if a > 10]),
                "best_date": {
                    "date": sorted_dates[0][0] if sorted_dates else None,
                    "accuracy": f"{sorted_dates[0][1]:.3f}%" if sorted_dates else "0%"
                },
                "worst_date": {
                    "date": sorted_dates[-1][0] if sorted_dates else None,
                    "accuracy": f"{sorted_dates[-1][1]:.3f}%" if sorted_dates else "0%"
                },
                "average_accuracy": f"{sum(date_accuracies.values()) / len(date_accuracies):.3f}%" if date_accuracies else "0%"
            }
        }

    def get_summary_stats(self):
        """Summary statistics"""
        summary_data = self._load_summary()
        if not summary_data:
            return {"error": "❌ No summary data available"}
        
        return {
            "📊 real_trading_data": {
                "total_real_trades": f"{summary_data.get('real_trades_count', 0):,}",
                "real_trading_days": summary_data.get('real_trading_days', 0),
                "avg_trades_per_day": f"{summary_data.get('avg_trades_per_day', 0):.1f}"
            },
            "🔬 simulation_data": {
                "available_simulation_files": summary_data.get('sim_files_count', 0),
                "date_range": summary_data.get('date_range', 'N/A')
            },
            "⚡ processing_performance": {
                "last_updated": summary_data.get('last_updated', 'N/A'),
                "processing_time_seconds": summary_data.get('processing_time_seconds', 0),
                "analysis_efficiency": summary_data.get('analysis_efficiency', 'N/A')
            }
        }

    def analyze_spike_addition(self, threshold):
        """Analyze SpikeAddition parameter impact"""
        return {
            "🎯 current_threshold": f"${threshold}",
            "📊 impact_analysis": {
                "price_tolerance_window": f"Trades match if price difference ≤ ${threshold}",
                "matching_sensitivity": {
                    "strict": {"range": "$10-15", "description": "Stricter matching, fewer false positives"},
                    "moderate": {"range": "$20-25", "description": "Balanced approach (current)"},
                    "lenient": {"range": "$30-50", "description": "More matches but less precise"}
                }
            },
            "🧪 recommended_testing": [
                {"threshold": 10, "type": "Ultra-strict", "use_case": "High precision"},
                {"threshold": 15, "type": "Strict", "use_case": "Good balance"},
                {"threshold": 20, "type": "Moderate-strict", "use_case": "Standard trading"},
                {"threshold": 25, "type": "Current baseline", "use_case": "Current setting"},
                {"threshold": 30, "type": "Lenient", "use_case": "More matches"}
            ],
            "⚙️ dynamic_strategy": {
                "low_volatility": "$10-15",
                "high_volatility": "$25-35",
                "news_events": "$35-50",
                "weekend_low_volume": "$15-20"
            },
            "💡 optimization_recommendations": [
                "Implement volatility-based dynamic thresholds",
                "Test with historical data using different values",
                "Consider percentage-based thresholds (0.1-0.3%)",
                "Add time-of-day adjustments",
                "Monitor false positive/negative rates"
            ],
            "📈 expected_results": [
                "Better matching accuracy",
                "Reduced false signals",
                "Improved capital efficiency",
                "More realistic simulation results"
            ]
        }

    # Helper methods
    def _load_quick_lookup(self):
        """Load quick lookup data"""
        file_path = f"{self.results_dir}/quick_lookup.json"
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def _load_detailed_results(self, date):
        """Load detailed results for date"""
        file_path = f"{self.results_dir}/detail_{date}.json"
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def _load_summary(self):
        """Load summary data"""
        file_path = f"{self.results_dir}/summary.json"
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def _get_available_dates(self):
        """Get list of available dates"""
        try:
            files = os.listdir(self.results_dir)
            dates = []
            for file in files:
                if file.startswith('detail_') and file.endswith('.json'):
                    date = file.replace('detail_', '').replace('.json', '')
                    dates.append(date)
            return sorted(dates)
        except:
            return []

    def _list_json_files(self):
        """List available JSON files"""
        try:
            files = []
            for file in os.listdir(self.results_dir):
                if file.endswith('.json'):
                    files.append(file)
            return files
        except:
            return []

    def _error_no_data(self):
        """Standard error for no data"""
        return {
            "error": "❌ No analysis data found",
            "solution": "Run the analysis script first: python3 improved_preprocess_data.py",
            "files_searched": f"{self.results_dir}/quick_lookup.json",
            "available_files": self._list_json_files()
        }

    def _get_critical_issues(self, stats, verdict):
        """Generate critical issues list"""
        if verdict == "NO":
            return [
                "🔴 Zero dates achieved acceptable performance (≥80%)",
                "🔴 Massive overtrading detected",
                "🔴 Poor execution quality",
                "🔴 System unsuitable for live trading"
            ]
        return ["🟢 Performance within acceptable parameters"]

    def _get_date_recommendations(self, accuracy, sim_trades, live_trades):
        """Generate date-specific recommendations"""
        if accuracy < 1:
            return [
                f"🔴 CRITICAL: {accuracy:.2f}% accuracy is unacceptable",
                f"📉 Reduce simulation trades from {sim_trades:,} to <{live_trades*2:,}",
                "🔍 Investigate algorithm parameters for this date",
                "📊 Consider market conditions and volatility"
            ]
        elif accuracy < 10:
            return [
                f"🟡 POOR: {accuracy:.2f}% accuracy needs improvement",
                "⚠️ Moderate overtrading detected",
                "⚙️ Fine-tune matching parameters",
                "🔧 Add additional filters"
            ]
        else:
            return [
                f"🟢 ACCEPTABLE: {accuracy:.2f}% accuracy",
                "📊 Monitor and maintain current parameters",
                "📈 Consider scaling up carefully"
            ]

async def main():
    """Main function to run the MCP server"""
    trading_server = TradingAnalysisServer()
    
    async with stdio_server() as (read_stream, write_stream):
        await trading_server.server.run(
            read_stream,
            write_stream,
            trading_server.server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())