#!/usr/bin/env python3
"""
Trading Analysis Server - Hybrid MCP/HTTP with Ngrok
Supports both local MCP and remote HTTP via Ngrok tunnel
Compatible with Claude's MCP connector system
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional
import asyncio
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

# Ngrok integration
try:
    from pyngrok import ngrok
    NGROK_AVAILABLE = True
except ImportError:
    NGROK_AVAILABLE = False

# MCP imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

class TradingAnalysisHTTPHandler(BaseHTTPRequestHandler):
    """HTTP handler for ngrok/web access"""
    
    def __init__(self, trading_server, *args, **kwargs):
        self.trading_server = trading_server
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        try:
            parsed_url = urllib.parse.urlparse(self.path)
            path = parsed_url.path
            params = urllib.parse.parse_qs(parsed_url.query)

            # Route requests to analysis methods
            if path == "/" or path == "/health":
                response = {
                    "status": "healthy",
                    "server": "Trading Analysis (HTTP+MCP)",
                    "timestamp": datetime.now().isoformat(),
                    "mcp_available": MCP_AVAILABLE,
                    "ngrok_available": NGROK_AVAILABLE
                }
            elif path == "/overall_analysis":
                response = self.trading_server.get_overall_analysis()
            elif path == "/date_analysis":
                date = params.get('date', [None])[0]
                response = self.trading_server.get_date_analysis(date)
            elif path == "/summary_stats":
                response = self.trading_server.get_summary_stats()
            elif path == "/problematic_dates":
                response = self.trading_server.get_problematic_dates()
            elif path == "/trading_recommendations":
                response = self.trading_server.get_trading_recommendations()
            elif path == "/available_dates":
                response = self.trading_server.get_available_dates()
            elif path == "/spike_analysis":
                threshold = float(params.get('threshold', [25])[0])
                response = self.trading_server.analyze_spike_addition_impact(threshold)
            else:
                response = {
                    "error": "Endpoint not found",
                    "available_endpoints": [
                        "/health - Server health check",
                        "/overall_analysis - Complete analysis overview", 
                        "/date_analysis?date=DD-MM-YYYY - Specific date analysis",
                        "/summary_stats - Summary statistics",
                        "/problematic_dates - Worst performing dates",
                        "/trading_recommendations - Improvement suggestions",
                        "/available_dates - All analyzed dates",
                        "/spike_analysis?threshold=25 - SpikeAddition analysis"
                    ]
                }
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(json.dumps(response, indent=2).encode())
            
        except Exception as e:
            self.send_error_response(str(e))
    
    def do_POST(self):
        """Handle POST requests for MCP-style calls"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                post_data = self.rfile.read(content_length)
                try:
                    request_data = json.loads(post_data.decode('utf-8'))
                except:
                    request_data = {}
            else:
                request_data = {}
            
            path = self.path
            
            if path == "/mcp/call":
                tool_name = request_data.get('tool')
                arguments = request_data.get('arguments', {})
                response = self.handle_tool_call(tool_name, arguments)
            else:
                response = {"error": "Unknown POST endpoint", "path": path}
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response, indent=2).encode())
            
        except Exception as e:
            self.send_error_response(f"POST error: {str(e)}")
    
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def send_error_response(self, error_message: str):
        """Send error response"""
        self.send_response(500)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        error_response = {
            "error": error_message,
            "timestamp": datetime.now().isoformat()
        }
        self.wfile.write(json.dumps(error_response, indent=2).encode())

    def handle_tool_call(self, tool_name: str, arguments: dict):
        """Handle tool calls via HTTP"""
        try:
            if tool_name == "get_overall_analysis":
                return {"result": self.trading_server.get_overall_analysis()}
            elif tool_name == "get_date_analysis":
                date = arguments.get("date")
                return {"result": self.trading_server.get_date_analysis(date)}
            elif tool_name == "get_summary_stats":
                return {"result": self.trading_server.get_summary_stats()}
            elif tool_name == "get_problematic_dates":
                return {"result": self.trading_server.get_problematic_dates()}
            elif tool_name == "get_trading_recommendations":
                return {"result": self.trading_server.get_trading_recommendations()}
            elif tool_name == "analyze_spike_addition_impact":
                threshold = arguments.get("threshold", 25)
                return {"result": self.trading_server.analyze_spike_addition_impact(threshold)}
            elif tool_name == "get_available_dates":
                return {"result": self.trading_server.get_available_dates()}
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            return {"error": f"Tool execution failed: {str(e)}"}

    def log_message(self, format, *args):
        """Custom log format"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {format % args}")

class TradingAnalysisServer:
    def __init__(self):
        self.results_dir = "analysis_results"
        
        # Initialize MCP server if available
        if MCP_AVAILABLE:
            self.server = Server("trading-analysis")
            self.setup_tools()
        else:
            self.server = None

    def setup_tools(self):
        """Setup MCP tools"""
        if not self.server:
            return
            
        @self.server.list_tools()
        async def list_tools():
            return [
                Tool(
                    name="get_overall_analysis",
                    description="Get comprehensive overall trading analysis and performance metrics",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                Tool(
                    name="get_date_analysis", 
                    description="Get detailed analysis for a specific trading date",
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
                    name="get_summary_stats",
                    description="Get summary statistics for trading data",
                    inputSchema={
                        "type": "object", 
                        "properties": {},
                        "required": []
                    }
                ),
                Tool(
                    name="get_problematic_dates",
                    description="Get analysis of worst performing trading dates",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                Tool(
                    name="get_trading_recommendations", 
                    description="Get actionable trading improvement recommendations",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                Tool(
                    name="analyze_spike_addition_impact",
                    description="Analyze the impact of different SpikeAddition threshold values",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "threshold": {
                                "type": "number",
                                "description": "SpikeAddition threshold value in dollars (default: 25)",
                                "default": 25
                            }
                        },
                        "required": []
                    }
                ),
                Tool(
                    name="get_available_dates",
                    description="Get list of all available analysis dates with performance metrics", 
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
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
                elif name == "get_summary_stats":
                    result = self.get_summary_stats()
                elif name == "get_problematic_dates":
                    result = self.get_problematic_dates()
                elif name == "get_trading_recommendations":
                    result = self.get_trading_recommendations()
                elif name == "analyze_spike_addition_impact":
                    threshold = arguments.get("threshold", 25)
                    result = self.analyze_spike_addition_impact(threshold)
                elif name == "get_available_dates":
                    result = self.get_available_dates()
                else:
                    result = {"error": f"Unknown tool: {name}"}
                
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
                
            except Exception as e:
                error_result = {"error": f"Tool execution failed: {str(e)}"}
                return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

    # Analysis methods (your existing code)
    def get_overall_analysis(self) -> Dict[str, Any]:
        """Get comprehensive overall analysis"""
        quick_data = self._load_quick_lookup()
        if not quick_data:
            return {"error": "No analysis data found. Run: python3 improved_preprocess_data.py"}
        
        stats = quick_data.get('quick_stats', {})
        verdict = quick_data.get('overall_verdict', 'UNKNOWN')
        
        return {
            "overall_verdict": verdict,
            "key_metrics": {
                "total_simulation_trades": stats.get('total_sim_trades', 0),
                "successful_matches": stats.get('total_accurate_matches', 0),
                "average_execution_accuracy": stats.get('avg_execution_accuracy', 0),
                "analyzed_dates": stats.get('successful_analyses', 0),
                "failed_analyses": stats.get('failed_analyses', 0)
            },
            "performance_breakdown": {
                "yes_verdicts": stats.get('yes_verdicts', 0),
                "partial_verdicts": stats.get('partial_verdicts', 0),
                "no_verdicts": stats.get('no_verdicts', 0),
                "overall_success_rate": stats.get('overall_accuracy_rate', 0)
            },
            "financial_impact": {
                "total_simulated_capital": stats.get('total_sim_ec', 0),
                "testable_trades": stats.get('total_testable_trades', 0),
                "match_rate_pct": (stats.get('total_accurate_matches', 0) / max(stats.get('total_sim_trades', 1), 1) * 100)
            },
            "critical_findings": self._get_critical_issues(stats, verdict),
            "next_steps": self._get_recommendations_summary(),
            "last_updated": quick_data.get('last_updated', 'Unknown')
        }
    
    def get_date_analysis(self, date: Optional[str]) -> Dict[str, Any]:
        """Get detailed analysis for specific date"""
        if not date:
            return {"error": "Please provide a date in DD-MM-YYYY format"}
        
        detail_data = self._load_detailed_results(date)
        if not detail_data:
            return {"error": f"No analysis data found for {date}"}
        
        validation = detail_data.get('validation_result', {})
        metrics = detail_data.get('metrics', {})
        temporal = detail_data.get('temporal_analysis', {})
        match_details = detail_data.get('match_details', [])
        
        return {
            "date": date,
            "execution_verdict": validation.get('verdict', 'UNKNOWN'),
            "accuracy_pct": validation.get('execution_accuracy_pct', 0),
            "grade": validation.get('grade', 'N/A'),
            "trade_metrics": {
                "live_trades": metrics.get('live_trade_count', 0),
                "simulation_trades": metrics.get('sim_trade_count', 0),
                "accurate_matches": metrics.get('accurate_matches', 0),
                "total_testable": metrics.get('total_testable', 0)
            },
            "capital_analysis": {
                "live_executed_capital": metrics.get('live_ec', 0),
                "simulation_executed_capital": metrics.get('sim_ec', 0),
                "volume_ratio": metrics.get('sim_ec', 0) / max(metrics.get('live_ec', 1), 1)
            },
            "temporal_coverage": {
                "real_trading_period": temporal.get('real_period', 'N/A'),
                "simulation_period": temporal.get('sim_period', 'N/A'),
                "overlap_duration_hours": temporal.get('overlap_duration_hours', 0)
            },
            "match_quality": self._analyze_match_details(match_details),
            "recommendations": self._get_date_recommendations(validation, metrics)
        }
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics"""
        summary_data = self._load_summary()
        if not summary_data:
            return {"error": "No summary data available"}
        
        return {
            "real_trading_data": {
                "total_real_trades": summary_data.get('real_trades_count', 0),
                "real_trading_days": summary_data.get('real_trading_days', 0),
                "avg_trades_per_day": summary_data.get('avg_trades_per_day', 0)
            },
            "simulation_data": {
                "available_simulation_files": summary_data.get('sim_files_count', 0),
                "date_range": summary_data.get('date_range', 'N/A')
            },
            "processing_performance": {
                "last_updated": summary_data.get('last_updated', 'N/A'),
                "processing_time_seconds": summary_data.get('processing_time_seconds', 0),
                "analysis_efficiency": summary_data.get('analysis_efficiency', 'N/A')
            }
        }
    
    def get_problematic_dates(self) -> Dict[str, Any]:
        """Get problematic dates analysis"""
        quick_data = self._load_quick_lookup()
        if not quick_data:
            return {"error": "No analysis data found"}
        
        date_accuracies = quick_data.get('date_accuracies', {})
        problematic = quick_data.get('problematic_dates', [])
        
        # Sort by accuracy (worst first)
        sorted_dates = sorted(date_accuracies.items(), key=lambda x: x[1])
        
        return {
            "worst_performing_dates": [
                {
                    "date": date,
                    "accuracy_pct": accuracy,
                    "status": "CRITICAL" if accuracy < 0.1 else "POOR" if accuracy < 0.2 else "WEAK"
                }
                for date, accuracy in sorted_dates[:10]
            ],
            "problem_summary": {
                "total_problematic_dates": len(problematic),
                "worst_accuracy": min(date_accuracies.values()) if date_accuracies else 0,
                "best_accuracy": max(date_accuracies.values()) if date_accuracies else 0,
                "average_accuracy": sum(date_accuracies.values()) / len(date_accuracies) if date_accuracies else 0
            },
            "common_issues": [
                "Over-trading (excessive simulation trades)",
                "Poor timing alignment", 
                "Market condition mismatches",
                "Parameter tuning needed"
            ]
        }
    
    def get_trading_recommendations(self) -> Dict[str, Any]:
        """Get trading improvement recommendations"""
        quick_data = self._load_quick_lookup()
        stats = quick_data.get('quick_stats', {}) if quick_data else {}
        
        total_sim_trades = stats.get('total_sim_trades', 0)
        total_matches = stats.get('total_accurate_matches', 0)
        accuracy = stats.get('avg_execution_accuracy', 0)
        
        return {
            "current_performance": {
                "simulation_trades": total_sim_trades,
                "successful_matches": total_matches,
                "accuracy_pct": accuracy
            },
            "priority_actions": [
                {
                    "priority": 1,
                    "category": "Reduce Over-Trading",
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
                    "category": "Optimize Matching Criteria", 
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
                    "category": "Enhance Signal Quality",
                    "actions": [
                        "Add trend strength filters (ADX > 25)",
                        "Require volume confirmation (>20-day avg)",
                        "Implement volatility adjustments",
                        "Add market session filters"
                    ]
                }
            ],
            "quick_wins": [
                "Trade frequency limit: Max 1 trade/minute",
                "Minimum trade size: $1,000",
                "Session filters: Avoid first/last 30min",
                "Basic stops: 2% max loss per trade"
            ],
            "expected_improvements": {
                "trade_count_reduction": "90%",
                "accuracy_target": "10-25%",
                "capital_efficiency": "10x improvement",
                "risk_management": "Controlled exposure"
            },
            "success_metrics": [
                "Daily trade count <1,000",
                "Win rate >40%",
                "Average trade size >$5,000",
                "Maximum drawdown <5%"
            ]
        }
    
    def analyze_spike_addition_impact(self, threshold: float) -> Dict[str, Any]:
        """Analyze SpikeAddition parameter impact"""
        return {
            "current_threshold": threshold,
            "impact_analysis": {
                "price_tolerance_window": f"Trades match if price difference ≤ ${threshold}",
                "matching_sensitivity": {
                    "strict": {"range": "$10-15", "description": "Stricter matching, fewer false positives"},
                    "moderate": {"range": "$20-25", "description": "Balanced approach (current)"},
                    "lenient": {"range": "$30-50", "description": "More matches but less precise"}
                }
            },
            "recommended_testing": [
                {"threshold": 10, "type": "Ultra-strict", "use_case": "High precision"},
                {"threshold": 15, "type": "Strict", "use_case": "Good balance"},
                {"threshold": 20, "type": "Moderate-strict", "use_case": "Standard trading"},
                {"threshold": 25, "type": "Current baseline", "use_case": "Current setting"},
                {"threshold": 30, "type": "Lenient", "use_case": "More matches"}
            ],
            "dynamic_strategy": {
                "low_volatility": "$10-15",
                "high_volatility": "$25-35",
                "news_events": "$35-50",
                "weekend_low_volume": "$15-20"
            },
            "optimization_recommendations": [
                "Implement volatility-based dynamic thresholds",
                "Test with historical data using different values",
                "Consider percentage-based thresholds (0.1-0.3%)",
                "Add time-of-day adjustments",
                "Monitor false positive/negative rates"
            ],
            "expected_results": [
                "Better matching accuracy",
                "Reduced false signals",
                "Improved capital efficiency",
                "More realistic simulation results"
            ]
        }
    
    def get_available_dates(self) -> Dict[str, Any]:
        """Get list of available analysis dates"""
        quick_data = self._load_quick_lookup()
        if not quick_data:
            return {"error": "No analysis data found"}
        
        date_verdicts = quick_data.get('date_verdicts', {})
        date_accuracies = quick_data.get('date_accuracies', {})
        
        # Sort dates by accuracy (best first)
        sorted_dates = sorted(date_accuracies.items(), key=lambda x: x[1], reverse=True)
        
        analyzed_dates = []
        for date, accuracy in sorted_dates:
            verdict = date_verdicts.get(date, 'UNKNOWN')
            analyzed_dates.append({
                "date": date,
                "accuracy_pct": accuracy,
                "verdict": verdict,
                "status": "SUCCESS" if verdict == "YES" else "PARTIAL" if verdict == "PARTIAL" else "FAILED"
            })
        
        # Get skipped dates
        skipped_dates = [date for date, verdict in date_verdicts.items() if verdict == "SKIPPED"]
        
        return {
            "analyzed_dates": analyzed_dates,
            "skipped_dates": skipped_dates[:10],  # Limit to first 10
            "summary": {
                "total_analyzed": len(sorted_dates),
                "total_skipped": len(skipped_dates),
                "best_date": {
                    "date": sorted_dates[0][0] if sorted_dates else None,
                    "accuracy": sorted_dates[0][1] if sorted_dates else 0
                },
                "worst_date": {
                    "date": sorted_dates[-1][0] if sorted_dates else None,
                    "accuracy": sorted_dates[-1][1] if sorted_dates else 0
                },
                "average_accuracy": sum(date_accuracies.values()) / len(date_accuracies) if date_accuracies else 0,
                "dates_above_1_pct": len([acc for acc in date_accuracies.values() if acc > 1])
            }
        }
    
    # Helper methods
    def _load_quick_lookup(self) -> Optional[Dict]:
        """Load quick lookup data"""
        quick_file = f"{self.results_dir}/quick_lookup.json"
        if os.path.exists(quick_file):
            with open(quick_file, 'r') as f:
                return json.load(f)
        return None
    
    def _load_detailed_results(self, date: str) -> Optional[Dict]:
        """Load detailed results for specific date"""
        detail_file = f"{self.results_dir}/detail_{date}.json"
        if os.path.exists(detail_file):
            with open(detail_file, 'r') as f:
                return json.load(f)
        return None
    
    def _load_summary(self) -> Optional[Dict]:
        """Load summary data"""
        summary_file = f"{self.results_dir}/summary.json"
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                return json.load(f)
        return None
    
    def _get_critical_issues(self, stats: Dict, verdict: str) -> list:
        """Generate critical issues list"""
        if verdict == "NO":
            return [
                "Zero dates achieved acceptable performance (≥80%)",
                "Massive over-trading detected",
                "Poor execution quality",
                "System unsuitable for live trading"
            ]
        return ["Performance within acceptable parameters"]
    
    def _get_recommendations_summary(self) -> list:
        """Generate recommendations summary"""
        return [
            "Halt live trading immediately",
            "Reduce trade frequency by 90%",
            "Optimize matching criteria",
            "Implement proper risk management"
        ]
    
    def _analyze_match_details(self, match_details: list) -> Dict[str, Any]:
        """Analyze match details"""
        if not match_details:
            return {"status": "No match details available"}
        
        time_diffs = [abs(float(m.get('time_diff', 0))) for m in match_details]
        price_diffs = [float(m.get('price_diff_abs', 0)) for m in match_details]
        
        if time_diffs and price_diffs:
            return {
                "successful_matches": len(match_details),
                "avg_timing_difference_sec": sum(time_diffs) / len(time_diffs),
                "avg_price_difference_usd": sum(price_diffs) / len(price_diffs),
                "timing_range": {"min": min(time_diffs), "max": max(time_diffs)},
                "price_range": {"min": min(price_diffs), "max": max(price_diffs)}
            }
        
        return {"status": "Match quality data incomplete"}
    
    def _get_date_recommendations(self, validation: Dict, metrics: Dict) -> list:
        """Generate date-specific recommendations"""
        accuracy = validation.get('execution_accuracy_pct', 0)
        sim_trades = metrics.get('sim_trade_count', 0)
        live_trades = metrics.get('live_trade_count', 0)
        
        if accuracy < 1:
            return [
                f"CRITICAL: {accuracy:.2f}% accuracy is unacceptable",
                f"Reduce simulation trades from {sim_trades:,} to <{live_trades*2:,}",
                "Investigate algorithm parameters for this date",
                "Consider market conditions and volatility"
            ]
        elif accuracy < 10:
            return [
                f"POOR: {accuracy:.2f}% accuracy needs improvement",
                "Moderate over-trading detected",
                "Fine-tune matching parameters",
                "Add additional filters"
            ]
        else:
            return [
                f"ACCEPTABLE: {accuracy:.2f}% accuracy",
                "Monitor and maintain current parameters",
                "Consider scaling up carefully"
            ]

def get_ngrok_token():
    """Get ngrok token from environment or .env file"""
    # Try environment variable first
    token = os.getenv('NGROK_AUTHTOKEN')
    if token:
        return token
    
    # Try .env file
    env_file = ".env"
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('NGROK_AUTHTOKEN='):
                    return line.split('=', 1)[1].strip()
    
    return None

def setup_ngrok(port):
    """Setup ngrok tunnel"""
    if not NGROK_AVAILABLE:
        return None
    
    token = get_ngrok_token()
    if not token:
        return None
    
    try:
        ngrok.set_auth_token(token)
        tunnel = ngrok.connect(port, "http")
        return tunnel.public_url
    except Exception as e:
        print(f"❌ Ngrok failed: {e}")
        return None

def run_http_server(trading_server, port=8000):
    """Run HTTP server with ngrok in background thread"""
    class HTTPHandlerWrapper(TradingAnalysisHTTPHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(trading_server, *args, **kwargs)
    
    server_address = ('', port)
    httpd = HTTPServer(server_address, HTTPHandlerWrapper)
    
    # Setup ngrok tunnel
    public_url = setup_ngrok(port)
    
    print(f"📡 Local HTTP: http://localhost:{port}")
    
    if public_url:
        print(f"🌐 Public URL: {public_url}")
        print()
        print("🔧 CLAUDE REMOTE MCP SETUP:")
        print("=" * 50)
        print("Go to: https://claude.ai/settings/connectors")
        print("Add new connector:")
        print(f"   Name: Trading Analysis")
        print(f"   Remote MCP URL: {public_url}")
        print()
        print("✅ Your server is now accessible to Claude!")
        
        # Test endpoints
        print("🔗 Test endpoints:")
        print(f"   Health: {public_url}/health")
        print(f"   Overall: {public_url}/overall_analysis")
        print(f"   Date: {public_url}/date_analysis?date=22-06-2025")
    else:
        print("⚠️  Ngrok not available - using localhost only")
        print("   For public access:")
        print("   1. Install: pip install pyngrok")
        print("   2. Get token: https://dashboard.ngrok.com/get-started/your-authtoken")
        print("   3. Add to .env: NGROK_AUTHTOKEN=your_token")
    
    print()
    print("🛑 Press Ctrl+C to stop")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 HTTP server stopped")
        if public_url and NGROK_AVAILABLE:
            try:
                ngrok.disconnect(public_url)
                print("🌐 Ngrok tunnel closed")
            except:
                pass
        httpd.server_close()

async def run_mcp_server(trading_server):
    """Run MCP server for local connections"""
    if not trading_server.server:
        print("❌ MCP not available - install with: pip install mcp")
        return
    
    print("🔧 MCP Server ready for local connections")
    async with stdio_server() as (read_stream, write_stream):
        await trading_server.server.run(
            read_stream,
            write_stream,
            trading_server.server.create_initialization_options()
        )

def main():
    """Main function - choose mode based on arguments"""
    mode = sys.argv[1] if len(sys.argv) > 1 else "http"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    
    trading_server = TradingAnalysisServer()
    
    print("🚀 Trading Analysis Server")
    print(f"Mode: {mode}")
    print(f"MCP Available: {MCP_AVAILABLE}")
    print(f"Ngrok Available: {NGROK_AVAILABLE}")
    print()
    
    if mode == "mcp":
        # Pure MCP mode for local connections
        if MCP_AVAILABLE:
            asyncio.run(run_mcp_server(trading_server))
        else:
            print("❌ MCP mode requires: pip install mcp")
            sys.exit(1)
    else:
        # HTTP mode with ngrok for remote connections
        runf_http_server(trading_server, port)

if __name__ == "__main__":
    main()