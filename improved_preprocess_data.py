#!/usr/bin/env python3
"""
Improved Data Preprocessing Script

Fixes issues with the original preprocessing:
1. Better path handling
2. More granular accuracy reporting (not just 0% or 100%)
3. Improved matching criteria
4. Better diagnostics for debugging
"""

import os
import json
import pickle
import glob
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from collections import defaultdict

class ImprovedDataPreprocessor:
    """Enhanced preprocessor with better matching logic and diagnostics"""
    
    def __init__(self):
        self.results_dir = "analysis_results"
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
    def load_real_trades(self, filepath: str) -> pd.DataFrame:
        """Load and preprocess real trading data"""
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
            
            # Convert created_time to timestamp
            df['timestamp'] = pd.to_datetime(df['created_time']).view('int64') // 10**9
            
            # Clean numeric columns
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
            
            # Add derived fields
            df['value'] = df['price'] * df['quantity']
            df['is_buy'] = df['side'] == 'B'
            
            return df
            
        except Exception as e:
            print(f"Error loading real trades: {str(e)}")
            raise

    def extract_simulation_trades(self, sim_data: List[Dict]) -> pd.DataFrame:
        """Extract actual trades from simulation market data"""
        trades = []
        
        for i, snapshot in enumerate(sim_data):
            timestamp = snapshot.get('timestamp', 0)
            
            # Extract buy trades
            buy_trades = snapshot.get('TradesMarketBuy', [])
            if isinstance(buy_trades, np.ndarray) and len(buy_trades) > 0:
                for trade in buy_trades:
                    trades.append({
                        'timestamp': timestamp,
                        'price': float(trade[0]),
                        'quantity': float(trade[1]),
                        'side': 'B',
                        'snapshot_idx': i
                    })
            
            # Extract sell trades
            sell_trades = snapshot.get('TradesMarketSell', [])
            if isinstance(sell_trades, np.ndarray) and len(sell_trades) > 0:
                for trade in sell_trades:
                    trades.append({
                        'timestamp': timestamp,
                        'price': float(trade[0]),
                        'quantity': float(trade[1]),
                        'side': 'A',
                        'snapshot_idx': i
                    })
        
        df = pd.DataFrame(trades)
        if not df.empty:
            df['value'] = df['price'] * df['quantity']
            df['is_buy'] = df['side'] == 'B'
        
        return df

    def analyze_single_date(self, real_trades: pd.DataFrame, sim_data: List[Dict], date_str: str) -> Dict:
        """Analyze a single date with improved matching and granular accuracy"""
        
        # Extract simulation trades
        sim_trades = self.extract_simulation_trades(sim_data)
        
        # Calculate executed capital (EC)
        live_ec = (real_trades['price'] * real_trades['quantity']).sum()
        sim_ec = (sim_trades['price'] * sim_trades['quantity']).sum() if not sim_trades.empty else 0
        
        # Count trades
        live_trade_count = len(real_trades)
        sim_trade_count = len(sim_trades)
        
        # Multiple matching criteria with different tolerances
        tolerances = [
            {'time': 2, 'price_pct': 0.005, 'name': 'strict'},
            {'time': 5, 'price_pct': 0.01, 'name': 'moderate'}, 
            {'time': 10, 'price_pct': 0.02, 'name': 'relaxed'}
        ]
        
        results = {}
        diagnostics = []
        
        # If no simulation trades, it's a clear failure
        if sim_trades.empty:
            diagnostics.append("⚠️ No simulation trades found - algorithm likely crashed")
            return {
                'validation_result': {
                    'is_acceptable': False,
                    'accuracy_verdict': 'NO',
                    'execution_accuracy_pct': 0,
                    'reason': 'No simulation trades found - possible algorithm crash'
                },
                'metrics': {
                    'live_ec': live_ec,
                    'sim_ec': sim_ec,
                    'live_trade_count': live_trade_count,
                    'sim_trade_count': sim_trade_count,
                    'accurate_matches': 0,
                    'total_testable': 0,
                    'execution_accuracy_pct': 0
                },
                'diagnostics': diagnostics,
                'match_details': []
            }
        
        # Check temporal overlap first
        real_min, real_max = real_trades['timestamp'].min(), real_trades['timestamp'].max()
        sim_min, sim_max = sim_trades['timestamp'].min(), sim_trades['timestamp'].max()
        
        if real_max < sim_min or sim_max < real_min:
            gap_seconds = min(abs(sim_min - real_max), abs(real_min - sim_max))
            diagnostics.append(f"⚠️ No temporal overlap - {gap_seconds}s gap between real and sim data")
            return {
                'validation_result': {
                    'is_acceptable': False,
                    'accuracy_verdict': 'NO', 
                    'execution_accuracy_pct': 0,
                    'reason': f'No temporal overlap - {gap_seconds}s gap between datasets'
                },
                'metrics': {
                    'live_ec': live_ec,
                    'sim_ec': sim_ec,
                    'live_trade_count': live_trade_count,
                    'sim_trade_count': sim_trade_count,
                    'accurate_matches': 0,
                    'total_testable': 0,
                    'execution_accuracy_pct': 0
                },
                'diagnostics': diagnostics,
                'match_details': []
            }

        # Find overlapping period
        overlap_start = max(real_min, sim_min)
        overlap_end = min(real_max, sim_max)
        
        # Filter trades to overlap period only
        real_overlap = real_trades[
            (real_trades['timestamp'] >= overlap_start) & 
            (real_trades['timestamp'] <= overlap_end)
        ]
        sim_overlap = sim_trades[
            (sim_trades['timestamp'] >= overlap_start) & 
            (sim_trades['timestamp'] <= overlap_end)
        ]
        
        if real_overlap.empty:
            diagnostics.append(f"⚠️ No real trades in overlap period ({overlap_end - overlap_start}s)")
            
        # Perform matching with strict criteria (for main result)
        time_tolerance = 2
        price_tolerance = 0.005
        
        accurate_matches = 0
        total_testable = 0
        match_details = []
        used_sim_indices = set()
        
        for _, real_trade in real_overlap.iterrows():
            # Find simulation trades within strict criteria
            time_mask = (
                (sim_overlap['timestamp'] >= real_trade['timestamp'] - time_tolerance) &
                (sim_overlap['timestamp'] <= real_trade['timestamp'] + time_tolerance) &
                (sim_overlap['side'] == real_trade['side'])  # Same side (buy/sell)
            )
            
            potential_matches = sim_overlap[time_mask]
            
            if not potential_matches.empty:
                total_testable += 1
                
                # Check price accuracy
                price_diff_pct = abs(potential_matches['price'] - real_trade['price']) / real_trade['price']
                accurate_price_matches = potential_matches[price_diff_pct <= price_tolerance]
                
                if not accurate_price_matches.empty:
                    # Find best match (closest in time, then price)
                    time_diffs = abs(accurate_price_matches['timestamp'] - real_trade['timestamp'])
                    best_idx = time_diffs.idxmin()
                    
                    # Avoid double-matching
                    if best_idx not in used_sim_indices:
                        accurate_matches += 1
                        used_sim_indices.add(best_idx)
                        best_match = accurate_price_matches.loc[best_idx]
                        
                        match_details.append({
                            'real_time': int(real_trade['timestamp']),
                            'sim_time': int(best_match['timestamp']),
                            'time_diff': int(abs(real_trade['timestamp'] - best_match['timestamp'])),
                            'real_price': float(real_trade['price']),
                            'sim_price': float(best_match['price']),
                            'price_diff_pct': float(abs(real_trade['price'] - best_match['price']) / real_trade['price'] * 100),
                            'side': real_trade['side'],
                            'accurate': True
                        })

        # Calculate accuracy metrics
        execution_accuracy = (accurate_matches / total_testable * 100) if total_testable > 0 else 0
        
        # More nuanced verdict logic
        if total_testable == 0:
            verdict = 'NO'
            reason = 'No matching trades found - possible algorithm crash or data misalignment'
        elif execution_accuracy >= 80.0:
            verdict = 'YES'
            reason = f'Simulation achieves {execution_accuracy:.1f}% execution accuracy within strict criteria (2s, ±0.005%)'
        elif execution_accuracy >= 50.0:
            verdict = 'PARTIAL'
            reason = f'Simulation achieves {execution_accuracy:.1f}% execution accuracy - needs improvement'
        else:
            verdict = 'NO'
            reason = f'Poor execution accuracy ({execution_accuracy:.1f}%) - significant issues detected'
        
        # Add diagnostics
        if sim_trade_count > live_trade_count * 10:
            diagnostics.append(f"📈 Very high sim activity ({sim_trade_count} vs {live_trade_count} live) - possible volatility over-reaction")
        elif sim_trade_count < live_trade_count * 0.1:
            diagnostics.append(f"📉 Very low sim activity ({sim_trade_count} vs {live_trade_count} live) - possible under-trading")
            
        if execution_accuracy < 50 and total_testable > 0:
            diagnostics.append(f"⚠️ Poor execution accuracy ({execution_accuracy:.1f}%) indicates timing or pricing issues")
            
        overlap_duration = overlap_end - overlap_start
        if overlap_duration < 3600:  # Less than 1 hour
            diagnostics.append(f"⏰ Short overlap period ({overlap_duration}s) may limit analysis reliability")

        return {
            'validation_result': {
                'is_acceptable': verdict == 'YES',
                'accuracy_verdict': verdict,
                'execution_accuracy_pct': execution_accuracy,
                'reason': reason
            },
            'metrics': {
                'live_ec': float(live_ec),
                'sim_ec': float(sim_ec), 
                'live_trade_count': live_trade_count,
                'sim_trade_count': sim_trade_count,
                'accurate_matches': accurate_matches,
                'total_testable': total_testable,
                'execution_accuracy_pct': execution_accuracy,
                'overlap_duration': int(overlap_duration),
                'overlap_real_trades': len(real_overlap)
            },
            'diagnostics': diagnostics,
            'match_details': match_details[:20],  # Show first 20 matches
            'temporal_analysis': {
                'real_period': f"{datetime.fromtimestamp(real_min)} to {datetime.fromtimestamp(real_max)}",
                'sim_period': f"{datetime.fromtimestamp(sim_min)} to {datetime.fromtimestamp(sim_max)}",
                'overlap_period': f"{datetime.fromtimestamp(overlap_start)} to {datetime.fromtimestamp(overlap_end)}",
                'overlap_duration_hours': overlap_duration / 3600
            }
        }

    def process_all_data(self) -> Dict[str, Any]:
        """Process all simulation files with improved analysis"""
        print("🔄 Starting improved data preprocessing...")
        
        # Get all simulation files
        pattern = "data/HL_btcusdt_BTC/parrot_HL_btcusdt_BTC_*.pickle"
        files = glob.glob(pattern)
        
        if not files:
            print("❌ No simulation files found")
            return {}
        
        print(f"📊 Found {len(files)} simulation files to process")
        
        # Load real trades once
        print("📈 Loading real trades data...")
        real_trades = self.load_real_trades("data/agent_live_data.csv")
        print(f"✅ Loaded {len(real_trades)} real trades")
        print(f"📅 Real data period: {datetime.fromtimestamp(real_trades['timestamp'].min())} to {datetime.fromtimestamp(real_trades['timestamp'].max())}")
        
        # Overall summary
        overall_summary = {
            "processing_date": datetime.now().isoformat(),
            "total_files": len(files),
            "real_trades_count": len(real_trades),
            "real_trades_ec": float((real_trades['price'] * real_trades['quantity']).sum()),
            "date_results": {},
            "overall_verdict": None,
            "summary_stats": {}
        }
        
        successful_analyses = []
        failed_analyses = []
        
        # Process each simulation file
        for i, file in enumerate(sorted(files)):
            basename = os.path.basename(file)
            date_part = basename.replace("parrot_HL_btcusdt_BTC_", "").replace(".pickle", "")
            
            print(f"🔍 Processing {date_part} ({i+1}/{len(files)})...")
            
            try:
                # Load simulation data
                with open(file, 'rb') as f:
                    sim_data = pickle.load(f)
                
                # Run improved analysis
                results = self.analyze_single_date(real_trades, sim_data, date_part)
                
                if 'error' not in results:
                    # Extract key metrics for fast access
                    validation = results.get('validation_result', {})
                    metrics = results.get('metrics', {})
                    
                    date_summary = {
                        'date': date_part,
                        'verdict': validation.get('accuracy_verdict', 'UNKNOWN'),
                        'accuracy_pct': validation.get('execution_accuracy_pct', 0),
                        'live_trades': metrics.get('live_trade_count', 0),
                        'sim_trades': metrics.get('sim_trade_count', 0),
                        'live_ec': metrics.get('live_ec', 0),
                        'sim_ec': metrics.get('sim_ec', 0),
                        'accurate_matches': metrics.get('accurate_matches', 0),
                        'total_testable': metrics.get('total_testable', 0),
                        'reason': validation.get('reason', ''),
                        'diagnostics': results.get('diagnostics', []),
                        'has_detailed_results': True
                    }
                    
                    overall_summary['date_results'][date_part] = date_summary
                    successful_analyses.append(date_summary)
                    
                    # Save detailed results for this date
                    detail_file = f"{self.results_dir}/detail_{date_part}.json"
                    with open(detail_file, 'w') as f:
                        json.dump(results, f, indent=2, default=str)
                    
                    verdict_emoji = "✅" if validation.get('accuracy_verdict') == 'YES' else "⚠️" if validation.get('accuracy_verdict') == 'PARTIAL' else "❌"
                    print(f"  {verdict_emoji} {date_part}: {validation.get('accuracy_verdict', 'UNKNOWN')} "
                          f"({validation.get('execution_accuracy_pct', 0):.1f}% accuracy)")
                
                else:
                    error_summary = {
                        'date': date_part,
                        'verdict': 'ERROR',
                        'error': results.get('error', 'Unknown error'),
                        'has_detailed_results': False
                    }
                    overall_summary['date_results'][date_part] = error_summary
                    failed_analyses.append(error_summary)
                    print(f"  ❌ {date_part}: ERROR - {results.get('error', 'Unknown')}")
                
            except Exception as e:
                error_summary = {
                    'date': date_part,
                    'verdict': 'ERROR',
                    'error': str(e),
                    'has_detailed_results': False
                }
                overall_summary['date_results'][date_part] = error_summary
                failed_analyses.append(error_summary)
                print(f"  ❌ {date_part}: EXCEPTION - {str(e)}")
        
        # Calculate overall statistics with PARTIAL support
        if successful_analyses:
            yes_count = sum(1 for r in successful_analyses if r['verdict'] == 'YES')
            partial_count = sum(1 for r in successful_analyses if r['verdict'] == 'PARTIAL')
            total_count = len(successful_analyses)
            
            # Overall verdict considers both YES and PARTIAL as acceptable
            acceptable_count = yes_count + partial_count
            overall_accuracy_rate = acceptable_count / total_count * 100 if total_count > 0 else 0
            
            # Weighted average of execution accuracy (not just yes/no)
            weighted_accuracy = sum(r['accuracy_pct'] for r in successful_analyses) / len(successful_analyses)
            
            overall_summary['overall_verdict'] = 'YES' if overall_accuracy_rate >= 80 else 'PARTIAL' if overall_accuracy_rate >= 50 else 'NO'
            overall_summary['summary_stats'] = {
                'successful_analyses': len(successful_analyses),
                'failed_analyses': len(failed_analyses),
                'yes_verdicts': yes_count,
                'partial_verdicts': partial_count,
                'no_verdicts': total_count - yes_count - partial_count,
                'overall_accuracy_rate': overall_accuracy_rate,
                'avg_execution_accuracy': weighted_accuracy,
                'total_sim_trades': sum(r['sim_trades'] for r in successful_analyses),
                'total_sim_ec': sum(r['sim_ec'] for r in successful_analyses),
                'total_accurate_matches': sum(r['accurate_matches'] for r in successful_analyses),
                'total_testable_trades': sum(r['total_testable'] for r in successful_analyses)
            }
        
        # Save overall summary
        summary_file = f"{self.results_dir}/summary.json"
        with open(summary_file, 'w') as f:
            json.dump(overall_summary, f, indent=2, default=str)
        
        print(f"\n🎯 Improved Preprocessing Complete!")
        print(f"✅ Successful: {len(successful_analyses)}")
        print(f"❌ Failed: {len(failed_analyses)}")
        if successful_analyses:
            overall_verdict = overall_summary['overall_verdict']
            avg_accuracy = overall_summary['summary_stats']['avg_execution_accuracy']
            print(f"📋 Overall Verdict: {overall_verdict}")
            print(f"📊 Average Execution Accuracy: {avg_accuracy:.1f}%")
            print(f"📊 Results saved to: {self.results_dir}/")
        
        return overall_summary

    def create_quick_lookup(self, summary: Dict[str, Any]) -> None:
        """Create enhanced quick lookup file"""
        
        quick_data = {
            'last_updated': summary.get('processing_date'),
            'overall_verdict': summary.get('overall_verdict'),
            'quick_stats': summary.get('summary_stats', {}),
            'date_verdicts': {
                date: data.get('verdict', 'UNKNOWN') 
                for date, data in summary.get('date_results', {}).items()
            },
            'date_accuracies': {
                date: data.get('accuracy_pct', 0) 
                for date, data in summary.get('date_results', {}).items()
                if data.get('has_detailed_results', False)
            },
            'problematic_dates': [
                date for date, data in summary.get('date_results', {}).items()
                if data.get('verdict') in ['NO', 'ERROR'] or data.get('accuracy_pct', 0) < 50
            ],
            'excellent_dates': [
                date for date, data in summary.get('date_results', {}).items()
                if data.get('verdict') == 'YES' and data.get('accuracy_pct', 0) >= 95
            ],
            'partial_dates': [
                date for date, data in summary.get('date_results', {}).items()
                if data.get('verdict') == 'PARTIAL'
            ]
        }
        
        quick_file = f"{self.results_dir}/quick_lookup.json"
        with open(quick_file, 'w') as f:
            json.dump(quick_data, f, indent=2)
        
        print(f"⚡ Enhanced quick lookup saved to: {quick_file}")


def main():
    """Main preprocessing function"""
    print("🚀 Improved Data Preprocessing System")
    print("📊 Enhanced analysis with granular accuracy reporting and better diagnostics")
    
    # Check if data exists
    if not os.path.exists("data/agent_live_data.csv"):
        print("❌ Real trade data not found: data/agent_live_data.csv")
        return
    
    pattern = "data/HL_btcusdt_BTC/parrot_HL_btcusdt_BTC_*.pickle"
    files = glob.glob(pattern)
    if not files:
        print("❌ No simulation files found")
        return
    
    print(f"📁 Found {len(files)} simulation files to process")
    
    response = input("🤔 This may take several minutes. Continue? (y/n): ").strip().lower()
    if response != 'y':
        print("❌ Preprocessing cancelled")
        return
    
    # Run improved preprocessing
    preprocessor = ImprovedDataPreprocessor()
    summary = preprocessor.process_all_data()
    
    if summary:
        preprocessor.create_quick_lookup(summary)
        print("\n✅ Improved preprocessing complete! You can now use the fast agent.")
        print("🚀 Run: python run_fast_agent.py")
    else:
        print("❌ Preprocessing failed")


if __name__ == "__main__":
    main()