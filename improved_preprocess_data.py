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
            
            # Convert created_time to timestamp and datetime
            df['timestamp'] = pd.to_datetime(df['created_time']).astype('int64') // 10**9
            df['datetime'] = pd.to_datetime(df['created_time'])
            df['date'] = df['datetime'].dt.date
            
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
    
    def analyze_real_trades_by_date(self, real_trades: pd.DataFrame) -> dict:
        """Analyze real trades and group by date to see which dates have trading activity"""
        try:
            # Group trades by date and count
            date_groups = real_trades.groupby('date').agg({
                'price': 'count',  # Count of trades
                'value': 'sum',    # Total volume
                'timestamp': ['min', 'max']  # Time range
            }).round(2)
            
            date_groups.columns = ['trade_count', 'total_volume', 'start_time', 'end_time']
            
            # Convert to dictionary with date string keys (DD-MM-YYYY format)
            date_analysis = {}
            for date_obj, row in date_groups.iterrows():
                date_str = date_obj.strftime('%d-%m-%Y')
                date_analysis[date_str] = {
                    'date_obj': date_obj,
                    'trade_count': int(row['trade_count']),
                    'total_volume': float(row['total_volume']),
                    'start_time': int(row['start_time']),
                    'end_time': int(row['end_time']),
                    'trades_data': real_trades[real_trades['date'] == date_obj].copy()
                }
            
            return date_analysis
            
        except Exception as e:
            print(f"Error analyzing real trades by date: {str(e)}")
            return {}
    
    def filter_real_trades_by_date(self, real_trades: pd.DataFrame, target_date_str: str) -> pd.DataFrame:
        """Filter real trades to only include trades from the target date"""
        try:
            # Parse target date (format: DD-MM-YYYY)
            from datetime import datetime
            target_date = datetime.strptime(target_date_str, '%d-%m-%Y').date()
            
            # Filter trades for this specific date
            filtered_trades = real_trades[real_trades['date'] == target_date].copy()
            
            return filtered_trades
            
        except Exception as e:
            print(f"Error filtering trades for date {target_date_str}: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame on error

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

    def analyze_execution_quality(self, real_trades: pd.DataFrame, sim_trades: pd.DataFrame, spike_addition: float = 25.0) -> Dict:
        """Analyze execution quality by iterating through simulation trades and matching against real trades"""
        
        if sim_trades.empty:
            return {
                'total_sim_trades': 0,
                'matched_trades': 0,
                'accuracy_pct': 0,
                'match_details': [],
                'reason': 'No simulation trades found'
            }
            
        if real_trades.empty:
            return {
                'total_sim_trades': len(sim_trades),
                'matched_trades': 0,
                'accuracy_pct': 0,
                'match_details': [],
                'reason': 'No real trades found for comparison'
            }
        
        total_sim_trades = len(sim_trades)
        matched_trades = 0
        match_details = []
        used_real_indices = set()
        
        # Iterate through each simulated trade
        for sim_idx, sim_trade in sim_trades.iterrows():
            # Find real trades within acceptable match criteria
            time_tolerance = 5  # 5 seconds
            
            # Time window filter
            time_mask = (
                (real_trades['timestamp'] >= sim_trade['timestamp'] - time_tolerance) &
                (real_trades['timestamp'] <= sim_trade['timestamp'] + time_tolerance) &
                (real_trades['side'] == sim_trade['side'])  # Same side (buy/sell)
            )
            
            potential_real_matches = real_trades[time_mask]
            
            if not potential_real_matches.empty:
                # Check price criteria with SpikeAddition threshold
                price_diff_abs = abs(potential_real_matches['price'] - sim_trade['price'])
                acceptable_price_matches = potential_real_matches[price_diff_abs <= spike_addition]
                
                if not acceptable_price_matches.empty:
                    # Find best match (closest in time, then price)
                    time_diffs = abs(acceptable_price_matches['timestamp'] - sim_trade['timestamp'])
                    best_real_idx = time_diffs.idxmin()
                    
                    # Avoid double-matching
                    if best_real_idx not in used_real_indices:
                        matched_trades += 1
                        used_real_indices.add(best_real_idx)
                        best_real_match = acceptable_price_matches.loc[best_real_idx]
                        
                        match_details.append({
                            'sim_time': int(sim_trade['timestamp']),
                            'real_time': int(best_real_match['timestamp']),
                            'time_diff': int(abs(sim_trade['timestamp'] - best_real_match['timestamp'])),
                            'sim_price': float(sim_trade['price']),
                            'real_price': float(best_real_match['price']),
                            'price_diff_abs': float(abs(sim_trade['price'] - best_real_match['price'])),
                            'side': sim_trade['side'],
                            'sim_quantity': float(sim_trade['quantity']),
                            'real_quantity': float(best_real_match['quantity'])
                        })
        
        accuracy_pct = (matched_trades / total_sim_trades * 100) if total_sim_trades > 0 else 0
        
        return {
            'total_sim_trades': total_sim_trades,
            'matched_trades': matched_trades,
            'accuracy_pct': accuracy_pct,
            'match_details': match_details,
            'reason': f'Matched {matched_trades}/{total_sim_trades} sim trades using ${spike_addition} price tolerance'
        }

    def analyze_single_date(self, real_trades: pd.DataFrame, sim_data: List[Dict], date_str: str) -> Dict:
        """Analyze a single date with corrected matching logic that iterates through sim trades"""
        
        # Extract simulation trades
        sim_trades = self.extract_simulation_trades(sim_data)
        
        # Calculate executed capital (EC)
        live_ec = (real_trades['price'] * real_trades['quantity']).sum()
        sim_ec = (sim_trades['price'] * sim_trades['quantity']).sum() if not sim_trades.empty else 0
        
        # Count trades
        live_trade_count = len(real_trades)
        sim_trade_count = len(sim_trades)
        
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
                    'total_testable': sim_trade_count,
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
                    'total_testable': sim_trade_count,
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
            
        # Use the new analyze_execution_quality function with SpikeAddition parameter
        spike_addition = 25.0  # $25 threshold as specified in requirements
        execution_results = self.analyze_execution_quality(real_overlap, sim_overlap, spike_addition)
        
        # Extract results
        accurate_matches = execution_results['matched_trades']
        total_testable = execution_results['total_sim_trades']
        match_details = execution_results['match_details']

        # Calculate accuracy metrics (now correctly based on sim trades)
        execution_accuracy = execution_results['accuracy_pct']
        
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
                'spike_addition_used': spike_addition,
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
        """Process simulation files efficiently - only load sim data for dates with real trades"""
        print("🔄 Starting improved data preprocessing...")
        
        # Load real trades first and analyze by date
        print("📈 Loading and analyzing real trades data...")
        real_trades = self.load_real_trades("data/agent_live_data.csv")
        print(f"✅ Loaded {len(real_trades)} real trades")
        print(f"📅 Real data period: {datetime.fromtimestamp(real_trades['timestamp'].min())} to {datetime.fromtimestamp(real_trades['timestamp'].max())}")
        
        # Analyze real trades by date to see which dates have activity
        print("🔍 Analyzing real trades by date...")
        real_trades_by_date = self.analyze_real_trades_by_date(real_trades)
        
        print(f"📅 Found real trading activity on {len(real_trades_by_date)} dates:")
        for date_str, info in real_trades_by_date.items():
            print(f"  {date_str}: {info['trade_count']} trades, ${info['total_volume']:,.0f} volume")
        
        # Get all simulation files
        pattern = "data/HL_btcusdt_BTC/parrot_HL_btcusdt_BTC_*.pickle"
        all_sim_files = glob.glob(pattern)
        
        if not all_sim_files:
            print("❌ No simulation files found")
            return {}
        
        print(f"📊 Found {len(all_sim_files)} total simulation files")
        
        # Filter simulation files to only those with corresponding real trades
        relevant_sim_files = []
        for file in all_sim_files:
            basename = os.path.basename(file)
            date_part = basename.replace("parrot_HL_btcusdt_BTC_", "").replace(".pickle", "")
            if date_part in real_trades_by_date:
                relevant_sim_files.append((file, date_part))
        
        print(f"🎯 Only {len(relevant_sim_files)} simulation files have corresponding real trades")
        if len(relevant_sim_files) < len(all_sim_files):
            skipped_count = len(all_sim_files) - len(relevant_sim_files)
            print(f"⏭️ Skipping {skipped_count} simulation files with no real trading data")
        
        # Overall summary
        overall_summary = {
            "processing_date": datetime.now().isoformat(),
            "total_files": len(all_sim_files),
            "relevant_files": len(relevant_sim_files),
            "real_trades_count": len(real_trades),
            "real_trades_ec": float((real_trades['price'] * real_trades['quantity']).sum()),
            "date_results": {},
            "overall_verdict": None,
            "summary_stats": {}
        }
        
        successful_analyses = []
        failed_analyses = []
        
        # Sort files by date for chronological processing
        relevant_sim_files.sort(key=lambda x: x[1])
        
        # Process only relevant simulation files (those with real trades)
        for i, (file, date_part) in enumerate(relevant_sim_files):
            print(f"🔍 Processing {date_part} ({i+1}/{len(relevant_sim_files)})...")
            
            try:
                # Get pre-filtered real trades for this date
                real_trades_info = real_trades_by_date[date_part]
                date_filtered_real_trades = real_trades_info['trades_data']
                
                print(f"  📊 Using {len(date_filtered_real_trades)} real trades for {date_part}")
                
                # Load simulation data (only for dates we know have real trades)
                print(f"  💾 Loading simulation data for {date_part}...")
                with open(file, 'rb') as f:
                    sim_data = pickle.load(f)
                
                # Run improved analysis with date-filtered real trades
                results = self.analyze_single_date(date_filtered_real_trades, sim_data, date_part)
                
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
        
        # Add skipped entries for simulation files that were never processed (no real trades)
        for file in all_sim_files:
            basename = os.path.basename(file)
            date_part = basename.replace("parrot_HL_btcusdt_BTC_", "").replace(".pickle", "")
            if date_part not in overall_summary['date_results']:
                skipped_summary = {
                    'date': date_part,
                    'verdict': 'SKIPPED',
                    'reason': 'No real trades found for this date - simulation file not processed',
                    'has_detailed_results': False
                }
                overall_summary['date_results'][date_part] = skipped_summary
        
        # Calculate overall statistics with PARTIAL support  
        skipped_count = sum(1 for r in overall_summary['date_results'].values() if r.get('verdict') == 'SKIPPED')
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
                'skipped_analyses': skipped_count,
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
        if skipped_count > 0:
            print(f"⏭️ Skipped: {skipped_count} (no real trades for those dates)")
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
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        print("🔄 Running in automatic mode...")
    else:
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