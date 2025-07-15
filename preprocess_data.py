#!/usr/bin/env python3
"""
Data Preprocessing Script

Analyzes all trading data upfront and saves results to JSON files.
This prevents RAM issues and allows instant agent responses.
"""

import os
import json
import pickle
import glob
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
from deep_analysis_engine import DeepAnalysisEngine

class DataPreprocessor:
    """Preprocesses all trading data and saves validation results"""
    
    def __init__(self):
        self.results_dir = "analysis_results"
        self.engine = DeepAnalysisEngine()
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
    def process_all_data(self) -> Dict[str, Any]:
        """Process all simulation files and save results"""
        print("🔄 Starting comprehensive data preprocessing...")
        
        # Get all simulation files
        pattern = "data/HL_btcusdt_BTC/parrot_HL_btcusdt_BTC_*.pickle"
        files = glob.glob(pattern)
        
        if not files:
            print("❌ No simulation files found")
            return {}
        
        print(f"📊 Found {len(files)} simulation files to process")
        
        # Load real trades once (cached)
        print("📈 Loading real trades data...")
        real_trades = self.engine.load_real_trades("data/agent_live_data.csv")
        print(f"✅ Loaded {len(real_trades)} real trades")
        
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
        for i, file in enumerate(files):
            basename = os.path.basename(file)
            date_part = basename.replace("parrot_HL_btcusdt_BTC_", "").replace(".pickle", "")
            
            print(f"🔍 Processing {date_part} ({i+1}/{len(files)})...")
            
            try:
                # Load simulation data
                with open(file, 'rb') as f:
                    sim_data = pickle.load(f)
                
                # Run analysis
                results = self.engine.analyze_discrepancies(
                    "data/agent_live_data.csv",
                    sim_data,
                    date_part
                )
                
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
                    
                    print(f"  ✅ {date_part}: {validation.get('accuracy_verdict', 'UNKNOWN')} "
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
        
        # Calculate overall statistics
        if successful_analyses:
            yes_count = sum(1 for r in successful_analyses if r['verdict'] == 'YES')
            total_count = len(successful_analyses)
            
            overall_summary['overall_verdict'] = 'YES' if yes_count >= total_count * 0.8 else 'NO'
            overall_summary['summary_stats'] = {
                'successful_analyses': len(successful_analyses),
                'failed_analyses': len(failed_analyses),
                'yes_verdicts': yes_count,
                'no_verdicts': total_count - yes_count,
                'overall_accuracy_rate': yes_count / total_count * 100 if total_count > 0 else 0,
                'avg_execution_accuracy': sum(r['accuracy_pct'] for r in successful_analyses) / len(successful_analyses),
                'total_sim_trades': sum(r['sim_trades'] for r in successful_analyses),
                'total_sim_ec': sum(r['sim_ec'] for r in successful_analyses),
                'total_accurate_matches': sum(r['accurate_matches'] for r in successful_analyses),
                'total_testable_trades': sum(r['total_testable'] for r in successful_analyses)
            }
        
        # Save overall summary
        summary_file = f"{self.results_dir}/summary.json"
        with open(summary_file, 'w') as f:
            json.dump(overall_summary, f, indent=2, default=str)
        
        print(f"\n🎯 Preprocessing Complete!")
        print(f"✅ Successful: {len(successful_analyses)}")
        print(f"❌ Failed: {len(failed_analyses)}")
        if successful_analyses:
            overall_verdict = overall_summary['overall_verdict']
            print(f"📋 Overall Verdict: {overall_verdict}")
            print(f"📊 Results saved to: {self.results_dir}/")
        
        return overall_summary
    
    def create_quick_lookup(self, summary: Dict[str, Any]) -> None:
        """Create a quick lookup file for instant responses"""
        
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
            ]
        }
        
        quick_file = f"{self.results_dir}/quick_lookup.json"
        with open(quick_file, 'w') as f:
            json.dump(quick_data, f, indent=2)
        
        print(f"⚡ Quick lookup saved to: {quick_file}")


def main():
    """Main preprocessing function"""
    print("🚀 Data Preprocessing System")
    print("📊 This will analyze all simulation data and save results for instant agent access")
    
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
    
    # Run preprocessing
    preprocessor = DataPreprocessor()
    summary = preprocessor.process_all_data()
    
    if summary:
        preprocessor.create_quick_lookup(summary)
        print("\n✅ Preprocessing complete! You can now use the fast agent.")
        print("🚀 Run: python run_fast_agent.py")
    else:
        print("❌ Preprocessing failed")


if __name__ == "__main__":
    main()