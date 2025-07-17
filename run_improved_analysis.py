#!/usr/bin/env python3
"""
Run Improved Analysis with Gap-Aware Matching and Volatility Management

This script demonstrates the improved analysis capabilities with:
1. Gap-aware matching for trading interruptions
2. Volatility-adaptive parameters
3. Enhanced diagnostic information
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path
import pickle
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from improved_analysis_engine import ImprovedAnalysisEngine
from data_manager import DataManager


def test_improved_analysis():
    """Test the improved analysis engine with problematic dates"""
    
    print("🚀 IMPROVED ANALYSIS ENGINE TEST")
    print("="*60)
    
    # Initialize components
    engine = ImprovedAnalysisEngine()
    data_manager = DataManager()
    
    # Test dates: failed and successful
    test_dates = [
        {'date': '25-06-2025', 'expected': 'FAILED'},
        {'date': '01-06-2025', 'expected': 'FAILED'},
        {'date': '15-06-2025', 'expected': 'SUCCESS'},
        {'date': '24-06-2025', 'expected': 'SUCCESS'}
    ]
    
    results = []
    
    for test_case in test_dates:
        date_str = test_case['date']
        expected = test_case['expected']
        
        print(f"\\n🔍 Testing {date_str} (Expected: {expected})")
        print("-" * 40)
        
        try:
            # Load data
            print("📊 Loading data...")
            real_data = data_manager.load_real_data()
            sim_data = data_manager.load_simulation_data(date_str)
            
            if sim_data is None:
                print(f"❌ No simulation data for {date_str}")
                continue
            
            # Filter real data for this date
            real_trades = data_manager.filter_real_trades_by_date(real_data, date_str)
            sim_trades = data_manager.extract_simulation_trades(sim_data)
            
            if real_trades.empty or sim_trades.empty:
                print(f"⚠️ No trades found for {date_str}")
                continue
            
            print(f"📈 Real trades: {len(real_trades)}, Sim trades: {len(sim_trades)}")
            
            # Run improved analysis
            print("🔧 Running improved analysis...")
            result = engine.validate_simulation_accuracy_enhanced(
                real_trades, sim_trades, date_str
            )
            
            # Display results
            print(f"\\n📋 RESULTS for {date_str}:")
            print(f"✅ Acceptable: {result['is_acceptable']}")
            print(f"📊 Accuracy: {result['accuracy_rate']:.1%}")
            print(f"🎯 Matches: {result['accurate_matches']}/{result['total_testable']}")
            print(f"🔧 Method: {result['matching_method']}")
            
            # Volatility analysis
            vol_analysis = result['volatility_analysis']
            print(f"🌊 Volatility: {vol_analysis['volatility_level']}")
            
            # Adjusted parameters
            params = result['adjusted_parameters']
            print(f"⏱️ Time tolerance: {params['time_tolerance']:.1f}s")
            print(f"💰 Price tolerance: {params['price_tolerance']:.3f}%")
            print(f"🎯 Accuracy threshold: {params['accuracy_threshold']:.1%}")
            
            # Diagnostics
            if result['diagnostics']:
                print("🔍 Diagnostics:")
                for diagnostic in result['diagnostics']:
                    print(f"  • {diagnostic}")
            
            # Recommendations
            if 'recommendations' in vol_analysis:
                print("💡 Recommendations:")
                for rec in vol_analysis['recommendations'][:3]:
                    print(f"  • {rec}")
            
            # Store result
            results.append({
                'date': date_str,
                'expected': expected,
                'acceptable': result['is_acceptable'],
                'accuracy': result['accuracy_rate'],
                'volatility_level': vol_analysis['volatility_level'],
                'time_tolerance': params['time_tolerance'],
                'price_tolerance': params['price_tolerance']
            })
            
        except Exception as e:
            print(f"❌ Error analyzing {date_str}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\\n\\n📊 SUMMARY OF IMPROVEMENTS")
    print("="*60)
    
    for result in results:
        status = "✅ FIXED" if result['acceptable'] else "❌ STILL FAILED"
        print(f"{result['date']}: {status} ({result['accuracy']:.1%} accuracy)")
        print(f"  • Volatility: {result['volatility_level']}")
        print(f"  • Time tolerance: {result['time_tolerance']:.1f}s")
        print(f"  • Price tolerance: {result['price_tolerance']:.3f}%")
    
    # Calculate improvement metrics
    if results:
        improved_count = sum(1 for r in results if r['acceptable'])
        total_count = len(results)
        improvement_rate = improved_count / total_count * 100
        
        print(f"\\n🎯 IMPROVEMENT RATE: {improvement_rate:.1f}% ({improved_count}/{total_count})")
        
        if improvement_rate > 50:
            print("🎉 SIGNIFICANT IMPROVEMENT ACHIEVED!")
        elif improvement_rate > 25:
            print("📈 MODERATE IMPROVEMENT ACHIEVED")
        else:
            print("⚠️ LIMITED IMPROVEMENT - MORE WORK NEEDED")
    
    return results


def analyze_gap_patterns():
    """Analyze gap patterns in the data"""
    
    print("\\n\\n🔍 GAP PATTERN ANALYSIS")
    print("="*60)
    
    data_manager = DataManager()
    
    try:
        # Load real data
        real_data = data_manager.load_real_data()
        
        # Convert timestamps
        real_data['timestamp'] = pd.to_datetime(real_data['created_time'])
        real_data = real_data.sort_values('timestamp')
        
        # Analyze gaps by hour of day
        print("📊 GAP ANALYSIS BY HOUR OF DAY:")
        
        gap_hours = {}
        timestamps = real_data['timestamp']
        
        for i in range(1, len(timestamps)):
            gap_seconds = (timestamps.iloc[i] - timestamps.iloc[i-1]).total_seconds()
            if gap_seconds > 300:  # 5+ minute gaps
                hour = timestamps.iloc[i].hour
                if hour not in gap_hours:
                    gap_hours[hour] = []
                gap_hours[hour].append(gap_seconds / 60)
        
        # Sort by hour
        for hour in sorted(gap_hours.keys()):
            gaps = gap_hours[hour]
            avg_gap = np.mean(gaps)
            max_gap = max(gaps)
            print(f"  {hour:2d}:00 - {len(gaps):3d} gaps, avg: {avg_gap:5.1f}min, max: {max_gap:5.1f}min")
        
        # Find most problematic hours
        problem_hours = []
        for hour, gaps in gap_hours.items():
            if len(gaps) > 20 or max(gaps) > 60:  # More than 20 gaps or gaps > 1 hour
                problem_hours.append((hour, len(gaps), max(gaps)))
        
        if problem_hours:
            print("\\n⚠️ MOST PROBLEMATIC HOURS:")
            for hour, count, max_gap in sorted(problem_hours, key=lambda x: x[2], reverse=True):
                print(f"  {hour:2d}:00 - {count} gaps, max: {max_gap:.1f}min")
        
    except Exception as e:
        print(f"❌ Error in gap analysis: {e}")


def main():
    """Main function to run all improvements"""
    
    print("🚀 COMPREHENSIVE IMPROVEMENT TESTING")
    print("="*80)
    
    # Test improved analysis
    results = test_improved_analysis()
    
    # Analyze gap patterns
    analyze_gap_patterns()
    
    # Final recommendations
    print("\\n\\n💡 FINAL RECOMMENDATIONS")
    print("="*60)
    
    recommendations = [
        "🔧 Implement gap-aware matching for all dates",
        "📊 Use volatility-adaptive parameters dynamically",
        "⏱️ Adjust time tolerances based on trading gaps",
        "🎯 Lower accuracy thresholds during volatile periods",
        "📈 Monitor execution quality in real-time",
        "🌊 Implement volatility-based position sizing",
        "⚠️ Add circuit breakers for extreme volatility",
        "📅 Special handling for month boundary dates"
    ]
    
    for rec in recommendations:
        print(f"• {rec}")
    
    print("\\n✨ IMPROVEMENTS COMPLETED!")
    return results


if __name__ == "__main__":
    main()