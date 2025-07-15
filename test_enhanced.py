#!/usr/bin/env python3
"""Test enhanced trading agent functionality."""

import pandas as pd
from enhanced_analysis import EnhancedTradeAnalyzer

def test_enhanced_analyzer():
    """Test enhanced analyzer with sample data."""
    
    # Create sample trade data
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2025-06-01', periods=100, freq='H'),
        'price': [50000 + i*10 + (i%10)*100 for i in range(100)],
        'quantity': [0.01 + (i%5)*0.005 for i in range(100)],
        'realized_pnl': [(i%10-5)*50 for i in range(100)],
        'side': ['B' if i%2==0 else 'A' for i in range(100)]
    })
    
    print("🧪 Testing Enhanced Analyzer with sample data...")
    print(f"📊 Sample data: {len(sample_data)} trades")
    
    analyzer = EnhancedTradeAnalyzer()
    
    # Test each analysis method
    tests = [
        ("Position Sizing", analyzer.analyze_position_sizing),
        ("Execution Quality", analyzer.analyze_execution_quality),
        ("Market Context", analyzer.analyze_market_context),
        ("Performance Attribution", analyzer.analyze_performance_attribution),
        ("Trade Sequences", analyzer.analyze_trade_sequences)
    ]
    
    for test_name, test_func in tests:
        try:
            print(f"\n🔬 Testing {test_name}...")
            result = test_func(sample_data)
            print(f"✅ {test_name}: {len(result)} chars output")
            if len(result) > 200:
                print(f"   Preview: {result[:200]}...")
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
    
    # Test comprehensive analysis
    try:
        print(f"\n🔬 Testing Comprehensive Analysis...")
        result = analyzer.get_comprehensive_analysis(sample_data)
        print(f"✅ Comprehensive: {len(result)} chars output")
        print("✅ All enhanced analysis features working!")
    except Exception as e:
        print(f"❌ Comprehensive analysis failed: {e}")

if __name__ == "__main__":
    test_enhanced_analyzer()