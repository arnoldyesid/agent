#!/usr/bin/env python3
"""Simple test script for the trading agent."""

import os
import sys
from dotenv import load_dotenv

# Load environment
load_dotenv()

def test_data_loading():
    """Test basic data loading functionality."""
    try:
        from data_manager import DataManager
        print("✅ DataManager imports successfully")
        
        dm = DataManager()
        print("✅ DataManager initializes successfully")
        
        # Test real data loading
        chunks = dm.get_real_trade_chunks()
        print(f"✅ Real data loaded: {len(chunks)} chunks")
        
        # Test available dates
        dates = dm.get_available_dates()
        print(f"✅ Available dates: {len(dates)} dates")
        
        if chunks:
            import pandas as pd
            all_data = pd.concat(chunks, ignore_index=True)
            print(f"✅ Total trades: {len(all_data):,}")
            
            # Check for essential columns
            required_cols = ['price', 'quantity', 'side', 'timestamp']
            missing_cols = [col for col in required_cols if col not in all_data.columns]
            if missing_cols:
                print(f"⚠️ Missing columns: {missing_cols}")
            else:
                print("✅ All required columns present")
                
            # Basic statistics
            if 'realized_pnl' in all_data.columns:
                total_pnl = all_data['realized_pnl'].sum()
                print(f"📊 Total PnL: ${total_pnl:,.2f}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error in data loading: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_analysis_engine():
    """Test analysis engine functionality."""
    try:
        from analysis_engine import AnalysisEngine
        from langchain_openai import ChatOpenAI
        
        print("✅ AnalysisEngine imports successfully")
        
        # Create a simple LLM instance
        llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
        engine = AnalysisEngine(llm)
        print("✅ AnalysisEngine initializes successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in analysis engine: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agent_tools():
    """Test individual agent tools."""
    try:
        from trading_agent import TradingAgent
        print("✅ TradingAgent imports successfully")
        
        agent = TradingAgent()
        print("✅ TradingAgent initializes successfully")
        
        # Test individual tools
        print("\n🔧 Testing individual tools:")
        
        # Test PnL calculation
        try:
            pnl_result = agent._calculate_pnl("")
            print("✅ PnL calculation works")
            print(f"   Preview: {pnl_result[:200]}...")
        except Exception as e:
            print(f"❌ PnL calculation failed: {e}")
        
        # Test trade statistics
        try:
            stats_result = agent._get_real_trade_stats("")
            print("✅ Trade statistics work")
            print(f"   Preview: {stats_result[:200]}...")
        except Exception as e:
            print(f"❌ Trade statistics failed: {e}")
        
        # Test available dates
        try:
            dates_result = agent._get_available_dates("")
            print("✅ Available dates work")
            print(f"   Result: {dates_result}")
        except Exception as e:
            print(f"❌ Available dates failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in agent tools: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Trading Agent Components\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment")
        return
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Analysis Engine", test_analysis_engine), 
        ("Agent Tools", test_agent_tools)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}:")
        print("-" * 40)
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n🎯 Summary: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Agent is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()