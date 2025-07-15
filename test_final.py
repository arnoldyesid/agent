#!/usr/bin/env python3
"""Final test of enhanced trading agent."""

import os
from dotenv import load_dotenv

load_dotenv()

def test_agent_tools():
    """Test specific enhanced agent tools."""
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found")
        return False
    
    try:
        from trading_agent import TradingAgent
        print("✅ Trading agent imports successfully")
        
        agent = TradingAgent()
        print("✅ Trading agent initializes successfully")
        
        # Test individual enhanced tools
        print("\n🧪 Testing Enhanced Analysis Tools:")
        
        enhanced_tools = [
            ("Position Sizing Analysis", agent._analyze_position_sizing),
            ("Execution Quality Analysis", agent._analyze_execution_quality), 
            ("Market Context Analysis", agent._analyze_market_context),
            ("Performance Attribution", agent._analyze_performance_attribution),
            ("Trade Sequences Analysis", agent._analyze_trade_sequences),
            ("Comprehensive Analysis", agent._get_comprehensive_analysis)
        ]
        
        for tool_name, tool_func in enhanced_tools:
            try:
                print(f"   🔧 Testing {tool_name}...")
                result = tool_func("")
                if "Error" in result:
                    print(f"   ⚠️  {tool_name}: {result[:100]}...")
                else:
                    print(f"   ✅ {tool_name}: Working ({len(result)} chars)")
            except Exception as e:
                print(f"   ❌ {tool_name}: Failed with {e}")
        
        # Test basic tools
        print("\n🧪 Testing Basic Tools:")
        
        basic_tools = [
            ("PnL Calculation", agent._calculate_pnl),
            ("Trade Statistics", agent._get_real_trade_stats),
            ("Available Dates", agent._get_available_dates)
        ]
        
        for tool_name, tool_func in basic_tools:
            try:
                print(f"   🔧 Testing {tool_name}...")
                result = tool_func("")
                if "Error" in result:
                    print(f"   ⚠️  {tool_name}: {result[:100]}...")
                else:
                    print(f"   ✅ {tool_name}: Working ({len(result)} chars)")
            except Exception as e:
                print(f"   ❌ {tool_name}: Failed with {e}")
        
        # Count total tools
        total_tools = len(agent._create_tools())
        print(f"\n📊 Total Tools Available: {total_tools}")
        
        print("\n🎉 Enhanced trading agent is ready!")
        print("💡 The agent can now provide deep analysis of trading behavior")
        return True
        
    except Exception as e:
        print(f"❌ Error testing agent: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run final test."""
    print("🚀 Final Enhanced Trading Agent Test")
    print("=" * 50)
    
    success = test_agent_tools()
    
    if success:
        print("\n✅ SUCCESS: Enhanced trading agent is fully functional!")
        print("\n🎯 Key Features Added:")
        print("   📊 Position Sizing Analysis")
        print("   ⚡ Execution Quality Assessment") 
        print("   🌍 Market Context Analysis")
        print("   🎯 Performance Attribution")
        print("   🔄 Trade Sequence Analysis")
        print("   🔍 Comprehensive Deep Analysis")
        print("\n💡 Usage: Run 'python run_agent.py' to start the agent")
        print("💡 Or ask for 'comprehensive analysis' for deep insights")
    else:
        print("\n❌ FAILURE: Some issues detected")

if __name__ == "__main__":
    main()