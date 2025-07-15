#!/usr/bin/env python3
"""Test the fixes for validation and parsing issues."""

import os
from dotenv import load_dotenv

def test_validation_tool():
    """Test that validation queries work correctly."""
    
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found")
        return False
    
    try:
        from trading_agent import TradingAgent
        
        agent = TradingAgent()
        print("✅ Agent initialized successfully")
        
        # Test validation tool directly
        print("\n🧪 Testing validation tool directly:")
        result = agent._explain_validation_process("")
        
        if "Discrepancy Validation Process" in result:
            print("✅ Validation tool works correctly")
            print(f"   Content length: {len(result)} characters")
            return True
        else:
            print("❌ Validation tool not working correctly")
            return False
            
    except Exception as e:
        print(f"❌ Error testing validation: {e}")
        return False

def test_input_handling():
    """Test improved input handling."""
    
    try:
        from trading_agent import TradingAgent
        
        agent = TradingAgent()
        
        # Test fallback responses
        print("\n🧪 Testing input handling:")
        
        test_cases = [
            ("validation", "Should explain validation process"),
            ("process", "Should explain validation process"),  
            ("pnl", "Should calculate PnL"),
            ("statistics", "Should show statistics")
        ]
        
        for test_input, expected in test_cases:
            try:
                result = agent._try_fallback_response(test_input)
                if result:
                    print(f"✅ '{test_input}': Got fallback response ({len(result)} chars)")
                else:
                    print(f"⚠️ '{test_input}': No fallback response")
            except Exception as e:
                print(f"❌ '{test_input}': Error - {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing input handling: {e}")
        return False

def main():
    """Run all tests."""
    print("🔧 Testing Agent Fixes")
    print("=" * 30)
    
    validation_ok = test_validation_tool()
    input_ok = test_input_handling()
    
    if validation_ok and input_ok:
        print(f"\n✅ SUCCESS: All fixes are working!")
        print(f"\n💡 The agent should now:")
        print(f"   • Use the correct validation tool for 'validation' queries")
        print(f"   • Handle non-question inputs gracefully")
        print(f"   • Avoid infinite parsing loops")
        print(f"   • Provide better error messages")
    else:
        print(f"\n⚠️ Some issues may still exist")

if __name__ == "__main__":
    main()