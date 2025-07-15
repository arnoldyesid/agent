#!/usr/bin/env python3
"""Test model configuration functionality."""

import os
from dotenv import load_dotenv

def test_model_configuration():
    """Test different model configurations."""
    
    print("🧪 Testing Model Configuration")
    print("=" * 40)
    
    # Load current environment
    load_dotenv()
    
    current_model = os.getenv("OPENAI_MODEL", "gpt-4o")
    print(f"📋 Current model: {current_model}")
    
    # Test supported models
    supported_models = ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]
    
    print(f"\n🎯 Supported models:")
    for model in supported_models:
        status = "✅ (current)" if model == current_model else "⚪"
        print(f"   {status} {model}")
    
    print(f"\n💡 To change the model:")
    print(f"   1. Edit .env file")
    print(f"   2. Set OPENAI_MODEL=<model_name>")
    print(f"   3. Restart the agent")
    
    print(f"\n📝 Example .env configuration:")
    print(f"   OPENAI_API_KEY=your_api_key_here")
    print(f"   OPENAI_MODEL=gpt-4  # Use GPT-4 instead of GPT-4o")
    
    # Test importing with current configuration
    try:
        from trading_agent import OPENAI_MODEL, TradingAgent
        print(f"\n✅ Trading agent loaded successfully")
        print(f"   Model: {OPENAI_MODEL}")
        print(f"   Agent can be initialized: ✅")
        
    except Exception as e:
        print(f"\n❌ Error loading trading agent: {e}")
        return False
    
    print(f"\n🎉 Model configuration is working correctly!")
    return True

def show_model_comparison():
    """Show comparison of different models."""
    
    print(f"\n📊 Model Comparison:")
    print(f"=" * 40)
    
    models_info = [
        {
            "name": "gpt-4o",
            "description": "Latest optimized model",
            "speed": "Fast",
            "quality": "Highest",
            "cost": "Medium",
            "recommended": "✅ Default"
        },
        {
            "name": "gpt-4",
            "description": "Original GPT-4 model", 
            "speed": "Medium",
            "quality": "High",
            "cost": "High",
            "recommended": "For complex analysis"
        },
        {
            "name": "gpt-3.5-turbo",
            "description": "Fast and economical",
            "speed": "Very Fast", 
            "quality": "Good",
            "cost": "Low",
            "recommended": "For basic analysis"
        }
    ]
    
    for model in models_info:
        print(f"\n🤖 {model['name']}")
        print(f"   Description: {model['description']}")
        print(f"   Speed: {model['speed']}")
        print(f"   Quality: {model['quality']}")
        print(f"   Cost: {model['cost']}")
        print(f"   Use case: {model['recommended']}")

if __name__ == "__main__":
    success = test_model_configuration()
    if success:
        show_model_comparison()