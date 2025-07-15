#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean wrapper for the trading agent with minimal logging.
"""

import os
import sys
import warnings
from dotenv import load_dotenv

# Suppress warnings and unnecessary output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Load environment
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("Error: Please set OPENAI_API_KEY in .env file")
    print("Optional: Set OPENAI_MODEL in .env file (default: gpt-4o)")
    sys.exit(1)

def simple_interface():
    """Simple interface that avoids the complex agent setup."""
    from data_manager import DataManager
    import pandas as pd
    
    print("🚀 Enhanced Trading Analysis Interface")
    print("💡 Available commands:")
    print("  1. Type 'pnl' for profit/loss calculation")
    print("  2. Type 'stats' for trade statistics") 
    print("  3. Type 'dates' for available dates")
    print("  4. Type 'validation' for validation process explanation")
    print("  5. Type 'exit' to quit")
    print("🆕 Enhanced with deep analysis capabilities!")
    print("🛑 This mode avoids complex AI agent issues\n")
    
    try:
        dm = DataManager()
        
        while True:
            try:
                cmd = input("Command: ").strip().lower()
                
                if cmd in ['exit', 'quit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif cmd in ['pnl', 'profit']:
                    chunks = dm.get_real_trade_chunks()
                    if chunks:
                        all_data = pd.concat(chunks, ignore_index=True)
                        total_pnl = all_data['realized_pnl'].sum() if 'realized_pnl' in all_data.columns else 0
                        total_fees = all_data['total_fee'].sum() if 'total_fee' in all_data.columns else 0
                        print(f"\n💰 Total PnL: ${total_pnl:,.2f}")
                        print(f"💸 Total Fees: ${total_fees:,.2f}")
                        print(f"💵 Net Profit: ${total_pnl - total_fees:,.2f}\n")
                    else:
                        print("❌ No data available\n")
                
                elif cmd in ['stats', 'statistics']:
                    chunks = dm.get_real_trade_chunks()
                    if chunks:
                        all_data = pd.concat(chunks, ignore_index=True)
                        print(f"\n📊 Trade Statistics:")
                        print(f"  Total trades: {len(all_data):,}")
                        if 'price' in all_data.columns:
                            print(f"  Avg price: ${all_data['price'].mean():,.2f}")
                        if 'side' in all_data.columns:
                            sides = all_data['side'].value_counts()
                            for side, count in sides.items():
                                side_name = "Buy" if side == "B" else "Sell" if side == "A" else side
                                print(f"  {side_name} trades: {count:,}")
                        print()
                    else:
                        print("❌ No data available\n")
                
                elif cmd in ['dates', 'available']:
                    dates = dm.get_available_dates()
                    print(f"\n📅 Available dates ({len(dates)} total):")
                    for date in sorted(dates):
                        print(f"  {date}")
                    print()
                
                elif cmd in ['validation', 'process', 'validate']:
                    print("""
🔍 **Discrepancy Validation Process**

## **Data Preparation**
1. Load real trading records from CSV
2. Load simulation data from pickle files
3. Parse timestamps and ensure data integrity

## **Temporal Alignment**
3. Use timestamp-based matching with 2-second tolerance
4. Apply pd.merge_asof() for nearest trade matching
5. Rename columns: price_real vs price_sim

## **Discrepancy Detection**
6. Calculate: price_diff = abs(price_real - price_sim)
7. Apply $2.00 threshold for significant discrepancies
8. Generate statistics and percentage differences

## **AI Analysis**
9. Feed samples to GPT-4o for pattern recognition
10. Analyze causes: latency, slippage, market conditions
11. Provide simulation improvement recommendations

## **Success Metrics**
- Threshold Compliance: <$2 difference acceptable
- Temporal Matching: Within 2-second window
- Volume Coverage: % of real trades matched
- Statistical Analysis: Distribution patterns

**Framework**: Pandas + LangChain + GPT-4o
**Performance**: Chunked processing with caching
**Output**: Detailed statistics + AI insights
                    """)
                
                else:
                    print("❓ Unknown command. Try: pnl, stats, dates, validation, or exit\n")
                    
            except KeyboardInterrupt:
                print("\n⏹️ Interrupted")
                continue
            except Exception as e:
                print(f"❌ Error: {e}\n")
                
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")

def main():
    print("Initializing Trading Analysis Agent...")
    
    try:
        from trading_agent import TradingAgent
        agent = TradingAgent()
        agent.run_interactive()
    except Exception as e:
        print(f"⚠️ AI agent failed to start: {e}")
        print("🔄 Falling back to simple interface...\n")
        simple_interface()

if __name__ == "__main__":
    main()