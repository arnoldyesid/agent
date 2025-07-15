#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Trading Analysis Module

Additional deep analysis capabilities for the trading agent.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class EnhancedTradeAnalyzer:
    """Enhanced analysis capabilities for deep trade insights."""
    
    def __init__(self):
        """Initialize the enhanced analyzer."""
        self.analysis_cache = {}
    
    def analyze_position_sizing(self, trade_data: pd.DataFrame) -> str:
        """Analyze position sizing patterns and recommendations."""
        try:
            if 'quantity' not in trade_data.columns:
                return "❌ No quantity data available for position sizing analysis."
            
            # Position size statistics
            qty_stats = trade_data['quantity'].describe()
            
            # Position size buckets
            buckets = pd.qcut(trade_data['quantity'], q=5, labels=['XS', 'S', 'M', 'L', 'XL'])
            bucket_performance = trade_data.groupby(buckets, observed=False).agg({
                'realized_pnl': ['sum', 'mean', 'count'] if 'realized_pnl' in trade_data.columns else 'count'
            })
            
            # Large position analysis
            large_threshold = qty_stats['75%']
            large_positions = trade_data[trade_data['quantity'] > large_threshold]
            
            result = f"""
📈 **Position Sizing Analysis**

## **Size Distribution**
🔢 **Average Position**: {qty_stats['mean']:.4f}
📊 **Median Position**: {qty_stats['50%']:.4f}
📈 **Largest Position**: {qty_stats['max']:.4f}
📉 **Smallest Position**: {qty_stats['min']:.4f}
📐 **Standard Deviation**: {qty_stats['std']:.4f}

## **Position Size Categories**
{self._format_bucket_analysis(bucket_performance)}

## **Large Position Analysis**
🎯 **Large Position Threshold**: {large_threshold:.4f}
📊 **Large Positions Count**: {len(large_positions)}
📊 **% of Total Trades**: {len(large_positions)/len(trade_data)*100:.1f}%
"""

            if 'realized_pnl' in trade_data.columns:
                large_pnl = large_positions['realized_pnl'].sum()
                result += f"💰 **Large Position PnL**: ${large_pnl:,.2f}\n"
                result += f"📊 **Avg PnL per Large Trade**: ${large_positions['realized_pnl'].mean():,.2f}\n"
            
            return result
            
        except Exception as e:
            return f"❌ Error in position sizing analysis: {str(e)}"
    
    def analyze_execution_quality(self, trade_data: pd.DataFrame) -> str:
        """Analyze trade execution quality and timing."""
        try:
            if 'timestamp' not in trade_data.columns:
                return "❌ No timestamp data available for execution analysis."
            
            # Convert timestamp to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(trade_data['timestamp']):
                trade_data['timestamp'] = pd.to_datetime(trade_data['timestamp'])
            
            # Time-based analysis
            trade_data['hour'] = trade_data['timestamp'].dt.hour
            trade_data['minute'] = trade_data['timestamp'].dt.minute
            trade_data['day_of_week'] = trade_data['timestamp'].dt.day_name()
            
            # Trade frequency analysis
            hourly_volume = trade_data.groupby('hour').size()
            peak_hour = hourly_volume.idxmax()
            peak_volume = hourly_volume.max()
            
            # Execution clustering (trades close in time)
            trade_data_sorted = trade_data.sort_values('timestamp')
            time_diffs = trade_data_sorted['timestamp'].diff().dt.total_seconds()
            rapid_trades = len(time_diffs[time_diffs < 60])  # Trades within 1 minute
            
            result = f"""
⚡ **Execution Quality Analysis**

## **Trading Timing**
🕐 **Peak Trading Hour**: {peak_hour}:00 ({peak_volume} trades)
⏱️ **Rapid Fire Trades**: {rapid_trades} (within 60s of each other)
📊 **Avg Time Between Trades**: {time_diffs.mean()/60:.1f} minutes

## **Hourly Distribution**
"""
            
            # Add hourly breakdown
            for hour in sorted(hourly_volume.index):
                volume = hourly_volume[hour]
                bar = "█" * int(volume / hourly_volume.max() * 20)
                result += f"   {hour:02d}:00 │{bar:<20}│ {volume:3d} trades\n"
            
            # Day of week analysis
            daily_volume = trade_data.groupby('day_of_week').size()
            result += f"\n## **Daily Distribution**\n"
            for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
                if day in daily_volume.index:
                    volume = daily_volume[day]
                    result += f"📅 **{day}**: {volume} trades\n"
            
            return result
            
        except Exception as e:
            return f"❌ Error in execution quality analysis: {str(e)}"
    
    def analyze_market_context(self, trade_data: pd.DataFrame) -> str:
        """Analyze trades in market context."""
        try:
            if 'price' not in trade_data.columns:
                return "❌ No price data available for market context analysis."
            
            # Price trend analysis
            trade_data_sorted = trade_data.sort_values('timestamp')
            trade_data_sorted['price_change'] = trade_data_sorted['price'].pct_change()
            trade_data_sorted['price_ma_10'] = trade_data_sorted['price'].rolling(window=10).mean()
            
            # Market condition categorization
            conditions = []
            for idx, row in trade_data_sorted.iterrows():
                if pd.isna(row['price_ma_10']):
                    conditions.append('Unknown')
                elif row['price'] > row['price_ma_10'] * 1.01:
                    conditions.append('Strong Up')
                elif row['price'] > row['price_ma_10']:
                    conditions.append('Up Trend')
                elif row['price'] < row['price_ma_10'] * 0.99:
                    conditions.append('Strong Down')
                else:
                    conditions.append('Down Trend')
            
            trade_data_sorted['market_condition'] = conditions
            
            # Performance by market condition
            condition_performance = trade_data_sorted.groupby('market_condition').agg({
                'realized_pnl': ['sum', 'mean', 'count'] if 'realized_pnl' in trade_data_sorted.columns else 'count',
                'price': ['mean']
            })
            
            result = f"""
🌍 **Market Context Analysis**

## **Price Movement**
📈 **Price Range**: ${trade_data['price'].min():,.2f} - ${trade_data['price'].max():,.2f}
📊 **Average Price**: ${trade_data['price'].mean():,.2f}
📉 **Price Volatility**: {trade_data['price'].std():,.2f}

## **Market Conditions Performance**
"""
            
            if 'realized_pnl' in trade_data_sorted.columns:
                for condition in condition_performance.index:
                    count = condition_performance.loc[condition, ('realized_pnl', 'count')]
                    total_pnl = condition_performance.loc[condition, ('realized_pnl', 'sum')]
                    avg_pnl = condition_performance.loc[condition, ('realized_pnl', 'mean')]
                    avg_price = condition_performance.loc[condition, ('price', 'mean')]
                    
                    result += f"📊 **{condition}**: {count} trades, ${total_pnl:,.2f} total, ${avg_pnl:,.2f} avg\n"
            
            return result
            
        except Exception as e:
            return f"❌ Error in market context analysis: {str(e)}"
    
    def analyze_performance_attribution(self, trade_data: pd.DataFrame) -> str:
        """Analyze what factors contribute most to performance."""
        try:
            if 'realized_pnl' not in trade_data.columns:
                return "❌ No PnL data available for performance attribution."
            
            # Trade size impact
            trade_data['size_bucket'] = pd.qcut(trade_data['quantity'], q=3, labels=['Small', 'Medium', 'Large'])
            size_performance = trade_data.groupby('size_bucket', observed=False)['realized_pnl'].agg(['sum', 'mean', 'count'])
            
            # Time of day impact
            trade_data['hour'] = pd.to_datetime(trade_data['timestamp']).dt.hour
            trade_data['time_bucket'] = pd.cut(trade_data['hour'], 
                                             bins=[0, 6, 12, 18, 24], 
                                             labels=['Night', 'Morning', 'Afternoon', 'Evening'])
            time_performance = trade_data.groupby('time_bucket', observed=False)['realized_pnl'].agg(['sum', 'mean', 'count'])
            
            # Side impact
            side_performance = trade_data.groupby('side')['realized_pnl'].agg(['sum', 'mean', 'count'])
            
            result = f"""
🎯 **Performance Attribution Analysis**

## **By Position Size**
"""
            
            for size in size_performance.index:
                total = size_performance.loc[size, 'sum']
                avg = size_performance.loc[size, 'mean']
                count = size_performance.loc[size, 'count']
                contrib = total / trade_data['realized_pnl'].sum() * 100
                result += f"📊 **{size} Positions**: ${total:,.2f} ({contrib:.1f}%) from {count} trades (${avg:,.2f} avg)\n"
            
            result += f"\n## **By Time of Day**\n"
            for time_bucket in time_performance.index:
                total = time_performance.loc[time_bucket, 'sum']
                avg = time_performance.loc[time_bucket, 'mean']
                count = time_performance.loc[time_bucket, 'count']
                contrib = total / trade_data['realized_pnl'].sum() * 100
                result += f"🕐 **{time_bucket}**: ${total:,.2f} ({contrib:.1f}%) from {count} trades (${avg:,.2f} avg)\n"
            
            result += f"\n## **By Trade Side**\n"
            for side in side_performance.index:
                side_name = "Buy" if side == "B" else "Sell" if side == "A" else side
                total = side_performance.loc[side, 'sum']
                avg = side_performance.loc[side, 'mean']
                count = side_performance.loc[side, 'count']
                contrib = total / trade_data['realized_pnl'].sum() * 100
                result += f"📈 **{side_name}**: ${total:,.2f} ({contrib:.1f}%) from {count} trades (${avg:,.2f} avg)\n"
            
            return result
            
        except Exception as e:
            return f"❌ Error in performance attribution: {str(e)}"
    
    def analyze_trade_sequences(self, trade_data: pd.DataFrame) -> str:
        """Analyze sequences and patterns in trading behavior."""
        try:
            if len(trade_data) < 5:
                return "❌ Not enough trades for sequence analysis."
            
            # Sort by timestamp
            trade_data_sorted = trade_data.sort_values('timestamp')
            
            # Consecutive wins/losses
            if 'realized_pnl' in trade_data_sorted.columns:
                trade_data_sorted['is_win'] = trade_data_sorted['realized_pnl'] > 0
                
                # Find streaks
                streaks = []
                current_streak = 1
                current_type = trade_data_sorted.iloc[0]['is_win']
                
                for i in range(1, len(trade_data_sorted)):
                    if trade_data_sorted.iloc[i]['is_win'] == current_type:
                        current_streak += 1
                    else:
                        streaks.append({'type': 'Win' if current_type else 'Loss', 'length': current_streak})
                        current_streak = 1
                        current_type = trade_data_sorted.iloc[i]['is_win']
                
                # Add final streak
                streaks.append({'type': 'Win' if current_type else 'Loss', 'length': current_streak})
                
                # Analyze streaks
                win_streaks = [s['length'] for s in streaks if s['type'] == 'Win']
                loss_streaks = [s['length'] for s in streaks if s['type'] == 'Loss']
                
                max_win_streak = max(win_streaks) if win_streaks else 0
                max_loss_streak = max(loss_streaks) if loss_streaks else 0
                avg_win_streak = np.mean(win_streaks) if win_streaks else 0
                avg_loss_streak = np.mean(loss_streaks) if loss_streaks else 0
            
            # Side sequences
            trade_data_sorted['side_change'] = trade_data_sorted['side'] != trade_data_sorted['side'].shift(1)
            direction_changes = trade_data_sorted['side_change'].sum()
            
            result = f"""
🔄 **Trade Sequence Analysis**

## **Winning/Losing Streaks**
🏆 **Longest Win Streak**: {max_win_streak} trades
📉 **Longest Loss Streak**: {max_loss_streak} trades
📊 **Average Win Streak**: {avg_win_streak:.1f} trades
📊 **Average Loss Streak**: {avg_loss_streak:.1f} trades

## **Trading Behavior**
🔄 **Direction Changes**: {direction_changes} times
📊 **Directional Consistency**: {(1 - direction_changes/len(trade_data))*100:.1f}%
📈 **Total Streaks**: {len(streaks)}
"""
            
            if len(streaks) > 0:
                result += f"🎯 **Average Streak Length**: {np.mean([s['length'] for s in streaks]):.1f} trades\n"
            
            return result
            
        except Exception as e:
            return f"❌ Error in sequence analysis: {str(e)}"
    
    def _format_bucket_analysis(self, bucket_performance) -> str:
        """Format bucket analysis results."""
        try:
            result = ""
            for bucket in bucket_performance.index:
                if pd.isna(bucket):
                    continue
                count = bucket_performance.loc[bucket, ('realized_pnl', 'count')] if 'realized_pnl' in bucket_performance.columns else bucket_performance.loc[bucket]
                result += f"📊 **{bucket} Positions**: {count} trades\n"
            return result
        except:
            return "📊 Bucket analysis data formatting error\n"
    
    def get_comprehensive_analysis(self, trade_data: pd.DataFrame) -> str:
        """Get comprehensive deep analysis combining all methods."""
        try:
            result = "🔍 **Comprehensive Deep Trading Analysis**\n"
            result += "=" * 50 + "\n\n"
            
            # Run all analysis methods
            analyses = [
                ("Position Sizing", self.analyze_position_sizing),
                ("Execution Quality", self.analyze_execution_quality),
                ("Market Context", self.analyze_market_context),
                ("Performance Attribution", self.analyze_performance_attribution),
                ("Trade Sequences", self.analyze_trade_sequences)
            ]
            
            for name, method in analyses:
                try:
                    analysis_result = method(trade_data)
                    result += f"{analysis_result}\n\n"
                except Exception as e:
                    result += f"❌ {name} analysis failed: {str(e)}\n\n"
            
            return result
            
        except Exception as e:
            return f"❌ Error in comprehensive analysis: {str(e)}"