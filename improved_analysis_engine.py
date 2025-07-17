#!/usr/bin/env python3
"""
Improved Analysis Engine with Volatility Management and Gap Handling

This module provides enhanced analysis capabilities that adapt to market conditions
and handle trading gaps more effectively.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import statistics
from volatility_manager import VolatilityManager
from deep_analysis_engine import DeepAnalysisEngine


class ImprovedAnalysisEngine(DeepAnalysisEngine):
    """
    Enhanced analysis engine with volatility management and gap handling
    """
    
    def __init__(self):
        super().__init__()
        self.volatility_manager = VolatilityManager()
        
        # Enhanced matching parameters
        self.adaptive_matching = True
        self.gap_aware_matching = True
        self.volatility_adaptive_thresholds = True
        
    def validate_simulation_accuracy_enhanced(self, real_trades: pd.DataFrame, 
                                            sim_trades: pd.DataFrame, 
                                            date_str: str) -> Dict:
        """
        Enhanced validation with volatility-aware and gap-aware matching
        
        Args:
            real_trades: DataFrame with real trading data
            sim_trades: DataFrame with simulation trading data  
            date_str: Date string for analysis
            
        Returns:
            Dictionary with enhanced validation results
        """
        # Load volatility analysis for this date
        volatility_analysis = self.volatility_manager.analyze_date_volatility(
            date_str, f'analysis_results/detail_{date_str}.json'
        )
        
        # Get adjusted parameters based on volatility
        adjusted_params = volatility_analysis['adjusted_parameters']
        
        # Use adaptive thresholds if enabled
        if self.volatility_adaptive_thresholds:
            time_tolerance = adjusted_params['time_tolerance']
            price_tolerance = adjusted_params['price_tolerance']
            accuracy_threshold = adjusted_params['accuracy_threshold']
        else:
            time_tolerance = 2.0
            price_tolerance = 0.005
            accuracy_threshold = 0.8
        
        # Perform gap-aware matching if enabled
        if self.gap_aware_matching:
            matches = self._gap_aware_matching(real_trades, sim_trades, time_tolerance, price_tolerance)
        else:
            matches = self._standard_matching(real_trades, sim_trades, time_tolerance, price_tolerance)
        
        # Calculate accuracy metrics
        total_testable = len(real_trades)
        accurate_matches = len([m for m in matches if m['accurate']])
        
        if total_testable > 0:
            accuracy_rate = accurate_matches / total_testable
            is_acceptable = accuracy_rate >= accuracy_threshold
        else:
            accuracy_rate = 0
            is_acceptable = False
        
        # Enhanced diagnostic information
        diagnostics = self._enhanced_diagnostics(
            real_trades, sim_trades, matches, volatility_analysis, date_str
        )
        
        return {
            'is_acceptable': is_acceptable,
            'accuracy_rate': accuracy_rate,
            'accurate_matches': accurate_matches,
            'total_testable': total_testable,
            'matches': matches,
            'volatility_analysis': volatility_analysis,
            'adjusted_parameters': adjusted_params,
            'diagnostics': diagnostics,
            'matching_method': 'gap_aware' if self.gap_aware_matching else 'standard'
        }
    
    def _gap_aware_matching(self, real_trades: pd.DataFrame, sim_trades: pd.DataFrame,
                           time_tolerance: float, price_tolerance: float) -> List[Dict]:
        """
        Gap-aware matching that accounts for trading interruptions
        
        Args:
            real_trades: DataFrame with real trading data
            sim_trades: DataFrame with simulation trading data
            time_tolerance: Time tolerance in seconds
            price_tolerance: Price tolerance as decimal (0.005 = 0.5%)
            
        Returns:
            List of match dictionaries
        """
        matches = []
        
        # Identify trading gaps in real data
        real_gaps = self._identify_trading_gaps(real_trades)
        
        # Create gap-aware time windows
        gap_adjusted_windows = self._create_gap_adjusted_windows(real_trades, real_gaps)
        
        # Match trades using gap-aware windows
        for _, real_trade in real_trades.iterrows():
            # Get adjusted time window for this trade
            adjusted_window = self._get_adjusted_window(
                real_trade, gap_adjusted_windows, time_tolerance
            )
            
            # Find potential matches within adjusted window
            potential_matches = self._find_potential_matches(
                real_trade, sim_trades, adjusted_window, price_tolerance
            )
            
            if potential_matches:
                # Select best match using enhanced scoring
                best_match = self._select_best_match(real_trade, potential_matches)
                
                # Create match record
                match_record = self._create_match_record(real_trade, best_match, price_tolerance)
                matches.append(match_record)
        
        return matches
    
    def _identify_trading_gaps(self, real_trades: pd.DataFrame) -> List[Dict]:
        """Identify significant trading gaps in real data"""
        gaps = []
        
        if len(real_trades) < 2:
            return gaps
        
        timestamps = real_trades['timestamp'].values
        
        for i in range(1, len(timestamps)):
            gap_seconds = (timestamps[i] - timestamps[i-1]).total_seconds()
            
            if gap_seconds > 300:  # 5+ minute gaps
                gaps.append({
                    'start_time': timestamps[i-1],
                    'end_time': timestamps[i],
                    'duration': gap_seconds,
                    'start_index': i-1,
                    'end_index': i
                })
        
        return gaps
    
    def _create_gap_adjusted_windows(self, real_trades: pd.DataFrame, gaps: List[Dict]) -> Dict:
        """Create adjusted time windows that account for trading gaps"""
        gap_adjusted_windows = {}
        
        for idx, row in real_trades.iterrows():
            trade_time = row['timestamp']
            
            # Check if this trade is near a gap
            near_gap = False
            gap_adjustment = 0
            
            for gap in gaps:
                # If trade is within 10 minutes of gap start
                if abs((trade_time - gap['start_time']).total_seconds()) < 600:
                    near_gap = True
                    gap_adjustment = min(gap['duration'] / 4, 120)  # Max 2 minutes extra
                    break
                # If trade is within 10 minutes of gap end
                elif abs((trade_time - gap['end_time']).total_seconds()) < 600:
                    near_gap = True
                    gap_adjustment = min(gap['duration'] / 4, 120)  # Max 2 minutes extra
                    break
            
            gap_adjusted_windows[idx] = {
                'base_time': trade_time,
                'gap_adjustment': gap_adjustment,
                'near_gap': near_gap
            }
        
        return gap_adjusted_windows
    
    def _get_adjusted_window(self, real_trade: pd.Series, gap_windows: Dict, 
                           base_tolerance: float) -> Tuple[float, float]:
        """Get adjusted time window for a specific trade"""
        trade_idx = real_trade.name
        
        if trade_idx in gap_windows:
            adjustment = gap_windows[trade_idx]['gap_adjustment']
            adjusted_tolerance = base_tolerance + adjustment
        else:
            adjusted_tolerance = base_tolerance
        
        trade_time = real_trade['timestamp']
        
        return (
            trade_time - pd.Timedelta(seconds=adjusted_tolerance),
            trade_time + pd.Timedelta(seconds=adjusted_tolerance)
        )
    
    def _find_potential_matches(self, real_trade: pd.Series, sim_trades: pd.DataFrame,
                               time_window: Tuple, price_tolerance: float) -> pd.DataFrame:
        """Find potential simulation matches within time window"""
        start_time, end_time = time_window
        
        # Time-based filtering
        time_mask = (
            (sim_trades['timestamp'] >= start_time) &
            (sim_trades['timestamp'] <= end_time) &
            (sim_trades['side'] == real_trade['side'])
        )
        
        potential_matches = sim_trades[time_mask]
        
        if potential_matches.empty:
            return potential_matches
        
        # Price-based filtering
        price_diff = abs(potential_matches['price'] - real_trade['price'])
        price_threshold = real_trade['price'] * price_tolerance
        price_mask = price_diff <= price_threshold
        
        return potential_matches[price_mask]
    
    def _select_best_match(self, real_trade: pd.Series, potential_matches: pd.DataFrame) -> pd.Series:
        """Select best match using enhanced scoring"""
        if len(potential_matches) == 1:
            return potential_matches.iloc[0]
        
        # Calculate composite score
        scores = []
        
        for _, sim_trade in potential_matches.iterrows():
            # Time difference score (lower is better)
            time_diff = abs((real_trade['timestamp'] - sim_trade['timestamp']).total_seconds())
            time_score = 1 / (1 + time_diff)
            
            # Price difference score (lower is better)
            price_diff = abs(real_trade['price'] - sim_trade['price'])
            price_score = 1 / (1 + price_diff)
            
            # Quantity difference score (lower is better)
            qty_diff = abs(real_trade['quantity'] - sim_trade['quantity'])
            qty_score = 1 / (1 + qty_diff)
            
            # Composite score (weighted)
            composite_score = (time_score * 0.5) + (price_score * 0.3) + (qty_score * 0.2)
            scores.append(composite_score)
        
        # Return best match
        best_idx = np.argmax(scores)
        return potential_matches.iloc[best_idx]
    
    def _create_match_record(self, real_trade: pd.Series, sim_trade: pd.Series, 
                           price_tolerance: float) -> Dict:
        """Create detailed match record"""
        time_diff = (real_trade['timestamp'] - sim_trade['timestamp']).total_seconds()
        price_diff = abs(real_trade['price'] - sim_trade['price'])
        price_diff_pct = price_diff / real_trade['price'] * 100
        
        # Determine if match is accurate
        is_accurate = (
            abs(time_diff) <= 2.0 and
            price_diff_pct <= (price_tolerance * 100)
        )
        
        return {
            'real_timestamp': real_trade['timestamp'],
            'sim_timestamp': sim_trade['timestamp'],
            'time_diff': time_diff,
            'real_price': real_trade['price'],
            'sim_price': sim_trade['price'],
            'price_diff': price_diff,
            'price_diff_pct': price_diff_pct,
            'real_quantity': real_trade['quantity'],
            'sim_quantity': sim_trade['quantity'],
            'side': real_trade['side'],
            'accurate': is_accurate
        }
    
    def _enhanced_diagnostics(self, real_trades: pd.DataFrame, sim_trades: pd.DataFrame,
                            matches: List[Dict], volatility_analysis: Dict, date_str: str) -> List[str]:
        """Generate enhanced diagnostic information"""
        diagnostics = []
        
        # Gap analysis
        gaps = self._identify_trading_gaps(real_trades)
        if gaps:
            max_gap = max(gap['duration'] for gap in gaps) / 60
            diagnostics.append(
                f"⚠️ Found {len(gaps)} trading gaps >5min (max: {max_gap:.0f}min) - possible algo crashes"
            )
        
        # Volatility analysis
        vol_level = volatility_analysis['volatility_level']
        if vol_level in ['HIGH', 'EXTREME']:
            diagnostics.append(f"🌊 {vol_level} volatility detected - adjusted parameters used")
        
        # Activity analysis
        if len(sim_trades) > 0:
            activity_ratio = len(sim_trades) / len(real_trades)
            if activity_ratio > 5:
                diagnostics.append(
                    f"📈 High sim activity ({len(sim_trades):,} vs {len(real_trades):,} live) - possible volatility over-reaction"
                )
            elif activity_ratio < 0.3:
                diagnostics.append(
                    f"📉 Low sim activity ({len(sim_trades):,} vs {len(real_trades):,} live) - possible under-reaction"
                )
        
        # Matching quality analysis
        if matches:
            accurate_matches = len([m for m in matches if m['accurate']])
            match_rate = accurate_matches / len(matches) * 100
            
            if match_rate < 50:
                diagnostics.append(f"❌ Low match rate ({match_rate:.1f}%) - timing issues detected")
            
            # Time synchronization analysis
            time_diffs = [m['time_diff'] for m in matches]
            avg_time_diff = np.mean(time_diffs)
            if abs(avg_time_diff) > 5:
                diagnostics.append(f"⏱️ Clock synchronization issue (avg: {avg_time_diff:+.1f}s)")
        
        # Date-specific diagnostics
        if date_str.endswith('01-06-2025'):
            diagnostics.append("📅 01-06-2025 is first simulation date - possible initialization issues")
        
        return diagnostics
    
    def _standard_matching(self, real_trades: pd.DataFrame, sim_trades: pd.DataFrame,
                         time_tolerance: float, price_tolerance: float) -> List[Dict]:
        """Standard matching algorithm (fallback)"""
        matches = []
        
        for _, real_trade in real_trades.iterrows():
            # Standard time window
            time_mask = (
                (sim_trades['timestamp'] >= real_trade['timestamp'] - pd.Timedelta(seconds=time_tolerance)) &
                (sim_trades['timestamp'] <= real_trade['timestamp'] + pd.Timedelta(seconds=time_tolerance)) &
                (sim_trades['side'] == real_trade['side'])
            )
            
            potential_matches = sim_trades[time_mask]
            
            if not potential_matches.empty:
                # Price filtering
                price_diff = abs(potential_matches['price'] - real_trade['price'])
                price_threshold = real_trade['price'] * price_tolerance
                price_mask = price_diff <= price_threshold
                
                valid_matches = potential_matches[price_mask]
                
                if not valid_matches.empty:
                    # Select best match
                    best_match = self._select_best_match(real_trade, valid_matches)
                    match_record = self._create_match_record(real_trade, best_match, price_tolerance)
                    matches.append(match_record)
        
        return matches


def main():
    """Test the improved analysis engine"""
    engine = ImprovedAnalysisEngine()
    
    # Test with a problematic date
    print("🔍 Testing Improved Analysis Engine")
    print("📅 Date: 25-06-2025")
    
    # This would need real data to test properly
    print("✅ Improved analysis engine created successfully")
    print("🎯 Features enabled:")
    print("  • Volatility-adaptive thresholds")
    print("  • Gap-aware matching")
    print("  • Enhanced diagnostics")


if __name__ == "__main__":
    main()