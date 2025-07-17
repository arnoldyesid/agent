#!/usr/bin/env python3
"""
Volatility Management System for Cryptocurrency Trading Simulation

This module provides dynamic volatility detection and parameter adjustment
to improve simulation accuracy during volatile market conditions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import statistics


class VolatilityManager:
    """
    Manages volatility detection and parameter adjustment for trading simulation
    """
    
    def __init__(self, config_path: str = "data/HL_btcusdt_BTC/metadata.json"):
        """Initialize volatility manager with configuration"""
        self.config_path = config_path
        self.load_config()
        
        # Volatility thresholds
        self.spread_volatility_threshold = 2.0  # Standard deviations
        self.volume_activity_threshold = 3.0    # Multiple of normal activity
        self.gap_threshold = 300                # 5 minutes in seconds
        self.price_volatility_threshold = 0.02  # 2% price volatility
        
        # Adaptive parameters
        self.base_time_tolerance = 2.0          # Base time tolerance in seconds
        self.base_price_tolerance = 0.005       # Base price tolerance (0.5%)
        self.max_time_tolerance = 10.0          # Maximum time tolerance
        self.max_price_tolerance = 0.02         # Maximum price tolerance (2%)
        
    def load_config(self):
        """Load configuration from metadata file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            print(f"⚠️ Config file not found: {self.config_path}")
            self.config = self._default_config()
    
    def _default_config(self) -> Dict:
        """Default configuration if file not found"""
        return {
            "fast_depth": 2000,
            "slow_depth": 15000,
            "symbol": "BTC/USDT",
            "timeframe": "1m"
        }
    
    def detect_volatility(self, market_data: Dict) -> Dict:
        """
        Detect volatility indicators from market data
        
        Args:
            market_data: Dictionary containing market data with keys:
                - spreads: List of bid-ask spreads
                - volumes: List of trading volumes
                - prices: List of prices
                - timestamps: List of timestamps
                
        Returns:
            Dictionary with volatility metrics and classification
        """
        volatility_metrics = {}
        
        # Spread volatility analysis
        if 'spreads' in market_data and market_data['spreads']:
            spreads = market_data['spreads']
            volatility_metrics['spread_volatility'] = np.std(spreads)
            volatility_metrics['avg_spread'] = np.mean(spreads)
            volatility_metrics['max_spread'] = np.max(spreads)
            volatility_metrics['spread_volatility_ratio'] = (
                volatility_metrics['spread_volatility'] / volatility_metrics['avg_spread']
                if volatility_metrics['avg_spread'] > 0 else 0
            )
        
        # Volume volatility analysis  
        if 'volumes' in market_data and market_data['volumes']:
            volumes = market_data['volumes']
            volatility_metrics['volume_volatility'] = np.std(volumes)
            volatility_metrics['avg_volume'] = np.mean(volumes)
            volatility_metrics['volume_volatility_ratio'] = (
                volatility_metrics['volume_volatility'] / volatility_metrics['avg_volume']
                if volatility_metrics['avg_volume'] > 0 else 0
            )
        
        # Price volatility analysis
        if 'prices' in market_data and market_data['prices']:
            prices = market_data['prices']
            returns = np.diff(prices) / prices[:-1]
            volatility_metrics['price_volatility'] = np.std(returns)
            volatility_metrics['price_volatility_annualized'] = (
                volatility_metrics['price_volatility'] * np.sqrt(525600)  # Annualized (minutes)
            )
        
        # Gap analysis
        if 'timestamps' in market_data and market_data['timestamps']:
            timestamps = market_data['timestamps']
            gaps = []
            for i in range(1, len(timestamps)):
                gap = timestamps[i] - timestamps[i-1]
                if gap > self.gap_threshold:
                    gaps.append(gap)
            
            volatility_metrics['trading_gaps'] = len(gaps)
            volatility_metrics['max_gap'] = max(gaps) if gaps else 0
            volatility_metrics['avg_gap'] = np.mean(gaps) if gaps else 0
        
        # Classify volatility level
        volatility_level = self._classify_volatility(volatility_metrics)
        volatility_metrics['volatility_level'] = volatility_level
        
        return volatility_metrics
    
    def _classify_volatility(self, metrics: Dict) -> str:
        """Classify volatility level based on metrics"""
        score = 0
        
        # Spread volatility contribution
        if 'spread_volatility_ratio' in metrics:
            if metrics['spread_volatility_ratio'] > 2.0:
                score += 3
            elif metrics['spread_volatility_ratio'] > 1.0:
                score += 2
            elif metrics['spread_volatility_ratio'] > 0.5:
                score += 1
        
        # Volume volatility contribution
        if 'volume_volatility_ratio' in metrics:
            if metrics['volume_volatility_ratio'] > 2.0:
                score += 3
            elif metrics['volume_volatility_ratio'] > 1.0:
                score += 2
            elif metrics['volume_volatility_ratio'] > 0.5:
                score += 1
        
        # Price volatility contribution
        if 'price_volatility' in metrics:
            if metrics['price_volatility'] > 0.05:  # 5%
                score += 3
            elif metrics['price_volatility'] > 0.02:  # 2%
                score += 2
            elif metrics['price_volatility'] > 0.01:  # 1%
                score += 1
        
        # Gap contribution
        if 'trading_gaps' in metrics:
            if metrics['trading_gaps'] > 50:
                score += 3
            elif metrics['trading_gaps'] > 20:
                score += 2
            elif metrics['trading_gaps'] > 10:
                score += 1
        
        # Classify based on total score
        if score >= 8:
            return "EXTREME"
        elif score >= 6:
            return "HIGH"
        elif score >= 4:
            return "MEDIUM"
        elif score >= 2:
            return "LOW"
        else:
            return "NORMAL"
    
    def adjust_parameters(self, volatility_level: str) -> Dict:
        """
        Adjust simulation parameters based on volatility level
        
        Args:
            volatility_level: Volatility classification (NORMAL, LOW, MEDIUM, HIGH, EXTREME)
            
        Returns:
            Dictionary with adjusted parameters
        """
        # Base parameters
        adjusted_params = {
            'time_tolerance': self.base_time_tolerance,
            'price_tolerance': self.base_price_tolerance,
            'fast_depth': self.config.get('fast_depth', 2000),
            'slow_depth': self.config.get('slow_depth', 15000),
            'accuracy_threshold': 0.8,
            'volatility_adjustments': {}
        }
        
        # Adjust based on volatility level
        if volatility_level == "EXTREME":
            adjusted_params['time_tolerance'] = min(self.max_time_tolerance, self.base_time_tolerance * 3.0)
            adjusted_params['price_tolerance'] = min(self.max_price_tolerance, self.base_price_tolerance * 2.5)
            adjusted_params['fast_depth'] = int(self.config.get('fast_depth', 2000) * 1.5)
            adjusted_params['slow_depth'] = int(self.config.get('slow_depth', 15000) * 1.3)
            adjusted_params['accuracy_threshold'] = 0.6
            adjusted_params['volatility_adjustments']['reason'] = "Extreme volatility detected"
            
        elif volatility_level == "HIGH":
            adjusted_params['time_tolerance'] = min(self.max_time_tolerance, self.base_time_tolerance * 2.0)
            adjusted_params['price_tolerance'] = min(self.max_price_tolerance, self.base_price_tolerance * 2.0)
            adjusted_params['fast_depth'] = int(self.config.get('fast_depth', 2000) * 1.3)
            adjusted_params['slow_depth'] = int(self.config.get('slow_depth', 15000) * 1.2)
            adjusted_params['accuracy_threshold'] = 0.7
            adjusted_params['volatility_adjustments']['reason'] = "High volatility detected"
            
        elif volatility_level == "MEDIUM":
            adjusted_params['time_tolerance'] = min(self.max_time_tolerance, self.base_time_tolerance * 1.5)
            adjusted_params['price_tolerance'] = min(self.max_price_tolerance, self.base_price_tolerance * 1.5)
            adjusted_params['fast_depth'] = int(self.config.get('fast_depth', 2000) * 1.1)
            adjusted_params['slow_depth'] = int(self.config.get('slow_depth', 15000) * 1.1)
            adjusted_params['accuracy_threshold'] = 0.75
            adjusted_params['volatility_adjustments']['reason'] = "Medium volatility detected"
            
        elif volatility_level == "LOW":
            # Tighten parameters for better accuracy
            adjusted_params['time_tolerance'] = max(1.0, self.base_time_tolerance * 0.8)
            adjusted_params['price_tolerance'] = max(0.001, self.base_price_tolerance * 0.8)
            adjusted_params['accuracy_threshold'] = 0.85
            adjusted_params['volatility_adjustments']['reason'] = "Low volatility - tightened parameters"
            
        # Add adjustment metadata
        adjusted_params['volatility_adjustments'].update({
            'original_time_tolerance': self.base_time_tolerance,
            'original_price_tolerance': self.base_price_tolerance,
            'adjustment_factor': adjusted_params['time_tolerance'] / self.base_time_tolerance,
            'volatility_level': volatility_level
        })
        
        return adjusted_params
    
    def analyze_date_volatility(self, date_str: str, analysis_file: str) -> Dict:
        """
        Analyze volatility for a specific date using analysis results
        
        Args:
            date_str: Date string in format 'DD-MM-YYYY'
            analysis_file: Path to analysis results file
            
        Returns:
            Dictionary with volatility analysis and recommendations
        """
        try:
            with open(analysis_file, 'r') as f:
                data = json.load(f)
            
            # Extract market data
            market_data = {}
            
            # Spread analysis
            if 'spread_analysis' in data:
                spread_data = data['spread_analysis']
                market_data['avg_spread'] = spread_data.get('avg_spread', 0)
                market_data['spread_volatility'] = spread_data.get('spread_volatility', 0)
                market_data['max_spread'] = spread_data.get('max_spread', 0)
            
            # Volume analysis
            if 'volume_analysis' in data:
                volume_data = data['volume_analysis']
                market_data['sim_volume'] = volume_data.get('sim_total_volume', 0)
                market_data['real_volume'] = volume_data.get('real_total_volume', 0)
                market_data['volume_ratio'] = (
                    market_data['sim_volume'] / market_data['real_volume']
                    if market_data['real_volume'] > 0 else 0
                )
            
            # Create volatility metrics
            volatility_metrics = {
                'spread_volatility_ratio': (
                    market_data['spread_volatility'] / market_data['avg_spread']
                    if market_data.get('avg_spread', 0) > 0 else 0
                ),
                'volume_activity_ratio': market_data.get('volume_ratio', 0),
                'max_spread': market_data.get('max_spread', 0)
            }
            
            # Classify volatility
            volatility_level = self._classify_volatility(volatility_metrics)
            
            # Get adjusted parameters
            adjusted_params = self.adjust_parameters(volatility_level)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                volatility_level, volatility_metrics, adjusted_params
            )
            
            return {
                'date': date_str,
                'volatility_metrics': volatility_metrics,
                'volatility_level': volatility_level,
                'adjusted_parameters': adjusted_params,
                'recommendations': recommendations
            }
            
        except Exception as e:
            return {
                'date': date_str,
                'error': str(e),
                'volatility_level': 'UNKNOWN',
                'recommendations': ['Unable to analyze volatility for this date']
            }
    
    def _generate_recommendations(self, volatility_level: str, metrics: Dict, params: Dict) -> List[str]:
        """Generate specific recommendations based on volatility analysis"""
        recommendations = []
        
        if volatility_level == "EXTREME":
            recommendations.extend([
                f"⚠️ EXTREME volatility detected - consider suspending automated trading",
                f"🔧 Increase time tolerance to {params['time_tolerance']:.1f}s",
                f"📊 Increase price tolerance to {params['price_tolerance']:.3f}%",
                f"🎯 Lower accuracy threshold to {params['accuracy_threshold']:.1f}",
                f"📈 Consider manual intervention for this period"
            ])
        elif volatility_level == "HIGH":
            recommendations.extend([
                f"⚠️ HIGH volatility detected - use cautious execution",
                f"🔧 Increase time tolerance to {params['time_tolerance']:.1f}s",
                f"📊 Increase price tolerance to {params['price_tolerance']:.3f}%",
                f"🎯 Lower accuracy threshold to {params['accuracy_threshold']:.1f}",
                f"📈 Monitor execution quality closely"
            ])
        elif volatility_level == "MEDIUM":
            recommendations.extend([
                f"⚠️ MEDIUM volatility detected - adjust parameters moderately",
                f"🔧 Increase time tolerance to {params['time_tolerance']:.1f}s",
                f"📊 Increase price tolerance to {params['price_tolerance']:.3f}%",
                f"🎯 Lower accuracy threshold to {params['accuracy_threshold']:.1f}"
            ])
        elif volatility_level == "LOW":
            recommendations.extend([
                f"✅ LOW volatility detected - tighten parameters for better accuracy",
                f"🔧 Decrease time tolerance to {params['time_tolerance']:.1f}s",
                f"📊 Decrease price tolerance to {params['price_tolerance']:.3f}%",
                f"🎯 Increase accuracy threshold to {params['accuracy_threshold']:.1f}"
            ])
        else:
            recommendations.extend([
                f"✅ NORMAL volatility detected - use standard parameters",
                f"🔧 Standard time tolerance: {params['time_tolerance']:.1f}s",
                f"📊 Standard price tolerance: {params['price_tolerance']:.3f}%"
            ])
        
        # Add specific metric-based recommendations
        if 'max_spread' in metrics and metrics['max_spread'] > 10:
            recommendations.append(f"💰 Wide spreads detected (max: ${metrics['max_spread']:.1f}) - consider wider price tolerances")
        
        if 'volume_activity_ratio' in metrics and metrics['volume_activity_ratio'] > 5:
            recommendations.append(f"📈 High simulation activity ({metrics['volume_activity_ratio']:.1f}x) - check for over-reaction")
        
        return recommendations


def main():
    """Example usage of VolatilityManager"""
    vm = VolatilityManager()
    
    # Analyze a specific problematic date
    result = vm.analyze_date_volatility('25-06-2025', 'analysis_results/detail_25-06-2025.json')
    
    print("🔍 VOLATILITY ANALYSIS RESULTS:")
    print(f"📅 Date: {result['date']}")
    print(f"📊 Volatility Level: {result['volatility_level']}")
    print(f"🎯 Adjusted Parameters: {result['adjusted_parameters']}")
    print("💡 Recommendations:")
    for rec in result['recommendations']:
        print(f"  • {rec}")


if __name__ == "__main__":
    main()