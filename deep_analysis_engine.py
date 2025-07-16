"""
Deep Analysis Engine for Trading Discrepancies
Provides accurate trade matching and insightful analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Optional
import logging
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


class DeepAnalysisEngine:
    """Advanced analysis engine that properly matches real trades with simulation data"""
    
    def __init__(self):
        self.real_trades_cache = None
        self.simulation_cache = {}
        
    def load_real_trades(self, filepath: str) -> pd.DataFrame:
        """Load and preprocess real trading data"""
        if self.real_trades_cache is not None:
            return self.real_trades_cache
            
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
            
            # Convert created_time to timestamp
            df['timestamp'] = pd.to_datetime(df['created_time']).astype(int) // 10**9
            
            # Clean numeric columns
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
            df['realized_pnl'] = pd.to_numeric(df['realized_pnl'], errors='coerce')
            
            # Add derived fields
            df['value'] = df['price'] * df['quantity']
            df['is_buy'] = df['side'] == 'B'
            
            self.real_trades_cache = df
            logger.info(f"Loaded {len(df)} real trades")
            return df
            
        except Exception as e:
            logger.error(f"Error loading real trades: {str(e)}")
            raise
    
    def extract_simulation_trades(self, sim_data: List[Dict]) -> pd.DataFrame:
        """Extract actual trades from simulation market data"""
        trades = []
        
        for i, snapshot in enumerate(sim_data):
            timestamp = snapshot.get('timestamp', 0)
            
            # Extract buy trades
            buy_trades = snapshot.get('TradesMarketBuy', [])
            if isinstance(buy_trades, np.ndarray) and len(buy_trades) > 0:
                for trade in buy_trades:
                    trades.append({
                        'timestamp': timestamp,
                        'price': float(trade[0]),
                        'quantity': float(trade[1]),
                        'side': 'B',
                        'snapshot_idx': i
                    })
            
            # Extract sell trades
            sell_trades = snapshot.get('TradesMarketSell', [])
            if isinstance(sell_trades, np.ndarray) and len(sell_trades) > 0:
                for trade in sell_trades:
                    trades.append({
                        'timestamp': timestamp,
                        'price': float(trade[0]),
                        'quantity': float(trade[1]),
                        'side': 'A',
                        'snapshot_idx': i
                    })
        
        df = pd.DataFrame(trades)
        if not df.empty:
            df['value'] = df['price'] * df['quantity']
            df['is_buy'] = df['side'] == 'B'
        
        return df
    
    def validate_simulation_accuracy(self, real_trades: pd.DataFrame, sim_trades: pd.DataFrame) -> Dict:
        """Validate if simulation has acceptable accuracy with YES/NO verdict"""
        
        # Strict accuracy criteria for validation
        time_tolerance = 2  # 2 seconds for accuracy validation
        price_tolerance = 0.005  # 0.005% price tolerance
        
        # Calculate executed capital (EC)
        live_ec = (real_trades['price'] * real_trades['quantity']).sum()
        sim_ec = (sim_trades['price'] * sim_trades['quantity']).sum() if not sim_trades.empty else 0
        
        # Count trades
        live_trade_count = len(real_trades)
        sim_trade_count = len(sim_trades)
        
        # Find accurate matches
        accurate_matches = 0
        total_testable = 0
        match_details = []
        
        for _, real_trade in real_trades.iterrows():
            # Find simulation trades within strict criteria
            time_mask = (
                (sim_trades['timestamp'] >= real_trade['timestamp'] - time_tolerance) &
                (sim_trades['timestamp'] <= real_trade['timestamp'] + time_tolerance) &
                (sim_trades['side'] == real_trade['side'])  # Same side (buy/sell)
            )
            
            potential_matches = sim_trades[time_mask]
            
            if not potential_matches.empty:
                total_testable += 1
                
                # Check price accuracy (±0.005%)
                price_diff_pct = abs(potential_matches['price'] - real_trade['price']) / real_trade['price']
                accurate_price_matches = potential_matches[price_diff_pct <= price_tolerance]
                
                if not accurate_price_matches.empty:
                    accurate_matches += 1
                    # Find best match for details
                    best_idx = price_diff_pct.idxmin()
                    best_match = potential_matches.loc[best_idx]
                    
                    match_details.append({
                        'real_time': real_trade['timestamp'],
                        'sim_time': best_match['timestamp'],
                        'time_diff': abs(real_trade['timestamp'] - best_match['timestamp']),
                        'real_price': real_trade['price'],
                        'sim_price': best_match['price'],
                        'price_diff_pct': abs(real_trade['price'] - best_match['price']) / real_trade['price'] * 100,
                        'side': real_trade['side'],
                        'accurate': True
                    })
        
        # Calculate accuracy metrics
        execution_accuracy = (accurate_matches / total_testable * 100) if total_testable > 0 else 0
        
        # Determine acceptability (>=80% accuracy threshold)
        is_acceptable = execution_accuracy >= 80.0 and total_testable > 0
        
        return {
            'validation_result': {
                'is_acceptable': is_acceptable,
                'accuracy_verdict': 'YES' if is_acceptable else 'NO',
                'execution_accuracy_pct': execution_accuracy,
                'reason': self._get_accuracy_reason(execution_accuracy, total_testable, live_trade_count, sim_trade_count)
            },
            'metrics': {
                'live_ec': live_ec,
                'sim_ec': sim_ec,
                'live_trade_count': live_trade_count,
                'sim_trade_count': sim_trade_count,
                'accurate_matches': accurate_matches,
                'total_testable': total_testable,
                'execution_accuracy_pct': execution_accuracy
            },
            'match_details': match_details[:10]  # Show first 10 matches
        }
    
    def _get_accuracy_reason(self, accuracy: float, testable: int, live_count: int, sim_count: int) -> str:
        """Generate reason for accuracy verdict"""
        if accuracy >= 80.0 and testable > 0:
            return f"Simulation achieves {accuracy:.1f}% execution accuracy within strict criteria (2s, ±0.005%)"
        elif testable == 0:
            return "No matching trades found - possible algorithm crash or data misalignment"
        elif accuracy < 50.0:
            return f"Poor accuracy ({accuracy:.1f}%) - significant timing or price discrepancies detected"
        elif sim_count == 0:
            return "No simulation trades found - algorithm may have crashed"
        elif sim_count < live_count * 0.5:
            return f"Low simulation activity ({sim_count} vs {live_count} live) - possible algo issues"
        elif sim_count > live_count * 2:
            return f"High simulation activity ({sim_count} vs {live_count} live) - possible over-trading in sim"
        else:
            return f"Moderate accuracy ({accuracy:.1f}%) - needs investigation of timing/price alignment"
    
    def analyze_execution_quality(self, real_trades: pd.DataFrame, sim_trades: pd.DataFrame) -> Dict:
        """Analyze execution quality differences between real and simulated trades"""
        results = {
            'execution_analysis': {},
            'price_improvement': {},
            'timing_analysis': {},
            'volume_analysis': {}
        }
        
        # Match trades by time window (5 second tolerance for general analysis)
        time_tolerance = 5
        
        matched_trades = []
        unmatched_real = []
        unmatched_sim = []
        
        # Create a copy to track matched sim trades
        sim_trades_copy = sim_trades.copy()
        matched_sim_indices = set()
        
        # Loop through sim trades to find acceptable matches
        for _, sim_trade in sim_trades_copy.iterrows():
            if sim_trade.name in matched_sim_indices:
                continue
                
            # Find real trades within time window
            time_mask = (
                (real_trades['timestamp'] >= sim_trade['timestamp'] - time_tolerance) &
                (real_trades['timestamp'] <= sim_trade['timestamp'] + time_tolerance) &
                (real_trades['side'] == sim_trade['side'])
            )
            
            potential_matches = real_trades[time_mask]
            
            if not potential_matches.empty:
                # Find best match by price and quantity
                price_diff = abs(potential_matches['price'] - sim_trade['price'])
                qty_diff = abs(potential_matches['quantity'] - sim_trade['quantity'])
                
                # Normalized distance metric
                distance = price_diff / sim_trade['price'] + qty_diff / sim_trade['quantity']
                best_match_idx = distance.idxmin()
                best_match = potential_matches.loc[best_match_idx]
                
                # Check if this real trade hasn't been matched already
                already_matched = any(
                    match['real_time'] == best_match['timestamp'] and 
                    match['real_price'] == best_match['price'] and
                    match['side'] == best_match['side']
                    for match in matched_trades
                )
                
                if not already_matched:
                    matched_trades.append({
                        'real_price': best_match['price'],
                        'sim_price': sim_trade['price'],
                        'real_qty': best_match['quantity'],
                        'sim_qty': sim_trade['quantity'],
                        'real_time': best_match['timestamp'],
                        'sim_time': sim_trade['timestamp'],
                        'side': sim_trade['side'],
                        'price_diff': best_match['price'] - sim_trade['price'],
                        'qty_diff': best_match['quantity'] - sim_trade['quantity'],
                        'time_diff': best_match['timestamp'] - sim_trade['timestamp']
                    })
                    
                    # Mark this sim trade as matched
                    matched_sim_indices.add(sim_trade.name)
        
        # Find unmatched real trades
        matched_real_times = {match['real_time'] for match in matched_trades}
        for _, real_trade in real_trades.iterrows():
            if real_trade['timestamp'] not in matched_real_times:
                unmatched_real.append(real_trade)
        
        # Analyze matched trades
        if matched_trades:
            matched_df = pd.DataFrame(matched_trades)
            
            # Execution quality metrics
            results['execution_analysis'] = {
                'total_matched': len(matched_trades),
                'match_rate': len(matched_trades) / len(real_trades) * 100,
                'avg_price_diff': matched_df['price_diff'].mean(),
                'avg_price_diff_pct': (matched_df['price_diff'] / matched_df['real_price']).mean() * 100,
                'price_improvement_count': len(matched_df[matched_df['price_diff'] > 0]),
                'price_deterioration_count': len(matched_df[matched_df['price_diff'] < 0]),
                'avg_time_diff': matched_df['time_diff'].mean()
            }
            
            # Price improvement analysis by side
            for side in ['B', 'A']:
                side_data = matched_df[matched_df['side'] == side]
                if not side_data.empty:
                    side_name = 'Buy' if side == 'B' else 'Sell'
                    improvement = side_data['price_diff'].mean()
                    
                    # For buys, negative diff is improvement (paid less)
                    # For sells, positive diff is improvement (sold for more)
                    if side == 'B':
                        improvement = -improvement
                    
                    results['price_improvement'][side_name] = {
                        'avg_improvement': improvement,
                        'avg_improvement_pct': improvement / side_data['real_price'].mean() * 100,
                        'improved_trades': len(side_data[side_data['price_diff'] * (-1 if side == 'B' else 1) > 0]),
                        'total_trades': len(side_data)
                    }
        
        # Timing analysis
        if matched_trades:
            time_diffs = [t['time_diff'] for t in matched_trades]
            results['timing_analysis'] = {
                'avg_latency': statistics.mean(time_diffs),
                'median_latency': statistics.median(time_diffs),
                'max_latency': max(time_diffs),
                'min_latency': min(time_diffs),
                'trades_ahead': sum(1 for t in time_diffs if t > 0),
                'trades_behind': sum(1 for t in time_diffs if t < 0)
            }
        
        # Calculate unmatched sim trades
        unmatched_sim_count = len(sim_trades) - len(matched_sim_indices)
        
        # Volume analysis
        results['volume_analysis'] = {
            'real_total_volume': real_trades['value'].sum(),
            'sim_total_volume': sim_trades['value'].sum() if not sim_trades.empty else 0,
            'real_trade_count': len(real_trades),
            'sim_trade_count': len(sim_trades),
            'unmatched_real_count': len(unmatched_real),
            'unmatched_sim_count': unmatched_sim_count
        }
        
        return results
    
    def analyze_market_conditions(self, sim_data: List[Dict], real_trades: pd.DataFrame) -> Dict:
        """Analyze market conditions during trading periods"""
        results = {
            'spread_analysis': {},
            'liquidity_analysis': {},
            'volatility_analysis': {},
            'market_impact': {}
        }
        
        # Extract order book data
        spreads = []
        bid_depths = []
        ask_depths = []
        mid_prices = []
        
        for snapshot in sim_data:
            bids = snapshot.get('BidsOB', [])
            asks = snapshot.get('AsksOB', [])
            
            if isinstance(bids, np.ndarray) and isinstance(asks, np.ndarray) and len(bids) > 0 and len(asks) > 0:
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                
                spread = best_ask - best_bid
                spreads.append(spread)
                mid_prices.append((best_bid + best_ask) / 2)
                
                # Calculate depth (top 5 levels)
                bid_depth = sum(float(b[1]) * float(b[0]) for b in bids[:5])
                ask_depth = sum(float(a[1]) * float(a[0]) for a in asks[:5])
                bid_depths.append(bid_depth)
                ask_depths.append(ask_depth)
        
        if spreads:
            results['spread_analysis'] = {
                'avg_spread': statistics.mean(spreads),
                'median_spread': statistics.median(spreads),
                'max_spread': max(spreads),
                'min_spread': min(spreads),
                'spread_volatility': statistics.stdev(spreads) if len(spreads) > 1 else 0
            }
        
        if bid_depths and ask_depths:
            results['liquidity_analysis'] = {
                'avg_bid_depth': statistics.mean(bid_depths),
                'avg_ask_depth': statistics.mean(ask_depths),
                'liquidity_imbalance': statistics.mean(bid_depths) - statistics.mean(ask_depths),
                'min_bid_depth': min(bid_depths),
                'min_ask_depth': min(ask_depths)
            }
        
        if mid_prices and len(mid_prices) > 1:
            # Calculate volatility
            returns = [(mid_prices[i] - mid_prices[i-1]) / mid_prices[i-1] 
                      for i in range(1, len(mid_prices))]
            
            if returns:
                results['volatility_analysis'] = {
                    'return_volatility': statistics.stdev(returns) * 100,
                    'price_range': max(mid_prices) - min(mid_prices),
                    'price_range_pct': (max(mid_prices) - min(mid_prices)) / statistics.mean(mid_prices) * 100
                }
        
        # Market impact analysis
        if not real_trades.empty and mid_prices:
            avg_mid = statistics.mean(mid_prices)
            buy_trades = real_trades[real_trades['is_buy']]
            sell_trades = real_trades[~real_trades['is_buy']]
            
            results['market_impact'] = {
                'avg_buy_price_vs_mid': buy_trades['price'].mean() - avg_mid if not buy_trades.empty else 0,
                'avg_sell_price_vs_mid': sell_trades['price'].mean() - avg_mid if not sell_trades.empty else 0,
                'estimated_total_slippage': sum(
                    (row['price'] - avg_mid) * row['quantity'] * (1 if row['is_buy'] else -1)
                    for _, row in real_trades.iterrows()
                )
            }
        
        return results
    
    def generate_insights(self, analysis_results: Dict) -> List[str]:
        """Generate actionable insights from analysis results"""
        insights = []
        
        # Execution quality insights
        exec_analysis = analysis_results.get('execution_analysis', {})
        if exec_analysis:
            match_rate = exec_analysis.get('match_rate', 0)
            if match_rate < 50:
                insights.append(f"⚠️ Low trade matching rate ({match_rate:.1f}%) suggests significant execution differences between real and simulated environments")
            
            avg_price_diff_pct = exec_analysis.get('avg_price_diff_pct', 0)
            if abs(avg_price_diff_pct) > 0.1:
                direction = "worse" if avg_price_diff_pct > 0 else "better"
                insights.append(f"📊 Real trades executed {abs(avg_price_diff_pct):.2f}% {direction} than simulation on average")
        
        # Price improvement insights
        for side, data in analysis_results.get('price_improvement', {}).items():
            if data.get('avg_improvement_pct', 0) != 0:
                improvement = data['avg_improvement_pct']
                if abs(improvement) > 0.05:
                    action = "improvement" if improvement > 0 else "deterioration"
                    insights.append(f"💰 {side} orders show {abs(improvement):.2f}% price {action} in real trading")
        
        # Timing insights
        timing = analysis_results.get('timing_analysis', {})
        if timing:
            avg_latency = timing.get('avg_latency', 0)
            if abs(avg_latency) > 1:
                direction = "ahead of" if avg_latency > 0 else "behind"
                insights.append(f"⏱️ Real trades execute {abs(avg_latency):.1f} seconds {direction} simulation")
        
        # Market condition insights
        spread = analysis_results.get('spread_analysis', {})
        if spread:
            avg_spread = spread.get('avg_spread', 0)
            if avg_spread > 10:
                insights.append(f"📈 High average spread (${avg_spread:.2f}) indicates volatile market conditions")
        
        liquidity = analysis_results.get('liquidity_analysis', {})
        if liquidity:
            imbalance = liquidity.get('liquidity_imbalance', 0)
            if abs(imbalance) > 10000:
                side = "bid" if imbalance > 0 else "ask"
                insights.append(f"💧 Significant liquidity imbalance favoring {side} side (${abs(imbalance):,.0f})")
        
        # Market impact insights
        impact = analysis_results.get('market_impact', {})
        if impact:
            slippage = impact.get('estimated_total_slippage', 0)
            if abs(slippage) > 100:
                loss_gain = "loss" if slippage > 0 else "gain"
                insights.append(f"💸 Estimated ${abs(slippage):,.2f} in slippage {loss_gain} from market impact")
        
        # Volume insights
        volume = analysis_results.get('volume_analysis', {})
        if volume:
            unmatched_pct = volume.get('unmatched_real_count', 0) / max(volume.get('real_trade_count', 1), 1) * 100
            if unmatched_pct > 20:
                insights.append(f"🔍 {unmatched_pct:.0f}% of real trades have no simulation equivalent - check strategy differences")
        
        return insights
    
    def diagnose_accuracy_issues(self, real_trades: pd.DataFrame, sim_trades: pd.DataFrame, date_str: str) -> List[str]:
        """Diagnose potential causes of accuracy issues"""
        diagnostics = []
        
        # Check for trading gaps (possible crashes)
        if not real_trades.empty:
            real_times = sorted(real_trades['timestamp'])
            gaps = []
            for i in range(1, len(real_times)):
                gap = real_times[i] - real_times[i-1]
                if gap > 300:  # 5+ minute gaps
                    gaps.append(gap)
            
            if gaps:
                max_gap = max(gaps) / 60  # Convert to minutes
                diagnostics.append(f"⚠️ Found {len(gaps)} trading gaps >5min (max: {max_gap:.0f}min) - possible algo crashes")
        
        # Check for volatility correlation
        if not sim_trades.empty and len(sim_trades) > 0:
            sim_times = sorted(sim_trades['timestamp'])
            live_times = sorted(real_trades['timestamp']) if not real_trades.empty else []
            
            # Check activity correlation
            if len(live_times) > 0:
                live_activity = len(live_times)
                sim_activity = len(sim_times)
                
                if sim_activity > live_activity * 3:
                    diagnostics.append(f"📈 High sim activity ({sim_activity} vs {live_activity} live) - possible volatility over-reaction")
                elif sim_activity < live_activity * 0.3:
                    diagnostics.append(f"📉 Low sim activity ({sim_activity} vs {live_activity} live) - possible conservative algo tuning")
        
        # Check for latency patterns
        time_diffs = []
        for _, real_trade in real_trades.iterrows():
            closest_sim = sim_trades[
                (abs(sim_trades['timestamp'] - real_trade['timestamp']) <= 10) &
                (sim_trades['side'] == real_trade['side'])
            ]
            if not closest_sim.empty:
                min_diff = abs(closest_sim['timestamp'] - real_trade['timestamp']).min()
                time_diffs.append(min_diff)
        
        if time_diffs:
            avg_latency = statistics.mean(time_diffs)
            if avg_latency > 3:
                diagnostics.append(f"⏱️ High average latency ({avg_latency:.1f}s) - possible network/execution delays")
            
            # Check for systematic bias
            positive_diffs = sum(1 for d in time_diffs if d > 0)
            if positive_diffs > len(time_diffs) * 0.8:
                diagnostics.append(f"🔄 Systematic timing bias - live trades consistently ahead of simulation")
        
        # Check date-specific issues
        if '01-06' in date_str:
            diagnostics.append(f"📅 {date_str} is first simulation date - possible initialization issues")
        
        return diagnostics
    
    def analyze_discrepancies(self, real_data_path: str, sim_data: List[Dict], date_str: str = "") -> Dict:
        """Main analysis function with accuracy validation"""
        try:
            # Load real trades
            real_trades = self.load_real_trades(real_data_path)
            
            # Extract simulation trades
            sim_trades = self.extract_simulation_trades(sim_data)
            
            # Perform validation first
            validation_results = self.validate_simulation_accuracy(real_trades, sim_trades)
            
            # Perform analyses
            results = {
                'summary': {
                    'real_trades_count': len(real_trades),
                    'sim_trades_count': len(sim_trades),
                    'analysis_period': {
                        'start': datetime.fromtimestamp(min(real_trades['timestamp']), tz=timezone.utc).isoformat() if not real_trades.empty else 'N/A',
                        'end': datetime.fromtimestamp(max(real_trades['timestamp']), tz=timezone.utc).isoformat() if not real_trades.empty else 'N/A'
                    }
                }
            }
            
            # Add validation results
            results.update(validation_results)
            
            # Run execution quality analysis (broader analysis)
            exec_results = self.analyze_execution_quality(real_trades, sim_trades)
            results.update(exec_results)
            
            # Run market condition analysis
            market_results = self.analyze_market_conditions(sim_data, real_trades)
            results.update(market_results)
            
            # Generate insights
            insights = self.generate_insights(results)
            results['insights'] = insights
            
            # Add diagnostics if accuracy is poor
            if not validation_results['validation_result']['is_acceptable']:
                diagnostics = self.diagnose_accuracy_issues(real_trades, sim_trades, date_str)
                results['diagnostics'] = diagnostics
            
            return results
            
        except Exception as e:
            logger.error(f"Error in deep analysis: {str(e)}")
            return {
                'error': str(e),
                'summary': {'error': 'Analysis failed'}
            }