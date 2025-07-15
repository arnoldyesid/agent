#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis Engine for Trading Discrepancy Analysis

Handles comparison between real and simulated trading data,
identifies discrepancies, and provides AI-powered insights.
"""

import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class AnalysisEngine:
    """Analyzes discrepancies between real and simulated trading data."""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.discrepancy_threshold = 2.0  # USD
        self.merge_tolerance = pd.Timedelta("2s")
    
    def analyze_chunk_discrepancy(self, real_chunk: pd.DataFrame, 
                                sim_df: pd.DataFrame) -> dict:
        """Analyze discrepancies between real and simulated data chunks.
        
        Args:
            real_chunk: DataFrame with real trading data
            sim_df: DataFrame with simulation data
            
        Returns:
            Dictionary with analysis results
        """
        if sim_df.empty or real_chunk.empty:
            return {
                "status": "no_data",
                "message": "No data available for comparison",
                "discrepancies": pd.DataFrame()
            }
        
        try:
            # Prepare data for comparison
            real_prepared = self._prepare_real_data(real_chunk)
            sim_prepared = self._prepare_sim_data(sim_df)
            
            # Merge data by timestamp
            merged = self._merge_data(real_prepared, sim_prepared)
            
            if merged.empty:
                return {
                    "status": "no_matches",
                    "message": "No temporal matches found between real and simulated data",
                    "discrepancies": pd.DataFrame()
                }
            
            # Calculate discrepancies
            discrepancies = self._calculate_discrepancies(merged)
            
            if discrepancies.empty:
                return {
                    "status": "no_discrepancies",
                    "message": f"No significant discrepancies (>${self.discrepancy_threshold}) found",
                    "discrepancies": pd.DataFrame(),
                    "total_comparisons": len(merged)
                }
            
            # Generate analysis
            analysis = self._generate_analysis(discrepancies)
            
            return {
                "status": "discrepancies_found",
                "message": f"Found {len(discrepancies)} significant discrepancies",
                "discrepancies": discrepancies,
                "total_comparisons": len(merged),
                "analysis": analysis,
                "statistics": self._calculate_statistics(discrepancies)
            }
            
        except Exception as e:
            logger.error(f"Error in discrepancy analysis: {e}")
            return {
                "status": "error",
                "message": f"Analysis failed: {str(e)}",
                "discrepancies": pd.DataFrame()
            }
    
    def _prepare_real_data(self, real_chunk: pd.DataFrame) -> pd.DataFrame:
        """Prepare real data for comparison."""
        df = real_chunk.copy()
        
        # Ensure price column exists
        if "price" not in df.columns:
            raise ValueError("'price' column not found in real data")
        
        # Rename price column to avoid conflicts
        df = df.rename(columns={"price": "price_real"})
        
        # Ensure timestamp column
        if "timestamp" not in df.columns:
            raise ValueError("'timestamp' column not found in real data")
        
        return df[["timestamp", "price_real"]].sort_values("timestamp")
    
    def _prepare_sim_data(self, sim_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare simulation data for comparison."""
        df = sim_df.copy()
        
        # Ensure price column exists
        if "price" not in df.columns:
            raise ValueError("'price' column not found in simulation data")
        
        # Rename price column to avoid conflicts
        df = df.rename(columns={"price": "price_sim"})
        
        # Ensure timestamp column
        if "timestamp" not in df.columns:
            raise ValueError("'timestamp' column not found in simulation data")
        
        # Include metadata for analysis
        cols = ["timestamp", "price_sim"]
        if "source_file" in df.columns:
            cols.append("source_file")
        if "meta_pair_symbol" in df.columns:
            cols.append("meta_pair_symbol")
        if "meta_broker" in df.columns:
            cols.append("meta_broker")
        if "meta_master" in df.columns:
            cols.append("meta_master")
        
        return df[cols].sort_values("timestamp")
    
    def _merge_data(self, real_df: pd.DataFrame, sim_df: pd.DataFrame) -> pd.DataFrame:
        """Merge real and simulation data by timestamp with tolerance."""
        return pd.merge_asof(
            real_df,
            sim_df,
            on="timestamp",
            direction="nearest",
            tolerance=self.merge_tolerance
        )
    
    def _calculate_discrepancies(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate and filter significant price discrepancies."""
        # Calculate absolute price difference
        merged_df["price_diff"] = (merged_df["price_real"] - merged_df["price_sim"]).abs()
        
        # Filter for significant discrepancies
        discrepancies = merged_df[merged_df["price_diff"] > self.discrepancy_threshold].copy()
        
        # Calculate percentage difference
        discrepancies["price_diff_pct"] = (
            discrepancies["price_diff"] / discrepancies["price_real"] * 100
        )
        
        return discrepancies
    
    def _calculate_statistics(self, discrepancies: pd.DataFrame) -> dict:
        """Calculate statistics for discrepancies."""
        if discrepancies.empty:
            return {}
        
        return {
            "count": len(discrepancies),
            "avg_discrepancy": discrepancies["price_diff"].mean(),
            "max_discrepancy": discrepancies["price_diff"].max(),
            "min_discrepancy": discrepancies["price_diff"].min(),
            "avg_discrepancy_pct": discrepancies["price_diff_pct"].mean(),
            "max_discrepancy_pct": discrepancies["price_diff_pct"].max()
        }
    
    def _generate_analysis(self, discrepancies: pd.DataFrame) -> Optional[str]:
        """Generate AI-powered analysis of discrepancies."""
        if not self.llm_client or discrepancies.empty:
            return None
        
        try:
            # Prepare sample data for analysis
            sample = discrepancies[[
                "timestamp", "price_real", "price_sim", "price_diff", "price_diff_pct"
            ]].head(10)
            
            # Include metadata if available
            meta_info = ""
            if "meta_pair_symbol" in discrepancies.columns:
                meta_data = discrepancies[[
                    "meta_pair_symbol", "meta_broker", "meta_master"
                ]].drop_duplicates().head(3)
                meta_info = f"\nMetadata:\n{meta_data.to_string(index=False)}\n"
            
            # Calculate statistics
            stats = self._calculate_statistics(discrepancies)
            
            # Create analysis prompt
            prompt = f"""
Analyze these trading discrepancies between real and simulated data:

Statistics:
- Total discrepancies: {stats['count']}
- Average discrepancy: ${stats['avg_discrepancy']:.2f} ({stats['avg_discrepancy_pct']:.2f}%)
- Maximum discrepancy: ${stats['max_discrepancy']:.2f} ({stats['max_discrepancy_pct']:.2f}%)
- Minimum discrepancy: ${stats['min_discrepancy']:.2f}
{meta_info}
Sample discrepancies:
{sample.to_string(index=False)}

Please provide:
1. Possible causes for these price discrepancies
2. Recommendations to improve simulation accuracy
3. Any patterns or trends you notice

Keep the analysis concise and actionable."""
            
            return self.llm_client.predict(prompt)
            
        except Exception as e:
            logger.error(f"Error generating AI analysis: {e}")
            return f"AI analysis failed: {str(e)}"
    
    def format_results(self, results: dict) -> str:
        """Format analysis results for display."""
        if results["status"] == "no_data":
            return "No data available for comparison."
        
        elif results["status"] == "no_matches":
            return "No temporal matches found between real and simulated data."
        
        elif results["status"] == "no_discrepancies":
            return f"No significant discrepancies found. Analyzed {results['total_comparisons']} data points."
        
        elif results["status"] == "error":
            return f"Analysis error: {results['message']}"
        
        elif results["status"] == "discrepancies_found":
            output = [f"Found {results['message']}"]
            
            if "statistics" in results:
                stats = results["statistics"]
                output.append(f"\nStatistics:")
                output.append(f"- Average discrepancy: ${stats['avg_discrepancy']:.2f} ({stats['avg_discrepancy_pct']:.2f}%)")
                output.append(f"- Maximum discrepancy: ${stats['max_discrepancy']:.2f} ({stats['max_discrepancy_pct']:.2f}%)")
            
            # Show sample discrepancies
            if not results["discrepancies"].empty:
                sample = results["discrepancies"][["timestamp", "price_real", "price_sim", "price_diff"]].head(5)
                output.append(f"\nSample discrepancies:\n{sample.to_string(index=False)}")
            
            # Add AI analysis if available
            if "analysis" in results and results["analysis"]:
                output.append(f"\nAI Analysis:\n{results['analysis']}")
            
            return "\n".join(output)
        
        return "Unknown analysis result."
