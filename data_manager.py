#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Manager for Trading Analysis Agent

Handles loading, caching, and processing of real and simulation trading data.
Implements efficient caching to avoid reloading data on every request.
"""

import os
import pandas as pd
import pickle
from typing import Set, List, Dict, Optional
from datetime import datetime
import logging
import json
import hashlib
from pathlib import Path

# Configure logging  
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class DataManager:
    """Manages trading data with intelligent caching and efficient loading."""
    
    def __init__(self):
        self.sim_dir = os.path.join("data", "HL_btcusdt_BTC")
        self.real_data_file = os.path.join("data", "agent_live_data.csv")
        self.cache_dir = os.path.join("data", ".cache")
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Cache for loaded data
        self._real_data_cache = None
        self._sim_data_cache = {}
        self._last_loaded_dates = set()
        
        # Load persistent cache if available
        self._load_persistent_cache()
        
        logger.info("DataManager initialized with persistent caching")
    
    def get_real_trade_chunks(self) -> List[pd.DataFrame]:
        """Load real trading data and split into daily chunks.
        
        Returns:
            List of DataFrames, one per day
        """
        if self._real_data_cache is None:
            self._load_real_data()
        
        chunks = []
        for date, group in self._real_data_cache.groupby(self._real_data_cache["timestamp"].dt.date):
            chunks.append(group.copy())
        
        logger.info(f"Retrieved {len(chunks)} daily chunks from cache")
        return chunks
    
    def get_simulation_data_for_dates(self, dates: Set[str]) -> pd.DataFrame:
        """Load simulation data for specific dates with caching.
        
        Args:
            dates: Set of date strings in format 'dd-mm-yyyy'
            
        Returns:
            Combined DataFrame with simulation data
        """
        # Check if we need to load new data
        new_dates = dates - self._last_loaded_dates
        
        if new_dates:
            if len(new_dates) == 1:
                print(f"📦 Loading simulation data for {list(new_dates)[0]}...")
            else:
                print(f"📦 Loading simulation data for {len(new_dates)} dates...")
            self._load_simulation_data(new_dates)
            self._last_loaded_dates.update(new_dates)
            print("✅ Loading complete!")
        # No message for cached data to keep output clean
        
        # Return combined data for requested dates
        relevant_data = []
        for date_str in dates:
            if date_str in self._sim_data_cache:
                relevant_data.append(self._sim_data_cache[date_str])
        
        if not relevant_data:
            raise ValueError(f"No simulation data found for dates: {dates}")
        
        return pd.concat(relevant_data, ignore_index=True)
    
    def get_available_dates(self) -> Set[str]:
        """Get available dates from real data.
        
        Returns:
            Set of available date strings in format 'dd-mm-yyyy'
        """
        if self._real_data_cache is None:
            self._load_real_data()
        
        return set(self._real_data_cache["timestamp"].dt.strftime("%d-%m-%Y").unique())
    
    def clear_cache(self):
        """Clear all cached data to force reload."""
        self._real_data_cache = None
        self._sim_data_cache.clear()
        self._last_loaded_dates.clear()
        
        # Clear persistent cache files
        cache_files = [
            os.path.join(self.cache_dir, "cache_metadata.json"),
            os.path.join(self.cache_dir, "real_data_cache.pkl"),
        ]
        
        for cache_file in cache_files:
            if os.path.exists(cache_file):
                os.remove(cache_file)
        
        # Clear simulation cache files
        sim_cache_pattern = os.path.join(self.cache_dir, "sim_*.pkl")
        import glob
        for cache_file in glob.glob(sim_cache_pattern):
            os.remove(cache_file)
        
        logger.info("Cache cleared (including persistent cache)")
    
    def _load_real_data(self):
        """Load real trading data from CSV file."""
        if not os.path.exists(self.real_data_file):
            raise FileNotFoundError(f"Real data file not found: {self.real_data_file}")
        
        logger.info("Loading real trading data...")
        df = pd.read_csv(self.real_data_file)
        
        # Handle timestamp column
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        elif "created_time" in df.columns:
            df["timestamp"] = pd.to_datetime(df["created_time"])
        else:
            raise ValueError("No timestamp column found in real data")
        
        self._real_data_cache = df
        self._save_real_cache(df)
        logger.info(f"Loaded {len(df)} real trading records")
    
    def _load_simulation_data(self, dates: Set[str]):
        """Load simulation data for specific dates.
        
        Args:
            dates: Set of date strings to load
        """
        if not os.path.exists(self.sim_dir):
            raise FileNotFoundError(f"Simulation data directory not found: {self.sim_dir}")
        
        for filename in os.listdir(self.sim_dir):
            if not filename.endswith(".pickle"):
                continue
            
            # Check if this file matches any requested date
            matching_date = None
            for date_str in dates:
                if date_str in filename:
                    matching_date = date_str
                    break
            
            if not matching_date:
                continue
            
            try:
                full_path = os.path.join(self.sim_dir, filename)
                loaded = pd.read_pickle(full_path)
                
                print(f"  📄 {matching_date}", end="", flush=True)
                
                # Handle different data formats
                if isinstance(loaded, dict) and "data" in loaded:
                    df = loaded["data"]
                    meta = loaded.get("meta", {})
                elif isinstance(loaded, pd.DataFrame):
                    df = loaded
                    meta = {}
                elif isinstance(loaded, list):
                    df = pd.DataFrame(loaded)
                    meta = {}
                else:
                    logger.warning(f"Unsupported format in {filename}: {type(loaded)}")
                    continue
                
                # Validate required columns
                if "timestamp" not in df.columns:
                    logger.warning(f"No timestamp column in {filename}")
                    continue
                
                if "price" not in df.columns:
                    logger.warning(f"No price column in {filename}")
                    continue
                
                # Add metadata
                df["source_file"] = filename
                df["meta_pair_symbol"] = meta.get("pair_symbol", "unknown")
                df["meta_broker"] = meta.get("broker", "unknown")
                df["meta_master"] = meta.get("Master", "unknown")
                
                # Convert timestamp
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                
                # Cache the data
                self._sim_data_cache[matching_date] = df
                self._save_simulation_cache(matching_date, df)
                print(" ✓", flush=True)  # Completion indicator for this file
                
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
                continue
    
    def _load_persistent_cache(self):
        """Load cached data from disk if available and fresh."""
        cache_metadata_file = os.path.join(self.cache_dir, "cache_metadata.json")
        
        if not os.path.exists(cache_metadata_file):
            return
        
        try:
            with open(cache_metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Check if cache is still valid (files haven't changed)
            if self._is_cache_valid(metadata):
                # Load real data cache
                real_cache_file = os.path.join(self.cache_dir, "real_data_cache.pkl")
                if os.path.exists(real_cache_file):
                    self._real_data_cache = pd.read_pickle(real_cache_file)
                    logger.info("Loaded real data from persistent cache")
                
                # Load simulation cache info
                self._last_loaded_dates = set(metadata.get('loaded_dates', []))
                
                # Load simulation data caches
                for date_str in self._last_loaded_dates:
                    sim_cache_file = os.path.join(self.cache_dir, f"sim_{date_str.replace('-', '_')}.pkl")
                    if os.path.exists(sim_cache_file):
                        self._sim_data_cache[date_str] = pd.read_pickle(sim_cache_file)
                
                if self._sim_data_cache:
                    logger.info(f"Loaded simulation data for {len(self._sim_data_cache)} dates from persistent cache")
            else:
                logger.info("Persistent cache is stale, will reload data")
                
        except Exception as e:
            logger.warning(f"Error loading persistent cache: {e}")
    
    def _is_cache_valid(self, metadata: dict) -> bool:
        """Check if cached data is still valid based on file modification times."""
        try:
            # Check real data file
            real_file_mtime = os.path.getmtime(self.real_data_file)
            if real_file_mtime != metadata.get('real_file_mtime'):
                return False
            
            # Check simulation directory
            sim_dir_mtime = os.path.getmtime(self.sim_dir)
            if sim_dir_mtime != metadata.get('sim_dir_mtime'):
                return False
            
            return True
            
        except (OSError, KeyError):
            return False
    
    def _save_real_cache(self, df: pd.DataFrame):
        """Save real data cache to disk."""
        try:
            cache_file = os.path.join(self.cache_dir, "real_data_cache.pkl")
            df.to_pickle(cache_file)
            self._update_cache_metadata()
        except Exception as e:
            logger.warning(f"Error saving real data cache: {e}")
    
    def _save_simulation_cache(self, date_str: str, df: pd.DataFrame):
        """Save simulation data cache for a specific date."""
        try:
            cache_file = os.path.join(self.cache_dir, f"sim_{date_str.replace('-', '_')}.pkl")
            df.to_pickle(cache_file)
            self._update_cache_metadata()
        except Exception as e:
            logger.warning(f"Error saving simulation cache for {date_str}: {e}")
    
    def _update_cache_metadata(self):
        """Update cache metadata file."""
        try:
            metadata = {
                'real_file_mtime': os.path.getmtime(self.real_data_file) if os.path.exists(self.real_data_file) else None,
                'sim_dir_mtime': os.path.getmtime(self.sim_dir) if os.path.exists(self.sim_dir) else None,
                'loaded_dates': list(self._last_loaded_dates),
                'cache_timestamp': datetime.now().isoformat()
            }
            
            cache_metadata_file = os.path.join(self.cache_dir, "cache_metadata.json")
            with open(cache_metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Error updating cache metadata: {e}")


# Global instance for the module
_data_manager = DataManager()


# Public API functions for backward compatibility
def get_real_trade_chunks() -> List[pd.DataFrame]:
    """Get real trading data chunks (cached)."""
    return _data_manager.get_real_trade_chunks()


def load_simulation_data_for_dates(dates: Set[str]) -> pd.DataFrame:
    """Load simulation data for dates (cached)."""
    return _data_manager.get_simulation_data_for_dates(dates)


def get_available_dates() -> Set[str]:
    """Get available dates from real data."""
    return _data_manager.get_available_dates()


def clear_data_cache():
    """Clear all cached data."""
    _data_manager.clear_cache()
