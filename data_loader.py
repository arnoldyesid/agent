#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
from typing import Set, List

# Directory containing simulation files
SIM_DIR = os.path.join("data", "HL_btcusdt_BTC")

# Real trading data file
REAL_DATA_FILE = os.path.join("data", "agent_live_data.csv")


def get_real_trade_chunks() -> List[pd.DataFrame]:
    """
    Load real trading data from CSV and split by day.
    """
    if not os.path.exists(REAL_DATA_FILE):
        raise FileNotFoundError(f"Real data file not found: {REAL_DATA_FILE}")
    
    df = pd.read_csv(REAL_DATA_FILE)

    # Use 'timestamp' if exists, fallback to 'created_time'
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    elif "created_time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["created_time"])
    else:
        raise ValueError("No timestamp or created_time column found in real data.")
    
    chunks = []
    for date, group in df.groupby(df["timestamp"].dt.date):
        chunks.append(group)
    
    return chunks


def load_simulation_data_for_dates(dates: Set[str]) -> pd.DataFrame:
    """
    Load simulation files for dates that appear in `dates`.
    """
    all_data = []

    for filename in os.listdir(SIM_DIR):
        if not filename.endswith(".pickle"):
            continue

        for date_str in dates:
            if date_str in filename:
                try:
                    full_path = os.path.join(SIM_DIR, filename)
                    loaded = pd.read_pickle(full_path)

                    # Removed verbose loading messages for cleaner output

                    if isinstance(loaded, dict) and "data" in loaded:
                        df = loaded["data"]
                        meta = loaded.get("meta", {})
                    elif isinstance(loaded, pd.DataFrame):
                        df = loaded
                        meta = {}
                    elif isinstance(loaded, list):
                        try:
                            df = pd.DataFrame(loaded)
                            meta = {}
                        except Exception as e:
                            print(f"Could not convert list to DataFrame in {filename}: {e}")
                            continue
                    else:
                        print(f"Unrecognized format in {filename}: {type(loaded)}")
                        continue

                    df["source_file"] = filename
                    df["meta_pair_symbol"] = meta.get("pair_symbol", "unknown")
                    df["meta_broker"] = meta.get("broker", "unknown")
                    df["meta_master"] = meta.get("Master", "unknown")

                    if "timestamp" not in df.columns:
                        print(f"File {filename} has no 'timestamp' column.")
                        continue

                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    all_data.append(df)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

    if not all_data:
        raise ValueError("No simulation files loaded for the given dates.")

    return pd.concat(all_data, ignore_index=True)
