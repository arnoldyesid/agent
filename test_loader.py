#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from data_loader import get_real_trade_chunks, load_simulation_data_for_dates
from pprint import pprint

print("\nLoading real data...")
chunks = get_real_trade_chunks()
print(f"{len(chunks)} real data chunks loaded.\n")

# Check unique dates
date_strs = set(chunk["timestamp"].dt.strftime("%d-%m-%Y").iloc[0] for chunk in chunks)
print("Dates found in real data:", date_strs)

print("\nLoading simulation data for those dates...")
try:
    sim_df = load_simulation_data_for_dates(date_strs)
    print(f"{len(sim_df)} simulation data rows loaded.")
    print(sim_df.head())
except Exception as e:
    print(f"Error loading simulation data: {e}")