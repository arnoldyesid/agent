#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from langchain_openai import ChatOpenAI


def analyze_discrepancy_chunk(real_chunk: pd.DataFrame, sim_df: pd.DataFrame, llm: ChatOpenAI) -> str:
    if sim_df.empty or real_chunk.empty:
        return "No data available to compare."

    # Renombrar para unificar nombres
    if "price" in real_chunk.columns:
        real_chunk = real_chunk.rename(columns={"price": "price_real"})
    else:
        return "'price' column not found in real_chunk."

    if "price" in sim_df.columns:
        sim_df = sim_df.rename(columns={"price": "price_sim"})
    else:
        return "'price' column not found in sim_df."

    # Asegúrate de que ambas tengan timestamps
    if "timestamp" not in real_chunk.columns or "timestamp" not in sim_df.columns:
        return "Missing 'timestamp' column in one of the dataframes."

    # Orden y merge por tiempo
    merged = pd.merge_asof(
        real_chunk.sort_values("timestamp"),
        sim_df.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("2s")
    )

    merged["price_diff"] = (merged["price_real"] - merged["price_sim"]).abs()
    filtered = merged[merged["price_diff"] > 2]

    if filtered.empty:
        return "No discrepancies > $2 found in this chunk."

    # Sample de discrepancias
    sample = filtered[
        ["timestamp", "price_real", "price_sim", "price_diff", "source_file"]
    ].head(10).to_string(index=False)

    # Metadatos (si existen)
    meta_info = sim_df[
        ["meta_pair_symbol", "meta_broker", "meta_master"]
    ].drop_duplicates().to_string(index=False)

    # Prompt para LLM
    prompt = (
        "Simulation metadata:\n"
        f"{meta_info}\n\n"
        "Discrepancies between real and simulated trades (price_diff > $2):\n"
        f"{sample}\n\n"
        "1. What are possible causes of these mismatches?\n"
        "2. How can simulation accuracy be improved?"
    )

    explanation = llm.predict(prompt)
    return f"Discrepancy Sample:\n{sample}\n\nAnalysis:\n{explanation}"
