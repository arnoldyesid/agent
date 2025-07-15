#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType

from data_loader import get_real_trade_chunks, load_simulation_data_for_dates
from discrepancy import analyze_discrepancy_chunk

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env")

# No system_message here
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=OPENAI_API_KEY
)

def analyze_all(_: str) -> str:
    all_results = []
    for real_chunk in get_real_trade_chunks():
        if real_chunk.empty:
            continue

        date_strs = real_chunk["timestamp"].dt.strftime("%d-%m-%Y").unique().tolist()
        sim_df = load_simulation_data_for_dates(set(date_strs))

        result = analyze_discrepancy_chunk(real_chunk, sim_df, llm)
        all_results.append(result)

    return "\n\n".join(all_results) if all_results else "No discrepancies found in any chunk."

def list_data(_: str) -> str:
    try:
        return "Files:\n" + "\n".join(os.listdir("data"))
    except Exception as e:
        return f"Error listing files: {e}"

def preview_file(file_name: str) -> str:
    import pandas as pd

    path = os.path.join("data", file_name.strip())
    if not os.path.exists(path):
        return f"File '{file_name}' not found."

    try:
        if file_name.endswith(".csv"):
            df = pd.read_csv(path)
        elif file_name.endswith(".pickle"):
            loaded = pd.read_pickle(path)
            if isinstance(loaded, dict) and "data" in loaded:
                df = loaded["data"]
            elif isinstance(loaded, pd.DataFrame):
                df = loaded
            else:
                return f"Unsupported pickle format in '{file_name}'. Expected a DataFrame or dict with 'data' key."
        return df.head().to_string(index=False)
    except Exception as e:
        return f"Error reading file: {e}"

tools = [
    Tool(name="AnalyzeAllDiscrepancies", func=analyze_all,
         description="Run full real vs sim trade analysis in batches."),
    Tool(name="ListDataFiles", func=list_data,
         description="List files in the 'data/' directory."),
    Tool(name="PreviewFile", func=preview_file,
         description="Preview the content of a CSV or Pickle file. Input should be the filename."),
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

def main():
    print("Trade Analysis Agent is ready. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        try:
            response = agent.run(user_input)
            print(f"\n{response}\n")
        except Exception as e:
            print(f"\nError: {e}\n")

if __name__ == "__main__":
    main()
