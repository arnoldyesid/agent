#!/usr/bin/env python3
"""
Script to get the exact paths needed for Claude MCP setup
"""

import os
import sys
import json

def get_setup_info():
    """Get all the information needed for MCP setup"""
    
    # Get current working directory
    current_dir = os.getcwd()
    
    # Get absolute path to the MCP server script
    script_name = "mcp_trading_server.py"
    script_path = os.path.join(current_dir, script_name)
    
    # Get Python executable path
    python_executable = sys.executable
    
    # Check if script exists
    script_exists = os.path.exists(script_path)
    
    # Get analysis results directory
    results_dir = os.path.join(current_dir, "analysis_results")
    results_exists = os.path.exists(results_dir)
    
    print("🔧 CLAUDE MCP CONNECTOR SETUP INFORMATION")
    print("=" * 60)
    print()
    
    print("📋 Copy these EXACT values into Claude MCP settings:")
    print("-" * 50)
    print(f"Server Name: Trading Analysis")
    print(f"Command: {python_executable}")
    print(f"Arguments: [\"{script_path}\"]")
    print(f"Working Directory: {current_dir}")
    print()
    
    print("📁 Path Details:")
    print(f"   Python executable: {python_executable}")
    print(f"   Script path: {script_path}")
    print(f"   Working directory: {current_dir}")
    print(f"   Results directory: {results_dir}")
    print()
    
    print("✅ File Status:")
    print(f"   MCP script exists: {'YES' if script_exists else 'NO - CREATE IT FIRST!'}")
    print(f"   Results dir exists: {'YES' if results_exists else 'NO - RUN ANALYSIS FIRST!'}")
    print()
    
    if not script_exists:
        print("⚠️  WARNING: mcp_trading_server.py not found!")
        print(f"   Save the MCP server code as: {script_path}")
        print()
    
    if not results_exists:
        print("⚠️  WARNING: analysis_results directory not found!")
        print("   Run your trading analysis script first to generate data")
        print()
    
    print("🚀 Setup Steps:")
    print("1. Go to: https://claude.ai/settings/connectors")
    print("2. Click 'Add MCP Server'")
    print("3. Copy the values above EXACTLY")
    print("4. Click 'Connect'")
    print()
    
    # Save to JSON file for easy reference
    setup_info = {
        "server_name": "Trading Analysis",
        "command": python_executable,
        "arguments": [script_path],
        "working_directory": current_dir,
        "script_exists": script_exists,
        "results_exists": results_exists
    }
    
    config_file = os.path.join(current_dir, "mcp_setup_config.json")
    with open(config_file, 'w') as f:
        json.dump(setup_info, f, indent=2)
    
    print(f"💾 Setup info saved to: {config_file}")
    print()
    
    # Windows vs Unix path handling
    if os.name == 'nt':  # Windows
        print("🖥️  Windows Detected:")
        print(f"   Use forward slashes in Arguments: [\"{script_path.replace(os.sep, '/')}\"]")
        print()
    
    return setup_info

if __name__ == "__main__":
    get_setup_info()