# main2.py
from main import main
from main1 import run_v2_analysis  # Assuming this function exists for V2 analysis
import json
import os 
import re 
from typing import Dict, Any

# Define the output file name
OUTPUT_FILE = "final_output.txt"  # Changed to .txt since we're using simple format

def combine_analysis(json_path: str) -> str:
    """
    Orchestrates the V1 analysis (calling main.main()), 
    and saves the result to a file.
    """
    print("Analysis are starting")
    
    # --- Step 1: Run V1 analysis (returns the raw LLM string output) ---
    a_json_string = main(json_path) 
    
    # Save the raw output to a file for debugging
    with open("raw_output.txt", 'w', encoding='utf-8') as f:
        f.write(a_json_string)
    print("Raw output saved to raw_output.txt for debugging")
    
    # --- Step 2: Save the output to the output file ---
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(a_json_string)
        print(f"✅ V1 Analysis result saved successfully to {OUTPUT_FILE}")
    except Exception as e:
        print(f"❌ Error saving file {OUTPUT_FILE}: {e}")
    
    # --- Step 3: Run V2 analysis (currently commented out) ---
    # # b = run_v2_analysis(json_path)
    # # print(b)
    
    print("Both analysis are complete") 
    
    # Return the original raw string output from main.py
    return a_json_string

# Example usage (uncomment and update path to run)
if __name__ == "__main__":
    path = "/home/anand/final_app/data/reddit_20251013_123358.json"
    print(path)
    combine_analysis(path)
    
    # After the analysis is complete, parse and display the results
    from simple_parser import parse_and_display
    print("Parsing and displaying final output...")
    parse_and_display("final_output.txt")
