# main2.py (Simplified with sequential workflow)
from main import main
from final_analysis import combined_analysis
import json
import os 
import re 
import time
from typing import Dict, Any
from datetime import datetime
import tempfile

# Import PDF generation functionality
from pdf_generator import pdf_generator

# Import RAG Chatbot
from rag_chatbot import create_rag_chatbot

# Define the output file name
OUTPUT_FILE = "final_output.txt"  # Changed to .txt since we're using simple format

def combine_analysis(json_path: str) -> str:
    """
    Orchestrates the V1 analysis (calling main.main()), 
    with cache validation and state tracking.
    """
    print("Analysis are starting")
    
    # Ensure cache directory exists
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Cache state file
    cache_state_file = os.path.join(cache_dir, "analysis_state.json")
    
    # Initialize or load cache state
    if os.path.exists(cache_state_file):
        try:
            with open(cache_state_file, 'r') as f:
                cache_state = json.load(f)
        except:
            cache_state = {"last_run": None, "input_hash": None}
    else:
        cache_state = {"last_run": None, "input_hash": None}
    
    # --- Step 1: Run V1 analysis (returns the raw LLM string output) ---
    a_json_string = main(json_path)
    
    # Save cache state
    import hashlib
    with open(json_path, 'rb') as f:
        input_hash = hashlib.md5(f.read()).hexdigest()
    
    cache_state.update({
        "last_run": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_hash": input_hash
    })
    
    with open(cache_state_file, 'w') as f:
        json.dump(cache_state, f, indent=2)
    
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

def generate_pdf_report():
    """Generate PDF from the parsed data in final_output.txt"""
    print("\n" + "="*50)
    print("Generating PDF Report...")
    print("="*50)
    
    # Use the global PDF generator instance
    pdf_path = pdf_generator.generate_pdf_from_output_file(OUTPUT_FILE)
    
    if pdf_path:
        print(f"\n📄 PDF report available at: {pdf_path}")
        print(f"📁 You can find the report in the current directory.")
        
        # Ask if user wants to open the PDF
        try:
            import subprocess
            import platform
            
            # Try to open the PDF with the default viewer
            if platform.system() == "Darwin":  # macOS
                subprocess.run(["open", pdf_path])
            elif platform.system() == "Windows":  # Windows
                os.startfile(pdf_path)
            else:  # Linux
                subprocess.run(["xdg-open", pdf_path])
            
            print(f"📂 Opening PDF report with default viewer...")
        except Exception as e:
            print(f"⚠️ Could not open PDF automatically: {str(e)}")
            print(f"Please open the file manually: {pdf_path}")
    else:
        print("\n❌ Failed to generate PDF report")

def start_rag_chatbot(json_path: str):
    """Initialize and start the RAG chatbot with the original data."""
    print("\n" + "="*50)
    print("Starting RAG Chatbot - Query Extension")
    print("="*50)
    print("You can now ask questions about the original data.")
    print("Type 'exit' to end the conversation.")
    print("="*50 + "\n")
    
    try:
        # Create and initialize the RAG chatbot
        chatbot = create_rag_chatbot(json_path)
        
        # Start the interactive chat
        chatbot.interactive_chat()
        
    except Exception as e:
        print(f"❌ Error starting RAG chatbot: {str(e)}")
        print("Please check the logs for more details.")

# Example usage (uncomment and update path to run)
if __name__ == "__main__":
    path = "/home/anand/final_app/data/youtube_search_output.json"
    print(path)
    
    # Step 1: Run the analysis
    combine_analysis(path)
    
    # Step 2: Parse and display the final output
    from simple_parser import parse_and_display
    print("\n" + "="*50)
    print("Parsing and displaying final output...")
    print("="*50)
    parse_and_display("final_output.txt")
    
    
    
    # Step 4: Start the RAG chatbot as an extension
    print("\n" + "="*50)
    print("Analysis Complete! Starting Chatbot Extension...")
    print("="*50)
    start_rag_chatbot(path)
    # Step 3: Generate PDF report
    generate_pdf_report()
