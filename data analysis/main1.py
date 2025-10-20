#!/usr/bin/env python3
"""
Main orchestration file for the analysis pipeline - V2 only.

The primary function has been renamed from 'main' to 'run_v2_analysis' 
to allow for clean import into Streamlit and distinguish it from main.py.
"""

import logging
import sys
import gc
import json
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import os
from pathlib import Path
import importlib
# Import configuration and modules (assuming these exist in your environment)
from config import (
    JSON_PATH, GGUF_PATH, PHI4_MAX_CONTEXT, NUM_MODELS, MAX_WORKERS,
    CHUNK_SIZE, CHUNK_OVERLAP, BATCH_SIZE_TOKENS, 
    TRANSLATION_MAX_LENGTH, TRANSLATION_RETRIES
)
from utils import (
    load_json, extract_texts, chunk_texts, create_processing_batches,
    translate, get_free_memory_mb
)

from model_manager import SingleModelManager
from analysis import process_batch, combined_analysis_V2 # combined_analysis_V2 generates the V2 schema

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout), # Log to console
        logging.FileHandler("analysis_log.txt")       # Log to file
    ]
)

def partial_results1(cache_partial_results):
    if cache_partial_results:
        return True
    else:
        return False


def run_v2_analysis(json_path=None):
    """
    Loads partial results from main.py's JSON file if present, otherwise imports main.py and uses its partial results for final output.
    """
    PARTIAL_RESULTS_JSON = "partial_results_main.json"
    try:
        # 1. Try to load partial results from main.py's JSON file
        if os.path.exists(PARTIAL_RESULTS_JSON):
            logging.info(f"Found {PARTIAL_RESULTS_JSON}, loading partial results and generating final output...")
            with open(PARTIAL_RESULTS_JSON, "r", encoding="utf-8") as f:
                partial_results = json.load(f)
            model_manager = SingleModelManager(GGUF_PATH, PHI4_MAX_CONTEXT)
            final_json_output = combined_analysis_V2(model_manager, partial_results, stream_llm_if_supported=True)
            if final_json_output:
                try:
                    parsed_output = json.loads(final_json_output)
                    if "multiverse_combined" not in parsed_output:
                        parsed_output = {"multiverse_combined": parsed_output}
                    final_json_output = json.dumps(parsed_output, indent=2)
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse JSON output: {e}")
                    error_output = {"multiverse_combined": {"error": f"Failed to generate valid JSON: {str(e)}"}}
                    final_json_output = json.dumps(error_output, indent=2)
            return final_json_output

        # 2. If not present, import main.py and use its partial results for final output
        logging.info("No partial results JSON found. Importing main.py to generate partial results...")
        main_module = importlib.import_module("main")
        # Run main to generate partial results and save them as JSON
        # (main.py will save the partial results as partial_results_main.json)
        main_module.main(json_path=json_path)
        # Now load the just-created partial results and generate final output
        if os.path.exists(PARTIAL_RESULTS_JSON):
            with open(PARTIAL_RESULTS_JSON, "r", encoding="utf-8") as f:
                partial_results = json.load(f)
            model_manager = SingleModelManager(GGUF_PATH, PHI4_MAX_CONTEXT)
            final_json_output = combined_analysis_V2(model_manager, partial_results, stream_llm_if_supported=True)
            if final_json_output:
                try:
                    parsed_output = json.loads(final_json_output)
                    if "multiverse_combined" not in parsed_output:
                        parsed_output = {"multiverse_combined": parsed_output}
                    final_json_output = json.dumps(parsed_output, indent=2)
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse JSON output: {e}")
                    error_output = {"multiverse_combined": {"error": f"Failed to generate valid JSON: {str(e)}"}}
                    final_json_output = json.dumps(error_output, indent=2)
            return final_json_output
        else:
            error_output = {"multiverse_combined": {"error": "Failed to generate partial results JSON from main.py."}}
            return json.dumps(error_output, indent=2)
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        error_output = {"multiverse_combined": {"error": f"Pipeline failed: {str(e)}"}}
        return json.dumps(error_output, indent=2)

if __name__ == "__main__":
    # If run directly, print the result to stdout
    print(run_v2_analysis())
