# main.py

#!/usr/bin/env python3
"""
Main orchestration file for the analysis pipeline.
This version focuses on orchestration only, using utils for batch processing.
"""

import logging
import sys
import gc
import json
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import configuration and modules
from config import (
    JSON_PATH, GGUF_PATH, PHI4_MAX_CONTEXT, NUM_MODELS, MAX_WORKERS
)
from utils import (
    load_json, extract_texts, chunk_texts, create_processing_batches,
    estimate_tokens, process_json_to_batches
)
from analysis import QwenModelManager, process_batch
from final_analysis import combined_analysis  # Import the combined_analysis function

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("analysis_log.txt")
    ]
)

PARTIAL_RESULTS_JSON = "partial_results_main.json"
FAILED_BATCHES_JSON = "failed_batches_main.json"

def save_results_to_file(results_list, filename):
    """Save a list of results to a JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results_list, f, indent=2)

def main(json_path: str = JSON_PATH):
    """
    The main pipeline orchestrator.
    """
    try:
        partial_results = {}
        
        # ----------------------------------------------------
        # Step 1: Check for Partial Results Cache
        # ----------------------------------------------------
        if os.path.exists(PARTIAL_RESULTS_JSON):
            logging.info(f"Partial results cache found at {PARTIAL_RESULTS_JSON}. Loading analysis from cache...")
            with open(PARTIAL_RESULTS_JSON, 'r', encoding='utf-8') as f:
                try:
                    partial_results_raw = json.load(f)
                except json.JSONDecodeError as e:
                    logging.error(f"Error parsing partial results JSON: {e}")
                    partial_results_raw = {}
            
            # Sanitize the loaded cache - ensure it's a dictionary
            if isinstance(partial_results_raw, list):
                logging.warning(f"Partial results is a list, converting to dictionary...")
                partial_results = {}
                for i, item in enumerate(partial_results_raw):
                    if isinstance(item, dict):
                        partial_results[f"batch_{i}"] = item
                    else:
                        logging.warning(f"Item {i} is not a dictionary, skipping...")
            elif isinstance(partial_results_raw, dict):
                partial_results = partial_results_raw
            else:
                logging.warning(f"Partial results is not a list or dictionary, creating empty dict...")
                partial_results = {}
            
            # Further sanitize the dictionary values
            for key, value in partial_results.items():
                if not isinstance(value, dict):
                    logging.warning(f"Sanitizing cache for {key}: expected dict, found {type(value)}. Converting to error dict.")
                    partial_results[key] = {"error": f"Corrupted batch result from cache. Raw content: {str(value)[:50]}..."}
            
            logging.info(f"Loaded {len(partial_results)} partial results from cache and sanitized.")
        else:
            logging.info("Partial results cache not found. Running full analysis pipeline...")
            
            # Step 2: Process input JSON into batches using utils
            temp_batches_file = "temp_batches.json"
            try:
                logging.info("Processing input JSON into batches...")
                num_batches = process_json_to_batches(
                    json_path,
                    temp_batches_file,
                    max_tokens_per_batch=PHI4_MAX_CONTEXT - 500  # Reserve space for prompt
                )
                logging.info(f"Successfully created {num_batches} batches")
                
                # Load the processed batches
                with open(temp_batches_file, 'r', encoding='utf-8') as f:
                    batches_data = json.load(f)
                
                # Convert the batches data into the format expected by the rest of the pipeline
                batches = []
                for batch_entry in batches_data["batches"]:
                    # Split the content into chunks at natural breaks
                    batch_chunks = batch_entry["content"].split("\n\n")
                    batches.append(batch_chunks)
                
                logging.info(f"Loaded {len(batches)} batches for processing")
                
            except Exception as e:
                logging.error(f"Error processing input JSON into batches: {e}")
                raise
            
            
            # Step 3: Initialize Model Managers
            model_managers = []
            for i in range(NUM_MODELS):
                logging.info(f"Initializing Qwen2.5 model {i+1}/{NUM_MODELS}...")
                model_managers.append(QwenModelManager(GGUF_PATH, PHI4_MAX_CONTEXT))
                
            # Step 4: Parallel Processing
            results_futures = {}
            logging.info(f"Starting analysis for {len(batches)} batches using {len(model_managers)} worker(s)...")

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                model_index = 0
                for i, batch in enumerate(batches):
                    # Rotate model managers for distribution
                    model_manager = model_managers[model_index % len(model_managers)]
                    model_index += 1
                    
                    future = executor.submit(process_batch, model_manager, batch, f"Batch {i}")
                    results_futures[future] = i

                # Collect results
                for future in tqdm(as_completed(results_futures), total=len(batches), desc="Processing Batches"):
                    batch_index = results_futures[future]
                    try:
                        result = future.result()
                        
                        # Ensure the result is a dictionary before saving
                        if isinstance(result, str):
                            try:
                                result = json.loads(result)
                            except json.JSONDecodeError:
                                logging.warning(f"Batch {batch_index} returned unparsable string. Saving as error.")
                                result = {"error": f"LLM output unparsable: {result[:100]}..."}
                        
                        if isinstance(result, dict):
                            partial_results[f"batch_{batch_index}"] = result
                        else:
                            logging.error(f"Batch {batch_index} returned unexpected type ({type(result)}). Saving as error.")
                            partial_results[f"batch_{batch_index}"] = {"error": f"Unexpected result type: {type(result)}"}
                        
                    except Exception as e:
                        logging.error(f"Error processing batch {batch_index}: {e}")
                        partial_results[f"batch_{batch_index}"] = {"error": str(e)}

            # Step 5: Save Partial Results to cache file
            logging.info(f"Saving {len(partial_results)} partial results to {PARTIAL_RESULTS_JSON}...")
            with open(PARTIAL_RESULTS_JSON, 'w', encoding='utf-8') as f:
                json.dump(partial_results, f, indent=2)
            logging.info("Partial results saved.")
            
            # Clean up model managers
            for manager in model_managers:
                manager.cleanup()
        
        # ----------------------------------------------------
        # Step 6: Generate the final report using the final_analysis module
        # ----------------------------------------------------
        logging.info("Generating the final report using final_analysis module...")
        
        # Check if we have enough successful batches
        total_batches = len(partial_results)
        successful_batches = sum(1 for v in partial_results.values() 
                                if isinstance(v, dict) and 'result' in v and 
                                not v.get('result', '').startswith('ERROR:'))
        
        if successful_batches == 0:
            logging.error("No successful batches available for final combined analysis.")
            return "ERROR: No successful batch analyses to generate final report."
        
        if successful_batches < total_batches * 0.3:
            logging.warning(f"Many batches failed: {successful_batches}/{total_batches} successful.")
        
        # Save the partial results to a temporary file if not already saved
        if not os.path.exists(PARTIAL_RESULTS_JSON):
            logging.info(f"Saving partial results to {PARTIAL_RESULTS_JSON} for final analysis...")
            with open(PARTIAL_RESULTS_JSON, 'w', encoding='utf-8') as f:
                json.dump(partial_results, f, indent=2)
        
        # Use the combined_analysis function from final_analysis.py
        # Pass the path to the partial results file
        final_analysis_result = combined_analysis(PARTIAL_RESULTS_JSON)
        
        # Return the final analysis result
        return final_analysis_result

    except Exception as e:
        logging.error(f"Pipeline failed at a high level: {e}")
        return f"ERROR: Pipeline startup failed. {e}"
    finally:
        # Clean up model managers to free memory
        if 'model_managers' in locals():
            for manager in model_managers:
                try:
                    manager.cleanup()
                except:
                    pass
            gc.collect()

if __name__ == "__main__":
    print(main())
