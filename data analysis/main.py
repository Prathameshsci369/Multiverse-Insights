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
from final_analysis import combined_analysis
# Import configuration and modules
from config import (
    JSON_PATH, GGUF_PATH, PHI4_MAX_CONTEXT, NUM_MODELS, MAX_WORKERS
)
from utils import (
    load_json, extract_texts, chunk_texts, create_processing_batches,
    estimate_tokens, process_json_to_batches
)
from model_manager import SingleModelManager
from analysis import process_batch

print(GGUF_PATH)
print(JSON_PATH)

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

def main(json_path: str = JSON_PATH):
    """
    The main pipeline orchestrator.
    """
    raw_analysis_output = ""
    
    try:
        partial_results = {}
        
        # ----------------------------------------------------
        # Step 1: Check for Partial Results Cache
        # ----------------------------------------------------
        if os.path.exists(PARTIAL_RESULTS_JSON):
            logging.info(f"Partial results cache found at {PARTIAL_RESULTS_JSON}. Loading analysis from cache...")
            with open(PARTIAL_RESULTS_JSON, 'r', encoding='utf-8') as f:
                partial_results_raw = json.load(f)
            
            # Sanitize the loaded cache
            partial_results = {}
            for key, value in partial_results_raw.items():
                if isinstance(value, dict):
                    partial_results[key] = value
                else:
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
                    max_tokens_per_batch=PHI4_MAX_CONTEXT - 3000  # Reserve space for prompt
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
            finally:
                # Clean up temporary files
                if os.path.exists(temp_batches_file):
                    try:
                        os.remove(temp_batches_file)
                    except Exception as e:
                        logging.warning(f"Could not remove temporary file {temp_batches_file}: {e}")
            
            # Step 3: Initialize Model Managers
            model_managers = []
            for i in range(NUM_MODELS):
                logging.info(f"Initializing model {i+1}/{NUM_MODELS}...")
                model_managers.append(SingleModelManager(GGUF_PATH, PHI4_MAX_CONTEXT))
                
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
            
        # ----------------------------------------------------
        # Step 6: Generate the final report
        # ----------------------------------------------------
        
        # Initialize cache directory if it doesn't exist
        cache_dir = os.path.join(os.path.dirname(PARTIAL_RESULTS_JSON), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Filter out batches that failed or contain explicit error messages
        final_results_list = []
        
        # First, validate partial results structure
        if not partial_results:
            logging.warning("No partial results found. Creating new cache structure.")
            partial_results = {}
        
        for key, value in partial_results.items():
            # Ensure we're working with dictionaries
            if not isinstance(value, dict):
                value = {"batch_id": key, "result": str(value)}
            
            # More robust error detection
            is_error_batch = False
            result_content = value.get('result', '')
            
            # Check various error conditions
            error_indicators = [
                lambda x: 'error' in value,
                lambda x: isinstance(x, str) and x.startswith('ERROR:'),
                lambda x: isinstance(x, str) and 'exceed context window' in x.lower(),
                lambda x: x is None or x == 'None',
                lambda x: isinstance(x, str) and len(x.strip()) < 10  # Too short to be valid
            ]
            
            for check in error_indicators:
                if check(result_content):
                    is_error_batch = True
                    break
            
            if is_error_batch:
                logging.warning(f"Skipping error batch {key} for final combined analysis. Reason: {result_content[:100]}...")
            else:
                # Only include non-error results
                if isinstance(result_content, str) and len(result_content.strip()) > 0:
                    final_results_list.append(result_content)
        
        # Check if we have enough successful batches
        total_batches = len(partial_results)
        successful_batches = len(final_results_list)
        
        if successful_batches == 0:
            logging.error("No successful batches available for final combined analysis.")
            return "ERROR: No successful batch analyses to generate final report."
        
        if successful_batches < total_batches * 0.3:
            logging.error(f"Too many batches failed: {successful_batches}/{total_batches} successful.")
            logging.error("Consider reducing the context limit and retrying the analysis.")
        
        # Ensure we have a model manager available for the final combined analysis
        if not ('model_managers' in locals() and model_managers):
            model_managers = [SingleModelManager(GGUF_PATH, PHI4_MAX_CONTEXT)]

        logging.info("Generating the final report...")
        
        # Pass the SAFE, filtered LIST OF STRINGS to combined_analysis
        raw_analysis_output =combined_analysis(
            PARTIAL_RESULTS_JSON
            
        )
        
        logging.info("Final report generation complete.")
        return raw_analysis_output

    except Exception as e:
        logging.error(f"Pipeline failed at a high level: {e}")
        return f"ERROR: Pipeline startup failed. {e}"
    finally:
        # Clean up model managers to free memory
        if 'model_managers' in locals():
            for manager in model_managers:
                try:
                    del manager
                except:
                    pass
            gc.collect()

if __name__ == "__main__":
    print(main())
