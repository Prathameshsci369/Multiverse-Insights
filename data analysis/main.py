#!/usr/bin/env python3
"""
Main orchestration file for the analysis pipeline.
This version processes batches sequentially to provide clean, linear streaming output.
It also includes a token count check to potentially skip batch processing for smaller inputs.
"""

import logging
import sys
import gc
import json
import os
from tqdm import tqdm

# Import configuration and modules
from config import (
    JSON_PATH, GGUF_PATH, PHI4_MAX_CONTEXT
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

def main(json_path: str = JSON_PATH):
    """
    The main pipeline orchestrator.
    """
    model_manager = None  # Initialize model_manager to None for the finally block
    try:
        # Load the input JSON data
        data = load_json(json_path)
        if not data:
            return "ERROR: Failed to load input JSON."
        
        # Extract all texts first
        all_texts = extract_texts(data)
        joined_text = " ".join(all_texts)
        total_tokens = estimate_tokens(joined_text)
        
        logging.info(f"Total tokens in input data: {total_tokens}")
        
        # Check if total tokens are less than 30k
        if total_tokens < 30000:
            logging.info(f"Total tokens ({total_tokens}) is less than 30k. Skipping batch processing for direct analysis.")
            
            # Create a special structure for direct analysis
            direct_analysis_data = {
                "direct_analysis": {
                    "text": joined_text,
                    "tokens": total_tokens
                }
            }
            
            # Call combined_analysis with the direct analysis flag
            final_analysis_result = combined_analysis(direct_analysis_data, is_direct=True)
            
            # Return the final analysis result
            return final_analysis_result
        else:
            # Proceed with existing batch processing logic
            logging.info(f"Total tokens ({total_tokens}) exceed 30k. Proceeding with batch processing.")
            
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
                
                # Step 3: Initialize a single Model Manager for sequential processing
                logging.info(f"Initializing Qwen2.5 model for sequential processing...")
                model_manager = QwenModelManager(GGUF_PATH, PHI4_MAX_CONTEXT)
                    
                # Step 4: Sequential Processing
                logging.info(f"Starting sequential analysis for {len(batches)} batches...")

                # Use tqdm to show a progress bar for the sequential processing
                for i, batch in tqdm(enumerate(batches), total=len(batches), desc="Processing Batches"):
                    batch_id = f"Batch {i}"
                    try:
                        # Process the batch directly and stream output to terminal
                        result = process_batch(model_manager, batch, batch_id)
                        
                        # Ensure the result is a dictionary before saving
                        if isinstance(result, str):
                            try:
                                result = json.loads(result)
                            except json.JSONDecodeError:
                                logging.warning(f"Batch {i} returned unparsable string. Saving as error.")
                                result = {"error": f"LLM output unparsable: {result[:100]}..."}
                        
                        if isinstance(result, dict):
                            partial_results[f"batch_{i}"] = result
                        else:
                            logging.error(f"Batch {i} returned unexpected type ({type(result)}). Saving as error.")
                            partial_results[f"batch_{i}"] = {"error": f"Unexpected result type: {type(result)}"}
                        
                    except Exception as e:
                        logging.error(f"Error processing batch {i}: {e}")
                        partial_results[f"batch_{i}"] = {"error": str(e)}

                # Step 5: Save Partial Results to cache file
                logging.info(f"Saving {len(partial_results)} partial results to {PARTIAL_RESULTS_JSON}...")
                with open(PARTIAL_RESULTS_JSON, 'w', encoding='utf-8') as f:
                    json.dump(partial_results, f, indent=2)
                logging.info("Partial results saved.")
                
                # Clean up the model manager
                if model_manager:
                    model_manager.cleanup()
                    model_manager = None
            
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
        # Clean up model manager to free memory if it still exists
        if model_manager:
            try:
                logging.info("Cleaning up model manager in finally block.")
                model_manager.cleanup()
            except Exception as e:
                logging.error(f"Error during final cleanup: {e}")
            gc.collect()

if __name__ == "__main__":
    print(main())
