#!/usr/bin/env python3
"""
Main orchestration file for the analysis pipeline.
This version includes improved batch processing, error handling, and retry mechanisms.
"""

import logging
import sys
import gc
import json
import re
import os
from pathlib import Path
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import configuration and modules
from config import (
    JSON_PATH, GGUF_PATH, PHI4_MAX_CONTEXT, NUM_MODELS, MAX_WORKERS,
    CHUNK_SIZE, CHUNK_OVERLAP, BATCH_SIZE_TOKENS, 
    TRANSLATION_MAX_LENGTH, TRANSLATION_RETRIES
)
from utils import (
    load_json, extract_texts, chunk_texts, create_processing_batches,
    translate, get_free_memory_mb, validate_and_reprocess_chunks, validate_batches,
    process_json_to_batches, estimate_tokens
)
from model_manager import SingleModelManager
from analysis import process_batch, combined_analysis

print(GGUF_PATH)

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

def get_original_batch(batch_id, original_batches):
    """
    Retrieve the original batch data for a given batch_id.
    Handles both legacy format and new JSON batch format.
    """
    try:
        # Extract the numeric part from batch_id (e.g., "Batch 0" -> 0 or "batch_1" -> 1)
        batch_num = int(''.join(filter(str.isdigit, batch_id)))
        
        if batch_num < len(original_batches):
            batch = original_batches[batch_num]
            # Convert to consistent format
            if isinstance(batch, dict) and 'content' in batch:
                return batch['content']
            elif isinstance(batch, (list, str)):
                return batch
        
        logging.error(f"Batch {batch_num} not found in original_batches")
        return None
    except (ValueError, IndexError) as e:
        logging.error(f"Could not parse batch_id {batch_id}: {e}")
        return None

def process_batch_with_limit(llm_manager, batch, batch_id, max_tokens):
    """
    Process a batch with a specific token limit.
    Handles batches in the new format where they may be a string or list of chunks.
    """
    # Convert batch to text content
    if isinstance(batch, list):
        joined_text = "\n\n".join(batch)
    elif isinstance(batch, str):
        joined_text = batch
    else:
        joined_text = str(batch)
    
    # Estimate tokens and truncate if necessary
    total_tokens = estimate_tokens(joined_text)
    safe_token_limit = max_tokens - 2000  # Reserve space for prompt and response
    
    if total_tokens > safe_token_limit:
        reduction_ratio = safe_token_limit / total_tokens
        cutoff_chars = int(len(joined_text) * reduction_ratio)
        joined_text = joined_text[:cutoff_chars]
        logging.info(f"[process_batch_with_limit] Batch {batch_id} truncated to {estimate_tokens(joined_text)} tokens.")
    
    prompt = f"""Analyze the following text and provide a structured summary, sentiment, and key topics.

Text:
{joined_text}

===BEGIN ANALYSIS===

1. SUMMARY:
[Write 2-3 sentences summarizing the key points]

2. SENTIMENT:
- Positive: [X]% - [brief explanation]
- Negative: [X]% - [brief explanation]
- Neutral: [X]% - [brief explanation]

3. KEY THEMES:
- Theme 1: [brief description]
- Theme 2: [brief description]
- Theme 3: [brief description]

===END ANALYSIS===
"""
    
    try:
        result = llm_manager.generate(prompt, max_tokens=1500)  # Reduced response size
        return {"batch_id": batch_id, "result": result}
    except Exception as e:
        error_msg = f"ERROR: Failed to process batch {batch_id} on retry: {str(e)}"
        logging.error(error_msg)
        return {"batch_id": batch_id, "result": error_msg}

def retry_failed_batches(failed_batches, model_manager, original_batches, max_retries=2):
    """
    Retry processing batches that failed due to context window issues.
    """
    retry_results = {}
    
    for batch_id, batch_data in failed_batches.items():
        if 'context window' in batch_data.get('result', ''):
            logging.info(f"Retrying batch {batch_id} with reduced content...")
            
            # Get the original batch data
            original_batch = get_original_batch(batch_id, original_batches)
            
            if original_batch:
                # Process with a much smaller context limit
                retry_result = process_batch_with_limit(
                    model_manager, 
                    original_batch, 
                    batch_id, 
                    max_tokens=PHI4_MAX_CONTEXT // 2  # Use half the context limit
                )
                
                retry_results[batch_id] = retry_result
            else:
                logging.error(f"Could not find original batch for {batch_id}")
    
    return retry_results

def main(json_path: str = JSON_PATH):
    """
    The main pipeline orchestrator with improved batch processing and error handling.
    """
    raw_analysis_output = ""
    original_batches = []  # Store original batches for potential retry
    
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
            
            # --- Steps 2 through 7 (Full Run) ---
            
            # Step 2: Process input JSON into batches using the new utility
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
                
                original_batches = batches.copy()  # Store for potential retry
                
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
            
            # Step 5: Initialize Model Managers
            model_managers = []
            for i in range(NUM_MODELS):
                logging.info(f"Initializing model {i+1}/{NUM_MODELS}...")
                model_managers.append(SingleModelManager(GGUF_PATH, PHI4_MAX_CONTEXT))
                
            # Step 6: Parallel Processing
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

            # Check error rate
            error_count = sum(1 for result in partial_results.values() if 'error' in result)
            total_batches = len(partial_results)
            
            if error_count > 0:
                logging.warning(f"Found {error_count}/{total_batches} batches with errors.")
                
                # Try to retry batches that failed due to context window issues
                failed_batches = {k: v for k, v in partial_results.items() 
                                if 'error' in v and 'context window' in v.get('result', '')}
                
                if failed_batches:
                    logging.info(f"Attempting to retry {len(failed_batches)} batches that failed due to context window issues...")
                    retry_results = retry_failed_batches(failed_batches, model_managers[0], original_batches)
                    
                    # Update partial_results with retry results
                    for batch_id, result in retry_results.items():
                        if 'error' not in result or 'context window' not in result.get('result', ''):
                            partial_results[batch_id] = result
                            logging.info(f"Successfully retried batch {batch_id}")
                        else:
                            logging.warning(f"Retry failed for batch {batch_id}")

            # Step 7: Save Partial Results to cache file
            logging.info(f"Saving {len(partial_results)} partial results to {PARTIAL_RESULTS_JSON}...")
            with open(PARTIAL_RESULTS_JSON, 'w', encoding='utf-8') as f:
                json.dump(partial_results, f, indent=2)
            logging.info("Partial results saved.")
            
            # --- End of Full Run ---
            
        # main.py

# ... (around line 245, before Step 8)

        # ----------------------------------------------------
        # Step 8: Generate the final report
        # ----------------------------------------------------
        
        # Initialize cache directory if it doesn't exist
        cache_dir = os.path.join(os.path.dirname(PARTIAL_RESULTS_JSON), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Filter out batches that failed or contain explicit error messages
        final_results_list = [] # <-- NEW: Change to a list of strings for minimal input
        
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
        # The combined_analysis function will then join this list into one clean text corpus.
        raw_analysis_output = combined_analysis(
            model_managers[0], 
            # Passing the list of ONLY the 'result' strings:
            final_results_list, 
            stream_llm_if_supported=True
        )
        
        logging.info("Final report generation complete.")
        return raw_analysis_output

# ... (rest of main.py)

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
