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
    translate, get_free_memory_mb, validate_and_reprocess_chunks, validate_batches
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
    """
    try:
        # Extract the numeric part from batch_id (e.g., "Batch 0" -> 0)
        batch_num = int(batch_id.split()[-1])
        return original_batches[batch_num]
    except (ValueError, IndexError):
        logging.error(f"Could not parse batch_id: {batch_id}")
        return None

def process_batch_with_limit(llm_manager, batch, batch_id, max_tokens):
    """
    Process a batch with a specific token limit.
    """
    joined_text = " ".join(batch)
    
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

---
Report:
- Summary:
- Sentiment:
- Topics:
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
            
            # Step 2: Load and preprocess data
            original_data = load_json(json_path)
            all_texts = extract_texts(original_data)
            
            # Step 3: Chunk texts for context
            chunks = chunk_texts(all_texts, CHUNK_SIZE, CHUNK_OVERLAP)
            
            # Validate and reprocess any chunks that are too large
            chunks = validate_and_reprocess_chunks(chunks, PHI4_MAX_CONTEXT - 3000)
            
            # Step 4: Create processing batches for the LLM
            batches = create_processing_batches(chunks, PHI4_MAX_CONTEXT)
            
            # Validate batches
            valid_batches, oversized_batches = validate_batches(batches, PHI4_MAX_CONTEXT - 3000)
            
            # If there are oversized batches, reprocess them
            if oversized_batches:
                logging.info(f"Reprocessing {len(oversized_batches)} oversized batches...")
                for batch_id, batch, tokens in oversized_batches:
                    # Split the batch into smaller chunks
                    new_chunks = []
                    for chunk in batch:
                        chunk_tokens = estimate_tokens(chunk)
                        if chunk_tokens > PHI4_MAX_CONTEXT - 3000:
                            # Split this chunk
                            words = chunk.split()
                            sub_chunk = []
                            sub_chunk_tokens = 0
                            
                            for word in words:
                                word_tokens = estimate_tokens(word)
                                if sub_chunk_tokens + word_tokens <= PHI4_MAX_CONTEXT - 3000:
                                    sub_chunk.append(word)
                                    sub_chunk_tokens += word_tokens
                                else:
                                    if sub_chunk:
                                        new_chunks.append(" ".join(sub_chunk))
                                    sub_chunk = [word]
                                    sub_chunk_tokens = word_tokens
                            
                            if sub_chunk:
                                new_chunks.append(" ".join(sub_chunk))
                        else:
                            new_chunks.append(chunk)
                    
                    # Create new batches from the reprocessed chunks
                    new_batches = create_processing_batches(new_chunks, PHI4_MAX_CONTEXT)
                    valid_batches.extend(new_batches)
                
                logging.info(f"After reprocessing, total batches: {len(valid_batches)}")
            
            # Use valid_batches for processing
            batches = valid_batches
            original_batches = batches.copy()  # Store for potential retry
            
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
            
        # ----------------------------------------------------
        # Step 8: Generate the final report
        # ----------------------------------------------------
        
        # Filter out batches that failed or contain explicit error messages
        safe_partial_results = {}
        for key, value in partial_results.items():
            # Ensure we're working with dictionaries
            if not isinstance(value, dict):
                value = {"batch_id": key, "result": str(value)}
            
            # Check for explicit 'error' key or if 'result' contains an error string
            is_error_batch = False
            
            if 'error' in value:
                is_error_batch = True
            elif isinstance(value.get('result'), str) and value.get('result').startswith('ERROR:'):
                 is_error_batch = True
            
            if is_error_batch:
                logging.warning(f"Skipping error batch {key} for final combined analysis.")
            else:
                safe_partial_results[key] = value
        
        # Check if we have enough successful batches
        if len(safe_partial_results) < len(partial_results) * 0.3:
            logging.error(f"Too many batches failed: {len(safe_partial_results)}/{len(partial_results)} successful.")
            logging.error("Consider reducing the context limit and retrying the analysis.")
        
        # Ensure we have a model manager available for the final combined analysis
        if not ('model_managers' in locals() and model_managers):
            model_managers = [SingleModelManager(GGUF_PATH, PHI4_MAX_CONTEXT)]

        logging.info("Generating the final report...")
        
        # Pass the SAFE, filtered results to combined_analysis
        raw_analysis_output = combined_analysis(model_managers[0], safe_partial_results, stream_llm_if_supported=True)
        
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