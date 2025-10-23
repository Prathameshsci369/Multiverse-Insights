#!/usr/bin/env python3
"""
Utility functions for the analysis pipeline.
"""

import json
import logging
import sys
import datetime
import hashlib
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from config import PHI4_MAX_CONTEXT, SAFETY_MARGIN, MIN_TOKENS_RESERVE


def validate_batches(batches, max_tokens_per_batch):
    """
    Validates batches and returns a list of valid batches and a list of oversized batches.
    """
    valid_batches = []
    oversized_batches = []
    
    for i, batch in enumerate(batches):
        batch_tokens = sum(estimate_tokens(chunk) for chunk in batch)
        
        if batch_tokens > max_tokens_per_batch:
            logging.warning(f"Batch {i} is too large ({batch_tokens} tokens > {max_tokens_per_batch}).")
            oversized_batches.append((i, batch, batch_tokens))
        else:
            valid_batches.append(batch)
    
    if oversized_batches:
        logging.warning(f"Found {len(oversized_batches)} oversized batches that need reprocessing.")
    
    return valid_batches, oversized_batches

def validate_and_reprocess_chunks(chunks, max_tokens):
    """
    Validates chunks and reprocesses any that exceed the max token limit.
    """
    valid_chunks = []
    oversized_count = 0
    
    for i, chunk in enumerate(chunks):
        chunk_tokens = estimate_tokens(chunk)
        
        if chunk_tokens > max_tokens:
            oversized_count += 1
            logging.warning(f"Chunk {i} is too large ({chunk_tokens} tokens > {max_tokens}). Splitting it.")
            
            # Split the oversized chunk
            words = chunk.split()
            sub_chunk = []
            sub_chunk_tokens = 0
            
            for word in words:
                word_tokens = estimate_tokens(word)
                if sub_chunk_tokens + word_tokens <= max_tokens:
                    sub_chunk.append(word)
                    sub_chunk_tokens += word_tokens
                else:
                    if sub_chunk:
                        valid_chunks.append(" ".join(sub_chunk))
                    sub_chunk = [word]
                    sub_chunk_tokens = word_tokens
            
            # Add the last sub-chunk if it has content
            if sub_chunk:
                valid_chunks.append(" ".join(sub_chunk))
        else:
            valid_chunks.append(chunk)
    
    if oversized_count > 0:
        logging.info(f"Reprocessed {oversized_count} oversized chunks. Total chunks now: {len(valid_chunks)}")
    
    return valid_chunks


def _iso_now():
    """Returns the current UTC time in ISO 8601 format."""
    return datetime.datetime.utcnow().isoformat() + 'Z'

def load_json(path):
    """Load and return JSON from `path`. On error, log and re-raise."""
    import json, logging, os
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"JSON file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load JSON from {path}: {e}")
        raise

def estimate_tokens(text):
    """Rough estimation of token count (1 token ≈ 4 characters for most models)."""
    if isinstance(text, (list, tuple)):
        text = " ".join(text)
    # integer division rounding up: (length + 3) // 4
    length = len(text)
    return (length + 3) // 4


def _truncate_text_to_token_limit(text: str, max_tokens: int) -> str:
    """
    Truncate `text` so the estimated token count <= max_tokens.
    Uses the same token heuristic as estimate_tokens (1 token ≈ 4 chars).
    """
    if not text:
        return text
    char_cutoff = max_tokens * 4
    if len(text) <= char_cutoff:
        return text
    truncated = text[:char_cutoff]
    return truncated


def extract_texts(data, max_tokens_from_file: int = 20000):
    """
    Extracts all text content from various JSON structures and returns only the
    first `max_tokens_from_file` tokens (default 20k) worth of text as a single
    consolidated text element (list with one string). This prevents the pipeline
    from processing huge input files beyond the budget.
    """
    texts = []

    def find_all_strings(obj):
        if isinstance(obj, str) and obj.strip():
            texts.append(obj)
        elif isinstance(obj, dict):
            for value in obj.values():
                find_all_strings(value)
        elif isinstance(obj, list):
            for item in obj:
                find_all_strings(item)

    find_all_strings(data)

    if not texts:
        return []

    consolidated = "\n\n".join(texts)

    total_tokens = estimate_tokens(consolidated)
    if total_tokens <= max_tokens_from_file:
        logging.info(f"extract_texts: extracted {total_tokens:,} tokens (<= {max_tokens_from_file:,}), using full text.")
        return [consolidated]

    logging.info(f"extract_texts: extracted {total_tokens:,} tokens (> {max_tokens_from_file:,}); truncating to {max_tokens_from_file:,} tokens.")
    truncated = _truncate_text_to_token_limit(consolidated, max_tokens_from_file)

    # try to cut at the last natural break near the end
    last_break = max(truncated.rfind("\n\n"), truncated.rfind("\n"), truncated.rfind(". "))
    if last_break > int(0.6 * len(truncated)):
        truncated = truncated[:last_break+1]

    final_tokens = estimate_tokens(truncated)
    logging.info(f"extract_texts: truncated text is {final_tokens:,} tokens.")
    return [truncated]



def chunk_texts(texts, size, overlap):
    """
    Splits texts into smaller, overlapping chunks for better context utilization.
    Uses a very conservative size limit and multiple splitting passes.
    """
    # Calculate an ultra-conservative chunk size
    # Reserve tokens for:
    # - Prompt overhead (500 tokens)
    # - Response generation (3500 tokens)
    # - Safety margin (1000 tokens)
    # Total reserved: 5000 tokens
    max_safe_tokens = min(size, (PHI4_MAX_CONTEXT - 5000))
    safe_chunk_size = max_safe_tokens * 4 // 4  # Convert tokens to chars, conservative estimate
    
    # Use multiple levels of separators for better splitting
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=safe_chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=[
            "\n\n\n",  # Large section breaks
            "\n\n",    # Paragraph breaks
            "\n",      # Line breaks
            ". ",      # Sentences
            ", ",      # Clauses
            " ",       # Words
            ""         # Characters
        ]
    )
    
    chunks = []
    for text in texts:
        # First split into large chunks
        text_chunks = splitter.split_text(text)
        
        # Then verify and potentially split large chunks again
        for chunk in text_chunks:
            chunk_tokens = estimate_tokens(chunk)
            if chunk_tokens > max_safe_tokens:
                # If still too large, split again with smaller size
                smaller_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=safe_chunk_size // 2,  # Use half the size
                    chunk_overlap=overlap // 2,
                    length_function=len,
                    separators=["\n", ". ", ", ", " ", ""]
                )
                sub_chunks = smaller_splitter.split_text(chunk)
                chunks.extend(sub_chunks)
            else:
                chunks.append(chunk)
    
    logging.info(f"Successfully created {len(chunks)} chunks from the consolidated data.")
    
    # Log and verify the size distribution of chunks
    chunk_sizes = [estimate_tokens(chunk) for chunk in chunks]
    avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
    max_size = max(chunk_sizes) if chunk_sizes else 0
    logging.info(f"Chunk token distribution: avg={avg_size:.0f}, max={max_size}")
    
    # Final safety check - split any remaining large chunks
    if max_size > max_safe_tokens:
        logging.warning(f"Found chunks above safe token limit. Performing final split...")
        safe_chunks = []
        for chunk in chunks:
            chunk_tokens = estimate_tokens(chunk)
            if chunk_tokens > max_safe_tokens:
                # Split into smaller parts based on ratio
                ratio = max_safe_tokens / chunk_tokens
                chars_per_part = int(len(chunk) * ratio)
                parts = [chunk[i:i + chars_per_part] for i in range(0, len(chunk), chars_per_part)]
                safe_chunks.extend(parts)
            else:
                safe_chunks.append(chunk)
        chunks = safe_chunks
        
        # Log final sizes
        chunk_sizes = [estimate_tokens(chunk) for chunk in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
        max_size = max(chunk_sizes) if chunk_sizes else 0
        logging.info(f"After final split - chunks: {len(chunks)}, avg tokens: {avg_size:.0f}, max tokens: {max_size}")
    
    return chunks

def create_processing_batches(chunks, max_tokens_per_batch):
    """
    Creates batches of chunks that fit within the token limit with multiple safety checks.
    Uses ultra-conservative limits and smart splitting strategies.
    """
    batches = []
    current_batch = []
    current_tokens = 0
    
    # Ultra-conservative token reservations:
    # - 1000 tokens for prompt overhead and system message
    # - 4000 tokens for response generation
    # - 1500 tokens safety margin
    # Total reserved: 6500 tokens to be extra safe
    safe_token_limit = max_tokens_per_batch - 6500
    
    # Additional validation
    if safe_token_limit <= 0:
        logging.error(f"Token limit too small: {max_tokens_per_batch}. Need at least 6500 tokens.")
        safe_token_limit = 8000  # Fallback to reasonable minimum
    
    logging.info(f"Starting batch creation with conservative limits (max {safe_token_limit} tokens per batch)...")
    
    def split_text_on_boundaries(text, target_tokens):
        """Helper to split text on semantic boundaries with enhanced safety."""
        from config import MIN_TOKENS_RESERVE, SAFETY_MARGIN
        
        # Add extra safety margin to target
        safe_target = max(100, target_tokens - SAFETY_MARGIN)
        
        if estimate_tokens(text) <= safe_target:
            return [text]
            
        parts = []
        current_part = []
        current_tokens = 0
        
        # Enhanced splitting with language awareness
        # First try splitting on paragraph and sentence boundaries
        for split_char in ["\n\n\n", "\n\n", "\n", ". ", "? ", "! ", "; ", ", "]:
            if split_char in text:
                segments = text.split(split_char)
                for segment in segments:
                    segment = segment.strip()
                    if not segment:
                        continue
                    segment_tokens = estimate_tokens(segment)
                    
                    # If single segment is too large, will handle in next pass
                    if segment_tokens > safe_target:
                        continue
                        
                    if current_tokens + segment_tokens <= safe_target:
                        current_part.append(segment)
                        current_tokens += segment_tokens
                    else:
                        if current_part:
                            parts.append(split_char.join(current_part))
                        current_part = [segment]
                        current_tokens = segment_tokens
                
                if current_part:
                    parts.append(split_char.join(current_part))
                return parts
                
        # If no good splitting found, fall back to character-based splitting
        return [text[i:i + int(len(text) * target_tokens/estimate_tokens(text))] 
                for i in range(0, len(text), int(len(text) * target_tokens/estimate_tokens(text)))]
    
    for i, chunk in enumerate(chunks):
        chunk_tokens = estimate_tokens(chunk)
        
        # If a single chunk is too large
        if chunk_tokens > safe_token_limit:
            logging.warning(f"Chunk {i} ({chunk_tokens} tokens) exceeds safe limit ({safe_token_limit}). Smart splitting...")
            
            # Save any current batch
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            
            # Split the large chunk intelligently
            sub_chunks = split_text_on_boundaries(chunk, safe_token_limit)
            
            # Verify each sub-chunk
            for sub_chunk in sub_chunks:
                sub_tokens = estimate_tokens(sub_chunk)
                if sub_tokens > safe_token_limit:
                    # If still too large, split again more aggressively
                    smaller_chunks = split_text_on_boundaries(sub_chunk, safe_token_limit // 2)
                    for small_chunk in smaller_chunks:
                        if estimate_tokens(small_chunk) <= safe_token_limit:
                            batches.append([small_chunk])
                else:
                    batches.append([sub_chunk])
        else:
            # For normal-sized chunks
            if current_tokens + chunk_tokens > safe_token_limit:
                # Current batch would be too large, save it and start new
                if current_batch:
                    batches.append(current_batch)
                current_batch = [chunk]
                current_tokens = chunk_tokens
            else:
                # Add to current batch
                current_batch.append(chunk)
                current_tokens += chunk_tokens
    
    # Save final batch if any
    if current_batch:
        batches.append(current_batch)
    
    # Final validation pass
    validated_batches = []
    for batch in batches:
        batch_tokens = sum(estimate_tokens(chunk) for chunk in batch)
        if batch_tokens > safe_token_limit:
            # Split this batch if needed
            logging.warning(f"Final check: Found batch with {batch_tokens} tokens. Splitting...")
            text = " ".join(batch)
            sub_batches = split_text_on_boundaries(text, safe_token_limit)
            for sub_batch in sub_batches:
                validated_batches.append([sub_batch])
        else:
            validated_batches.append(batch)
    
    logging.info(f"Final batch count: {len(validated_batches)}")
    # Log token distribution
    batch_tokens = [sum(estimate_tokens(chunk) for chunk in batch) for batch in validated_batches]
    avg_tokens = sum(batch_tokens) / len(batch_tokens) if batch_tokens else 0
    max_tokens = max(batch_tokens) if batch_tokens else 0
    logging.info(f"Batch token distribution: avg={avg_tokens:.0f}, max={max_tokens}")
    
    return validated_batches
    
    logging.info(f"Total batches created: {len(batches)}")
    return batches

def detect_language_script(txt):
    """
    Detects the primary script/language of the text.
    Returns: tuple (script_name, is_english)
    """
    if not txt or not isinstance(txt, str):
        return "unknown", True
        
    # Count character ranges
    ascii_count = sum(1 for c in txt if ord(c) < 128)
    devanagari_count = sum(1 for c in txt if 0x0900 <= ord(c) <= 0x097F)  # Devanagari range
    
    total_chars = len(txt)
    if total_chars == 0:
        return "unknown", True
        
    # Calculate ratios
    ascii_ratio = ascii_count / total_chars
    devanagari_ratio = devanagari_count / total_chars
    
    # Determine primary script
    if devanagari_ratio > 0.3:  # If significant Devanagari presence
        return "devanagari", False
    elif ascii_ratio > 0.7:  # Mostly ASCII/English
        return "latin", True
    else:
        return "mixed", False

def is_english(txt):
    """Checks if a string is predominantly English."""
    script, is_eng = detect_language_script(txt)
    return is_eng

def translate_chunk(chunk, translator, max_length, retries):
    """Recursively translates a large chunk by splitting it into smaller parts."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=max_length * 4, chunk_overlap=0)
    sub_chunks = splitter.split_text(chunk)
    
    translated_sub_chunks = []
    for sub_chunk in sub_chunks:
        for retry_count in range(retries):
            try:
                translated_sub_chunks.append(translator(sub_chunk, max_length=max_length)[0]["translation_text"])
                break  # Success, move to the next sub-chunk
            except Exception as e:
                logging.warning(f"Translation failed on attempt {retry_count + 1}/{retries} for sub-chunk: {e}")
                time.sleep(1) # Wait before retrying
        else:
            logging.error(f"Max retries reached. Using original sub-chunk without translation: {sub_chunk[:50]}...")
            translated_sub_chunks.append(sub_chunk)
            
    return " ".join(translated_sub_chunks)

def translate(chunks, max_length, retries):
    """Translates non-English text chunks to English using an ML model."""
    try:
        tok = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
        model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
        trans = pipeline("translation", model=model, tokenizer=tok)
    except Exception as e:
        logging.error(f"Failed to load translation model: {e}")
        return chunks
    out = []
    for c in tqdm(chunks, desc="Translating"):
        if is_english(c):
            out.append(c)
        else:
            translated_chunk = translate_chunk(c, trans, max_length, retries)
            out.append(translated_chunk)
    return out

def parallel_map(func, inputs, max_workers=4):
    """
    Executes a function on a list of inputs in parallel using ThreadPoolExecutor.
    Returns results in the original order and logs errors.
    """
    results = [None] * len(inputs)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(func, i): idx for idx, i in enumerate(inputs)}
        
        for future in tqdm(as_completed(futures), total=len(inputs), desc="Processing in parallel"):
            original_idx = futures[future]
            try:
                results[original_idx] = future.result()
            except Exception as e:
                logging.error(f"Parallel task for input at index {original_idx} failed: {e}")
                results[original_idx] = None
    
    return results

def get_free_memory_mb():
    """Returns the amount of free memory in MB and includes swap."""
    import psutil
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    total_available = vm.available + swap.free
    return total_available / (1024 * 1024)

def check_memory_requirements(required_mb, buffer_mb=1000):
    """
    Checks if there's enough memory available for operation.
    Args:
        required_mb: Required memory in MB
        buffer_mb: Additional buffer to maintain in MB
    Returns:
        tuple: (bool, str) - (has_enough_memory, message)
    """
    available = get_free_memory_mb()
    total_needed = required_mb + buffer_mb
    
    if available >= total_needed:
        return True, f"Sufficient memory available: {available:.2f}MB >= {total_needed}MB needed"
    else:
        return False, f"Insufficient memory: {available:.2f}MB available, {total_needed}MB needed"


def process_json_to_batches(input_json_path, output_json_path, max_tokens_per_batch=11000):
    """
    Process a JSON input file into batches with token limits and save to a new JSON file.
    Each batch will contain content and its token count.
    
    Args:
        input_json_path (str): Path to input JSON file
        output_json_path (str): Path to save batches JSON file
        max_tokens_per_batch (int): Maximum tokens per batch (default 11000)
                                  Note: Using conservative limit to account for:
                                  - Prompt overhead (~500 tokens)
                                  - Response overhead (~3000 tokens)
                                  - Safety margin (~1500 tokens)
    """
    try:
        # Load input JSON
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract text content
        texts = extract_texts(data)
        if not texts:
            raise ValueError("No text content found in input JSON")
            
        # Create chunks
        chunks = chunk_texts(texts, max_tokens_per_batch * 4, overlap=50)  # *4 for char estimation
        logging.info(f"Created {len(chunks)} initial chunks")
        
        # Initialize batches list
        batches_data = {
            "metadata": {
                "created_at": _iso_now(),
                "input_file": input_json_path,
                "max_tokens_per_batch": max_tokens_per_batch
            },
            "batches": []
        }
        
        # Process chunks into batches
        current_batch = []
        current_tokens = 0
        batch_number = 1
        
        for chunk in chunks:
            chunk_tokens = estimate_tokens(chunk)
            
            # If adding this chunk would exceed limit, save current batch and start new one
            if current_tokens + chunk_tokens > max_tokens_per_batch:
                if current_batch:
                    batch_entry = {
                        "batch_no": batch_number,
                        "content": "\n\n".join(current_batch),
                        "token_count": current_tokens
                    }
                    batches_data["batches"].append(batch_entry)
                    batch_number += 1
                    
                current_batch = [chunk]
                current_tokens = chunk_tokens
            else:
                current_batch.append(chunk)
                current_tokens += chunk_tokens
        
        # Add final batch if it has content
        if current_batch:
            batch_entry = {
                "batch_no": batch_number,
                "content": "\n\n".join(current_batch),
                "token_count": current_tokens
            }
            batches_data["batches"].append(batch_entry)
        
        # Save batches to output JSON file
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(batches_data, f, indent=2, ensure_ascii=False)
            
        logging.info(f"Successfully created {len(batches_data['batches'])} batches")
        logging.info(f"Saved batches to {output_json_path}")
        
        # Clean up chunks (they are no longer needed)
        chunks.clear()
        
        return len(batches_data["batches"])
        
    except Exception as e:
        logging.error(f"Error processing JSON to batches: {e}")
        raise



def process_multiple_json_files(input_json_paths, output_json_path, max_tokens_per_batch=15500):
    """
    Process multiple JSON input files into a single batches file while maintaining token limits.
    
    Args:
        input_json_paths (list): List of paths to input JSON files
        output_json_path (str): Path to save combined batches JSON file
        max_tokens_per_batch (int): Maximum tokens per batch (default 15500)
        
    Returns:
        int: Total number of batches created
    """
    try:
        # Initialize combined batches data
        batches_data = {
            "metadata": {
                "created_at": _iso_now(),
                "input_files": input_json_paths,
                "max_tokens_per_batch": max_tokens_per_batch,
                "file_count": len(input_json_paths)
            },
            "batches": []
        }
        
        # Track global batch number
        global_batch_number = 1
        total_chunks = []
        
        # Process each input file
        for file_path in tqdm(input_json_paths, desc="Processing JSON files"):
            logging.info(f"Processing file: {file_path}")
            
            try:
                # Load and extract text from each file
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                texts = extract_texts(data)
                if texts:
                    # Create chunks for this file
                    file_chunks = chunk_texts(texts, max_tokens_per_batch * 4, overlap=50)
                    logging.info(f"Created {len(file_chunks)} chunks from {file_path}")
                    total_chunks.extend(file_chunks)
                else:
                    logging.warning(f"No text content found in {file_path}")
                    
            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")
                continue
        
        # Process all chunks into batches
        logging.info(f"Processing {len(total_chunks)} total chunks into batches")
        
        current_batch = []
        current_tokens = 0
        
        # Process all chunks into batches
        for chunk in total_chunks:
            chunk_tokens = estimate_tokens(chunk)
            
            # If adding this chunk would exceed limit, save current batch and start new one
            if current_tokens + chunk_tokens > max_tokens_per_batch:
                if current_batch:
                    batch_entry = {
                        "batch_no": global_batch_number,
                        "content": "\n\n".join(current_batch),
                        "token_count": current_tokens
                    }
                    batches_data["batches"].append(batch_entry)
                    global_batch_number += 1
                    
                current_batch = [chunk]
                current_tokens = chunk_tokens
            else:
                current_batch.append(chunk)
                current_tokens += chunk_tokens
        
        # Add final batch if it has content
        if current_batch:
            batch_entry = {
                "batch_no": global_batch_number,
                "content": "\n\n".join(current_batch),
                "token_count": current_tokens
            }
            batches_data["batches"].append(batch_entry)
        
        # Add summary statistics to metadata
        batches_data["metadata"].update({
            "total_batches": len(batches_data["batches"]),
            "average_tokens_per_batch": sum(b["token_count"] for b in batches_data["batches"]) / len(batches_data["batches"]) if batches_data["batches"] else 0,
            "total_tokens": sum(b["token_count"] for b in batches_data["batches"])
        })
        
        # Save combined batches to output JSON file
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(batches_data, f, indent=2, ensure_ascii=False)
            
        logging.info(f"Successfully created {len(batches_data['batches'])} total batches")
        logging.info(f"Saved combined batches to {output_json_path}")
        
        # Clean up
        total_chunks.clear()
        
        return len(batches_data["batches"])
        
    except Exception as e:
        logging.error(f"Error processing multiple JSON files: {e}")
        raise


if __name__ == "__main__":
    # Example usage for multiple files
    json_files = [
        "/home/anand/final_app/data/reddit_20251013_123358.json",
        "/home/anand/final_app/data/youtube_search_output.json",
        
    ]
    
    num_batches = process_multiple_json_files(
        input_json_paths=json_files,
        output_json_path="combined_batches.json",
        max_tokens_per_batch=15500
    )
    print(f"Created {num_batches} combined batches.")
