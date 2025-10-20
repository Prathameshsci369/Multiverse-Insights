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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from config import PHI4_MAX_CONTEXT


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
    Splits texts into larger, overlapping chunks for better context utilization.
    This improved version ensures chunks don't exceed the specified size.
    """
    # Calculate a safe chunk size that accounts for token estimation
    # We'll use a conservative estimate of 4 characters per token
    safe_chunk_size = min(size, (PHI4_MAX_CONTEXT - 3000) * 4 // 4)  # Convert tokens to chars
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=safe_chunk_size, 
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    for text in texts:
        text_chunks = splitter.split_text(text)
        chunks.extend(text_chunks)
    
    logging.info(f"Successfully created {len(chunks)} chunks from the consolidated data.")
    
    # Log the size distribution of chunks
    chunk_sizes = [estimate_tokens(chunk) for chunk in chunks]
    avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
    max_size = max(chunk_sizes) if chunk_sizes else 0
    logging.info(f"Chunk token distribution: avg={avg_size:.0f}, max={max_size}")
    
    return chunks

def create_processing_batches(chunks, max_tokens_per_batch):
    """
    Creates batches of chunks that fit within the token limit.
    This improved version handles chunks that exceed the limit by splitting them.
    """
    batches = []
    current_batch = []
    current_tokens = 0
    
    # Reserve tokens for the prompt and response
    safe_token_limit = max_tokens_per_batch - 3000  # Reserve space for prompt and response
    
    logging.info(f"Starting batch creation with sliding window strategy...")
    
    for i, chunk in enumerate(chunks):
        chunk_tokens = estimate_tokens(chunk)
        
        # If a single chunk exceeds the safe limit, we need to split it
        if chunk_tokens > safe_token_limit:
            logging.warning(f"Chunk {i} ({chunk_tokens} tokens) exceeds safe limit ({safe_token_limit}). Splitting chunk.")
            
            # Save current batch if it has content
            if current_batch:
                batches.append(current_batch)
                logging.info(f"Batch {len(batches)} created with {len(current_batch)} chunks.")
                current_batch = []
                current_tokens = 0
            
            # Split the oversized chunk into smaller pieces
            words = chunk.split()
            sub_chunk = []
            sub_chunk_tokens = 0
            
            for word in words:
                word_tokens = estimate_tokens(word)
                if sub_chunk_tokens + word_tokens <= safe_token_limit:
                    sub_chunk.append(word)
                    sub_chunk_tokens += word_tokens
                else:
                    if sub_chunk:
                        batches.append([" ".join(sub_chunk)])
                        logging.info(f"Batch {len(batches)} created with 1 chunk (split from oversized chunk).")
                    sub_chunk = [word]
                    sub_chunk_tokens = word_tokens
            
            # Add the last sub-chunk if it has content
            if sub_chunk:
                batches.append([" ".join(sub_chunk)])
                logging.info(f"Batch {len(batches)} created with 1 chunk (split from oversized chunk).")
        else:
            # Check if adding this chunk would exceed the limit
            if current_tokens + chunk_tokens > safe_token_limit:
                # Save current batch and start a new one
                if current_batch:
                    batches.append(current_batch)
                    logging.info(f"Batch {len(batches)} created with {len(current_batch)} chunks.")
                current_batch = [chunk]
                current_tokens = chunk_tokens
            else:
                # Add chunk to current batch
                current_batch.append(chunk)
                current_tokens += chunk_tokens
    
    # Add the last batch if it has content
    if current_batch:
        batches.append(current_batch)
        logging.info(f"Batch {len(batches)} created with {len(current_batch)} chunks.")
    
    logging.info(f"Total batches created: {len(batches)}")
    return batches

def is_english(txt):
    """Checks if a string is predominantly English."""
    return sum(1 for c in txt if ord(c) < 128) / max(1, len(txt)) > 0.9

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
    """Returns the amount of free memory in MB."""
    import psutil
    return psutil.virtual_memory().available / (1024 * 1024)


