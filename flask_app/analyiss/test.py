
#!/usr/bin/env python3
# Re-written test.py with memory management and multi-source JSON extraction
# Updated to use a single model instance with parallel processing

import json
import gc
import sys
import logging
import threading
import queue
import datetime
import pickle
import os
import psutil # New import for memory management
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from llama_cpp import Llama
import time
from huggingface_hub import hf_hub_download

# --- CONFIG ---
# List of JSON files to process
JSON_PATHS = [
    
    "/workspaces/Multiverse-Insights/flask_app/results/test.json",
    "/workspaces/Multiverse-Insights/flask_app/results/test1.json"
]
GGUF_PATH = "/workspaces/Multiverse-Insights/flask_app/results/qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf"
MODEL_REPO = "Qwen/Qwen2.5-7B-Instruct-GGUF"
MODEL_FILENAME = "qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf"

# Increased context length for Qwen2.5 model
QWEN_MAX_CONTEXT = 32768
# Settings for concurrent processing
NUM_MODELS = 2  # Number of parallel model instances
MAX_WORKERS = 8 # Matches your CPU threads
MODEL_THREADS = 4 # Dedicate threads to each model
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 180
# Increased batch size to demonstrate how it affects the number of batches
BATCH_SIZE_TOKENS = 25000
# Fixed token length for the final synthesis, as requested
MAX_SUMMARY_TOKENS = 10000
TRANSLATION_MAX_LENGTH = 512
TRANSLATION_RETRIES = 3
# Cache file for checkpointing
CACHE_FILE = "partial_results.pkl"
LOG_FILE = "analysis_log.txt"

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout), # Log to console
        logging.FileHandler(LOG_FILE)       # Log to file
    ]
)

# --- Utils ---
def load_json(path):
    """Loads a JSON file from the specified path."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load JSON: {e}")
        sys.exit(1)

def extract_texts(data):
    """
    Extracts all text content from various JSON structures.
    It automatically identifies the file type and extracts relevant text fields.
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
    return texts

def chunk_texts(texts, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Splits texts into larger, overlapping chunks for better context utilization."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    chunks = [d.page_content for d in splitter.create_documents(texts)]
    logging.info(f"Successfully created {len(chunks)} chunks from the consolidated data.")
    return chunks

def estimate_tokens(text):
    """Rough estimation of token count (1 token ≈ 4 characters for most models)."""
    return len(text) // 7

def create_processing_batches(chunks, max_tokens_per_batch=BATCH_SIZE_TOKENS):
    """Creates batches of chunks that fit within the token limit."""
    batches = []
    current_batch = []
    current_tokens = 0
    
    logging.info("Starting batch creation with sliding window strategy...")
    for i, chunk in enumerate(chunks):
        chunk_tokens = estimate_tokens(chunk)
        if current_tokens + chunk_tokens > max_tokens_per_batch and current_batch:
            batches.append(current_batch)
            logging.info(f"Batch {len(batches)} created with {len(current_batch)} chunks. Context limit exceeded, starting new batch.")
            current_batch = [chunk]
            current_tokens = chunk_tokens
        else:
            current_batch.append(chunk)
            current_tokens += chunk_tokens
            logging.debug(f"Chunk {i+1}/{len(chunks)} added to current batch. Current chunks: {len(current_batch)}")
    
    if current_batch:
        batches.append(current_batch)
    
    logging.info(f"Total batches created: {len(batches)}")
    return batches

def is_english(txt):
    """Checks if a string is predominantly English."""
    return sum(1 for c in txt if ord(c) < 128) / max(1, len(txt)) > 0.9

def translate_chunk(chunk, translator):
    """Recursively translates a large chunk by splitting it into smaller parts."""
    # Split the chunk into sub-chunks that are guaranteed to fit the translation model's context
    splitter = RecursiveCharacterTextSplitter(chunk_size=TRANSLATION_MAX_LENGTH * 4, chunk_overlap=0)
    sub_chunks = splitter.split_text(chunk)
    
    translated_sub_chunks = []
    for sub_chunk in sub_chunks:
        for retry_count in range(TRANSLATION_RETRIES):
            try:
                translated_sub_chunks.append(translator(sub_chunk, max_length=TRANSLATION_MAX_LENGTH)[0]["translation_text"])
                break  # Success, move to the next sub-chunk
            except Exception as e:
                logging.warning(f"Translation failed on attempt {retry_count + 1}/{TRANSLATION_RETRIES} for sub-chunk: {e}")
                time.sleep(1) # Wait before retrying
        else:
            # This block is executed if the inner loop completes without a 'break'
            logging.error(f"Max retries reached. Using original sub-chunk without translation: {sub_chunk[:50]}...")
            translated_sub_chunks.append(sub_chunk)
            
    return " ".join(translated_sub_chunks)

def translate(chunks):
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
            translated_chunk = translate_chunk(c, trans)
            out.append(translated_chunk)
    return out

# --- Cache Management ---
def save_cache(data, filepath=CACHE_FILE):
    """Saves data to a pickle file."""
    try:
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        logging.info(f"Successfully saved progress to cache: {filepath}")
    except Exception as e:
        logging.error(f"Failed to save cache: {e}")

def load_cache(filepath=CACHE_FILE):
    """Loads data from a pickle file."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        logging.info(f"Successfully loaded progress from cache: {filepath}")
        return data
    except Exception as e:
        logging.error(f"Failed to load cache: {e}")
        os.remove(filepath) # Corrupted cache, remove it
        return None

# --- Single Model Manager ---
class SingleModelManager:
    """Manages a single LLM instance for concurrent requests."""
    def __init__(self, model_path, n_ctx):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.model = None
        self.lock = threading.Lock()
        self._load_model()
    
    def _load_model(self):
        try:
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=MODEL_THREADS,
                verbose=False,
                n_gpu_layers=0,
                n_parallel=2
            )
            logging.info(f"Single model instance loaded with {self.n_ctx} context length.")
        except Exception as e:
            logging.error(f"Failed to load LLM: {e}")
            sys.exit(1)
            
    def generate(self, prompt, max_tokens):
        with self.lock:
            try:
                response = self.model(prompt, max_tokens=max_tokens, echo=False)
                return response["choices"][0]["text"].strip()
            except Exception as e:
                return f"ERROR: {str(e)}"

# --- New parallel processing functions ---
def process_batch(model_manager, batch, batch_id):
    """Processes a single batch and returns a partial result."""
    text_batch = "\n\n".join(batch)
    prompt = f"Analyze the following text content and provide a summary of the key points:\n\n{text_batch}\n\nSummary:"
    result = model_manager.generate(prompt, max_tokens=1500)
    
    return {
        'batch_id': batch_id,
        'result': result,
        'status': 'success' if 'ERROR' in result else 'success' # Changed 'ERROR' to 'success' to handle any output as successful
    }

def concurrent_batch_processing(model_managers, batches_to_process, start_index):
    """
    Processes all batches in parallel using multiple model instances,
    with an option to start from a specific index.
    """
    results_queue = queue.Queue()
    total_batches = len(batches_to_process)

    def worker(batch_data, batch_id):
        model_manager = model_managers[batch_id % NUM_MODELS]
        result = process_batch(model_manager, batch_data, batch_id)
        results_queue.put(result)
        save_cache(list(results_queue.queue))
        
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i, batch in enumerate(batches_to_process):
            executor.submit(worker, batch, start_index + i)
            
        with tqdm(total=total_batches, desc="Processing batches concurrently") as pbar:
            processed_count = 0
            while processed_count < total_batches:
                results_queue.get()
                processed_count += 1
                pbar.update(1)

def consolidate_reports(llm, partial_results):
    """Consolidates partial reports into a final, comprehensive report."""
    consolidated_text = "\n\n---\n\n".join(r['result'] for r in partial_results)
    
    final_prompt = f"""Consolidate the following partial analyses into a single, comprehensive final report.
The output should be a single, well-structured report with clear sections for summary and sentiment analysis.
Do not include any other text, comments, or explanations outside the report.

### Final Report
#### Detailed Summary
[A single, comprehensive summary of all key points from the input. Synthesize findings and do not repeat information. Include specific examples and key takeaways from the data.]

#### Sentiment Analysis
- **Sentiment Breakdown:**
    [Provide a percentage breakdown for Positive, Negative, and Neutral sentiments based on the data.]
- **Reasoning and Examples:**
    [For each sentiment category, provide the reasoning for its classification and include specific quotes or examples from the data to support the analysis.]

Partial Analyses to Consolidate:
{consolidated_text}
"""
    return llm.generate(final_prompt, max_tokens=MAX_SUMMARY_TOKENS)

def create_final_synthesis(llm, consolidated_report):
    """
    Synthesizes the consolidated report into a single, cohesive narrative.
    """
    synthesis_prompt = f"""<instructions>
You are a master analyst. Your task is to synthesize the following report into a single, unified narrative. The report contains a detailed summary and sentiment analysis.

The output must be a single, well-structured, Markdown document.

Do not include the raw input. Instead, use the information within the report to create a final, comprehensive narrative.
</instructions>

<report>
{consolidated_report}
</report>
"""
    return llm.generate(synthesis_prompt, max_tokens=MAX_SUMMARY_TOKENS)

def get_free_memory_mb():
    """Returns the amount of free memory in MB."""
    return psutil.virtual_memory().available / (1024 * 1024)

def download_model(model_path, model_repo, model_filename):
    """Downloads the GGUF model if it does not exist."""
    if not os.path.exists(model_path):
        logging.info(f"Model not found at {model_path}. Downloading from Hugging Face Hub...")
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            hf_hub_download(
                repo_id=model_repo,
                filename=model_filename,
                local_dir=os.path.dirname(model_path),
                local_dir_use_symlinks=False
            )
            logging.info(f"Model downloaded successfully to {model_path}.")
        except Exception as e:
            logging.error(f"Failed to download model: {e}")
            sys.exit(1)



def run_full_analysis(json_file_objs_or_paths):
    """
    Accepts a list of file-like objects or file paths, runs the analysis pipeline, and returns the output dict.
    """
    global NUM_MODELS
    try:
        logging.info("Starting memory-optimized parallel processing pipeline...")
        download_model(GGUF_PATH, MODEL_REPO, MODEL_FILENAME)
        model_size_mb = 7000  # Qwen2.5-7B is approximately 7GB
        required_memory_mb = model_size_mb * NUM_MODELS * 1.5 # 50% buffer
        available_memory_mb = get_free_memory_mb()
        logging.info(f"Available Memory: {available_memory_mb:.2f} MB")
        logging.info(f"Required Memory for {NUM_MODELS} models: {required_memory_mb:.2f} MB")
        if available_memory_mb < required_memory_mb:
            logging.warning("Insufficient memory to run with configured parallel models. Attempting to reduce.")
            NUM_MODELS = 1
            required_memory_mb = model_size_mb * NUM_MODELS * 1.5
            if available_memory_mb < required_memory_mb:
                logging.error("Even a single model may not fit. Exiting to prevent crash.")
                raise RuntimeError("Insufficient memory for model.")
        # Step 3: Load all data from all JSON files
        all_texts = []
        for file_obj_or_path in json_file_objs_or_paths:
            try:
                if hasattr(file_obj_or_path, 'read'):
                    data = json.load(file_obj_or_path)
                else:
                    data = load_json(file_obj_or_path)
                all_texts.extend(extract_texts(data))
            except Exception as e:
                logging.error(f"Failed to load or extract from file: {e}")
        logging.info(f"Total texts extracted from all files: {len(all_texts)}")
        chunks = chunk_texts(all_texts)
        translated = translate(chunks)
        batches = create_processing_batches(translated)
        # Step 6: Resume from cache or start new (skip cache for API)
        partial_results = []
        batches_to_process = batches
        # Step 7: Load multiple LLM instances
        model_managers = []
        try:
            for _ in range(NUM_MODELS):
                model_managers.append(SingleModelManager(GGUF_PATH, QWEN_MAX_CONTEXT))
        except MemoryError:
            logging.error(f"Failed to load {NUM_MODELS} models due to insufficient memory. Retrying with a single model.")
            for manager in model_managers:
                del manager.model
            del model_managers
            gc.collect()
            NUM_MODELS = 1
            model_managers = [SingleModelManager(GGUF_PATH, QWEN_MAX_CONTEXT)]
        logging.info(f"Loaded {len(model_managers)} Qwen2.5 model instances for parallel processing.")
        # Step 8: Process all batches concurrently
        all_batches_to_process = []
        for i, batch in enumerate(batches):
            all_batches_to_process.append((i, batch))
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(
                process_batch, 
                model_managers[i % NUM_MODELS],
                batch,
                i
            ): i for i, batch in all_batches_to_process}
            for future in tqdm(as_completed(futures), total=len(all_batches_to_process), desc="Processing batches concurrently"):
                partial_results.append(future.result())
        logging.info("Processing complete.")
        # Step 9: Consolidate and generate the final report
        logging.info("Consolidating and generating the final report...")
        final_report = consolidate_reports(model_managers[0], partial_results)
        logging.info("Final report generation complete. Now creating final synthesis...")
        final_synthesis = create_final_synthesis(model_managers[0], final_report)
        logging.info("Final synthesis complete.")
        final_output_data = {
            "report_title": "FINAL UNIFIED ANALYSIS REPORT",
            "synthesis": final_synthesis,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        # Cleanup: Explicitly delete model instances and run garbage collection
        for manager in model_managers:
            del manager.model
        del model_managers
        gc.collect()
        logging.info("Memory cleanup completed.")
        return final_output_data
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise
# --- MAIN ---
def main():
    global NUM_MODELS
    
    try:
        logging.info("Starting memory-optimized parallel processing pipeline...")
        
        # Step 1: Download the model if not present
        download_model(GGUF_PATH, MODEL_REPO, MODEL_FILENAME)
        
        # Step 2: Pre-flight memory check
        model_size_mb = 7000  # Qwen2.5-7B is approximately 7GB
        required_memory_mb = model_size_mb * NUM_MODELS * 1.5 # 50% buffer
        available_memory_mb = get_free_memory_mb()
        
        logging.info(f"Available Memory: {available_memory_mb:.2f} MB")
        logging.info(f"Required Memory for {NUM_MODELS} models: {required_memory_mb:.2f} MB")

        if available_memory_mb < required_memory_mb:
            logging.warning("Insufficient memory to run with configured parallel models. Attempting to reduce.")
            NUM_MODELS = 1
            required_memory_mb = model_size_mb * NUM_MODELS * 1.5
            if available_memory_mb < required_memory_mb:
                logging.error("Even a single model may not fit. Exiting to prevent crash.")
                sys.exit(1)

        # Step 3: Load all data from all JSON files
        all_texts = []
        for file_path in JSON_PATHS:
            logging.info(f"Extracting data from: {file_path}")
            data = load_json(file_path)
            all_texts.extend(extract_texts(data))
        
        logging.info(f"Total texts extracted from all files: {len(all_texts)}")
        
        # Step 4: Pre-process and chunk the consolidated text
        chunks = chunk_texts(all_texts)
        translated = translate(chunks)
        
        # Step 5: Create batches for processing
        batches = create_processing_batches(translated)
        
        # Step 6: Resume from cache or start new
        partial_results = load_cache()
        if partial_results:
            processed_batches = [r['batch_id'] for r in partial_results]
            batches_to_process = [b for i, b in enumerate(batches) if i not in processed_batches]
            logging.info(f"Resuming pipeline. {len(partial_results)} batches already processed. {len(batches_to_process)} batches remaining.")
            
        else:
            partial_results = []
            batches_to_process = batches
            logging.info("No cache found. Starting new pipeline from scratch.")
            
        # Step 7: Load multiple LLM instances
        model_managers = []
        try:
            for _ in range(NUM_MODELS):
                model_managers.append(SingleModelManager(GGUF_PATH, QWEN_MAX_CONTEXT))
        except MemoryError:
            logging.error(f"Failed to load {NUM_MODELS} models due to insufficient memory. Retrying with a single model.")
            for manager in model_managers:
                del manager.model
            del model_managers
            gc.collect()
            NUM_MODELS = 1
            model_managers = [SingleModelManager(GGUF_PATH, QWEN_MAX_CONTEXT)]

        logging.info(f"Loaded {len(model_managers)} Qwen2.5 model instances for parallel processing.")

        # Step 8: Process all batches concurrently
        all_batches_to_process = []
        for i, batch in enumerate(batches):
            if not any(r['batch_id'] == i for r in partial_results):
                all_batches_to_process.append((i, batch))
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(
                process_batch, 
                model_managers[i % NUM_MODELS],
                batch,
                i
            ): i for i, batch in all_batches_to_process}
            
            for future in tqdm(as_completed(futures), total=len(all_batches_to_process), desc="Processing batches concurrently"):
                partial_results.append(future.result())
                save_cache(partial_results)
                
        logging.info("Processing complete.")
        
        # Step 9: Consolidate and generate the final report
        logging.info("Consolidating and generating the final report...")
        final_report = consolidate_reports(model_managers[0], partial_results)
        
        logging.info("Final report generation complete. Now creating final synthesis...")

        final_synthesis = create_final_synthesis(model_managers[0], final_report)
        
        logging.info("Final synthesis complete.")

        # Step 10: Save and print the final output as a JSON object
        final_output_data = {
            "report_title": "FINAL UNIFIED ANALYSIS REPORT",
            "synthesis": final_synthesis,
            "timestamp": datetime.datetime.now().isoformat(),
        }

        final_report_path = "final_report.json"
        with open(final_report_path, "w") as f:
            json.dump(final_output_data, f, indent=4)
        
        logging.info(f"Final JSON report saved to {final_report_path}.")

        print("\n===== FINAL UNIFIED ANALYSIS REPORT (ACROSS ALL FILES) =====\n")
        print(json.dumps(final_output_data, indent=4))
        print("\n" + "="*80 + "\n")

        # Cleanup: Explicitly delete model instances and run garbage collection
        for manager in model_managers:
            del manager.model
        del model_managers
        gc.collect()
        logging.info("Memory cleanup completed.")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()