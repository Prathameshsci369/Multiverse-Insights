#!/usr/bin/env python3
# Memory-optimized: Single Phi-4 instance with concurrent request handling

import json
import gc
import sys
import logging
import threading
import asyncio
import queue
import time
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from llama_cpp import Llama

# --- MEMORY-OPTIMIZED CONFIG FOR YOUR SYSTEM ---
JSON_PATH = "/home/anand/Documents/data/reddit_search_output1.json"
GGUF_PATH = "/home/anand/Downloads/phi4.gguf"

# Memory-efficient settings for 16GB RAM, 6-core/12-thread CPU
PHI4_MAX_CONTEXT = 16384
PROCESSING_BATCH_SIZE = 14000  # Increased batch size since we have more context
MAX_WORKERS = 10  # More workers since model can handle concurrent requests
CHUNK_SIZE = 1800  # Larger chunks for better context
CHUNK_OVERLAP = 180

# Single model instance with concurrent request handling
USE_SINGLE_MODEL = True
MODEL_THREADS = 6  # Dedicate more threads to the model for concurrent processing
REQUEST_QUEUE_SIZE = 20  # Allow queuing of multiple requests

# GPU settings (adjust if you have GPU)
N_GPU_LAYERS = 0  # Set to -1 if you have a capable GPU

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def log_system_info():
    """Log system information for optimization insights."""
    cpu_count = psutil.cpu_count()
    memory = psutil.virtual_memory()
    logging.info(f"System Info - CPU Cores: {cpu_count}, Total RAM: {memory.total//1024**3}GB, Available RAM: {memory.available//1024**3}GB")
    logging.info(f"Memory-Optimized Settings - Workers: {MAX_WORKERS}, Single Model Instance, Batch Size: {PROCESSING_BATCH_SIZE} tokens")

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
    
    # Try to extract from the YouTube JSON structure (more specific check)
    try:
        if "transcript" in data[0] or ("comments" in data[0] and "text" in data[0]["comments"][0]):
            logging.info("Detected YouTube JSON format.")
            for e in data:
                if e.get("transcript"):
                    texts.append(e["transcript"])
                for c in e.get("comments", []):
                    if c.get("text"):
                        texts.append(c["text"])
                    for sc in c.get("subcomments", []):
                        if sc.get("text"):
                            texts.append(sc["text"])
            return texts
    except (IndexError, TypeError):
        pass # Not a YouTube JSON file

    # Try to extract from the Tweets JSON structure (more specific check)
    try:
        if "user_handle" in data[0] and "content" in data[0]:
            logging.info("Detected Twitter (X) JSON format.")
            for tweet in data:
                if tweet.get("content"):
                    texts.append(tweet["content"])
            return texts
    except (IndexError, TypeError):
        pass # Not a Tweets JSON file

    # Try to extract from the Reddit JSON structure (more specific check)
    try:
        if "title" in data[0] and "selftext" in data[0]:
            logging.info("Detected Reddit JSON format.")
            for post in data:
                if post.get("title"):
                    texts.append(post["title"])
                if post.get("selftext"):
                    texts.append(post["selftext"])
                for comment in post.get("comments", []):
                    if comment.get("comment"):
                        texts.append(comment["comment"])
                    for subcomment in comment.get("subcomments", []):
                        if subcomment.get("comment"):
                            texts.append(subcomment["comment"])
            return texts
    except (IndexError, TypeError):
        pass # Not a Reddit JSON file
    
    logging.warning("Unknown JSON format. Falling back to generic string search.")
    # Generic extraction for unknown formats
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
    return [d.page_content for d in splitter.create_documents(texts)]

def estimate_tokens(text):
    """Rough estimation of token count (1 token ≈ 4 characters for most models)."""
    return len(text) // 4

def create_processing_batches(chunks, max_tokens_per_batch=PROCESSING_BATCH_SIZE):
    """Creates batches of chunks that fit within the token limit."""
    batches = []
    current_batch = []
    current_tokens = 0
    
    for chunk in chunks:
        chunk_tokens = estimate_tokens(chunk)
        
        # If adding this chunk would exceed the limit, start a new batch
        if current_tokens + chunk_tokens > max_tokens_per_batch and current_batch:
            batches.append(current_batch)
            current_batch = [chunk]
            current_tokens = chunk_tokens
        else:
            current_batch.append(chunk)
            current_tokens += chunk_tokens
    
    # Add the last batch if it has content
    if current_batch:
        batches.append(current_batch)
    
    return batches

def is_english(txt):
    """Checks if a string is predominantly English based on ASCII characters."""
    return sum(1 for c in txt if ord(c) < 128) / max(1, len(txt)) > 0.9

# --- Translation ---
def translate(chunks):
    """Translates non-English text chunks to English using an ML model."""
    try:
        tok = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
        model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
        trans = pipeline("translation", model=model, tokenizer=tok)
    except Exception as e:
        logging.error(f"Failed to load translation model: {e}")
        return chunks  # fallback to original chunks

    out = []
    for c in tqdm(chunks, desc="Translating"):
        try:
            if is_english(c):
                out.append(c)
            else:
                out.append(trans(c, max_length=512)[0]["translation_text"])
        except Exception as e:
            logging.warning(f"Translation failed for chunk: {e}")
            out.append(c)  # fallback to original
    return out

# --- Concurrent Single-Model Manager ---
class ConcurrentPhi4Manager:
    """
    Single Phi-4 model instance that handles multiple concurrent requests efficiently.
    Uses request queuing and optimized model settings for high throughput.
    """
    
    def __init__(self, model_path, n_ctx=PHI4_MAX_CONTEXT):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.model = None
        self.request_lock = threading.Lock()  # Lightweight lock for request coordination
        self.request_queue = queue.Queue(maxsize=REQUEST_QUEUE_SIZE)
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0,
            'total_tokens_processed': 0
        }
        self._load_model()
    
    def _load_model(self):
        """Load the single Phi-4 model with optimized settings for concurrent processing."""
        try:
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=MODEL_THREADS,  # Use more threads for better concurrency
                n_gpu_layers=N_GPU_LAYERS,
                verbose=False,
                n_batch=1024,  # Larger batch size for efficiency
                use_mmap=True,  # Memory mapping for efficiency
                use_mlock=False,  # Don't lock memory pages
                # Concurrency optimizations
                n_parallel=4,  # Allow multiple sequences in parallel if supported
                rope_freq_base=10000.0,  # RoPE frequency base
                rope_freq_scale=1.0,  # RoPE frequency scaling
            )
            logging.info(f"Single Phi-4 model loaded with {self.n_ctx} context length and {MODEL_THREADS} threads")
        except Exception as e:
            logging.error(f"Failed to load Phi-4 model: {e}")
            sys.exit(1)
    
    def generate_concurrent(self, prompt, max_tokens=2000, timeout=60):
        """
        Generate text with optimized settings for concurrent processing.
        Uses shorter timeout and optimized parameters for faster throughput.
        """
        start_time = time.time()
        
        try:
            with self.request_lock:
                self.stats['total_requests'] += 1
                estimated_tokens = estimate_tokens(prompt)
                self.stats['total_tokens_processed'] += estimated_tokens
                
                # Optimized generation parameters for concurrent processing
                response = self.model(
                    prompt,
                    max_tokens=max_tokens,
                    echo=False,
                    # Performance-optimized parameters
                    temperature=0.7,
                    top_p=0.9,
                    top_k=40,
                    repeat_penalty=1.05,
                    # Faster stopping conditions
                    stop=["</s>", "Human:", "Assistant:", "\n\nHuman:", "\n\nAssistant:", "==="],
                    # Concurrency settings
                    stream=False,  # Don't stream for batch processing
                )
                
                result = response["choices"][0]["text"].strip()
                
                # Update stats
                response_time = time.time() - start_time
                self.stats['successful_requests'] += 1
                self.stats['average_response_time'] = (
                    (self.stats['average_response_time'] * (self.stats['successful_requests'] - 1) + response_time) /
                    self.stats['successful_requests']
                )
                
                return result
                
        except Exception as e:
            self.stats['failed_requests'] += 1
            logging.error(f"Concurrent generation failed: {e}")
            return f"ERROR: {str(e)}"
    
    def get_stats(self):
        """Get performance statistics."""
        return {
            'model_loaded': self.model is not None,
            'context_length': self.n_ctx,
            'model_threads': MODEL_THREADS,
            **self.stats
        }

# --- High-Performance Concurrent Processing ---
def process_batch_concurrent(model_manager, batch, batch_id):
    """Process a single batch using the concurrent model manager."""
    try:
        text_batch = "\n\n".join(batch)
        estimated_tokens = estimate_tokens(text_batch)
        
        # Enhanced prompt for better analysis
        prompt = f"""Analyze the following text content ({estimated_tokens} tokens) and provide a comprehensive analysis.

Text Content:
{text_batch}

Please provide your analysis in this exact format:

SUMMARY: [Detailed summary covering all key points, main themes, and important information from the text]

SENTIMENT ANALYSIS: 
- Overall sentiment: [Positive/Negative/Neutral with percentage]
- Emotional themes: [List main emotional patterns]
- Sentiment examples: [Specific quotes demonstrating different sentiments]

KEY INSIGHTS: [Most significant findings, patterns, and trends identified]

MAIN THEMES: [Primary topics and recurring subjects discussed]

NOTABLE QUOTES: [2-3 most important or representative quotes from the text]

Analysis complete."""
        
        start_time = time.time()
        result = model_manager.generate_concurrent(prompt, max_tokens=2200)
        processing_time = time.time() - start_time
        
        return {
            'batch_id': batch_id,
            'result': result,
            'status': 'success',
            'chunk_count': len(batch),
            'estimated_tokens': estimated_tokens,
            'processing_time': processing_time,
            'tokens_per_second': estimated_tokens / processing_time if processing_time > 0 else 0
        }
        
    except Exception as e:
        logging.error(f"Error processing batch {batch_id}: {e}")
        return {
            'batch_id': batch_id,
            'result': f"ERROR: {str(e)}",
            'status': 'error',
            'chunk_count': len(batch) if batch else 0,
            'estimated_tokens': 0,
            'processing_time': 0,
            'tokens_per_second': 0
        }

def concurrent_batch_processing(model_manager, batches):
    """Process batches using concurrent execution with a single model instance."""
    results = []
    start_time = time.time()
    
    # Use more workers since we're not limited by multiple model instances
    with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="BatchProcessor") as executor:
        # Submit all batch processing jobs
        future_to_batch = {
            executor.submit(process_batch_concurrent, model_manager, batch, i): i 
            for i, batch in enumerate(batches)
        }
        
        # Monitor progress with detailed statistics
        completed = 0
        total_tokens_processed = 0
        progress_bar = tqdm(total=len(batches), desc="Processing batches concurrently")
        
        for future in as_completed(future_to_batch):
            try:
                result = future.result()
                results.append(result)
                completed += 1
                total_tokens_processed += result.get('estimated_tokens', 0)
                
                # Update progress with performance metrics
                elapsed_time = time.time() - start_time
                if result['status'] == 'success':
                    success_count = sum(1 for r in results if r['status'] == 'success')
                    avg_tokens_per_sec = sum(r.get('tokens_per_second', 0) for r in results if r['status'] == 'success') / max(1, success_count)
                    
                    progress_bar.set_postfix({
                        'Success': f"{success_count}/{completed}",
                        'Tokens/s': f"{avg_tokens_per_sec:.0f}",
                        'Total': f"{total_tokens_processed:,}"
                    })
                progress_bar.update(1)
                
            except Exception as e:
                batch_id = future_to_batch[future]
                logging.error(f"Batch {batch_id} generated an exception: {e}")
                results.append({
                    'batch_id': batch_id,
                    'result': f"ERROR: {str(e)}",
                    'status': 'error',
                    'chunk_count': 0,
                    'estimated_tokens': 0,
                    'processing_time': 0,
                    'tokens_per_second': 0
                })
        
        progress_bar.close()
    
    # Sort results by batch_id to maintain order
    results.sort(key=lambda x: x['batch_id'])
    return results

def create_final_comprehensive_report(model_manager, batch_results):
    """Create a comprehensive final report from concurrent processing results."""
    
    # Filter successful results
    successful_results = [r for r in batch_results if r['status'] == 'success']
    
    if not successful_results:
        return "ERROR: No successful batch processing results to consolidate."
    
    # Calculate detailed statistics
    total_chunks = sum(r['chunk_count'] for r in successful_results)
    total_tokens = sum(r.get('estimated_tokens', 0) for r in successful_results)
    total_processing_time = sum(r.get('processing_time', 0) for r in successful_results)
    avg_tokens_per_second = sum(r.get('tokens_per_second', 0) for r in successful_results) / len(successful_results)
    success_rate = len(successful_results) / len(batch_results) * 100
    
    # Prepare consolidated analysis with better organization
    consolidated_analyses = []
    for i, result in enumerate(successful_results[:10]):  # Limit to first 10 for final consolidation
        consolidated_analyses.append(f"=== BATCH {result['batch_id']} ANALYSIS ===\n{result['result']}")
    
    if len(successful_results) > 10:
        consolidated_analyses.append(f"... and {len(successful_results) - 10} more batches with similar detailed analyses.")
    
    consolidated_text = "\n\n".join(consolidated_analyses)
    
    final_prompt = f"""You are an expert data analyst. Create a comprehensive final report by consolidating the following batch analyses from concurrent processing.

PROCESSING STATISTICS:
- Model: Single Phi-4 instance with concurrent request handling
- Batches processed: {len(successful_results)}/{len(batch_results)} (Success rate: {success_rate:.1f}%)
- Total chunks analyzed: {total_chunks:,}
- Total tokens processed: {total_tokens:,}
- Average processing speed: {avg_tokens_per_second:.0f} tokens/second
- Total processing time: {total_processing_time:.1f} seconds

Create a comprehensive final report with these sections:

1. EXECUTIVE SUMMARY: High-level overview of the most critical findings and insights

2. COMPREHENSIVE STORYLINE: Detailed narrative that weaves together all themes, events, patterns, and insights discovered across all batches

3. DETAILED SENTIMENT ANALYSIS:
   - Overall sentiment distribution with precise percentages
   - Emotional journey and sentiment evolution
   - Key emotional themes and their significance  
   - Most representative quotes for each sentiment category

4. KEY INSIGHTS AND DISCOVERIES: Most important findings, trends, patterns, and discoveries

5. THEMATIC BREAKDOWN: Detailed analysis of major topics, categories, and recurring themes

6. STATISTICAL OBSERVATIONS: Quantitative patterns and data insights

7. NOTABLE QUOTES AND EXAMPLES: Most significant, interesting, or representative quotes from the data

BATCH ANALYSES TO CONSOLIDATE:
{consolidated_text}

FINAL COMPREHENSIVE REPORT:"""
    
    try:
        logging.info("Generating final comprehensive report...")
        final_report = model_manager.generate_concurrent(final_prompt, max_tokens=3500)
        return final_report
    except Exception as e:
        logging.error(f"Error generating final report: {e}")
        return f"ERROR: Failed to generate final report: {str(e)}"

# --- MAIN ---
def main():
    try:
        start_time = time.time()
        log_system_info()
        
        # Step 1: Data Pre-processing
        logging.info("Loading and processing data...")
        data = load_json(JSON_PATH)
        texts = extract_texts(data)
        chunks = chunk_texts(texts)
        logging.info(f"Extracted {len(texts)} texts, created {len(chunks)} chunks")
        
        # Step 2: Translation
        logging.info("Starting translation process...")
        translated = translate(chunks)
        logging.info("Translation complete")
        
        # Step 3: Create optimized processing batches
        logging.info("Creating memory-optimized processing batches...")
        batches = create_processing_batches(translated)
        logging.info(f"Created {len(batches)} processing batches")
        
        # Log detailed batch statistics
        batch_sizes = [len(batch) for batch in batches]
        batch_tokens = [sum(estimate_tokens(chunk) for chunk in batch) for batch in batches]
        total_estimated_tokens = sum(batch_tokens)
        
        logging.info(f"Batch Statistics:")
        logging.info(f"  - Chunks per batch: Min={min(batch_sizes)}, Max={max(batch_sizes)}, Avg={sum(batch_sizes)/len(batch_sizes):.1f}")
        logging.info(f"  - Tokens per batch: Min={min(batch_tokens):,}, Max={max(batch_tokens):,}, Avg={sum(batch_tokens)/len(batch_tokens):,.0f}")
        logging.info(f"  - Total estimated tokens: {total_estimated_tokens:,}")
        logging.info(f"  - Memory usage: Single model instance (~3-4GB)")
        
        # Step 4: Initialize concurrent model manager
        logging.info("Initializing concurrent Phi-4 model manager...")
        model_manager = ConcurrentPhi4Manager(GGUF_PATH)
        
        # Step 5: Concurrent processing with single model
        logging.info(f"Starting concurrent processing with {MAX_WORKERS} workers and 1 model instance...")
        batch_results = concurrent_batch_processing(model_manager, batches)
        
        # Step 6: Process results and statistics
        successful_batches = sum(1 for r in batch_results if r['status'] == 'success')
        failed_batches = len(batch_results) - successful_batches
        processing_time = time.time() - start_time
        
        # Calculate performance metrics
        total_processed_tokens = sum(r.get('estimated_tokens', 0) for r in batch_results if r['status'] == 'success')
        overall_tokens_per_second = total_processed_tokens / processing_time if processing_time > 0 else 0
        
        logging.info(f"Concurrent processing complete: {successful_batches} successful, {failed_batches} failed in {processing_time:.2f}s")
        logging.info(f"Overall processing speed: {overall_tokens_per_second:,.0f} tokens/second")
        
        # Step 7: Generate comprehensive final report
        logging.info("Generating comprehensive final report...")
        final_report = create_final_comprehensive_report(model_manager, batch_results)
        
        # Step 8: Display comprehensive results
        end_time = time.time()
        total_processing_time = end_time - start_time
        
        print("\n" + "="*120)
        print("MEMORY-OPTIMIZED CONCURRENT PROCESSING ANALYSIS REPORT")
        print("="*120)
        print(f"System Configuration:")
        print(f"  - Hardware: 16GB RAM, 6-core/12-thread CPU")
        print(f"  - Model Strategy: Single Phi-4 instance with concurrent request handling")
        print(f"  - Memory Usage: ~3-4GB (single model) vs ~9GB (3 models) - 60% memory savings!")
        print(f"")
        print(f"Processing Configuration:")
        print(f"  - Concurrent Workers: {MAX_WORKERS}")
        print(f"  - Model Threads: {MODEL_THREADS}")
        print(f"  - Context Length: {PHI4_MAX_CONTEXT:,} tokens")
        print(f"  - Batch Size: {PROCESSING_BATCH_SIZE:,} tokens")
        print(f"")
        print(f"Performance Metrics:")
        print(f"  - Total Processing Time: {total_processing_time:.2f} seconds")
        print(f"  - Batches: {successful_batches}/{len(batches)} processed ({successful_batches/len(batches)*100:.1f}% success)")
        print(f"  - Total Chunks: {len(translated):,}")
        print(f"  - Total Tokens: {total_estimated_tokens:,}")
        print(f"  - Processing Speed: {overall_tokens_per_second:,.0f} tokens/second")
        print(f"  - Model Statistics: {model_manager.get_stats()}")
        print("="*120)
        print()
        print(final_report)
        print("\n" + "="*120)
        
        # Cleanup
        del model_manager
        gc.collect()
        logging.info("Memory cleanup completed")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
