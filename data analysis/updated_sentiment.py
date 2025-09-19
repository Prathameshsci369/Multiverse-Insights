#!/usr/bin/env python3
# Optimized test.py for high-end system: 16GB RAM, 6-core/12-thread CPU, 11GB available

import json
import gc
import sys
import logging
import threading
import queue
import time
import psutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from llama_cpp import Llama

# --- OPTIMIZED CONFIG FOR YOUR SYSTEM ---
JSON_PATH = "/home/anand/Documents/data/reddit_search_output1.json"
GGUF_PATH = "/home/anand/Downloads/phi4.gguf"

# System-optimized settings for 16GB RAM, 6-core/12-thread CPU
PHI4_MAX_CONTEXT = 16384
PROCESSING_BATCH_SIZE = 12288  # Increased to 12K tokens per batch
MAX_WORKERS = 8  # Utilizing more of your 12 threads
CHUNK_SIZE = 1500  # Larger chunks for better context
CHUNK_OVERLAP = 150

# Model loading strategy
USE_MULTIPLE_MODEL_INSTANCES = True  # Load multiple model instances for true parallelism
MODEL_INSTANCES = 3  # Load 3 model instances (each ~3GB RAM = 9GB total, leaving 2GB buffer)

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
    logging.info(f"Optimization Settings - Workers: {MAX_WORKERS}, Model Instances: {MODEL_INSTANCES}, Batch Size: {PROCESSING_BATCH_SIZE} tokens")

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

# --- Advanced Model Management ---
class Phi4ModelPool:
    """Pool of Phi-4 model instances for true parallel processing."""
    
    def __init__(self, model_path, pool_size=MODEL_INSTANCES, n_ctx=PHI4_MAX_CONTEXT):
        self.model_path = model_path
        self.pool_size = pool_size
        self.n_ctx = n_ctx
        self.models = queue.Queue()
        self.model_stats = {'total_requests': 0, 'active_models': 0}
        self._load_models()
    
    def _load_models(self):
        """Load multiple model instances into the pool."""
        logging.info(f"Loading {self.pool_size} Phi-4 model instances...")
        
        for i in range(self.pool_size):
            try:
                model = Llama(
                    model_path=self.model_path,
                    n_ctx=self.n_ctx,
                    n_threads=2,  # 2 threads per model instance
                    n_gpu_layers=N_GPU_LAYERS,
                    verbose=False,
                    n_batch=512,
                    use_mmap=True,  # Use memory mapping for efficiency
                    use_mlock=False  # Don't lock memory pages
                )
                self.models.put(model)
                self.model_stats['active_models'] += 1
                logging.info(f"Model instance {i+1}/{self.pool_size} loaded successfully")
                
            except Exception as e:
                logging.error(f"Failed to load model instance {i+1}: {e}")
                # Continue with fewer models if some fail to load
        
        if self.model_stats['active_models'] == 0:
            logging.error("No model instances could be loaded!")
            sys.exit(1)
        
        logging.info(f"Model pool ready with {self.model_stats['active_models']} instances")
    
    def get_model(self, timeout=30):
        """Get a model from the pool (blocks if all models are busy)."""
        try:
            model = self.models.get(timeout=timeout)
            self.model_stats['total_requests'] += 1
            return model
        except queue.Empty:
            raise TimeoutError("No model available within timeout period")
    
    def return_model(self, model):
        """Return a model to the pool."""
        self.models.put(model)
    
    def generate(self, prompt, max_tokens=2000, timeout=30):
        """Generate text using any available model from the pool."""
        model = self.get_model(timeout)
        try:
            response = model(
                prompt,
                max_tokens=max_tokens,
                echo=False,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.1,
                stop=["</s>", "Human:", "Assistant:", "\n\nHuman:", "\n\nAssistant:"]
            )
            return response["choices"][0]["text"].strip()
        except Exception as e:
            logging.error(f"LLM generation failed: {e}")
            return "ERROR: Generation failed"
        finally:
            self.return_model(model)
    
    def get_stats(self):
        """Get pool statistics."""
        return {
            'pool_size': self.pool_size,
            'active_models': self.model_stats['active_models'],
            'total_requests': self.model_stats['total_requests'],
            'available_models': self.models.qsize()
        }

# --- High-Performance Processing Functions ---
def process_single_batch(model_pool, batch, batch_id):
    """Process a single batch using the model pool."""
    try:
        text_batch = "\n\n".join(batch)
        estimated_tokens = estimate_tokens(text_batch)
        
        prompt = f"""Analyze the following text content ({estimated_tokens} tokens) and provide a detailed analysis.

Text Content:
{text_batch}

Provide your analysis in the following format:

SUMMARY: [Comprehensive summary of key points, themes, and important information]

SENTIMENT ANALYSIS: 
- Overall emotional tone with percentage breakdown
- Key emotional themes and their prevalence  
- Notable sentiment patterns or shifts
- Specific examples from the text

KEY INSIGHTS: [Most important findings, trends, and patterns]

THEMES: [Major topics and recurring themes identified]
"""
        
        result = model_pool.generate(prompt, max_tokens=2000)
        
        return {
            'batch_id': batch_id,
            'result': result,
            'status': 'success',
            'chunk_count': len(batch),
            'estimated_tokens': estimated_tokens
        }
        
    except Exception as e:
        logging.error(f"Error processing batch {batch_id}: {e}")
        return {
            'batch_id': batch_id,
            'result': f"ERROR: {str(e)}",
            'status': 'error',
            'chunk_count': len(batch) if batch else 0,
            'estimated_tokens': 0
        }

def high_performance_batch_processing(model_pool, batches):
    """Process batches using high-performance parallel execution."""
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all batch processing jobs
        future_to_batch = {
            executor.submit(process_single_batch, model_pool, batch, i): i 
            for i, batch in enumerate(batches)
        }
        
        # Monitor progress and collect results
        completed = 0
        progress_bar = tqdm(total=len(batches), desc="Processing batches")
        
        for future in as_completed(future_to_batch):
            try:
                result = future.result()
                results.append(result)
                completed += 1
                
                # Update progress with additional info
                if result['status'] == 'success':
                    progress_bar.set_postfix({
                        'Success': f"{sum(1 for r in results if r['status'] == 'success')}/{completed}",
                        'Tokens/s': f"{sum(r.get('estimated_tokens', 0) for r in results)/(time.time()-start_time):.0f}"
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
                    'estimated_tokens': 0
                })
        
        progress_bar.close()
    
    # Sort results by batch_id to maintain order
    results.sort(key=lambda x: x['batch_id'])
    return results

def create_comprehensive_final_report(model_pool, batch_results):
    """Create a comprehensive final report from all batch results."""
    
    # Filter successful results
    successful_results = [r for r in batch_results if r['status'] == 'success']
    
    if not successful_results:
        return "ERROR: No successful batch processing results to consolidate."
    
    # Calculate statistics
    total_chunks = sum(r['chunk_count'] for r in successful_results)
    total_tokens = sum(r.get('estimated_tokens', 0) for r in successful_results)
    success_rate = len(successful_results) / len(batch_results) * 100
    
    # Prepare consolidated analysis
    consolidated_analyses = []
    for result in successful_results:
        consolidated_analyses.append(f"=== BATCH {result['batch_id']} ANALYSIS ===\n{result['result']}")
    
    consolidated_text = "\n\n".join(consolidated_analyses)
    
    final_prompt = f"""You are an expert data analyst. Create a comprehensive final report by consolidating the following {len(successful_results)} batch analyses.

ANALYSIS STATISTICS:
- Total batches processed: {len(successful_results)}/{len(batch_results)}
- Success rate: {success_rate:.1f}%
- Total chunks analyzed: {total_chunks}
- Estimated tokens processed: {total_tokens:,}

Create a final report with these sections:

1. EXECUTIVE SUMMARY: High-level overview of key findings and insights

2. GLOBAL STORYLINE: Comprehensive narrative integrating all themes, events, and insights across all batches

3. COMPREHENSIVE SENTIMENT ANALYSIS:
   - Overall sentiment distribution with percentages
   - Detailed emotional themes and patterns
   - Sentiment evolution or notable shifts
   - Representative quotes and examples

4. KEY INSIGHTS AND PATTERNS: Most significant findings, trends, and recurring themes

5. THEMATIC ANALYSIS: Major topics, categories, and subject areas identified

6. STATISTICAL INSIGHTS: Quantitative observations about the data patterns

BATCH ANALYSES TO CONSOLIDATE:
{consolidated_text}

COMPREHENSIVE FINAL REPORT:"""
    
    try:
        final_report = model_pool.generate(final_prompt, max_tokens=3000)
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
        logging.info("Creating optimized processing batches...")
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
        
        # Step 4: Initialize model pool
        logging.info("Initializing Phi-4 model pool...")
        model_pool = Phi4ModelPool(GGUF_PATH)
        
        # Step 5: High-performance parallel processing
        logging.info(f"Starting high-performance processing with {MAX_WORKERS} workers and {MODEL_INSTANCES} model instances...")
        batch_results = high_performance_batch_processing(model_pool, batches)
        
        # Step 6: Process results
        successful_batches = sum(1 for r in batch_results if r['status'] == 'success')
        failed_batches = len(batch_results) - successful_batches
        processing_time = time.time() - start_time
        
        logging.info(f"Processing complete: {successful_batches} successful, {failed_batches} failed in {processing_time:.2f}s")
        
        # Step 7: Generate comprehensive final report
        logging.info("Generating comprehensive final report...")
        final_report = create_comprehensive_final_report(model_pool, batch_results)
        
        # Step 8: Display comprehensive results
        end_time = time.time()
        total_processing_time = end_time - start_time
        tokens_per_second = total_estimated_tokens / total_processing_time if total_processing_time > 0 else 0
        
        print("\n" + "="*100)
        print("HIGH-PERFORMANCE COMPREHENSIVE ANALYSIS REPORT")
        print("="*100)
        print(f"System Specs: 16GB RAM, 6-core/12-thread CPU")
        print(f"Processing Configuration:")
        print(f"  - Workers: {MAX_WORKERS}")
        print(f"  - Model Instances: {MODEL_INSTANCES}")
        print(f"  - Context Length: {PHI4_MAX_CONTEXT:,} tokens")
        print(f"  - Batch Size: {PROCESSING_BATCH_SIZE:,} tokens")
        print(f"")
        print(f"Performance Metrics:")
        print(f"  - Total Processing Time: {total_processing_time:.2f} seconds")
        print(f"  - Batches Processed: {successful_batches}/{len(batches)} ({successful_batches/len(batches)*100:.1f}% success)")
        print(f"  - Total Chunks: {len(translated):,}")
        print(f"  - Total Tokens Processed: {total_estimated_tokens:,}")
        print(f"  - Processing Speed: {tokens_per_second:,.0f} tokens/second")
        print(f"  - Model Pool Stats: {model_pool.get_stats()}")
        print("="*100)
        print()
        print(final_report)
        print("\n" + "="*100)
        
        # Cleanup
        del model_pool
        gc.collect()

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
