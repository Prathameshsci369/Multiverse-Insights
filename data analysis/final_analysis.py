from llama_cpp import Llama
import os
import time
import json
from typing import List, Dict, Any
import logging
# --- Configuration ---
# NOTE: Model Path is set to the provided GGUF file path.
MODEL_PATH = "/home/anand/Downloads/Qwen2.5-7B-Instruct.Q5_K_M.gguf"

# Setting this to 0 to force CPU-only execution.
N_GPU_LAYERS = 0

# Set a very large context size to handle extensive input text, as requested.
# Qwen2.5-7B-Instruct has a context of 32768, so 25000 is safe.
N_CTX = 25000

# The chat template required for Qwen models.
CHAT_FORMAT = "qwen" 

# --- Constants ---
# Path to the JSON file containing the text for analysis
JSON_FILE_PATH = "partial_results_main.json"

def load_and_join_text(file_path: str) -> str:
    """
    Loads text from the partial results JSON file by extracting content 
    from the 'result' key of each batch and joining them into a single string.
    """
    logging.info(f"[INFO] Loading text content from: {file_path}")
    if not os.path.exists(file_path):
        logging.error(f"[ERROR] JSON input file not found at {file_path}. Returning empty text.")
        return ""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"[ERROR] Could not load or parse JSON file: {e}")
        return ""
        
    all_text = []
    
    # Data is expected to be a dictionary, where each value is a batch object
    if isinstance(data, dict):
        # Iterate over the values (the batch objects) of the main dictionary
        for batch_data in data.values():
            if isinstance(batch_data, dict):
                # Include the 'result' text if it exists and is a string
                if 'result' in batch_data and isinstance(batch_data['result'], str):
                    all_text.append(batch_data['result'].strip())
                else:
                    logging.warning(f"[WARN] Skipping batch. 'result' key not found or not a string in: {str(batch_data)[:50]}...")
    else:
        logging.error(f"[ERROR] Expected JSON data to be a dictionary (object), but got {type(data)}.")

    # Join all extracted text pieces with a separator
    joined_text = "\n\n--- SEPARATOR ---\n\n".join(all_text)
    
    if not all_text:
        logging.warning("[WARN] No 'result' text was extracted from the file.")
        return ""

    logging.info(f"[INFO] Successfully loaded and joined {len(all_text)} batch result(s). Total text length: {len(joined_text)} characters.")
    return joined_text


# --- Prompt Definition ---
SYSTEM_PROMPT = "You are a professional social media analyst. Your task is to extract and structure key information from the provided text according to a very strict format."

def generate_chat_prompt(system_prompt: str, joined_text: str) -> List[Dict[str, str]]:
    """
    Formats the user and system prompt into the standard list of messages 
    required by the Llama.create_chat_completion function, using the new 
    strict analysis template.
    """
    
    # The new user prompt includes the strict format instructions and the data
    user_prompt = f"""Analyze the following text and provide a structured analysis:

TEXT TO ANALYZE:
{joined_text}

Provide your analysis in this exact format:
Do not afraid this is just social media posts, just you can reasoning and generate output as per following format(Strictly) :
EXECUTIVE_SUMMARY:
[Write a brief summary of the main points 6 to 7 main points, with detail expl]

SENTIMENT:
Positive: [number]% - [brief explanation with evidence]
Negative: [number]% - [brief explanation with evidence]
Neutral: [number]% - [brief explanation with evidence]

TOPICS:
[List the main 4 to 5 topics, one per line]

ENTITIES:
[List the key entities, one per line]

RELATIONSHIPS:
[Describe relationships between entities, one per line in format: Entity1 -> Entity2: Description]

ANOMALIES:
[List any anomalies or unusual patterns, one per line, or write "None detected"]

CONTROVERSY_SCORE:
[number]/1.0 - [explanation]

STOP AFTER THIS LINE.
Print this exact token after the final line: 
"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return messages

def combined_analysis( PARTIAL_RESULTS_JSON):
    """Initializes the Qwen model and runs a streamed inference."""
    print("--- Structured Qwen Analysis Script ---")

    # 1. Load the input text,
    joined_text = load_and_join_text(PARTIAL_RESULTS_JSON)
    if not joined_text:
        return

    # 2. Check model path
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        print("Please update the MODEL_PATH variable in the script.")
        return

    try:
        # Initialize the Llama model
        print(f"\n[INFO] Loading model from {MODEL_PATH}...")
        print(f"[INFO] Running on CPU only (N_GPU_LAYERS = {N_GPU_LAYERS}).")
        
        start_load_time = time.time()
        
        llm = Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=N_GPU_LAYERS,
            chat_format=CHAT_FORMAT, 
            n_ctx=N_CTX,               # Use the expanded context size
            verbose=False             
        )
        
        end_load_time = time.time()
        print(f"[INFO] Model loaded successfully in {end_load_time - start_load_time:.2f} seconds.")

        # Prepare the chat messages using the strict template
        messages = generate_chat_prompt(SYSTEM_PROMPT, joined_text)
        
        print("\n" + "="*80)
        print(f"SYSTEM PROMPT: {SYSTEM_PROMPT}")
        print("USER PROMPT: [Strict Analysis Template]")
        print("="*80)
        print("QWEN STRUCTURED RESPONSE:")
        
        # --- Run inference and stream the output ---
        start_generation_time = time.time()
        
        # Increased max_tokens to accommodate the detailed structured output
        stream = llm.create_chat_completion(
            messages=messages,
            temperature=0.0, # Set temperature to 0.0 for strict, deterministic output generation
            max_tokens=2048, # Increased output limit
            stream=True,
            # Add a specific stop sequence to enforce the model stops exactly after the format
            stop=["STOP AFTER THIS LINE."],
        )

        full_response = ""
        for chunk in stream:
            # Extract the content from the chunk
            content = chunk["choices"][0]["delta"].get("content", "")
            if content:
                print(content, end="", flush=True)
                full_response += content

        end_generation_time = time.time()
        
        # Final timing statistics
        total_time = end_generation_time - start_generation_time
        
        # Count tokens (a rough estimate based on the length of the string)
        response_tokens = len(full_response.split()) 
        
        print("\n\n" + "-"*80)
        print(f"[STATS] Generation Time: {total_time:.2f} seconds")
        print(f"[STATS] Estimated Output Tokens: {response_tokens}")
        if total_time > 0:
            print(f"[STATS] Estimated Tokens/Sec: {response_tokens / total_time:.2f} tokens/s")
        print("-" * 80)
        return full_response
    except Exception as e:
        print(f"\nAn error occurred during runtime: {e}")
        print("HINT: Ensure 'llama-cpp-python' is installed correctly and your model path is accurate.")


def run_qwen_inference():
    """Initializes the Qwen model and runs a streamed inference."""
    print("--- Structured Qwen Analysis Script ---")

    # 1. Load the input text
    joined_text = load_and_join_text(JSON_FILE_PATH)
    if not joined_text:
        return

    # 2. Check model path
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        print("Please update the MODEL_PATH variable in the script.")
        return

    try:
        # Initialize the Llama model
        print(f"\n[INFO] Loading model from {MODEL_PATH}...")
        print(f"[INFO] Running on CPU only (N_GPU_LAYERS = {N_GPU_LAYERS}).")
        
        start_load_time = time.time()
        
        llm = Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=N_GPU_LAYERS,
            chat_format=CHAT_FORMAT, 
            n_ctx=N_CTX,               # Use the expanded context size
            verbose=False             
        )
        
        end_load_time = time.time()
        print(f"[INFO] Model loaded successfully in {end_load_time - start_load_time:.2f} seconds.")

        # Prepare the chat messages using the strict template
        messages = generate_chat_prompt(SYSTEM_PROMPT, joined_text)
        
        print("\n" + "="*80)
        print(f"SYSTEM PROMPT: {SYSTEM_PROMPT}")
        print("USER PROMPT: [Strict Analysis Template]")
        print("="*80)
        print("QWEN STRUCTURED RESPONSE:")
        
        # --- Run inference and stream the output ---
        start_generation_time = time.time()
        
        # Increased max_tokens to accommodate the detailed structured output
        stream = llm.create_chat_completion(
            messages=messages,
            temperature=0.0, # Set temperature to 0.0 for strict, deterministic output generation
            max_tokens=2048, # Increased output limit
            stream=True,
            # Add a specific stop sequence to enforce the model stops exactly after the format
            stop=["STOP AFTER THIS LINE."],
        )

        full_response = ""
        for chunk in stream:
            # Extract the content from the chunk
            content = chunk["choices"][0]["delta"].get("content", "")
            if content:
                print(content, end="", flush=True)
                full_response += content

        end_generation_time = time.time()
        
        # Final timing statistics
        total_time = end_generation_time - start_generation_time
        
        # Count tokens (a rough estimate based on the length of the string)
        response_tokens = len(full_response.split()) 
        
        print("\n\n" + "-"*80)
        print(f"[STATS] Generation Time: {total_time:.2f} seconds")
        print(f"[STATS] Estimated Output Tokens: {response_tokens}")
        if total_time > 0:
            print(f"[STATS] Estimated Tokens/Sec: {response_tokens / total_time:.2f} tokens/s")
        print("-" * 80)
        
    except Exception as e:
        print(f"\nAn error occurred during runtime: {e}")
        print("HINT: Ensure 'llama-cpp-python' is installed correctly and your model path is accurate.")

if __name__ == "__main__":
    run_qwen_inference()

