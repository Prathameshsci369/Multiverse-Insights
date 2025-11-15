import logging
import sys
import gc
import json
import os
from llama_cpp import Llama

# --- Configuration ---
# NOTE: This model is loaded *separately* from the models in main.py
# This is the "Final Analyst" model.
MODEL_PATH = "/home/anand/Downloads/Qwen2.5-7B-Instruct.Q5_K_M.gguf"
INPUT_TOKENS = 16384  # 16k
OUTPUT_TOKENS = 4096  # 4k
N_GPU_LAYERS = 0      # 0 for CPU-only, -1 to offload all layers to GPU
# --- End Configuration ---

def _load_model(model_path, n_ctx, n_gpu_layers):
    """
    (Internal) Loads the GGUF model from the specified path.
    """
    print("\n[Final Analysis] Loading final analysis model... This may take a moment.")
    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False  # Disables verbose logging from llama_cpp
        )
        print(f"[Final Analysis] Model loaded successfully. Context: {INPUT_TOKENS}k, Max Output: {OUTPUT_TOKENS}k.")
        return llm
    except Exception as e:
        print(f"[Final Analysis] Error: Failed to load model from {model_path}.")
        print(f"Details: {e}")
        sys.exit(1)

def _load_json_data(filepath):
    """
    (Internal) Tries to load and parse a JSON file from the given filepath.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"[Final Analysis] Successfully loaded partial results from {filepath}")
            return data
    except FileNotFoundError:
        print(f"[Final Analysis] Error: JSON file not found at {filepath}")
    except json.JSONDecodeError:
        print(f"[Final Analysis] Error: Could not decode JSON. Please check the file's format.")
    except Exception as e:
        print(f"[Final Analysis] An unexpected error occurred while reading the file: {e}")
    return None

def _extract_text_from_json(data):
    """
    (Internal) Attempts to extract a meaningful block of text from various JSON structures.
    For this module, it will just serialize the partial results.
    """
    # In the context of final analysis, we *always* want to analyze the full
    # JSON of partial results, not just a single text field.
    return json.dumps(data, indent=2)

def _get_analysis_messages(json_data, is_direct=False):
    """
    (Internal) Creates the system and user prompts for the final analysis.
    """
    if is_direct:
        # For direct analysis, use the full text
        joined_text = json_data["direct_analysis"]["text"]
        analysis_type = "full text"
    else:
        # For partial results, serialize the data
        joined_text = _extract_text_from_json(json_data)
        analysis_type = "partial analyses from other models"
    
    # 1. "Extract" text (which is just serializing the partial results)
    # joined_text = _extract_text_from_json(json_data)
    
    # 2. Set the new system prompt
    system_prompt = (
        "You are a specialized text analyst. You must follow the user's "
        "instructions precisely, providing a structured analysis of the provided data "
        f"(which consists of {analysis_type}) in the exact "
        "format they request. You must stop at the indicated line."
    )
    
    # 3. Set the user's requested prompt
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

CONTROVERSIAL_SCORE:
[number]/1.0 - [explanation]

STOP AFTER THIS LINE.
Print this exact token after the final line:A
"""
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

def _run_analysis(llm, messages):
    """
    (Internal) Runs a single, streaming analysis and prints/returns the output.
    This function now explicitly flushes the output buffer after every token.
    """
    print("\n[Final Analysis] Assistant: Generating final report... \n", end="", flush=True)
    full_response = ""
    try:
        response_stream = llm.create_chat_completion(
            messages=messages,
            max_tokens=OUTPUT_TOKENS,
            stream=True,
            # We can use the prompt's stop instruction as a stop token
            stop=["STOP AFTER THIS LINE."] 
        )
        
        for chunk in response_stream:
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                token = delta["content"]
                print(token, end="") # Using end="" to avoid extra newlines
                sys.stdout.flush()   # Explicitly flush the buffer
                full_response += token
        
        print() # Newline after full response
        return full_response.strip()

    except Exception as e:
        print(f"\n[Final Analysis] An error occurred during analysis: {e}")
        return None

# --- Main Public Function ---

def combined_analysis(data, is_direct=False):
    """
    Orchestrates the final analysis of partial results or direct text.
    
    This function is called by main.py. It loads the partial results
    from the given JSON file or uses direct analysis data, loads a GGUF model, runs a
    structured analysis prompt over the data, streams the result,
    and then returns the complete analysis as a string.
    
    Args:
        data: Either a path to a JSON file with partial results, or a dictionary with direct analysis data
        is_direct: Boolean flag indicating if this is a direct analysis
    """
    
    if is_direct:
        # Handle direct analysis
        logging.info("Performing direct analysis on full text.")
        json_data = data
    else:
        # Handle partial results analysis (existing logic)
        json_data = _load_json_data(data)
        if not json_data:
            return "ERROR: [Final Analysis] Failed to load partial results JSON."

    # 2. Create initial prompts
    messages = _get_analysis_messages(json_data, is_direct)

    # 3. Load the model
    llm = _load_model(MODEL_PATH, INPUT_TOKENS, N_GPU_LAYERS)
    
    # 4. Run the single analysis
    final_report = _run_analysis(llm, messages)

    # 5. Release resources and exit
    print("\n[Final Analysis] Analysis complete. Releasing model resources.")
    del llm  # Explicitly delete the Llama object to free memory
    print("[Final Analysis] Resources released.")
    
    return final_report

# Note: No __main__ block, as this is now a module to be imported.
def main():
    json_path = "/home/anand/Documents/data/tweets_output365498.json"
    combined_analysis(json_path)
