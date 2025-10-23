#!/usr/bin/env python3
"""
Analysis functions for the analysis pipeline.
"""
import re
import logging
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import PHI4_MAX_CONTEXT, FINAL_REPORT_TOKENS
from utils import estimate_tokens, _iso_now
from llm_interface import call_llm_and_stream_to_terminal_and_file

# --- add these imports near the top of analysis.py ---
import math
from typing import List, Dict


def extract_first_analysis_block(text):
    """
    Return the first valid analysis block up to CONTROVERSY_SCORE (inclusive)
    or up to the unique end token '===END_OF_ANALYSIS===' if present.
    If nothing matches, return best-effort trimmed text.
    """
    # 1) If model echoed the end token, prefer slicing up to it
    end_token = "===END_OF_ANALYSIS==="
    if end_token in text:
        first_part = text.split(end_token)[0].strip()
        return first_part

    # 2) Otherwise try to capture from EXECUTIVE_SUMMARY ... CONTROVERSY_SCORE (first occurrence)
    pattern = re.compile(
        r'EXECUTIVE_SUMMARY:.*?CONTROVERSY_SCORE:.*?(?:/1\.0\s*-\s*.*?$)',
        re.DOTALL | re.MULTILINE
    )
    m = pattern.search(text)
    if m:
        return m.group(0).strip()

    # 3) Fallback: capture from start until the first repeated EXECUTIVE_SUMMARY (to avoid repeats)
    first_exec = text.find("EXECUTIVE_SUMMARY:")
    if first_exec != -1:
        # find next EXECUTIVE_SUMMARY after the first — if exists, cut before it
        second_exec = text.find("EXECUTIVE_SUMMARY:", first_exec + 1)
        if second_exec != -1:
            return text[first_exec:second_exec].strip()
        # else return from first to end, but remove obvious repeated suffixes
        return text[first_exec:].strip()

    # 4) Last fallback: return the whole text trimmed
    return text.strip()
# --- helper: summarize a list of texts using a small summarization prompt ---
def _summarize_texts_with_llm(llm, texts: List[str], max_summary_tokens: int = 800, prompt_note: str = "") -> str:
    """
    Summarize the provided texts into a single concise summary using the LLM instance.
    Returns the summary string.
    - llm: model manager (must implement generate or generate_stream as used elsewhere)
    - texts: list of strings to summarize
    - max_summary_tokens: tokens allowed for the summary (small)
    """
    joined = "\n\n".join(texts)
    prompt = f"""Summarize the following texts into a concise, single-paragraph summary that preserves the most important points.
Do NOT add opinions or invent facts. Keep the summary short and factual.

{prompt_note}

Texts:
{joined}

Summary:
"""
    # Use a non-streaming short call (safe)
    try:
        # if llm has 'generate' method (SingleModelManager), use it
        if hasattr(llm, "generate"):
            raw = llm.generate(prompt, max_tokens=max_summary_tokens)
            # raw may be dict or string
            if isinstance(raw, dict) and "choices" in raw and len(raw["choices"]) > 0:
                return raw["choices"][0].get("text", "").strip()
            elif isinstance(raw, str):
                return raw.strip()
            else:
                return str(raw).strip()
        else:
            # fallback to lm_call_cached if llm object is a client expected elsewhere
            return lm_call_cached(llm, prompt, max_tokens=max_summary_tokens, stream=False)
    except Exception as e:
        logging.warning(f"Summarization call failed: {e}")
        # as fallback return joined truncated
        return joined[:1000] + ("..." if len(joined) > 1000 else "")

# --- helper: recursively compress partial_results until under token budget ---
def _ensure_under_token_budget(llm, partial_results: List[Dict], token_budget: int,
                               summary_max_tokens: int = 800, group_size: int = 8,
                               min_group_size: int = 2, max_rounds: int = 6) -> List[Dict]:
    """
    Reduce partial_results by summarizing groups until combined tokens <= token_budget.
    - partial_results: list of {'batch_id': int, 'result': str}
    - token_budget: allowed tokens for final joined_text
    - summary_max_tokens: max tokens to allocate to each summarization output
    - group_size: initial number of items to group per summarization
    Returns a new partial_results list (may contain summaries instead of raw results).
    """
    def combined_tokens(items):
        text = "\n\n".join([it.get("result", "") for it in items])
        return estimate_tokens(text)

    round_num = 0
    current = partial_results

    while combined_tokens(current) > token_budget and round_num < max_rounds:
        round_num += 1
        logging.info(f"Compression round {round_num}: total tokens {combined_tokens(current):,} > budget {token_budget:,}")
        new_items = []
        i = 0
        n = len(current)
        # dynamically adapt group size if too big
        gs = group_size
        if gs < min_group_size:
            gs = min_group_size
        while i < n:
            group = current[i:i+gs]
            # if group is small (like last tail) and too small to summarize, just append raw
            if len(group) == 1:
                # single element — keep as is
                new_items.append(group[0])
            else:
                texts = [g.get("result", "") for g in group]
                prompt_note = f"(This is an automatic intermediate summary round {round_num}. Summarize in concise bullet points or ~1-3 sentences.)"
                summary = _summarize_texts_with_llm(llm, texts, max_summary_tokens=summary_max_tokens, prompt_note=prompt_note)
                # create synthetic batch_id to indicate aggregated item (use min-max of group ids)
                batch_ids = [g.get("batch_id") for g in group if "batch_id" in g]
                agg_id = f"{batch_ids[0]}-{batch_ids[-1]}" if batch_ids else i
                new_items.append({"batch_id": agg_id, "result": summary})
            i += gs

        # If compression didn't reduce tokens (rare), lower group size to produce smaller summaries
        if combined_tokens(new_items) >= combined_tokens(current):
            if gs > min_group_size:
                group_size = max(min_group_size, gs // 2)
                logging.info(f"No token reduction this round; decreasing group_size to {group_size} and retrying.")
                # try again in next round with smaller groups
                current = current  # no change
            else:
                logging.warning("Cannot further compress results effectively. Breaking compression loop.")
                break
        else:
            # accept compressed version
            current = new_items

    logging.info(f"Compression finished after {round_num} rounds; final tokens: {combined_tokens(current):,}")
    return current


from utils import estimate_tokens
from config import PHI4_MAX_CONTEXT

def process_batch(llm_manager, batch, batch_id):
    """Processes a single batch using a specific LLM instance, with strict token management."""
    from config import MIN_TOKENS_RESERVE, SAFETY_MARGIN, PHI4_MAX_CONTEXT
    
    # Join batch texts and clean
    joined_text = " ".join(text.strip() for text in batch if text)
    
    # Ultra-conservative token management
    prompt_overhead = 2000  # System message and prompt structure
    expected_response = 4000  # Response generation
    safety_margin = 1000  # Additional safety buffer
    max_safe_tokens = PHI4_MAX_CONTEXT - prompt_overhead - expected_response - safety_margin
    
    # Pre-validate token count
    total_tokens = estimate_tokens(joined_text)
    if total_tokens > max_safe_tokens:
        logging.warning(f"Batch {batch_id} requires {total_tokens} tokens, exceeding safe limit of {max_safe_tokens}")

    # Estimate total tokens
    total_tokens = estimate_tokens(joined_text)

    if total_tokens > max_safe_tokens:
        # Calculate a more conservative truncation
        safe_ratio = (max_safe_tokens - 1000) / total_tokens  # Extra 1000 token safety margin
        cutoff_chars = int(len(joined_text) * safe_ratio)
        
        # Try to cut at a sentence boundary
        if cutoff_chars < len(joined_text):
            # Look for last sentence boundary
            for marker in [". ", "! ", "? ", "\n", ". \n"]:
                last_boundary = joined_text[:cutoff_chars].rfind(marker)
                if last_boundary != -1 and last_boundary > cutoff_chars * 0.8:  # At least 80% of content
                    cutoff_chars = last_boundary + 1
                    break
                    
        joined_text = joined_text[:cutoff_chars]
        logging.warning(f"[process_batch] Batch {batch_id} ({total_tokens} tokens) truncated to fit context window.")

    prompt = f"""Analyze the following text and provide a structured summary, sentiment, and key topics.

Text:
{joined_text}

---
Report:
- Summary:
- Sentiment:
- Topics:
"""

    # Generate safely
    result = llm_manager.generate(prompt, max_tokens=3000)
    return {"batch_id": batch_id, "result": result}

# --- main combined_analysis (replace existing one) ---
def combined_analysis(llm, partial_results, token_budget_single_call=PHI4_MAX_CONTEXT - 1000, 
                     combined_token_budget=FINAL_REPORT_TOKENS, show_progress=True, 
                     stream_llm_if_supported=True, custom_stream_handler=None):
    """
    Performs a combined analysis on the input data.
    Now expects a clean list of result strings.
    """
    # Log what we received
    logging.info(f"Received partial results of type: {type(partial_results)}")
    
    try:
        # Ensure we have a list of strings
        all_results = []
        
        if isinstance(partial_results, (list, tuple)):
            # Filter and clean the results
            for item in partial_results:
                if isinstance(item, str) and item.strip():
                    # Skip any items that look like errors
                    if not (item.startswith('ERROR:') or 'exceed context window' in item.lower()):
                        all_results.append(item.strip())
                elif isinstance(item, dict) and 'result' in item:
                    result = item['result']
                    if isinstance(result, str) and result.strip():
                        if not (result.startswith('ERROR:') or 'exceed context window' in result.lower()):
                            all_results.append(result.strip())
                            
        elif isinstance(partial_results, dict):
            # Extract results from dictionary values
            for value in partial_results.values():
                if isinstance(value, str) and value.strip():
                    if not (value.startswith('ERROR:') or 'exceed context window' in value.lower()):
                        all_results.append(value.strip())
                elif isinstance(value, dict) and 'result' in value:
                    result = value['result']
                    if isinstance(result, str) and result.strip():
                        if not (result.startswith('ERROR:') or 'exceed context window' in result.lower()):
                            all_results.append(result.strip())
        else:
            # Single item case
            if isinstance(partial_results, str) and partial_results.strip():
                if not (partial_results.startswith('ERROR:') or 'exceed context window' in partial_results.lower()):
                    all_results.append(partial_results.strip())
                    
        if not all_results:
            return "ERROR: No valid analysis results to combine."
            
    except Exception as e:
        logging.error(f"Error processing partial results: {e}")
        return f"ERROR: Failed to process analysis results: {str(e)}"
    
    # Join all results
    joined_text = "\n\n".join(all_results)
    
    # Debug: Print what we're about to analyze
    logging.info(f"Text to analyze length: {len(joined_text)} characters")
    logging.info(f"Text to analyze tokens: {estimate_tokens(joined_text):,}")
    logging.info(f"Text preview: {joined_text[:300]}...")
    
    if not joined_text.strip():
        logging.warning("No content found to analyze.")
        return "No data to analyze."
    
    total_tokens = estimate_tokens(joined_text)
    
    if show_progress:
        if custom_stream_handler:
            custom_stream_handler(f"Starting combined analysis on {total_tokens:,} tokens...")
        print(f"[{_iso_now()}] Starting combined analysis on {total_tokens:,} tokens...")
        print("---")
    
    # Simple prompt with clear format
    prompt = f"""Analyze the following text and provide a structured analysis:

TEXT TO ANALYZE:
{joined_text}

Provide your analysis in this exact format:

EXECUTIVE_SUMMARY:
[Write a brief summary of the main points]

SENTIMENT:
Positive: [number]% - [brief explanation]
Negative: [number]% - [brief explanation]
Neutral: [number]% - [brief explanation]

TOPICS:
[List the main topics, one per line]

ENTITIES:
[List the key entities, one per line]

RELATIONSHIPS:
[Describe relationships between entities, one per line in format: Entity1 -> Entity2: Description]

ANOMALIES:
[List any anomalies or unusual patterns, one per line, or write "None detected"]

CONTROVERSY_SCORE:
[number]/1.0 - [explanation]

STOP AFTER THIS LINE.
Print this exact token after the final line: ===END_OF_ANALYSIS===
DO NOT OUTPUT ANYTHING AFTER ===END_OF_ANALYSIS===."""
    
    # Use the streaming function with a strict token limit
    raw_response = call_llm_and_stream_to_terminal_and_file(
        llm, 
        prompt, 
        max_tokens=2000,  # Reduced token limit to prevent excessive generation
        out_filepath="final_analysis_report.json",
        enable_stream=stream_llm_if_supported,
        custom_stream_handler=custom_stream_handler
    )
    
    # Post-process to extract only the first analysis block
    cleaned = extract_first_analysis_block(raw_response)
    
    # Save the cleaned response to a separate file for debugging
    with open("final_analysis_report_clean.txt", "w", encoding="utf-8") as f:
        f.write(cleaned)
    
    return cleaned.strip()
        
        
        
        
def combined_analysis_V2(llm, partial_results, token_budget_single_call=PHI4_MAX_CONTEXT - 1000, 
                     combined_token_budget=FINAL_REPORT_TOKENS, show_progress=True, 
                     stream_llm_if_supported=True, custom_stream_handler=None):
    """
    Performs a combined analysis (summarization, sentiment, topics) on the input data.
    Modified to support custom stream handlers with enhanced error handling.
    """
    # Enhanced partial results processing with validation
    results_list = []
    
    def extract_result(item):
        """Helper to safely extract result from various formats"""
        if item is None:
            return None
        if isinstance(item, dict):
            # Try multiple common result keys
            for key in ['result', 'text', 'content', 'analysis']:
                if key in item:
                    value = item[key]
                    if value and isinstance(value, str):
                        return value.strip()
            # If no recognized keys, try to convert whole dict to string
            return str(item)
        if isinstance(item, str):
            return item.strip()
        return str(item)
    
    try:
        if isinstance(partial_results, dict):
            for k, v in partial_results.items():
                result = extract_result(v)
                if result:
                    results_list.append(result)
                    
        elif isinstance(partial_results, (list, tuple)):
            for item in partial_results:
                result = extract_result(item)
                if result:
                    results_list.append(result)
                    
        else:
            result = extract_result(partial_results)
            if result:
                results_list.append(result)
                
        # Remove any items that look like errors or empty content
        results_list = [r for r in results_list if r and 
                       not r.startswith('ERROR:') and
                       not 'exceed context window' in r.lower() and
                       len(r.strip()) > 0]
                       
        if not results_list:
            logging.warning("No valid results found in partial_results")
            return json.dumps({"multiverse_combined": {"error": "No valid partial results found for analysis"}})
            
    except Exception as e:
        logging.error(f"Error processing partial results: {e}")
        return json.dumps({
            "multiverse_combined": {
                "error": f"Failed to process partial results: {str(e)}",
                "details": "Exception occurred while extracting results"
            }
        })

    joined_text = "\n\n".join([r for r in results_list if r])
    
    if not joined_text.strip():
        logging.warning("No content found in partial results. Cannot generate final report.")
        return "{\"multiverse_combined\": \"No data to analyze.\"}"
    
    total_tokens = estimate_tokens(joined_text)
    
    if show_progress:
        if custom_stream_handler:
            custom_stream_handler(f"Starting combined analysis on {total_tokens:,} tokens...")
        print(f"[{_iso_now()}] Starting combined analysis on {total_tokens:,} tokens...")
        print("---")
    
    # The prompt is modified to request JSON output
    prompt = f"""<instructions>
You are a highly specialized AI analyst. Your ONLY task is to analyze the provided text corpus and generate a comprehensive synthesis in a single, well-formed JSON document.

You MUST STRICTLY adhere to the following rules:
Rules: Output only valid JSON starting with '{{' and ending with '}}', having one key "multiverse_combined". Base analysis strictly on given text only. Include exactly one entity_recognition section, one relationship_extraction section, one anomaly_detection section, and one controversy_score section. Use correct JSON syntax with proper quoting and escaping. No extra text or commentary outside JSON.
If you not got the text for the analysis, respond with an empty structure with zeroes  proper words to not get data. 
</instructions>

<analysis_request>

  "entity_recognition": [
  "min_tokens": 400,
   "max_tokens": 600,
    {{
      
      "type": "PERSON|ORGANIZATION|LOCATION|DATE",
      "name": "string"
    }}
  ],

  "relationship_extraction": [
    {{
      "entity1": "string",
      "relationship": "string",
      "entity2": "string"
    }}
  ],

  "anomaly_detection": [
    {{
      "section": "string",
      "description": "string"
    }}
  ],

  "controversy_score": {{
    "value": 0.0,
    "explanation": "string"
  }}

</analysis_request>

<corpus>
{joined_text}
</corpus>

{{"multiverse_combined": """
        
    # Use the streaming function for the final report
    response = call_llm_and_stream_to_terminal_and_file(
        llm, 
        prompt, 
        max_tokens=combined_token_budget,
        out_filepath="final_analysis_report.json",
        enable_stream=stream_llm_if_supported,
        custom_stream_handler=custom_stream_handler
    )

    # Clean and validate the response
    response = response.strip()
    
    # Try to extract valid JSON from the response
    try:
        # Clean the response first
        clean_response = response.strip()
        
        # Remove any non-JSON prefix
        start_idx = clean_response.find('{')
        if start_idx == -1:
            raise ValueError("No JSON object found in response")
        clean_response = clean_response[start_idx:]
        
        # Find matching closing brace
        brace_count = 0
        end_idx = -1
        in_string = False
        escape_next = False
        
        for i, char in enumerate(clean_response):
            if escape_next:
                escape_next = False
                continue
                
            if char == '\\':
                escape_next = True
                continue
                
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
                
            if not in_string:  # Only count braces outside strings
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i
                        break
        
        if end_idx == -1:
            raise ValueError("Incomplete JSON object")
        
        # Extract the JSON
        json_str = clean_response[:end_idx + 1]
        
        # Try to fix common JSON issues
        json_str = json_str.replace('\n', ' ').replace('\r', '')
        json_str = re.sub(r'(?<!\\)"(?![:,}\]])', '\\"', json_str)  # Escape unescaped quotes
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)  # Remove trailing commas
        
        # Validate that it's valid JSON
        parsed_json = json.loads(json_str)
        
        # Ensure the expected structure
        if "multiverse_combined" not in parsed_json:
            parsed_json = {"multiverse_combined": parsed_json}
            json_str = json.dumps(parsed_json)
        
    except (ValueError, json.JSONDecodeError) as e:
        logging.error(f"Failed to extract valid JSON from response: {e}")
        logging.debug(f"Raw response: {response[:500]}...")
        
        # Return a simple valid JSON with error message
        return '{"multiverse_combined": {"error": "Failed to generate valid JSON analysis", "raw_response": "' + response.replace('"', '\\"') + '"}}'
        
        
        
def combined_analysis123(llm, partial_results, token_budget_single_call=PHI4_MAX_CONTEXT - 1000, 
                     combined_token_budget=FINAL_REPORT_TOKENS, show_progress=True, 
                     stream_llm_if_supported=True, custom_stream_handler=None):
    """
    Performs a combined analysis (entity recognition, relationship extraction, anomaly detection, controversy score) on the input data.
    Modified to support custom stream handlers.
    """
    # Accept partial_results as either a dict (id->item), a list of items, or a single string
    results_list = []
    if isinstance(partial_results, dict):
        for k, v in partial_results.items():
            if isinstance(v, dict):
                results_list.append(v.get('result', '') or '')
            elif isinstance(v, str):
                results_list.append(v)
            else:
                results_list.append(str(v))
    elif isinstance(partial_results, list):
        for item in partial_results:
            if isinstance(item, dict):
                results_list.append(item.get('result', '') or '')
            elif isinstance(item, str):
                results_list.append(item)
            else:
                results_list.append(str(item))
    else:
        results_list = [str(partial_results)]

    joined_text = "\n\n".join([r for r in results_list if r])
    
    if not joined_text.strip():
        logging.warning("No content found in partial results. Cannot generate final report.")
        return "{\"multiverse_combined\": \"No data to analyze.\"}"
    
    total_tokens = estimate_tokens(joined_text)
    
    if show_progress:
        if custom_stream_handler:
            custom_stream_handler(f"Starting combined analysis on {total_tokens:,} tokens...")
        print(f"[{_iso_now()}] Starting combined analysis on {total_tokens:,} tokens...")
        print("---")
    print(joined_text[:1000] + "\n...")
    
    # The prompt is modified to request JSON output with clear examples
    prompt = f"""You are an expert analyst. Analyze ONLY the text below and return your analysis as a SINGLE JSON OBJECT. 
DO NOT return any text, headings, or content outside of the JSON structure.

TEXT TO ANALYZE:
{joined_text}

Your entire response must be a single valid JSON object following this exact structure:

{{
  "multiverse_combined": {{
    "executive_summary": "Brief overview of key points from the text",
    "sentiment_analysis": {{
      "positive": {{
        "percentage": 25,
        "reasoning": "Evidence from text showing positive aspects"
      }},
      "negative": {{
        "percentage": 45,
        "reasoning": "Evidence from text showing negative aspects"
      }},
      "neutral": {{
        "percentage": 30,
        "reasoning": "Evidence from text showing neutral aspects"
      }}
    }},
    "topics": [
      "Topic 1",
      "Topic 2"
    ],
    "entities": [
      "Entity 1",
      "Entity 2"
    ],
    "relationships": [
      {{
        "entity1": "First Entity",
        "relationship": "Type of connection",
        "entity2": "Second Entity",
        "description": "Details from text"
      }}
    ],
    "anomalies": [
      {{
        "description": "Any unusual patterns or outliers"
      }}
    ],
    "controversy_score": {{
      "score": 0.7,
      "explanation": "Reasoning from the text"
    }}
  }}
}}

CRITICAL RULES:
1. Output MUST be valid JSON that can be parsed by json.loads()
2. Analysis MUST be based ONLY on the provided text
3. Do not include ANY text outside the JSON structure
4. Use ONLY double quotes for strings
5. Ensure all JSON keys and values are properly quoted
6. Remove any trailing commas
7. Include the exact structure shown above
8. Fill in real values based on your analysis"""

        
    # Use the streaming function for the final report
    response = call_llm_and_stream_to_terminal_and_file(
        llm, 
        prompt, 
        max_tokens=combined_token_budget,
        out_filepath="final_analysis_report.json",
        enable_stream=stream_llm_if_supported,
        custom_stream_handler=custom_stream_handler
    )

    # Clean and validate the response
    response = response.strip()
    
    # Try to extract valid JSON from the response
    try:
        # Remove any leading numbers or text
        fixed_response = response
        if response and not response.startswith('{'):
            # Find the first occurrence of '{'
            first_brace = response.find('{')
            if first_brace != -1:
                fixed_response = response[first_brace:]
        
        # Remove any XML tags
        fixed_response = re.sub(r'<[^>]+>', '', fixed_response)
        
        # Find the start of JSON object
        start_idx = fixed_response.find('{')
        if start_idx == -1:
            raise ValueError("No JSON object found in response")
        
        # Find the end of JSON object
        brace_count = 0
        end_idx = -1
        for i in range(start_idx, len(fixed_response)):
            if fixed_response[i] == '{':
                brace_count += 1
            elif fixed_response[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break
        
        if end_idx == -1:
            raise ValueError("Incomplete JSON object")
        
        # Extract the JSON
        json_str = fixed_response[start_idx:end_idx+1]
        
        # Validate that it's valid JSON
        parsed_json = json.loads(json_str)
        
        # Check for template/placeholder values
        template_indicators = [
            '"type": "string"',
            '"name": "string"',
            '"entity1": "string"',
            '"entity2": "string"',
            '"description": "string"',
            'PERSON|ORGANIZATION|LOCATION|DATE',
            'DO NOT COPY',
            'Example'
        ]
        
        json_str_lower = json_str.lower()
        template_matches = sum(1 for indicator in template_indicators 
                             if indicator.lower() in json_str_lower)
        
        if template_matches >= 2:  # If we find 2 or more template indicators
            logging.warning("Detected template response instead of actual analysis")
            return json.dumps({
                "multiverse_combined": {
                    "error": "Analysis contains template values instead of actual content",
                    "details": "The model returned example/placeholder values instead of real analysis",
                    "raw_response": json_str[:500] + "..." if len(json_str) > 500 else json_str
                }
            }, indent=2)
        
        # If no template detected, ensure expected structure
        if "multiverse_combined" not in parsed_json:
            logging.warning("Output doesn't contain 'multiverse_combined' key. Wrapping response.")
            return json.dumps({"multiverse_combined": parsed_json}, indent=2)
        
        return json.dumps(parsed_json, indent=2)
        
    except (ValueError, json.JSONDecodeError) as e:
        logging.error(f"Failed to extract valid JSON from response: {e}")
        logging.debug(f"Raw response: {response[:500]}...")
        
        # Try to fix common JSON issues
        try:
            # Remove any leading numbers or text
            fixed_response = response
            if response and not response.startswith('{'):
                # Find the first occurrence of '{'
                first_brace = response.find('{')
                if first_brace != -1:
                    fixed_response = response[first_brace:]
            
            # Remove any XML tags
            fixed_response = re.sub(r'<[^>]+>', '', fixed_response)
            
            # Try to parse the fixed response
            parsed_json = json.loads(fixed_response)
            
            # Ensure the output has the expected structure
            if "multiverse_combined" not in parsed_json:
                logging.warning("Output doesn't contain 'multiverse_combined' key. Wrapping response.")
                return json.dumps({"multiverse_combined": parsed_json}, indent=2)
            
            return json.dumps(parsed_json, indent=2)
        except:
            # If all attempts fail, return a simple valid JSON with error message
            return json.dumps({
                "multiverse_combined": {
                    "error": "Failed to generate valid JSON analysis",
                    "raw_response": response[:500] + "..." if len(response) > 500 else response
                }
            }, indent=2)
