# utils.py
import os
import json
import tempfile
from main2 import combine_analysis
from simple_parser import parse_simple_format

def clear_cache():
    """Clear all cache files to ensure fresh analysis"""
    cache_files = [
        "partial_results_main.json",
        "raw_output.txt",
        "final_output.txt",
        "analysis_results.json"
    ]
    
    cleared_files = []
    errors = []
    
    for file in cache_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                cleared_files.append(file)
            except Exception as e:
                errors.append(f"Could not clear {file}: {str(e)}")
    
    return cleared_files, errors

def analyze_data(file_path, save_results=True):
    """Analyze data from a file and return parsed results"""
    try:
        # Call the analysis function
        raw_output = combine_analysis(file_path)
        
        # Check if the analysis was successful
        if not raw_output or raw_output.strip() == "":
            return None, "Analysis failed to produce output. Please try clearing the cache and running the analysis again."
        
        # Parse the output
        parsed_data = parse_simple_format(raw_output)
        
        # Save results if requested
        if save_results:
            with open("analysis_results.json", "w", encoding="utf-8") as f:
                json.dump(parsed_data, f, indent=4, ensure_ascii=False)
        
        return parsed_data, None
    except Exception as e:
        return None, f"Error during analysis: {str(e)}"

def create_sample_data():
    """Create sample JSON data for testing"""
    sample_data = {
        "articles": [
            {
                "title": "Sample Article 1",
                "content": "This is a sample article about technology and its impact on society. It discusses both positive and negative aspects."
            },
            {
                "title": "Sample Article 2",
                "content": "Another sample article focusing on environmental issues and climate change. It presents a critical view of current policies."
            }
        ]
    }
    
    # Convert to JSON string
    return json.dumps(sample_data, indent=2)

def process_uploaded_file(uploaded_file):
    """Process an uploaded file and return the path to a temporary file"""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name
