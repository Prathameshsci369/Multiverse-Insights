
from flask import Flask, request, jsonify, Response, stream_with_context
import io
import sys
import os
import json
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from test import run_full_analysis

app = Flask(__name__)

# --- API Key for basic security ---
API_KEY = os.environ.get("ANALYSIS_API_KEY", "changeme123")

def validate_api_key(req):
    key = req.headers.get("X-API-KEY")
    return key == API_KEY

def is_json_file(file_storage):
    return file_storage.filename.endswith('.json') and file_storage.mimetype == 'application/json'

def validate_json_content(file_storage, max_size_mb=5):
    file_storage.seek(0, io.SEEK_END)
    size = file_storage.tell()
    file_storage.seek(0)
    if size > max_size_mb * 1024 * 1024:
        return False, "File too large. Limit is {} MB.".format(max_size_mb)
    try:
        json.load(file_storage)
        file_storage.seek(0)
        return True, None
    except Exception as e:
        return False, f"Invalid JSON: {e}"

def progress_stream(files):
    yield json.dumps({"status": "started", "message": "Analysis started."}) + "\n"
    try:
        # Validate files
        for f in files:
            if not is_json_file(f):
                yield json.dumps({"status": "error", "message": f"Invalid file type: {f.filename}"}) + "\n"
                return
            valid, err = validate_json_content(f)
            if not valid:
                yield json.dumps({"status": "error", "message": f"File {f.filename}: {err}"}) + "\n"
                return
        yield json.dumps({"status": "progress", "message": "Files validated. Running analysis..."}) + "\n"
        # Run analysis and stream progress
        # (simulate progress for demonstration)
        for i in range(3):
            time.sleep(1)
            yield json.dumps({"status": "progress", "message": f"Analysis step {i+1}/3..."}) + "\n"
        # Actual analysis
        result = run_full_analysis(files)
        yield json.dumps({"status": "complete", "result": result}) + "\n"
    except GeneratorExit:
        # Client disconnected
        yield json.dumps({"status": "error", "message": "Connection lost during analysis."}) + "\n"
    except Exception as e:
        yield json.dumps({"status": "error", "message": str(e)}) + "\n"

@app.route('/analyze', methods=['POST'])
def analyze_endpoint():
    # API Key check
    if not validate_api_key(request):
        return jsonify({'error': 'Unauthorized. Provide X-API-KEY header.'}), 401
    if 'files' not in request.files:
        return jsonify({'error': 'No files part in the request. Use form field name "files".'}), 400
    files = request.files.getlist('files')
    if not files or len(files) == 0:
        return jsonify({'error': 'No files uploaded.'}), 400
    # Stream progress and result
    return Response(stream_with_context(progress_stream(files)), mimetype='application/json')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
