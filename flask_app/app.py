import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, request, send_file, url_for, redirect
import json
import csv
from reddit import RedditScraper
from main_youtube import YouTubeDataExtractor
from advance_twitter import TwitterScraper
import requests
import time

app = Flask(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

PLATFORMS = ["Reddit", "YouTube", "Twitter"]

ANALYSIS_API_URL = "https://refactored-chainsaw-jxgjxgwrj5p3qj6w-5000.app.github.dev/analyze"
ANALYSIS_API_KEY = os.environ.get("ANALYSIS_API_KEY", "changeme123")

def save_results(platform, data):
    json_path = os.path.join(RESULTS_DIR, f"{platform.lower()}_results.json")
    csv_path = os.path.join(RESULTS_DIR, f"{platform.lower()}_results.csv")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    if isinstance(data, list) and data:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
    return json_path, csv_path

# Helper to get all JSON files in the results directory
def get_all_json_files():
    return [os.path.join(RESULTS_DIR, f) for f in os.listdir(RESULTS_DIR) if f.endswith('.json')]

def send_files_for_analysis():
    files = get_all_json_files()
    files_payload = [('files', (os.path.basename(f), open(f, 'rb'), 'application/json')) for f in files]
    headers = {"X-API-KEY": ANALYSIS_API_KEY}
    try:
        response = requests.post(ANALYSIS_API_URL, files=files_payload, headers=headers, timeout=600)  # Wait up to 10 minutes
        if response.status_code == 200:
            # Save the analysis result
            analysis_result_path = os.path.join(RESULTS_DIR, 'analysis_result.json')
            with open(analysis_result_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            return analysis_result_path, response.json()
        else:
            return None, {"error": f"API returned status {response.status_code}: {response.text}"}
    except requests.Timeout:
        return None, {"error": "Analysis API timed out. Please try again later."}
    except Exception as e:
        return None, {"error": str(e)}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/home', methods=['GET', 'POST'])
def index():
    results = {}
    download_links = {}
    query = ''
    selected_platforms = []
    error = None
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        selected_platforms = request.form.getlist('platforms')
        if not query or not selected_platforms:
            error = "Please enter a query and select at least one platform."
        else:
            for platform in selected_platforms:
                if platform == "Reddit":
                    scraper = RedditScraper()
                    if scraper.reddit:
                        data = scraper.search_and_fetch_top_posts(query, limit=5)
                        json_path, csv_path = save_results('reddit', data)
                        results['Reddit'] = data
                        download_links['Reddit'] = {'json': url_for('download_file', filename=os.path.basename(json_path)), 'csv': url_for('download_file', filename=os.path.basename(csv_path))}
                elif platform == "YouTube":
                    extractor = YouTubeDataExtractor()
                    data = extractor.fetch_and_process_videos(query, max_results=5)
                    json_path, csv_path = save_results('youtube', data)
                    results['YouTube'] = data
                    download_links['YouTube'] = {'json': url_for('download_file', filename=os.path.basename(json_path)), 'csv': url_for('download_file', filename=os.path.basename(csv_path))}
                elif platform == "Twitter":
                    output_dir = RESULTS_DIR
                    scraper = TwitterScraper(
                        search_query=query,
                        cookies_path=os.path.join("twitter_cookies.json"),
                        json_output="twitter_results.json",
                        output_dir=output_dir
                    )
                    scraper.run_pipeline()
                    json_path = os.path.join(output_dir, "twitter_results.json")
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                    except Exception:
                        data = []
                    json_path, csv_path = save_results('twitter', data)
                    results['Twitter'] = data
                    download_links['Twitter'] = {'json': url_for('download_file', filename=os.path.basename(json_path)), 'csv': url_for('download_file', filename=os.path.basename(csv_path))}
    return render_template('index.html', platforms=PLATFORMS, results=results, download_links=download_links, query=query, selected_platforms=selected_platforms, error=error)

@app.route('/report')
@app.route('/report/<section>')
def report(section=None):
    if section is None:
        section = 'sentiment'
    return render_template('report.html', section=section)

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "File not found", 404

@app.route('/run_analysis')
def run_analysis():
    # Show a waiting page while analysis is running
    from flask import render_template_string
    waiting_html = '''
    <html><head><title>Running Analysis...</title></head>
    <body style="font-family:Arial;text-align:center;padding-top:80px;">
    <h2>Running analysis on all data...<br>Please wait, this may take several minutes.</h2>
    <div style="margin-top:40px;font-size:1.5em;">⏳</div>
    </body></html>
    '''
    # Render waiting page immediately
    import threading
    import flask
    def do_analysis():
        path, result = send_files_for_analysis()
        flask.session['analysis_result_path'] = path
        flask.session['analysis_result'] = result
    thread = threading.Thread(target=do_analysis)
    thread.start()
    # Wait for thread to finish (blocking, but shows waiting page)
    thread.join()
    # After analysis, redirect to report page
    return flask.redirect(flask.url_for('report', section='sentiment'))

if __name__ == '__main__':
    app.run(debug=True)
