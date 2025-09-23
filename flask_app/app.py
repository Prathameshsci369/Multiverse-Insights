import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, request, send_file, url_for
import json
import csv
from reddit import RedditScraper
from main_youtube import YouTubeDataExtractor
from advance_twitter import TwitterScraper

app = Flask(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

PLATFORMS = ["Reddit", "YouTube", "Twitter"]

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

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "File not found", 404

if __name__ == '__main__':
    app.run(debug=True)
