import os
import json
from datetime import datetime
from reddit import RedditScraper
from advance_twitter import TwitterScraper
from youtube import YouTubeScraper

def scrape_reddit_data(query):
    """
    Scrapes Reddit for a given query and saves the results to a JSON file.

    Args:
        query (str): The search query for Reddit.

    Returns:
        tuple: A tuple containing (file_path, error_message).
               file_path is the path to the saved JSON file on success, otherwise None.
               error_message is None on success, otherwise a string describing the error.
    """
    try:
        scraper = RedditScraper()
        if not scraper.reddit:
            return None, "Failed to initialize Reddit scraper. Please check credentials."
        
        final_query = query     
        
        scraped_data = scraper.search_and_fetch(final_query, limit=10)
        
        if not scraped_data:
            return None, "No data found for the given query."
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"reddit_search_{timestamp}.json"
        
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(scraped_data, f, indent=4, ensure_ascii=False)
            
        return os.path.abspath(output_filename), None
    except Exception as e:
        return None, f"Error during Reddit scraping: {str(e)}"

def scrape_twitter_data(query, start_date=None, end_date=None):
    """
    Scrapes Twitter for a given query and saves the results to a JSON file.

    Args:
        query (str): The search query for Twitter.
        start_date (str, optional): The start date for the search (YYYY-MM-DD). Defaults to None.
        end_date (str, optional): The end date for the search (YYYY-MM-DD). Defaults to None.

    Returns:
        tuple: A tuple containing (file_path, error_message).
               file_path is the path to the saved JSON file on success, otherwise None.
               error_message is None on success, otherwise a string describing the error.
    """
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"twitter_{timestamp}.json"
        
        scraper = TwitterScraper(
            search_query=query,
            start_date=start_date,
            end_date=end_date,
            json_output=filename
        )
        
        scraper.run_pipeline()
        
        if os.path.exists(filename):
            abs_filename = os.path.abspath(filename)
            return abs_filename, None
        else:
            outputs_path = os.path.join("outputs", filename)
            if os.path.exists(outputs_path):
                abs_outputs_path = os.path.abspath(outputs_path)
                import shutil
                shutil.move(abs_outputs_path, filename)
                abs_filename = os.path.abspath(filename)
                return abs_filename, None
            else:
                return None, f"Twitter scraping completed but output file '{filename}' was not found."
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error during Twitter scraping: {str(e)}"

def scrape_youtube_data(query, max_results=10):
    """
    Scrapes YouTube for a given query and saves the results to a JSON file.

    Args:
        query (str): The search query for YouTube.
        max_results (int, optional): The maximum number of video results to fetch. Defaults to 10.

    Returns:
        tuple: A tuple containing (file_path, error_message).
               file_path is the path to the saved JSON file on success, otherwise None.
               error_message is None on success, otherwise a string describing the error.
    """
    try:
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"youtube_{timestamp}.json"
        output_path = os.path.join(output_dir, filename)
        
        scraper = YouTubeScraper()
        videos = scraper.fetch_youtube_videos(query, max_results=max_results)
        
        if not videos:
            return None, "No videos found for the given query."
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(videos, f, indent=4, ensure_ascii=False)
            
        return os.path.abspath(output_path), None
    except Exception as e:
        return None, f"Error during YouTube scraping: {str(e)}"
