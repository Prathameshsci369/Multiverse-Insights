import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import os
from datetime import datetime
import time
import ast
import re
import asyncio
import nest_asyncio
import pandas as pd
from reddit import RedditScraper
from main_youtube import YouTubeDataExtractor
from twitter_playwright import TwitterPDFScraper
from telegram_scraper import TelegramScraper

# Configure environment
os.environ['PYTORCH_JIT'] = '0'  # Disable JIT
nest_asyncio.apply()

# Basic page config
st.set_page_config(
    page_title="Multiverse Insights",
    page_icon="🌌",
    layout="wide"
)

# Custom CSS for styling with animations
st.markdown("""
<style>
    @keyframes glowing {
        0% { text-shadow: 0 0 5px #FF4B4B; }
        50% { text-shadow: 0 0 20px #FF4B4B, 0 0 30px #FF4B4B; }
        100% { text-shadow: 0 0 5px #FF4B4B; }
    }
    
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    @keyframes borderGlow {
        0% { box-shadow: 0 0 5px #FF4B4B; }
        50% { box-shadow: 0 0 20px #FF4B4B; }
        100% { box-shadow: 0 0 5px #FF4B4B; }
    }
    
    .title {
        font-size: 3.5em;
        color: #FF4B4B;
        text-align: center;
        font-weight: bold;
        padding: 10px 0;
        margin-bottom: 20px;
        animation: glowing 2s ease-in-out infinite, float 3s ease-in-out infinite;
        display: inline-block;
    }
    
    .summary-box {
        background-color: #262730;
        border-radius: 10px;
        padding: 25px;
        margin: 20px 0;
        border: 2px solid #FF4B4B;
        transition: all 0.3s ease;
    }
    
    .summary-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(255,75,75,0.2);
        border-color: #FF6B6B;
    }
    
    .feature-list {
        list-style-type: none;
        padding-left: 10px;
    }
    
    .feature-list li {
        margin: 12px 0;
        padding: 12px;
        background-color: #0E1117;
        border-radius: 8px;
        border-left: 4px solid #FF4B4B;
        color: #FAFAFA;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .feature-list li:hover {
        transform: translateX(10px);
        background-color: #1A1E23;
        border-left-width: 8px;
    }
    
    .welcome-text {
        color: #FF4B4B;
        font-size: 1.5em;
        margin-bottom: 15px;
        animation: glowing 2s ease-in-out infinite;
        display: inline-block;
    }
    
    .summary-text {
        color: #FAFAFA;
        font-size: 1.1em;
        margin-bottom: 15px;
        transition: color 0.3s ease;
    }
    
    .summary-text:hover {
        color: #FF4B4B;
    }
    
    .title-container {
        text-align: center;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'query' not in st.session_state:
    st.session_state.query = ""

if 'platforms' not in st.session_state:
    st.session_state.platforms = {
        "Reddit": False,
        "YouTube": False,
        "Twitter": False,
        "Telegram": False
    }

def check_json_files():
    """Check for and return a list of available JSON files."""
    json_files = [
        "reddit_data.json",
        "tweets_output.json",
        "youtube_search_output.json",
        "telegram_data.json"
    ]
    return [f for f in json_files if os.path.exists(f)]

def run_platform_search(platform, query, start_date=None, end_date=None):
    """Run search for a specific platform."""
    try:
        if platform == "Reddit":
            scraper = RedditScraper()
            if scraper.reddit:
                results = scraper.search_and_fetch_top_posts(query, limit=5)
                if results:
                    scraper.dump_to_json(results)
                    return True
        elif platform == "YouTube":
            extractor = YouTubeDataExtractor()
            videos = extractor.fetch_and_process_videos(
                query,
                max_results=5,
                start_date=start_date,
                end_date=end_date
            )
            if videos:
                extractor.dump_to_json(videos, output_file="youtube_search_output.json")
                return True
        elif platform == "Twitter":
            scraper = TwitterPDFScraper(
                search_query=query,
                start_date=str(start_date) if start_date else None,
                end_date=str(end_date) if end_date else None
            )
            scraper.run_pipeline()
            return True
        elif platform == "Telegram":
            scraper = TelegramScraper(keywords=[query])
            asyncio.run(scraper.run())
            return True
        return False
    except Exception as e:
        st.error(f"Error in {platform} search: {str(e)}")
        return False

def display_results(files):
    """Display search results and visualizations."""
    if not files:
        st.info("No data available yet. Start searching to see results.")
        return

    # Show metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Available Files", len(files))
    with col2:
        try:
            total_records = sum(len(json.load(open(f))) for f in files)
            st.metric("Total Records", total_records)
        except:
            st.metric("Total Records", "Error")

    # Data preview
    st.subheader("Data Preview")
    for file in files:
        with st.expander(f"Preview {file}"):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        df = pd.DataFrame(data)
                        st.dataframe(df)
                    else:
                        st.json(data)
            except Exception as e:
                st.error(f"Error reading {file}: {str(e)}")

def main():
    # Adding space for better visibility
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Animated title with continuous glow and float effect
    st.markdown("""
        <div class="title-container">
            <h1 class="title">🌌 Multiverse Insights ⚡</h1>
        </div>
    """, unsafe_allow_html=True)
    
    # Summary Box with enhanced visibility
    st.markdown("""
    <div class="summary-box">
        <h3 class="welcome-text">⚡ Welcome to Multiverse Insights! ⚡</h3>
        <p class="summary-text">Your ultimate social media data analysis powerhouse that brings you:</p>
        <ul class="feature-list">
            <li>🔍 Multi-Platform Search Engine: Unified search across all major social platforms</li>
            <li>📊 Advanced Analytics: Deep dive into trends and discussions</li>
            <li>📅 Time Travel: Filter content by custom date ranges</li>
            <li>📈 Smart Visualization: Interactive data previews and insights</li>
        </ul>
        <p class="summary-text">Ready to explore? Start your journey below! 🚀</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2 = st.tabs(["Search", "Results"])
    
    with tab1:
        # Search query
        st.subheader("Enter Search Query")
        query = st.text_input(
            "What would you like to search for?",
            value=st.session_state.query,
            help="Enter keywords to search across platforms"
        )
        
        # Platform selection
        st.subheader("Choose Platforms")
        col1, col2 = st.columns(2)
        
        platforms = {
            "Reddit": "Search Reddit discussions",
            "YouTube": "Search YouTube videos",
            "Twitter": "Search Twitter posts",
            "Telegram": "Search Telegram channels"
        }
        
        for i, (platform, description) in enumerate(platforms.items()):
            with col1 if i % 2 == 0 else col2:
                st.session_state.platforms[platform] = st.checkbox(
                    platform,
                    value=st.session_state.platforms[platform],
                    help=description
                )
        
        # Date range
        st.subheader("Date Range (Optional)")
        start_date = st.text_input("Start Date (YYYY-MM-DD)", "")
        end_date = st.text_input("End Date (YYYY-MM-DD)", "")
        
        # Validate date format if dates are provided
        date_error = False
        if start_date or end_date:
            try:
                if start_date:
                    start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
                if end_date:
                    end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
            except ValueError:
                st.error("Please enter dates in YYYY-MM-DD format")
                date_error = True
        
        # Search button
        if st.button("Start Search"):
            if not query:
                st.error("Please enter a search query")
            elif not any(st.session_state.platforms.values()):
                st.warning("Please select at least one platform")
            elif date_error:
                st.error("Please fix the date format errors before searching")
            else:
                selected_platforms = [p for p, selected in st.session_state.platforms.items() if selected]
                progress = st.progress(0)
                status = st.empty()
                
                for i, platform in enumerate(selected_platforms):
                    status.info(f"Searching {platform}...")
                    success = run_platform_search(platform, query, start_date, end_date)
                    progress.progress((i + 1) / len(selected_platforms))
                    
                    if success:
                        st.success(f"✅ {platform} search complete")
                    else:
                        st.error(f"❌ Error searching {platform}")
                
                status.success("Search complete!")
                st.balloons()
    
    with tab2:
        st.subheader("Search Results")
        with st.spinner('Loading Results...'):
            files = check_json_files()
            display_results(files)

if __name__ == "__main__":
    main()
