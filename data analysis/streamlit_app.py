import streamlit as st
import os
import json
import tempfile
from typing import Dict, Any

# Import from our modules
try:
    from ui_components import (
        load_custom_css, 
        display_analysis_results,
        display_sentiment_analysis,
        display_entities,
        display_relationships,
        display_anomalies,
        display_controversy_score
    )
    from data_scrapers import (
        scrape_twitter_data,
        scrape_reddit_data,
        scrape_youtube_data
    )
    from utils1 import (
        clear_cache,
        analyze_data,
        create_sample_data,
        process_uploaded_file
    )
except ImportError:
    st.error("Error: Could not import custom modules (ui_components, data_scrapers, utils1). Please ensure they are in the same directory as the app.")
    # Define dummy functions to prevent crashes if imports fail
    def load_custom_css(): pass
    def display_analysis_results(data): st.write("Analysis results would show here.")
    def scrape_reddit_data(query): return None, "Scraper module not found."
    def scrape_twitter_data(query, start, end): return None, "Scraper module not found."
    def scrape_youtube_data(query, max_results): return None, "Scraper module not found."
    def clear_cache(): return [], ["Cache module not found."]
    def analyze_data(file): return None, "Analysis module not found."
    def create_sample_data(): return "{}"
    def process_uploaded_file(file): return ""

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Multiverse Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Custom CSS for the page ---
def load_page_css():
    """Loads custom CSS for the entire page with animations and hover effects."""
    st.markdown("""
        <style>
            /* Change entire page background */
            body {
                background-color: #0f1419 !important;
            }
            
            /* Override Streamlit's default app background */
            .stApp {
                background-color: #0f1419 !important;
            }
            
            /* Override Streamlit's main content area */
            .main .block-container {
                background-color: #0f1419 !important;
                max-width: 1200px !important;
                padding-top: 2rem !important;
            }
            
            /* Override Streamlit's sidebar */
            .css-1d391kg {
                background-color: #192734 !important;
            }
            
            /* Override Streamlit's sidebar content */
            .css-1lcbmhc {
                background-color: #192734 !important;
            }
            
            /* Horizontal layout container */
            .horizontal-container {
                display: flex !important;
                gap: 2rem !important;
                margin: 2rem 0 !important;
                width: 100% !important;
                justify-content: center !important;
            }
            
            /* Left and right columns */
            .column {
                flex: 1 !important;
                max-width: 500px !important;
            }
            
            /* Main container for the project card */
            .project-card {
                background-color: #192734 !important;
                border-radius: 12px !important;
                padding: 2rem !important;
                border: 1px solid #38444d !important;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3) !important;
                transition: all 0.3s ease !important;
                text-align: center !important;
                height: 100% !important;
                display: flex !important;
                flex-direction: column !important;
            }
            
            /* Hover effect for the project card */
            .project-card:hover {
                transform: translateY(-5px) !important;
                box-shadow: 0 10px 20px rgba(0, 0, 0, 0.4) !important;
                background-color: #22303c !important;
            }
            
            /* Style for the main title inside the card */
            .card-title {
                font-size: 1.8rem !important;
                font-weight: 700 !important;
                color: #1da1f2 !important;
                margin-bottom: 1rem !important;
                transition: color 0.3s ease !important;
            }
            
            /* Hover effect for the title */
            .project-card:hover .card-title {
                color: #1a91da !important;
            }
            
            /* Style for paragraphs inside the card - LARGER, BOLD, WHITE */
            .card-text {
                font-size: 1.1rem !important;
                font-weight: 600 !important;
                color: #ffffff !important;
                line-height: 1.7 !important;
                margin-bottom: 1rem !important;
                background-color: #38444d !important;
                padding: 1rem !important;
                border-radius: 8px !important;
                flex-grow: 1 !important;
            }
            
            /* Style for the features section */
            .features-section {
                margin-top: 1.5rem !important;
                flex-grow: 1 !important;
                display: flex !important;
                flex-direction: column !important;
            }
            
            /* Style for the features title */
            .features-title {
                font-size: 1.5rem !important;
                font-weight: 700 !important;
                color: #1da1f2 !important;
                margin-bottom: 1rem !important;
                transition: color 0.3s ease !important;
            }
            
            /* Hover effect for the features title */
            .project-card:hover .features-title {
                color: #1a91da !important;
            }
            
            /* Style for the features list */
            .features-list {
                list-style-type: none;
                padding: 0;
                margin: 0;
                flex-grow: 1 !important;
            }
            
            /* Style for individual feature items - DARK BACKGROUND */
            .feature-item {
                background-color: #38444d !important;
                border-radius: 8px !important;
                padding: 0.75rem 1rem !important;
                margin-bottom: 0.5rem !important;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
                transition: all 0.3s ease !important;
                position: relative !important;
                overflow: hidden !important;
                color: #ffffff !important;
                font-weight: 600 !important;
                font-size: 1rem !important;
            }
            
            /* Hover effect for feature items */
            .feature-item:hover {
                transform: translateX(10px) !important;
                background-color: #4a5568 !important;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3) !important;
            }
            
            /* Animation for feature items */
            @keyframes fadeInUp {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            /* Apply animation to feature items with delay */
            .feature-item:nth-child(1) { animation: fadeInUp 0.5s ease-out 0.1s both !important; }
            .feature-item:nth-child(2) { animation: fadeInUp 0.5s ease-out 0.2s both !important; }
            .feature-item:nth-child(3) { animation: fadeInUp 0.5s ease-out 0.3s both !important; }
            .feature-item:nth-child(4) { animation: fadeInUp 0.5s ease-out 0.4s both !important; }
            .feature-item:nth-child(5) { animation: fadeInUp 0.5s ease-out 0.5s both !important; }
            
            /* Add a subtle gradient background to the feature items */
            .feature-item::before {
                content: '' !important;
                position: absolute !important;
                top: 0 !important;
                left: 0 !important;
                width: 5px !important;
                height: 100% !important;
                background: linear-gradient(to bottom, #1da1f2, #0d8bd9) !important;
                opacity: 0 !important;
                transition: opacity 0.3s ease !important;
            }
            
            /* Show the gradient on hover */
            .feature-item:hover::before {
                opacity: 1 !important;
            }
            
            /* Center the entire content */
            .center-content {
                display: flex !important;
                flex-direction: column !important;
                align-items: center !important;
                justify-content: center !important;
            }
            
            /* Override Streamlit's default container styles */
            div[data-testid="stVerticalBlock"] {
                width: 100% !important;
            }
            
            /* Style for tabs */
            .stTabs [data-baseweb="tab-list"] {
                background-color: #192734 !important;
                border-radius: 8px !important;
            }
            
            .stTabs [data-baseweb="tab"] {
                color: #ffffff !important;
            }
            
            /* Style for tab content */
            .stTabs [data-baseweb="tab-panel"] {
                background-color: #192734 !important;
                border-radius: 8px !important;
                padding: 1rem !important;
            }
            
            /* Style for headers */
            h1, h2, h3, h4, h5, h6 {
                color: #ffffff !important;
            }
            
            /* Style for text */
            p, div, span {
                color: #ffffff !important;
            }
            
            /* Style for buttons */
            .stButton > button {
                background-color: #1da1f2 !important;
                color: #ffffff !important;
                border: none !important;
                border-radius: 4px !important;
                font-weight: 600 !important;
            }
            
            .stButton > button:hover {
                background-color: #1a91da !important;
            }
            
            /* Style for inputs */
            .stTextInput > div > div > input {
                background-color: #38444d !important;
                color: #ffffff !important;
                border: 1px solid #4a5568 !important;
            }
            
            /* Style for selectboxes */
            .stSelectbox > div > div > select {
                background-color: #38444d !important;
                color: #ffffff !important;
                border: 1px solid #4a5568 !important;
            }
            
            /* Style for sliders */
            .stSlider > div > div > div {
                background-color: #1da1f2 !important;
            }
            
            /* Style for file uploaders */
            .stFileUploader > div > div {
                background-color: #38444d !important;
                border: 1px dashed #4a5568 !important;
            }
            
            /* Style for info boxes */
            .stInfo {
                background-color: #192734 !important;
                border-left-color: #1da1f2 !important;
            }
            
            /* Style for success boxes */
            .stSuccess {
                background-color: #192734 !important;
                border-left-color: #17bf63 !important;
            }
            
            /* Style for warning boxes */
            .stWarning {
                background-color: #192734 !important;
                border-left-color: #ffad1f !important;
            }
            
            /* Style for error boxes */
            .stError {
                background-color: #192734 !important;
                border-left-color: #e0245e !important;
            }
        </style>
    """, unsafe_allow_html=True)

# --- 5. Main Application Logic ---

def main():
    # Load the custom CSS
    load_page_css()
    load_custom_css()
    
    # Create a centered container for the main title
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; flex-direction: column; margin-bottom: 2rem;">
        <h1 style="font-size: 2.5rem; font-weight: 700; color: #1da1f2; text-align: center;">📊 Multiverse Insights</h1>
        <p style="font-size: 1.2rem; color: #ffffff; text-align: center;">A Real-time Social Media Analyzer</p>
    </div>
    """, unsafe_allow_html=True)

    # Create horizontal layout for Project Overview and Key Capabilities
    st.markdown('<div class="horizontal-container">', unsafe_allow_html=True)
    
    # Left Column - Project Overview
    st.markdown('<div class="column">', unsafe_allow_html=True)
    st.markdown('<div class="project-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="card-title">Project Overview</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="center-content">
    <p class="card-text">
    Multiverse Insights is an advanced data analysis and intelligence platform 
    designed to process massive, multilingual, and unstructured data from diverse 
    online sources like Reddit, Twitter, and Telegram.
    </p>
    
    <p class="card-text">
    It integrates automated scraping, translation, sliding-window summarization, 
    and sentiment analysis to extract global insights and evolving narratives 
    from continuous data streams.
    </p>

    <p class="card-text">
    Built for scalability and accuracy, it combines AI-driven summarization 
    with local processing—making large-scale document and social data 
    analysis possible even on minimal hardware.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Right Column - Key Capabilities
    st.markdown('<div class="column">', unsafe_allow_html=True)
    st.markdown('<div class="project-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="card-title">Key Capabilities</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="features-section">', unsafe_allow_html=True)
    
    # Create animated feature items
    features = [
        "Process massive, multilingual, and unstructured data",
        "Automated scraping from Reddit, Twitter, and Telegram",
        "Sliding-window summarization for continuous data streams",
        "AI-driven sentiment analysis",
        "Scalable and accurate local processing"
    ]
    
    for feature in features:
        st.markdown(f'<div class="feature-item">{feature}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Initialize session state
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    
    # Add tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Social Media Analysis", "Custom File Analysis", "Settings"])
    
    with tab1:
        st.markdown("### 🔍 Social Media Search and Analysis")
        
        # Social media platform selection
        platform = st.radio(
            "Choose your platform:",
            ["Reddit", "Twitter", "YouTube"],
            horizontal=True,
            help="Select which social media platform to search"
        )
        
        # Search input
        search_query = st.text_input(
            f"Enter your search query for {platform}:", 
            help=f"Enter what you want to search for on {platform}"
        )
        
        # Date range for Twitter and YouTube
        if platform in ["Twitter", "YouTube"]:
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date (optional)", None)
            with col2:
                if platform == "YouTube":
                     end_date = st.slider("Number of results", min_value=1, max_value=20, value=10)
                else:
                     end_date = st.date_input("End Date (optional)", None)
        
        if st.button(f"Search {platform} & Analyze"):
            if search_query:
                with st.spinner(f"Fetching data from {platform}..."):
                    if platform == "Reddit":
                        filename, error = scrape_reddit_data(search_query)
                    elif platform == "Twitter":
                        start_date_str = start_date.strftime("%Y-%m-%d") if start_date else None
                        end_date_str = end_date.strftime("%Y-%m-%d") if end_date else None
                        filename, error = scrape_twitter_data(search_query, start_date_str, end_date_str)
                    else:  # YouTube
                        max_results = end_date
                        filename, error = scrape_youtube_data(search_query, max_results=max_results)
                    
                    if error:
                        st.error(error)
                    else:
                        st.success(f"Data saved to {filename}")
                        
                        with st.spinner("Analyzing the data..."):
                            parsed_data, error = analyze_data(filename)
                            if error:
                                st.error(error)
                            else:
                                st.session_state.analysis_data = parsed_data
                                st.session_state.analysis_complete = True
                                st.success("Analysis complete! View results below.")
            else:
                st.warning("Please enter a search query.")
        
        if st.session_state.get('analysis_complete'):
            st.markdown("---")
            st.markdown("### 📊 Analysis Results")
            
            analysis_data = st.session_state.analysis_data.get("multiverse_combined", {})
            display_analysis_results(analysis_data)
    
    with tab2:
        st.markdown("### 📂 Custom File Analysis")
        uploaded_file = st.file_uploader(
            "Upload a JSON file for analysis",
            type=["json"],
            help="Upload a JSON file containing text data to be analyzed"
        )
        
        if uploaded_file is not None:
            st.sidebar.subheader("File Information")
            st.sidebar.write(f"Filename: {uploaded_file.name}")
            st.sidebar.write(f"File size: {uploaded_file.size / 1024:.2f} KB")
            
            tmp_file_path = process_uploaded_file(uploaded_file)
            
            try:
                if st.button("Clear Cache & Analyze"):
                    cleared_files, errors = clear_cache()
                    if cleared_files:
                        st.success(f"Cleared cache files: {', '.join(cleared_files)}")
                    if errors:
                        for error in errors:
                            st.warning(error)
                    st.success("Cache cleared! Starting fresh analysis...")
                
                with st.spinner("Analyzing your data... This may take a few minutes."):
                    parsed_data, error = analyze_data(tmp_file_path)
                    
                    if error:
                        st.error(error)
                    else:
                        st.success("Analysis complete!")
                        
                        analysis_data = None
                        
                        if "multiverse_combined" in parsed_data:
                            analysis_data = parsed_data["multiverse_combined"]
                        elif "analysis_results" in parsed_data:
                            analysis_data = parsed_data["analysis_results"]
                        else:
                            analysis_data = parsed_data
                        
                        if st.session_state.debug_mode:
                            st.markdown('### 🐛 Debug Information')
                            st.write("Parsed data structure:")
                            st.json(parsed_data)
                        
                        if analysis_data:
                            display_analysis_results(analysis_data)
                        else:
                            st.error("No valid analysis data found. Please check your input file and try again.")
            
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                st.write("Please check your input file and try again.")
            
            finally:
                if tmp_file_path and os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
        else:
            st.markdown("""
            ## Welcome to the Text Analysis Dashboard! 📊
            
            This tool analyzes text data from JSON files and provides insights including:
            - Executive summaries
            - Social media search and analysis
            - Sentiment analysis
            - Key topics
            - Entity recognition
            - Relationship extraction
            - Anomaly detection
            - Controversy scoring
            
            ### How to use:
            1. Upload a JSON file containing text data
            2. Click "Clear Cache & Analyze" to start the analysis
            3. View the results below
            
            ### Troubleshooting:
            If you encounter issues with the analysis, try clicking "Clear Cache & Restart Analysis" before uploading your file again.
            """)
            
            st.markdown("### Need a sample file?")
            st.markdown("Download a sample JSON file to test the analysis:")
            
            st.download_button(
                label="Download Sample JSON",
                data=create_sample_data(),
                file_name="sample_data.json",
                mime="application/json"
            )
    
    with tab3:
        st.markdown("### ⚙️ Settings")
        
        st.subheader("Twitter Settings")
        cookies_status = "✅ Found" if os.path.exists("twitter_cookies.json") else "❌ Not Found"
        st.info(f"Twitter Cookies Status: {cookies_status}")
        
        uploaded_cookies = st.file_uploader(
            "Upload Twitter Cookies File (twitter_cookies.json)",
            type=["json"],
            help="Upload your Twitter cookies file for authentication"
        )
        
        if uploaded_cookies:
            with open("twitter_cookies.json", "wb") as f:
                f.write(uploaded_cookies.getvalue())
            st.success("Twitter cookies file updated successfully!")
        
        st.subheader("YouTube Settings")
        st.slider("Default number of results", min_value=1, max_value=20, value=10, key="youtube_max_results")
        
        cookies_status = "✅ Found" if os.path.exists("youtube_cookies.json") else "❌ Not Found"
        st.info(f"YouTube Cookies Status: {cookies_status}")
        
        uploaded_cookies = st.file_uploader(
            "Upload YouTube Cookies File (youtube_cookies.json)",
            type=["json"],
            help="Upload your YouTube cookies file for authentication"
        )
        
        if uploaded_cookies:
            with open("youtube_cookies.json", "wb") as f:
                f.write(uploaded_cookies.getvalue())
            st.success("YouTube cookies file updated successfully!")
    
    st.sidebar.title("Analysis Settings")
    
    st.sidebar.subheader("Analysis Options")
    show_raw_output = st.sidebar.checkbox("Show Raw Output", value=False)
    save_results = st.sidebar.checkbox("Save Results to File", value=True)
    st.session_state.debug_mode = st.sidebar.checkbox("Debug Mode", value=False)
    
    if st.sidebar.button("Clear Cache & Restart Analysis"):
        cleared_files, errors = clear_cache()
        if cleared_files:
            st.sidebar.success(f"Cleared cache files: {', '.join(cleared_files)}")
        if errors:
            for error in errors:
                st.sidebar.warning(error)
        st.rerun()

if __name__ == "__main__":
    main()
