import streamlit as st
import os
import json
import tempfile
import logging
from typing import Dict, Any
import pandas as pd
import altair as alt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from our modules
from ui_components import (
    load_custom_css, 
    load_page_css,
    display_analysis_results,
    display_entities,
    display_relationships,
    display_anomalies,
    display_controversy_score,
    initialize_chatbot,
    display_chatbot_interface,
    load_sample_data_to_ui,
    create_sentiment_chart,
    handle_social_media_analysis,
    handle_custom_file_analysis,
    handle_settings,
    handle_sidebar
)

# Import data processing modules
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

# Import the RAG chatbot
from rag_chatbot import RAGChatbot

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Multiverse Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 5. Main Application Logic ---

def main():
    # Load the custom CSS
    load_page_css()
    load_custom_css()
    
    # Initialize all session state variables
    initialize_session_state()
    
    # Check if we're in chat mode
    if st.session_state.get('chat_mode', False):
        display_chat_page()
        return
    
    # Display the main title and overview
    display_main_title()
    display_project_overview()
    display_key_capabilities()
    display_applications()
    
    # Add tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Social Media Analysis", "Custom File Analysis", "Settings"])
    
    with tab1:
        handle_social_media_analysis()
    
    with tab2:
        handle_custom_file_analysis()
    
    with tab3:
        handle_settings()
    
    # Handle sidebar
    handle_sidebar()

def display_chat_page():
    """Display the dedicated chat page with enhanced UI"""
    # Set page title
    st.markdown("# 🤖 Chat with your Data")
    
    # Display the chatbot interface
    display_chatbot_interface()

def initialize_session_state():
    """Initialize all session state variables"""
    session_vars = {
        'debug_mode': False,
        'analysis_complete': False,
        'analysis_data': None,
        'current_json_file': None,
        'file_uploaded': False,
        'current_view': None,
        'analysis_in_progress': False,
        'uploaded_file_path': None,
        'chat_messages': [],  # Initialize chat messages
        'chat_input': "",  # Initialize chat input
        'chat_mode': False  # Initialize chat mode
    }
    
    for var, default in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default
            logger.info(f"Initialized session state variable: {var}")

def display_main_title():
    """Display the main title with animation"""
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; flex-direction: column; margin-bottom: 2rem;">
        <h1 class="animated-title" style="font-size: 2.5rem; font-weight: 700; color: #1da1f2; text-align: center;">📊 Multiverse Insights</h1>
        <p style="font-size: 1.2rem; color: #ffffff; text-align: center;">A Real-time Social Media Analyzer</p>
    </div>
    """, unsafe_allow_html=True)

def display_project_overview():
    """Display the project overview in three columns"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h2 class="card-title">Track Trends</h2>
            <p class="card-text">
            Discover what's trending across social media platforms in real-time. 
            Stay ahead of the curve by identifying emerging topics before they go viral.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h2 class="card-title">Understand Sentiment</h2>
            <p class="card-text">
            Gauge public opinion and emotional responses to brands, products, or events. 
            Make data-driven decisions based on how people truly feel.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card">
            <h2 class="card-title">Competitive Edge</h2>
            <p class="card-text">
            Monitor your competitors' performance and reception. 
            Identify opportunities and threats in your market landscape.
            </p>
        </div>
        """, unsafe_allow_html=True)

def display_key_capabilities():
    """Display key capabilities in five columns"""
    st.markdown("### Key Capabilities")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    capabilities = [
        ("Multi-Platform", "Analyze content from Reddit, Twitter, YouTube, and more in one place"),
        ("Real-Time", "Get instant insights as conversations happen across platforms"),
        ("Visual Reports", "Transform complex data into easy-to-understand visualizations"),
        ("Smart Alerts", "Receive notifications when important trends or sentiments emerge"),
        ("Easy Export", "Share your findings with customizable reports and data exports")
    ]
    
    for i, (title, text) in enumerate(capabilities):
        with [col1, col2, col3, col4, col5][i]:
            st.markdown(f"""
            <div class="card">
                <h2 class="card-title">{title}</h2>
                <p class="card-text">{text}</p>
            </div>
            """, unsafe_allow_html=True)

def display_applications():
    """Display applications in three columns"""
    st.markdown("### Applications")
    col1, col2, col3 = st.columns(3)
    
    applications = [
        ("For Marketers", "Track campaign performance, identify influencer partnerships, and understand audience reactions to your messaging"),
        ("For Researchers", "Gather comprehensive data for academic studies, market research, and social science analysis"),
        ("For Businesses", "Monitor brand reputation, customer feedback, and industry trends to make informed decisions")
    ]
    
    for i, (title, text) in enumerate(applications):
        with [col1, col2, col3][i]:
            st.markdown(f"""
            <div class="card">
                <h2 class="card-title">{title}</h2>
                <p class="card-text">{text}</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
