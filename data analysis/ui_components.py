# ui_components.py
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List
import streamlit as st
import os
import json
import tempfile
import logging
from typing import Dict, Any
import pandas as pd
import altair as alt

import streamlit as st
import os
import json
import tempfile
import logging
from typing import Dict, Any
import pandas as pd
import altair as alt

# Import from utils1 for functions used in ui_components
from utils1 import (
    clear_cache,
    analyze_data,
    create_sample_data,
    process_uploaded_file
)

# Import data processing modules
from data_scrapers import (
    scrape_twitter_data,
    scrape_reddit_data,
    scrape_youtube_data
)

# Import the RAG chatbot
from rag_chatbot import RAGChatbot

# Set up logging
logger = logging.getLogger(__name__)
def load_custom_css():
    """Load custom CSS for the application"""
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
        }
        .section-header {
            font-size: 1.5rem;
            color: #ff7f0e;
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
        }
        .metric-card {
            background-color: #ffffff;
            padding: 1rem;
            border-radius: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .sentiment-positive {
            color: #000000;
            background-color: #2ca02c;
            border-left: 4px solid #2ca02c;
        }
        .sentiment-negative {
            color: #000000;
            background-color: #c23f3f;
            border-left: 4px solid #d62728;
        }
        .sentiment-neutral {
            color: #000000;
            background-color: #7f7f7f;
            border-left: 4px solid #7f7f7f;
        }
        .entity-tag {
            display: inline-block;
            background-color: #000000;
            color: #ffffff;
            padding: 0.25rem 0.5rem;
            margin: 0.25rem;
            border-radius: 0.25rem;
            font-size: 0.9rem;
        }
        .relationship-card {
            background-color: #1a1a1a;
            color: #ffffff;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            border-left: 4px solid #4a90e2;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        .relationship-header {
            display: flex;
            align-items: center;
            margin-bottom: 0.5rem;
            font-weight: bold;
        }
        .relationship-source {
            color: #4a90e2;
        }
        .relationship-arrow {
            margin: 0 0.5rem;
            color: #888888;
        }
        .relationship-target {
            color: #4a90e2;
        }
        .relationship-description {
            font-size: 0.9rem;
            color: #cccccc;
        }
        .topic-tag {
            display: inline-block;
            background-color: #000000;
            color: #ffffff;
            padding: 0.25rem 0.5rem;
            margin: 0.25rem;
            border-radius: 0.25rem;
            font-size: 0.9rem;
        }
        .anomaly-card {
            background-color: #000000;
            color: #ffffff;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
            border-left: 4px solid #d62728;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .controversy-meter {
            height: 30px;
            background-color: #f0f0f0;
            border-radius: 15px;
            overflow: hidden;
            margin: 1rem 0;
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.2);
        }
        .controversy-fill {
            height: 100%;
            background: linear-gradient(90deg, #2ca02c, #ff7f0e, #d62728);
            transition: width 1s ease-in-out;
        }
        .reasoning-text {
            font-style: italic;
            color: #333333;
            margin-top: 0.5rem;
            font-size: 1rem;
            line-height: 1.4;
        }
        .chart-container {
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

def load_page_css():
    """Loads minimal CSS and JavaScript for the page layout"""
    st.markdown("""
    <style>
        /* Basic page styling */
        body {
            background-color: #0f1419 !important;
        }
        
        .stApp {
            background-color: #0f1419 !important;
        }
        
        .main .block-container {
            background-color: #0f1419 !important;
            max-width: 1200px !important;
            padding-top: 2rem !important;
        }
        
        .css-1d391kg {
            background-color: #192734 !important;
        }
        
        .css-1lcbmhc {
            background-color: #192734 !important;
        }
        
        /* Card styling with equal height */
        .card {
            background-color: #192734;
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid #38444d;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
            text-align: center;
            height: 100%;
            display: flex;
            flex-direction: column;
            min-height: 200px; /* Ensure minimum height */
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.4);
            background-color: #22303c;
        }
        
        .card-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: #1da1f2;
            margin-bottom: 1rem;
            transition: color 0.3s ease;
        }
        
        .card:hover .card-title {
            color: #1a91da;
        }
        
        .card-text {
            font-size: 10.1rem;
            font-weight: 600;
            color: #ffffff;
            line-height: 1.7;
            background-color: #38444d;
            padding: 1rem;
            border-radius: 8px;
            flex-grow: 1;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        /* Animated title */
        @keyframes titleGlow {
            0% {
                text-shadow: 0 0 10px #1da1f2, 0 0 20px #1da1f2, 0 0 30px #1da1f2;
            }
            50% {
                text-shadow: 0 0 20px #1da1f2, 0 0 30px #1da1f2, 0 0 40px #1da1f2;
            }
            100% {
                text-shadow: 0 0 10px #1da1f2, 0 0 20px #1da1f2, 0 0 30px #1da1f2;
            }
        }
        
        .animated-title {
            animation: titleGlow 2s infinite alternate;
        }
        
        /* General styling - MODIFIED */
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff !important;
        }
        
        /* MORE SPECIFIC RULES FOR WHITE TEXT - EXCLUDES SENTIMENT CARDS */
        .stApp > div > div > div > div > div > p, 
        .stApp > div > div > div > div > div > span {
            color: #ffffff !important;
        }
        
        /* ENSURE SENTIMENT CARDS HAVE BLACK TEXT */
        .sentiment-positive p, .sentiment-positive span,
        .sentiment-negative p, .sentiment-negative span,
        .sentiment-neutral p, .sentiment-neutral span {
            color: #000000 !important;
        }
        
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
        
        .stTextInput > div > div > input {
            background-color: #38444d !important;
            color: #ffffff !important;
            border: 1px solid #4a5568 !important;
        }
        
        .stSelectbox > div > div > select {
            background-color: #38444d !important;
            color: #ffffff !important;
            border: 1px solid #4a5568 !important;
        }
        
        .stSlider > div > div > div {
            background-color: #1da1f2 !important;
        }
        
        .stFileUploader > div > div {
            background-color: #38444d !important;
            border: 1px dashed #4a5568 !important;
        }
        
        .stInfo {
            background-color: #192734 !important;
            border-left-color: #1da1f2 !important;
        }
        
        .stSuccess {
            background-color: #192734 !important;
            border-left-color: #17bf63 !important;
        }
        
        .stWarning {
            background-color: #192734 !important;
            border-left-color: #ffad1f !important;
        }
        
        .stError {
            background-color: #192734 !important;
            border-left-color: #e0245e !important;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            background-color: #192734 !important;
            border-radius: 8px !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            color: #ffffff !important;
        }
        
        .stTabs [data-baseweb="tab-panel"] {
            background-color: #192734 !important;
            border-radius: 8px !important;
            padding: 1rem !important;
        }
    </style>
    
    <script>
        // JavaScript to ensure horizontal layout and equal height
        document.addEventListener('DOMContentLoaded', function() {
            // Function to make all cards in a row the same height
            function equalizeCardHeights() {
                // Find all column containers
                const columnContainers = document.querySelectorAll('[data-testid="stVerticalBlock"]');
                
                columnContainers.forEach(container => {
                    // Get all cards in this container
                    const cards = container.querySelectorAll('.card');
                    
                    if (cards.length > 1) {
                        let maxHeight = 0;
                        
                        // Find the tallest card
                        cards.forEach(card => {
                            const height = card.offsetHeight;
                            if (height > maxHeight) {
                                maxHeight = height;
                            }
                        });
                        
                        // Set all cards to the same height
                        cards.forEach(card => {
                            card.style.height = maxHeight + 'px';
                        });
                    }
                });
            }
            
            // Run equalizeCardHeights on load
            equalizeCardHeights();
            
            // Run equalizeCardHeights when window is resized
            window.addEventListener('resize', equalizeCardHeights);
            
            // Run equalizeCardHeights after a short delay to ensure all content is loaded
            setTimeout(equalizeCardHeights, 500);
        });
    </script>
    """, unsafe_allow_html=True)

def display_sentiment_analysis(sentiment_data):
    """Display sentiment analysis with improved formatting"""
    if not sentiment_data:
        st.warning("No sentiment data available")
        return
    
    # Create columns for sentiment cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        positive = sentiment_data.get("positive", {})
        pos_percentage = positive.get("percentage", 0)
        pos_reasoning = positive.get("reasoning", "")
        
        st.markdown(f'''
        <div class="metric-card sentiment-positive">
            <h3 style="font-size: 1.2rem; font-weight: 700; margin-bottom: 0.5rem; color: #2ca02c;">Positive: {pos_percentage}%</h3>
            <p class="reasoning-text">{pos_reasoning}</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        negative = sentiment_data.get("negative", {})
        neg_percentage = negative.get("percentage", 0)
        neg_reasoning = negative.get("reasoning", "")
        
        st.markdown(f'''
        <div class="metric-card sentiment-negative">
            <h3 style="font-size: 1.2rem; font-weight: 700; margin-bottom: 0.5rem; color: #d62728;">Negative: {neg_percentage}%</h3>
            <p class="reasoning-text">{neg_reasoning}</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        neutral = sentiment_data.get("neutral", {})
        neu_percentage = neutral.get("percentage", 0)
        neu_reasoning = neutral.get("reasoning", "")
        
        st.markdown(f'''
        <div class="metric-card sentiment-neutral">
            <h3 style="font-size: 1.2rem; font-weight: 700; margin-bottom: 0.5rem; color: #7f7f7f;">Neutral: {neu_percentage}%</h3>
            <p class="reasoning-text">{neu_reasoning}</p>
        </div>
        ''', unsafe_allow_html=True)

def display_entities(entities):
    """Display entities as bullet points"""
    if not entities:
        st.info("No entities identified")
        return
    
    # Debug information
    if st.session_state.get('debug_mode', False):
        st.write(f"Debug: entities type: {type(entities)}")
        st.write(f"Debug: entities content: {entities}")
    
    # Handle different formats of entities
    if isinstance(entities, list):
        for entity in entities:
            if isinstance(entity, str):
                # Check if the entity contains a description (separated by -)
                if " - " in entity:
                    parts = entity.split(" - ", 1)
                    if len(parts) == 2:
                        name = parts[0].strip()
                        description = parts[1].strip()
                        st.markdown(f'<span class="entity-tag">{name}: {description}</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="entity-tag">{entity}</span>', unsafe_allow_html=True)
                else:
                    # Just a plain entity name
                    st.markdown(f'<span class="entity-tag">{entity}</span>', unsafe_allow_html=True)
            elif isinstance(entity, dict):
                if "name" in entity and "description" in entity:
                    st.markdown(f'<span class="entity-tag">{entity["name"]}: {entity["description"]}</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="entity-tag">{str(entity)}</span>', unsafe_allow_html=True)
    elif isinstance(entities, str):
        # If entities is a single string, split it into individual entities
        entity_list = [e.strip() for e in entities.split('\n') if e.strip()]
        for entity in entity_list:
            if " - " in entity:
                parts = entity.split(" - ", 1)
                if len(parts) == 2:
                    name = parts[0].strip()
                    description = parts[1].strip()
                    st.markdown(f'<span class="entity-tag">{name}: {description}</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="entity-tag">{entity}</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="entity-tag">{entity}</span>', unsafe_allow_html=True)
    else:
        st.info(f"Unexpected entities format: {type(entities)}")

def display_relationships(relationships):
    """Display relationships with improved formatting"""
    if not relationships:
        st.info("No relationships identified")
        return
    
    # Debug information
    if st.session_state.get('debug_mode', False):
        st.write(f"Debug: relationships type: {type(relationships)}")
        st.write(f"Debug: relationships content: {relationships}")
    
    # Handle different formats of relationships
    if isinstance(relationships, list):
        for rel in relationships:
            if isinstance(rel, dict):
                # Handle the specific format with entity1, entity2, and relationship keys
                if "entity1" in rel and "entity2" in rel and "relationship" in rel:
                    entity1 = rel.get("entity1", "")
                    entity2 = rel.get("entity2", "")
                    relationship_text = rel.get("relationship", "")
                    
                    st.markdown(f'''
                    <div class="relationship-card">
                        <div class="relationship-header">
                            <span class="relationship-source">{entity1}</span>
                            <span class="relationship-arrow">→</span>
                            <span class="relationship-target">{entity2}</span>
                        </div>
                        <div class="relationship-description">{relationship_text}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                # Handle other dictionary formats
                elif "entity" in rel and "description" in rel:
                    st.markdown(f'''
                    <div class="relationship-card">
                        <div class="relationship-header">
                            <span class="relationship-source">{rel["entity"]}</span>
                        </div>
                        <div class="relationship-description">{rel["description"]}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                elif "name" in rel and "description" in rel:
                    st.markdown(f'''
                    <div class="relationship-card">
                        <div class="relationship-header">
                            <span class="relationship-source">{rel["name"]}</span>
                        </div>
                        <div class="relationship-description">{rel["description"]}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    # Handle dictionary with unknown structure
                    st.markdown(f'''
                    <div class="relationship-card">
                        {str(rel)}
                    </div>
                    ''', unsafe_allow_html=True)
            elif isinstance(rel, str):
                # Handle string format - split on colon if present
                if ":" in rel:
                    parts = rel.split(":", 1)
                    if len(parts) == 2:
                        st.markdown(f'''
                        <div class="relationship-card">
                            <div class="relationship-header">
                                <span class="relationship-source">{parts[0].strip()}</span>
                            </div>
                            <div class="relationship-description">{parts[1].strip()}</div>
                        </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'''
                        <div class="relationship-card">
                            {rel}
                        </div>
                        ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="relationship-card">
                        {rel}
                    </div>
                    ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="relationship-card">
                    {str(rel)}
                </div>
                ''', unsafe_allow_html=True)
    elif isinstance(relationships, str):
        # Handle if relationships is a single string
        if ":" in relationships:
            parts = relationships.split(":", 1)
            if len(parts) == 2:
                st.markdown(f'''
                <div class="relationship-card">
                    <div class="relationship-header">
                        <span class="relationship-source">{parts[0].strip()}</span>
                    </div>
                    <div class="relationship-description">{parts[1].strip()}</div>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="relationship-card">
                    {relationships}
                </div>
                ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="relationship-card">
                {relationships}
            </div>
            ''', unsafe_allow_html=True)
    else:
        st.info(f"Unexpected relationships format: {type(relationships)}")

def display_anomalies(anomalies):
    """Display anomalies with improved formatting"""
    if not anomalies:
        st.info("No anomalies detected")
        return
    
    for anomaly in anomalies:
        if "description" in anomaly:
            st.markdown(f'''
            <div class="anomaly-card">
                <strong>Anomaly Detected</strong><br>
                {anomaly["description"]}
            </div>
            ''', unsafe_allow_html=True)
        elif isinstance(anomaly, str):
            st.markdown(f'''
            <div class="anomaly-card">
                {anomaly}
            </div>
            ''', unsafe_allow_html=True)

def display_controversy_score(controversy_data):
    """Display controversy score with improved formatting"""
    if not controversy_data:
        st.warning("No controversy score available")
        return
    
    score = controversy_data.get("value", 0)
    explanation = controversy_data.get("explanation", "")
    
    # Cap the score at 1.0 for display purposes
    display_score = min(score, 1.0)
    
    # Determine color based on score
    if display_score < 0.3:
        color = "#2ca02c"  # Green
        label = "Low Controversy"
    elif display_score < 0.7:
        color = "#ff7f0e"  # Orange
        label = "Medium Controversy"
    else:
        color = "#d62728"  # Red
        label = "High Controversy"
    
    # Display the score as a progress bar
    st.markdown(f'''
    <div class="controversy-meter">
        <div class="controversy-fill" style="width: {display_score*100}%; background: {color};"></div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Display score and explanation
    col1, col2 = st.columns([1, 3])
    with col1:
        # Show the actual score, not the capped one
        st.markdown(f"**Score:** {score}/1.0")
        st.markdown(f"**Level:** {label}")
        
        # Add a warning if the score is above 1.0
        if score > 1.0:
            st.warning(f"Score {score} is above the maximum expected value of 1.0")
    
    with col2:
        st.markdown(f"**Explanation:** {explanation}")

def display_analysis_results(analysis_data):
    """Display all analysis results in a structured way"""
    # Executive Summary
    st.markdown('<h2 class="section-header">📝 Executive Summary</h2>', unsafe_allow_html=True)
    exec_summary = analysis_data.get('executive_summary', 'No summary available')
    
    # Check if the summary contains bullet points
    if "- " in exec_summary:
        # Split by bullet points and create a formatted list
        points = [point.strip() for point in exec_summary.split('- ') if point.strip()]
        st.write("**Key Points:**")
        for point in points:
            st.markdown(f"• {point}")
    else:
        st.markdown(f"**{exec_summary}**")
    
    # Sentiment Analysis (moved after Executive Summary)
    st.markdown('<h2 class="section-header">😊 Sentiment Analysis</h2>', unsafe_allow_html=True)
    display_sentiment_analysis(analysis_data.get("sentiment_analysis", {}))
    
    # Topics
    st.markdown('<h2 class="section-header">🏷️ Key Topics</h2>', unsafe_allow_html=True)
    
    if "topics" in analysis_data:
        topics = analysis_data["topics"]
        if isinstance(topics, list):
            for topic in topics:
                st.markdown(f'<span class="topic-tag">{topic}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="topic-tag">{topics}</span>', unsafe_allow_html=True)
    else:
        st.info("No topics identified")
    
    # Entity Recognition
    st.markdown('<h2 class="section-header">👥 Recognized Entities</h2>', unsafe_allow_html=True)
    display_entities(analysis_data.get("entity_recognition", []))
    
    # Relationships
    st.markdown('<h2 class="section-header">🔗 Relationships</h2>', unsafe_allow_html=True)
    display_relationships(analysis_data.get("relationship_extraction", []))
    
    # Anomalies
    st.markdown('<h2 class="section-header">⚠️ Detected Anomalies</h2>', unsafe_allow_html=True)
    display_anomalies(analysis_data.get("anomaly_detection", []))
    
    # Controversy Score
    st.markdown('<h2 class="section-header">🌡️ Controversy Score</h2>', unsafe_allow_html=True)
    display_controversy_score(analysis_data.get("controversy_score", {}))
    


def initialize_chatbot(json_file_path):
    """Initialize the RAG chatbot with the provided JSON file"""
    if 'chatbot' not in st.session_state or st.session_state.get('current_json_file') != json_file_path:
        with st.spinner("Initializing chatbot... This may take a few minutes."):
            try:
                chatbot = RAGChatbot()
                chatbot.initialize(json_file_path)
                st.session_state.chatbot = chatbot
                st.session_state.current_json_file = json_file_path
                st.session_state.chat_messages = []  # Initialize chat history
                st.success("Chatbot initialized successfully!")
                logger.info("Chatbot initialized successfully")
                return True
            except Exception as e:
                st.error(f"Failed to initialize chatbot: {str(e)}")
                logger.error(f"Failed to initialize chatbot: {str(e)}")
                return False
    else:
        logger.info("Chatbot already initialized")
        return True

def display_chatbot_interface():
    """Display the chatbot interface with enhanced UI"""
    st.markdown("### 🤖 Chat with your Data")
    
    # Check if chatbot is initialized
    if 'chatbot' not in st.session_state:
        st.info("Please upload a file and click 'Chat with data' first to initialize the chatbot.")
        logger.warning("Chatbot not initialized in session state")
        return
    
    # Initialize chat history if not exists
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
        logger.info("Initialized chat messages in session state")
    
    # Create a container for the chat interface with enhanced styling
    chat_container = st.container()
    
    with chat_container:
        # Add a header with back button
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("← Back to Main App", key="back_to_main"):
                st.session_state.chat_mode = False
                st.rerun()
        with col2:
            st.markdown("### 💬 Conversation")
        
        # Display chat messages with enhanced styling
        message_container = st.container()
        with message_container:
            for i, message in enumerate(st.session_state.chat_messages):
                if message["role"] == "user":
                    st.markdown(f"""
                    <div style="display: flex; justify-content: flex-end; margin-bottom: 15px;">
                        <div style="background-color: #1DA1F2; color: white; padding: 12px 15px; border-radius: 18px 18px 4px 18px; max-width: 80%; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                            <strong>You:</strong> {message["content"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="display: flex; justify-content: flex-start; margin-bottom: 15px;">
                        <div style="background-color: #192734; color: white; padding: 12px 15px; border-radius: 18px 18px 18px 4px; max-width: 80%; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                            <strong>Bot:</strong> {message["content"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Add a separator
        st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)
        
        # Chat input section with enhanced styling
        input_container = st.container()
        with input_container:
            # Use a form to handle the submission
            with st.form(key="chat_form", clear_on_submit=True):
                col1, col2 = st.columns([4, 1])
                with col1:
                    user_input = st.text_input("Type your question:", placeholder="Ask me anything about the data...", key="user_question")
                with col2:
                    submit_button = st.form_submit_button("Send", use_container_width=True)
                
                if submit_button and user_input:
                    # Add user message to chat history
                    st.session_state.chat_messages.append({"role": "user", "content": user_input})
                    logger.info(f"User message: {user_input}")
                    
                    # Get bot response
                    with st.spinner("Thinking..."):
                        try:
                            response = st.session_state.chatbot.query_with_streaming(user_input)
                            bot_response = response.get("answer", "Sorry, I couldn't process your request.")
                            
                            # Add bot response to chat history
                            st.session_state.chat_messages.append({"role": "assistant", "content": bot_response})
                            logger.info(f"Bot response: {bot_response}")
                            
                            # Rerun to update the chat interface
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                            logger.error(f"Error in chatbot response: {str(e)}")
            
            # Add clear chat button
            if st.button("Clear Chat History", key="clear_chat"):
                st.session_state.chat_messages = []
                logger.info("Cleared chat history")
                st.rerun()
            
            # Add some helpful tips
            st.markdown("""
            <div style="background-color: #192734; padding: 10px; border-radius: 10px; margin-top: 15px; font-size: 0.9rem;">
                <strong>Tips:</strong> Ask specific questions about the data, such as "What are the main topics?" or "Summarize the findings."
            </div>
            """, unsafe_allow_html=True)

def load_sample_data_to_ui():
    """Load sample data directly into the UI"""
    sample_data = {
        "executive_summary": "The newsletter provides a detailed and alarming overview of the current global situation, focusing on escalating environmental, economic, and political crises. Key points include:\n\nEnvironmental Degradation: The text highlights severe climate disasters such as wildfires, floods, and heatwaves, indicating an accelerating climate crisis.\nEconomic Instability: It addresses growing economic instability, financial risks, and inflation, suggesting a fragile global economy.\nGeopolitical Conflicts: Ongoing conflicts in regions like the Middle East and Ukraine are emphasized, along with emerging tensions worldwide.\nHealth Crises: The newsletter discusses both ongoing pandemics (like the coronavirus) and new public health challenges such as neglected tropical diseases.\nSocio-Economic Inequality: It points to increasing wealth disparities and social disintegration, indicating a widening gap between different socio-economic groups.\nPolicy Failures: The text examines the inadequacies of current climate policies and the lack of adequate climate finance, suggesting that global leaders are failing to address critical issues effectively.\nThe overall sentiment is highly pessimistic and alarming, with a sense of urgency and despair as the author emphasizes the dire consequences of inaction and the rapid deterioration of the global situation.",
        "sentiment_analysis": {
            "positive": {
                "percentage": 5,
                "reasoning": "The text mentions a detailed overview which can be seen as informative and helpful."
            },
            "negative": {
                "percentage": 90,
                "reasoning": "The overall tone is highly pessimistic, with emphasis on global collapse, conflict, environmental degradation, economic instability, and political tensions."
            },
            "neutral": {
                "percentage": 5,
                "reasoning": "The text includes neutral information such as specific events and issues without expressing a clear positive or negative sentiment."
            }
        },
        "topics": [
            "Climate Change and Environmental Degradation",
            "Natural Disasters and Extreme Weather",
            "Economic Instability and Financial Risks",
            "Geopolitical Conflicts and Wars",
            "Pandemics and Public Health Crises"
        ],
        "entity_recognition": [
            "related disasters 5. Pandemics 6. Socio-economic challenges 7. Technological and security threats 8. Refugee and migration crises 9. Political instability and authoritarianism 10. Climate policy failures"
        ],
        "relationship_extraction": [
            {
                "entity1": "Environmental Degradation",
                "entity2": "None",
                "relationship": "The newsletter discusses the accelerating climate crisis and its impact on environmental degradation."
            }
        ],
        "anomaly_detection": [],
        "controversy_score": {
            "value": 0.8,
            "explanation": "The content discusses highly controversial topics including climate change, economic instability, and geopolitical conflicts, which often elicit strong opinions and disagreements."
        }
    }
    
    # Store the sample data in session state
    st.session_state.analysis_data = sample_data
    st.session_state.analysis_complete = True
    st.success("Sample data loaded successfully!")
    
    # Create a temporary JSON file for the chatbot
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        json.dump(sample_data, tmp)
        tmp_path = tmp.name
    
    # Initialize the chatbot with the sample data
    initialize_chatbot(tmp_path)

def create_sentiment_chart(sentiment_data):
    """Create a sentiment analysis chart using Altair"""
    if not sentiment_data:
        return None
    
    # Extract sentiment data and create DataFrame
    sentiment_list = []
    for sentiment_type, data in sentiment_data.items():
        if isinstance(data, dict) and "percentage" in data:
            sentiment_list.append({
                "Sentiment": sentiment_type.capitalize(),
                "Percentage": data["percentage"]
            })
    
    if not sentiment_list:
        return None
    
    sentiment_df = pd.DataFrame(sentiment_list)
    
    # Define color scheme for different sentiments
    sentiment_colors = {
        "Positive": "#2ca02c",  # Green
        "Negative": "#d62728",  # Red
        "Neutral": "#7f7f7f"    # Gray
    }
    
    # Create the Altair Chart object
    chart = alt.Chart(sentiment_df).mark_bar().encode(
        # Set the x-axis to the Percentage column and label it
        x=alt.X('Percentage:Q', axis=alt.Axis(title='Percentage (%)')),
        # Set the y-axis to the Sentiment column
        y=alt.Y('Sentiment:N'),
        # Color based on sentiment for better visual distinction
        color=alt.Color('Sentiment:N', 
                       scale=alt.Scale(domain=list(sentiment_colors.keys()), 
                                      range=list(sentiment_colors.values()))),
        # Add tooltips for interactivity
        tooltip=['Sentiment', 'Percentage']
    ).properties(
        # Set specific chart width and height
        width=400,  # Set exact pixel width
        height=250  # Set exact pixel height
    ).configure_view(
        # Remove chart stroke
        strokeWidth=0
    ).configure_axis(
        # Configure axis labels
        labelFontSize=12,
        titleFontSize=14,
        labelColor='white',
        titleColor='white',
        tickColor='white',
        gridColor='#38444d'
    ).configure_legend(
        # Configure legend
        labelColor='white',
        titleColor='white'
    ).interactive()  # Allows zooming/panning
    
    return chart
    
    


# Set up logging
logger = logging.getLogger(__name__)
def handle_social_media_analysis():
    """Handle the social media analysis tab"""
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
    
    # Only show the search button if analysis is not in progress
    if not st.session_state.analysis_in_progress:
        if st.button(f"Search {platform} & Analyze"):
            if search_query:
                st.session_state.analysis_in_progress = True
                logger.info(f"Starting search on {platform} for query: {search_query}")
                
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
                        logger.error(f"Error fetching data: {error}")
                        st.session_state.analysis_in_progress = False
                    else:
                        st.success(f"Data saved to {filename}")
                        logger.info(f"Data saved to {filename}")
                        st.session_state.uploaded_file_path = filename
                        st.session_state.file_uploaded = True
                        st.session_state.current_view = None
                        st.session_state.analysis_in_progress = False
                        st.success("Data fetched successfully! Please select an option below.")
                        st.rerun()
            else:
                st.warning("Please enter a search query.")
    else:
        st.info("Analysis is in progress. Please wait...")
    
    # Show options after file is uploaded but not analyzed yet
    if st.session_state.get('file_uploaded') and not st.session_state.get('analysis_complete') and st.session_state.get('current_view') is None:
        st.markdown("---")
        st.markdown("### What would you like to do with this data?")
        
        # Create two columns for the options
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Analyze this data", type="primary", key="tab1_analyze"):
                if st.session_state.uploaded_file_path:
                    st.session_state.analysis_in_progress = True
                    with st.spinner("Analyzing the data..."):
                        parsed_data, error = analyze_data(st.session_state.uploaded_file_path)
                        if error:
                            st.error(error)
                            logger.error(f"Error analyzing data: {error}")
                            st.session_state.analysis_in_progress = False
                        else:
                            st.session_state.analysis_data = parsed_data.get("multiverse_combined", {})
                            st.session_state.analysis_complete = True
                            st.session_state.current_view = "analysis"
                            st.session_state.analysis_in_progress = False
                            st.success("Analysis complete!")
                            logger.info("Analysis completed successfully")
                            st.rerun()
        
        with col2:
            if st.button("🤖 Chat with data", type="secondary", key="tab1_chat"):
                if st.session_state.uploaded_file_path:
                    logger.info(f"Initializing chatbot with file: {st.session_state.uploaded_file_path}")
                    # Initialize chatbot when chat is selected
                    if initialize_chatbot(st.session_state.uploaded_file_path):
                        st.session_state.chat_mode = True
                        st.rerun()
                    else:
                        st.error("Failed to initialize chatbot. Please try again.")
    
    # Display content based on selected view
    if st.session_state.get('current_view') == "analysis":
        st.markdown("---")
        st.markdown("### 📊 Analysis Results")
        
        # Check if analysis_data exists before trying to use it
        if st.session_state.get('analysis_data'):
            analysis_data = st.session_state.analysis_data
            # Display all analysis results (sentiment analysis is now included in this)
            display_analysis_results(analysis_data)
        else:
            st.error("No analysis data available. Please analyze the file first.")

def handle_custom_file_analysis():
    """Handle the custom file analysis tab"""
    st.markdown("### 📂 Custom File Analysis")
    
    # Add a button to load sample data
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("📊 Load Sample Data", type="primary"):
            load_sample_data_to_ui()
            st.session_state.file_uploaded = True
            st.session_state.current_view = None  # Reset view to show selection buttons
            st.rerun()
    
    with col2:
        st.markdown("""
        <div style="padding-top: 0.5rem;">
            Click the button to load sample analysis data directly into the interface.
            This will demonstrate all the features with pre-populated data.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "Upload a JSON file for analysis",
        type=["json"],
        help="Upload a JSON file containing text data to be analyzed"
    )
    
    if uploaded_file is not None:
        st.sidebar.subheader("File Information")
        st.sidebar.write(f"Filename: {uploaded_file.name}")
        st.sidebar.write(f"File size: {uploaded_file.size / 1024:.2f} KB")
        
        # Process the uploaded file and store its path
        tmp_file_path = process_uploaded_file(uploaded_file)
        st.session_state.uploaded_file_path = tmp_file_path
        st.session_state.file_uploaded = True
        st.session_state.current_view = None
        st.session_state.analysis_complete = False
        st.success("File uploaded successfully! Please select an option below.")
        logger.info(f"File uploaded: {uploaded_file.name}")
    
    # Show options after file is uploaded but not analyzed yet
    if st.session_state.get('file_uploaded') and not st.session_state.get('analysis_complete') and st.session_state.get('current_view') is None:
        st.markdown("---")
        st.markdown("### What would you like to do with this data?")
        
        # Create two columns for the options
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Analyze this data", type="primary", key="tab2_analyze"):
                if st.session_state.uploaded_file_path:
                    st.session_state.analysis_in_progress = True
                    with st.spinner("Analyzing your data... This may take a few minutes."):
                        parsed_data, error = analyze_data(st.session_state.uploaded_file_path)
                        
                        if error:
                            st.error(error)
                            logger.error(f"Error analyzing data: {error}")
                            st.session_state.analysis_in_progress = False
                        else:
                            st.success("Analysis complete!")
                            
                            analysis_data = None
                            
                            if "multiverse_combined" in parsed_data:
                                analysis_data = parsed_data["multiverse_combined"]
                            elif "analysis_results" in parsed_data:
                                analysis_data = parsed_data["analysis_results"]
                            else:
                                analysis_data = parsed_data
                            
                            st.session_state.analysis_data = analysis_data
                            st.session_state.analysis_complete = True
                            st.session_state.current_view = "analysis"
                            st.session_state.analysis_in_progress = False
                            
                            if st.session_state.debug_mode:
                                st.markdown('### 🐛 Debug Information')
                                st.write("Parsed data structure:")
                                st.json(parsed_data)
                            
                            logger.info("Analysis completed successfully")
                            st.rerun()
        
        with col2:
            if st.button("🤖 Chat with data", type="secondary", key="tab2_chat"):
                if st.session_state.uploaded_file_path:
                    logger.info(f"Initializing chatbot with file: {st.session_state.uploaded_file_path}")
                    # Initialize chatbot when chat is selected
                    if initialize_chatbot(st.session_state.uploaded_file_path):
                        st.session_state.chat_mode = True
                        st.rerun()
                    else:
                        st.error("Failed to initialize chatbot. Please try again.")
    
    # Display content based on selected view
    if st.session_state.get('current_view') == "analysis":
        st.markdown("---")
        st.markdown("### 📊 Analysis Results")
        
        # Check if analysis_data exists before trying to use it
        if st.session_state.get('analysis_data'):
            analysis_data = st.session_state.analysis_data
            # Display all analysis results (sentiment analysis is now included in this)
            display_analysis_results(analysis_data)
        else:
            st.error("No analysis data available. Please analyze the file first.")
    
    # Display welcome message if no file is uploaded
    elif not st.session_state.get('file_uploaded'):
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
        1. Click "Load Sample Data" to see a demonstration
        2. Or upload a JSON file containing text data
        3. Choose to either analyze the data or chat with it
        4. View the results below
        
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
        
        
def handle_settings():
    """Handle the settings tab"""
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
        logger.info("Twitter cookies file updated")
    
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
        logger.info("YouTube cookies file updated")

def handle_sidebar():
    """Handle the sidebar"""
    st.sidebar.title("Analysis Settings")
    
    st.sidebar.subheader("Analysis Options")
    show_raw_output = st.sidebar.checkbox("Show Raw Output", value=False)
    save_results = st.sidebar.checkbox("Save Results to File", value=True)
    st.session_state.debug_mode = st.sidebar.checkbox("Debug Mode", value=False)
    
    if st.sidebar.button("Clear Cache & Restart Analysis"):
        cleared_files, errors = clear_cache()
        if cleared_files:
            st.sidebar.success(f"Cleared cache files: {', '.join(cleared_files)}")
            logger.info(f"Cleared cache files: {', '.join(cleared_files)}")
        if errors:
            for error in errors:
                st.sidebar.warning(error)
                logger.warning(f"Cache clearing error: {error}")
        
        # Clear the analysis data from session state
        session_vars_to_clear = [
            'analysis_data', 'analysis_complete', 'chatbot', 'chat_messages',
            'current_json_file', 'file_uploaded', 'current_view',
            'analysis_in_progress', 'uploaded_file_path'
        ]
        
        for var in session_vars_to_clear:
            if var in st.session_state:
                del st.session_state[var]
                logger.info(f"Cleared session state variable: {var}")
        
        # Clean up temp file if it exists
        if st.session_state.uploaded_file_path and os.path.exists(st.session_state.uploaded_file_path):
            os.unlink(st.session_state.uploaded_file_path)
            logger.info(f"Removed temp file: {st.session_state.uploaded_file_path}")
        
        st.rerun()
        
        
        
def initialize_chatbot(json_file_path):
    """Initialize the RAG chatbot with the provided JSON file"""
    if 'chatbot' not in st.session_state or st.session_state.get('current_json_file') != json_file_path:
        with st.spinner("Initializing chatbot... This may take a few minutes."):
            try:
                chatbot = RAGChatbot()
                chatbot.initialize(json_file_path)
                st.session_state.chatbot = chatbot
                st.session_state.current_json_file = json_file_path
                st.session_state.chat_messages = []  # Initialize chat history
                st.success("Chatbot initialized successfully!")
                logger.info("Chatbot initialized successfully")
                return True
            except Exception as e:
                st.error(f"Failed to initialize chatbot: {str(e)}")
                logger.error(f"Failed to initialize chatbot: {str(e)}")
                return False
    else:
        logger.info("Chatbot already initialized")
        return True
