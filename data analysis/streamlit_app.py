# streamlit_app.py
import streamlit as st
import os
import tempfile
import json
from typing import Dict, Any
import shutil

# Import your analysis functions
from main2 import combine_analysis
from simple_parser import parse_simple_format

# Configure the page
st.set_page_config(
    page_title="Text Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .sentiment-positive {
        color: #2ca02c;
        border-left: 4px solid #2ca02c;
    }
    .sentiment-negative {
        color: #d62728;
        border-left: 4px solid #d62728;
    }
    .sentiment-neutral {
        color: #7f7f7f;
        border-left: 4px solid #7f7f7f;
    }
    .entity-tag {
        display: inline-block;
        background-color: #000000;  /* Changed to black */
        color: #ffffff;  /* Changed text to white for readability */
        padding: 0.25rem 0.5rem;
        margin: 0.25rem;
        border-radius: 0.25rem;
        font-size: 0.9rem;
    }
    .topic-tag {
        display: inline-block;
        background-color: #000000;  /* Changed to black */
        color: #ffffff;  /* Changed text to white for readability */
        padding: 0.25rem 0.5rem;
        margin: 0.25rem;
        border-radius: 0.25rem;
        font-size: 0.9rem;
    }
    .relationship-card {
        background-color: #000000;  /* Changed to black */
        color: #ffffff;  /* Changed text to white for readability */
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .anomaly-card {
        background-color: #000000;  /* Changed to black background */
        color: #ffffff;  /* White text for readability */
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
        color: #555;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def clear_cache():
    """Clear all cache files to ensure fresh analysis"""
    cache_files = [
        "partial_results_main.json",
        "raw_output.txt",
        "final_output.txt",
        "analysis_results.json"
    ]
    
    for file in cache_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                st.sidebar.success(f"Cleared cache file: {file}")
            except Exception as e:
                st.sidebar.warning(f"Could not clear {file}: {str(e)}")

def display_sentiment_analysis(sentiment_data):
    """Display sentiment analysis with improved formatting"""
    if not sentiment_data:
        st.warning("No sentiment data available")
        return
    
    # Create columns for sentiment
    col1, col2, col3 = st.columns(3)
    
    with col1:
        positive = sentiment_data.get("positive", {})
        pos_percentage = positive.get("percentage", 0)
        pos_reasoning = positive.get("reasoning", "")
        st.markdown(f'''
        <div class="metric-card sentiment-positive">
            <h3>Positive: {pos_percentage}%</h3>
            <p class="reasoning-text">{pos_reasoning}</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        negative = sentiment_data.get("negative", {})
        neg_percentage = negative.get("percentage", 0)
        neg_reasoning = negative.get("reasoning", "")
        st.markdown(f'''
        <div class="metric-card sentiment-negative">
            <h3>Negative: {neg_percentage}%</h3>
            <p class="reasoning-text">{neg_reasoning}</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        neutral = sentiment_data.get("neutral", {})
        neu_percentage = neutral.get("percentage", 0)
        neu_reasoning = neutral.get("reasoning", "")
        st.markdown(f'''
        <div class="metric-card sentiment-neutral">
            <h3>Neutral: {neu_percentage}%</h3>
            <p class="reasoning-text">{neu_reasoning}</p>
        </div>
        ''', unsafe_allow_html=True)

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
                    
                    # Create a title for the relationship
                    if entity1 and entity2:
                        title = f"{entity1} ↔ {entity2}"
                    elif entity1:
                        title = entity1
                    elif entity2:
                        title = entity2
                    else:
                        # Extract a title from the relationship text if no entities are available
                        if ":" in relationship_text:
                            title = relationship_text.split(":", 1)[0].strip()
                        else:
                            title = "Relationship"
                    
                    st.markdown(f'''
                    <div class="relationship-card">
                        <strong>{title}</strong><br>
                        {relationship_text}
                    </div>
                    ''', unsafe_allow_html=True)
                # Handle other dictionary formats
                elif "entity" in rel and "description" in rel:
                    st.markdown(f'''
                    <div class="relationship-card">
                        <strong>{rel["entity"]}</strong><br>
                        {rel["description"]}
                    </div>
                    ''', unsafe_allow_html=True)
                elif "name" in rel and "description" in rel:
                    st.markdown(f'''
                    <div class="relationship-card">
                        <strong>{rel["name"]}</strong><br>
                        {rel["description"]}
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
                            <strong>{parts[0].strip()}</strong><br>
                            {parts[1].strip()}
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
                    <strong>{parts[0].strip()}</strong><br>
                    {parts[1].strip()}
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
    
    # Determine color based on score
    if score < 0.3:
        color = "#2ca02c"  # Green
        label = "Low Controversy"
    elif score < 0.7:
        color = "#ff7f0e"  # Orange
        label = "Medium Controversy"
    else:
        color = "#d62728"  # Red
        label = "High Controversy"
    
    # Display the score as a progress bar
    st.markdown(f'''
    <div class="controversy-meter">
        <div class="controversy-fill" style="width: {score*100}%; background: {color};"></div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Display score and explanation
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(f"**Score:** {score:.1f}/1.0")
        st.markdown(f"**Level:** {label}")
    
    with col2:
        st.markdown(f"**Explanation:** {explanation}")

def main():
    # Initialize session state
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    
    # Header
    st.markdown('<h1 class="main-header">📊 Text Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Analysis Settings")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload a JSON file for analysis",
        type=["json"],
        help="Upload a JSON file containing text data to be analyzed"
    )
    print(uploaded_file)
    # Analysis options
    st.sidebar.subheader("Analysis Options")
    show_raw_output = st.sidebar.checkbox("Show Raw Output", value=False)
    save_results = st.sidebar.checkbox("Save Results to File", value=True)
    st.session_state.debug_mode = st.sidebar.checkbox("Debug Mode", value=False)
    
    # Add a button to clear cache
    if st.sidebar.button("Clear Cache & Restart Analysis"):
        clear_cache()
        st.sidebar.success("Cache cleared! Please upload your file again.")
        st.rerun()
    
    # Main content
    if uploaded_file is not None:
        # Display file info
        st.sidebar.subheader("File Information")
        st.sidebar.write(f"Filename: {uploaded_file.name}")
        st.sidebar.write(f"File size: {uploaded_file.size / 1024:.2f} KB")
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Clear cache before analysis if requested
            if st.sidebar.button("Clear Cache & Analyze"):
                clear_cache()
                st.sidebar.success("Cache cleared! Starting fresh analysis...")
            
            # Run analysis
            with st.spinner("Analyzing your data... This may take a few minutes."):
                # Call the analysis function
                raw_output = combine_analysis(tmp_file_path)
                
                # Check if the analysis was successful
                if not raw_output or raw_output.strip() == "":
                    st.error("Analysis failed to produce output. Please try clearing the cache and running the analysis again.")
                    return
                
                # Parse the output
                parsed_data = parse_simple_format(raw_output)
                
                # Save results if requested
                if save_results:
                    with open("analysis_results.json", "w", encoding="utf-8") as f:
                        json.dump(parsed_data, f, indent=4, ensure_ascii=False)
                    st.sidebar.success("Results saved to analysis_results.json")
            
            # Display results
            # Check for different possible structures in the parsed data
            analysis_data = None
            
            if "multiverse_combined" in parsed_data:
                analysis_data = parsed_data["multiverse_combined"]
            elif "analysis_results" in parsed_data:
                analysis_data = parsed_data["analysis_results"]
            else:
                # Use the entire parsed_data if no specific section is found
                analysis_data = parsed_data
            
            if st.session_state.debug_mode:
                st.markdown('<h2 class="section-header">🐛 Debug Information</h2>', unsafe_allow_html=True)
                st.write("Parsed data structure:")
                st.json(parsed_data)
            
            if analysis_data:
                # Executive Summary
                st.markdown('<h2 class="section-header">📝 Executive Summary</h2>', unsafe_allow_html=True)
                st.markdown(f"**{analysis_data.get('executive_summary', 'No summary available')}**")
                
                # Sentiment Analysis
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
                
                if "entity_recognition" in analysis_data:
                    entities = analysis_data["entity_recognition"]
                    if isinstance(entities, list):
                        for entity in entities:
                            st.markdown(f'<span class="entity-tag">{entity}</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="entity-tag">{entities}</span>', unsafe_allow_html=True)
                else:
                    st.info("No entities identified")
                
                # Relationship Extraction
                st.markdown('<h2 class="section-header">🔗 Relationships</h2>', unsafe_allow_html=True)
                display_relationships(analysis_data.get("relationship_extraction", []))
                
                # Anomaly Detection
                st.markdown('<h2 class="section-header">⚠️ Detected Anomalies</h2>', unsafe_allow_html=True)
                display_anomalies(analysis_data.get("anomaly_detection", []))
                
                # Controversy Score
                st.markdown('<h2 class="section-header">🌡️ Controversy Score</h2>', unsafe_allow_html=True)
                display_controversy_score(analysis_data.get("controversy_score", {}))
                
                # Raw output (if requested)
                if show_raw_output:
                    st.markdown('<h2 class="section-header">📄 Raw Output</h2>', unsafe_allow_html=True)
                    st.text_area("Raw Analysis Output", raw_output, height=300)
            else:
                st.error("No valid analysis data found. Please check your input file and try again.")
        
        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            st.write("Please check your input file and try again.")
        
        finally:
            # Clean up the temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    else:
        # Instructions for the user
        st.markdown("""
        ## Welcome to the Text Analysis Dashboard! 📊
        
        This tool analyzes text data from JSON files and provides insights including:
        - Executive summaries
        - Sentiment analysis
        - Key topics
        - Entity recognition
        - Relationship extraction
        - Anomaly detection
        - Controversy scoring
        
        ### How to use:
        1. Upload a JSON file using the sidebar on the left
        2. Configure your analysis options
        3. Click "Clear Cache & Analyze" to start a fresh analysis
        4. View the results below
        
        ### Supported JSON format:
        Your JSON file should contain text data that can be analyzed. The system will extract all text content from the JSON structure.
        
        ### Troubleshooting:
        If you encounter issues with the analysis, try clicking "Clear Cache & Restart Analysis" before uploading your file again.
        """)
        
        # Sample file download
        st.markdown("### Need a sample file?")
        st.markdown("Download a sample JSON file to test the analysis:")
        
        # Create a sample JSON
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
        sample_json = json.dumps(sample_data, indent=2)
        
        # Create a download button
        st.download_button(
            label="Download Sample JSON",
            data=sample_json,
            file_name="sample_data.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()
