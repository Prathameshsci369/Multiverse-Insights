# ui_components.py
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List

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
            color: #555;
            margin-top: 0.5rem;
        }
        .chart-container {
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

def create_sentiment_pie_chart(sentiment_data):
    """Create a pie chart for sentiment analysis"""
    if not sentiment_data:
        return None
    
    # Extract sentiment percentages
    labels = []
    sizes = []
    colors = []
    
    if "positive" in sentiment_data:
        labels.append("Positive")
        sizes.append(sentiment_data["positive"].get("percentage", 0))
        colors.append("#2ca02c")
    
    if "negative" in sentiment_data:
        labels.append("Negative")
        sizes.append(sentiment_data["negative"].get("percentage", 0))
        colors.append("#d62728")
    
    if "neutral" in sentiment_data:
        labels.append("Neutral")
        sizes.append(sentiment_data["neutral"].get("percentage", 0))
        colors.append("#7f7f7f")
    
    # Create the pie chart
    fig, ax = plt.subplots(figsize=(6, 4))
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    
    # Add a title
    ax.set_title("Sentiment Analysis Distribution", fontsize=16, pad=20)
    
    # Improve text appearance
    for text in texts:
        text.set_fontsize(12)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_weight('bold')
    
    return fig

def create_sentiment_bar_chart(sentiment_data):
    """Create a bar chart for sentiment analysis"""
    if not sentiment_data:
        return None
    
    # Extract sentiment percentages
    labels = []
    values = []
    colors = []
    
    if "positive" in sentiment_data:
        labels.append("Positive")
        values.append(sentiment_data["positive"].get("percentage", 0))
        colors.append("#2ca02c")
    
    if "negative" in sentiment_data:
        labels.append("Negative")
        values.append(sentiment_data["negative"].get("percentage", 0))
        colors.append("#d62728")
    
    if "neutral" in sentiment_data:
        labels.append("Neutral")
        values.append(sentiment_data["neutral"].get("percentage", 0))
        colors.append("#7f7f7f")
    
    # Create the bar chart
    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(labels, values, color=colors)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height}%', ha='center', va='bottom', fontsize=12)
    
    # Set y-axis limit
    ax.set_ylim(0, max(values) * 1.2 if values else 100)
    
    # Add a title and labels
    ax.set_title("Sentiment Analysis Distribution", fontsize=16, pad=20)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    
    # Remove x-axis ticks
    ax.tick_params(axis='x', which='both', length=0)
    
    return fig

def create_individual_sentiment_chart(sentiment_type, percentage, color):
    """Create a small chart for an individual sentiment"""
    # Create a simple gauge chart
    fig, ax = plt.subplots(figsize=(4, 2))
    
    # Create a gauge-like visualization
    theta = np.linspace(0, np.pi, 100)
    r = 0.5
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Draw the background arc
    ax.plot(x, y, color='lightgray', linewidth=20)
    
    # Draw the filled arc based on percentage
    fill_theta = np.linspace(0, np.pi * (percentage / 100), 100)
    fill_x = r * np.cos(fill_theta)
    fill_y = r * np.sin(fill_theta)
    ax.plot(fill_x, fill_y, color=color, linewidth=20)
    
    # Add the percentage text
    ax.text(0, -0.2, f"{percentage}%", ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Remove axes
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.3, 0.6)
    ax.axis('off')
    
    return fig

def display_sentiment_analysis(sentiment_data):
    """Display sentiment analysis with improved formatting and graphs"""
    if not sentiment_data:
        st.warning("No sentiment data available")
        return
    
    # Display the overall sentiment chart
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    # Add tabs for different chart types
    chart_tab1, chart_tab2 = st.tabs(["Pie Chart", "Bar Chart"])
    
    with chart_tab1:
        pie_chart = create_sentiment_pie_chart(sentiment_data)
        if pie_chart:
            st.pyplot(pie_chart)
    
    with chart_tab2:
        bar_chart = create_sentiment_bar_chart(sentiment_data)
        if bar_chart:
            st.pyplot(bar_chart)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create columns for sentiment cards with individual charts
    col1, col2, col3 = st.columns(3)
    
    with col1:
        positive = sentiment_data.get("positive", {})
        pos_percentage = positive.get("percentage", 0)
        pos_reasoning = positive.get("reasoning", "")
        
        # Create individual chart for positive sentiment
        pos_chart = create_individual_sentiment_chart("Positive", pos_percentage, "#2ca02c")
        
        st.markdown(f'''
        <div class="metric-card sentiment-positive">
            <h3>Positive: {pos_percentage}%</h3>
        </div>
        ''', unsafe_allow_html=True)
        
        if pos_chart:
            st.pyplot(pos_chart)
        
        st.markdown(f'<p class="reasoning-text">{pos_reasoning}</p>', unsafe_allow_html=True)
    
    with col2:
        negative = sentiment_data.get("negative", {})
        neg_percentage = negative.get("percentage", 0)
        neg_reasoning = negative.get("reasoning", "")
        
        # Create individual chart for negative sentiment
        neg_chart = create_individual_sentiment_chart("Negative", neg_percentage, "#d62728")
        
        st.markdown(f'''
        <div class="metric-card sentiment-negative">
            <h3>Negative: {neg_percentage}%</h3>
        </div>
        ''', unsafe_allow_html=True)
        
        if neg_chart:
            st.pyplot(neg_chart)
        
        st.markdown(f'<p class="reasoning-text">{neg_reasoning}</p>', unsafe_allow_html=True)
    
    with col3:
        neutral = sentiment_data.get("neutral", {})
        neu_percentage = neutral.get("percentage", 0)
        neu_reasoning = neutral.get("reasoning", "")
        
        # Create individual chart for neutral sentiment
        neu_chart = create_individual_sentiment_chart("Neutral", neu_percentage, "#7f7f7f")
        
        st.markdown(f'''
        <div class="metric-card sentiment-neutral">
            <h3>Neutral: {neu_percentage}%</h3>
        </div>
        ''', unsafe_allow_html=True)
        
        if neu_chart:
            st.pyplot(neu_chart)
        
        st.markdown(f'<p class="reasoning-text">{neu_reasoning}</p>', unsafe_allow_html=True)

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
