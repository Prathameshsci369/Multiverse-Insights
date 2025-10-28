# complete_pdf_generator.py
import os
import io
import base64
from datetime import datetime
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics import renderPDF
import tempfile
import re

# Set matplotlib to use Agg backend for non-interactive plotting
import matplotlib
matplotlib.use('Agg')

class PDFReportGenerator:
    """Class to generate PDF reports from analysis data"""
    
    def __init__(self):
        self.temp_files = []  # Track temporary files for cleanup
    
    def parse_text_output(self, text_content):
        """Parse the text output directly without relying on simple_parser"""
        parsed_data = {}
        
        # Define section patterns
        sections = {
            "EXECUTIVE_SUMMARY": r"EXECUTIVE_SUMMARY:\s*\n(.*?)(?=\n\n[A-Z_]+:|\nSENTIMENT:|\nTOPICS:|\nENTITIES:|\nRELATIONSHIPS:|\nANOMALIES:|\nCONTROVERSY_SCORE:|$)",
            "SENTIMENT": r"SENTIMENT:\s*\n(.*?)(?=\n\n[A-Z_]+:|\nTOPICS:|\nENTITIES:|\nRELATIONSHIPS:|\nANOMALIES:|\nCONTROVERSY_SCORE:|$)",
            "TOPICS": r"TOPICS:\s*\n(.*?)(?=\n\n[A-Z_]+:|\nENTITIES:|\nRELATIONSHIPS:|\nANOMALIES:|\nCONTROVERSY_SCORE:|$)",
            "ENTITIES": r"ENTITIES:\s*\n(.*?)(?=\n\n[A-Z_]+:|\nRELATIONSHIPS:|\nANOMALIES:|\nCONTROVERSY_SCORE:|$)",
            "RELATIONSHIPS": r"RELATIONSHIPS:\s*\n(.*?)(?=\n\n[A-Z_]+:|\nANOMALIES:|\nCONTROVERSY_SCORE:|$)",
            "ANOMALIES": r"ANOMALIES:\s*\n(.*?)(?=\n\n[A-Z_]+:|\nCONTROVERSY_SCORE:|$)",
            "CONTROVERSY_SCORE": r"CONTROVERSY_SCORE:\s*\n(.*?)(?=\n\n[A-Z_]+:|$)"
        }
        
        # Extract each section
        for section, pattern in sections.items():
            match = re.search(pattern, text_content, re.DOTALL)
            if match:
                content = match.group(1).strip()
                
                # Special handling for different sections
                if section == "EXECUTIVE_SUMMARY":
                    parsed_data["executive_summary"] = content
                elif section == "SENTIMENT":
                    # Parse sentiment data
                    sentiment_data = {}
                    sentiment_patterns = {
                        "positive": r"Positive:\s*(\d+)%\s*-\s*(.*)",
                        "negative": r"Negative:\s*(\d+)%\s*-\s*(.*)",
                        "neutral": r"Neutral:\s*(\d+)%\s*-\s*(.*)"
                    }
                    
                    for sentiment_type, pattern in sentiment_patterns.items():
                        sentiment_match = re.search(pattern, content)
                        if sentiment_match:
                            percentage = int(sentiment_match.group(1))
                            reasoning = sentiment_match.group(2).strip()
                            sentiment_data[sentiment_type] = {
                                "percentage": percentage,
                                "reasoning": reasoning
                            }
                    
                    parsed_data["sentiment_analysis"] = sentiment_data
                elif section == "TOPICS":
                    # Parse topics
                    topics = []
                    topic_lines = content.split('\n')
                    for line in topic_lines:
                        line = line.strip()
                        if line and line[0].isdigit():
                            # Remove numbering and clean up
                            topic = re.sub(r'^\d+\.\s*', '', line).strip()
                            if topic:
                                topics.append(topic)
                    parsed_data["topics"] = topics
                elif section == "ENTITIES":
                    # Parse entities
                    entities = []
                    entity_lines = content.split('\n')
                    for line in entity_lines:
                        line = line.strip()
                        if line and line[0].isdigit():
                            # Remove numbering and clean up
                            entity = re.sub(r'^\d+\.\s*', '', line).strip()
                            if entity:
                                entities.append(entity)
                    parsed_data["entity_recognition"] = entities
                elif section == "RELATIONSHIPS":
                    # Parse relationships - fixed to handle both numbered and unnumbered items
                    relationships = []
                    relationship_lines = content.split('\n')
                    for line in relationship_lines:
                        line = line.strip()
                        if line:
                            # Remove numbering if present
                            relationship = re.sub(r'^\d+\.\s*', '', line).strip()
                            if relationship:
                                # Try to parse the relationship format
                                rel_match = re.match(r'(.+?)\s*->\s*(.+?):\s*(.+)', relationship)
                                if rel_match:
                                    entity1 = rel_match.group(1).strip()
                                    entity2 = rel_match.group(2).strip()
                                    relationship_text = rel_match.group(3).strip()
                                    relationships.append({
                                        "entity1": entity1,
                                        "entity2": entity2,
                                        "relationship": relationship_text
                                    })
                                else:
                                    # If it doesn't match the expected format, add it as a string
                                    relationships.append(relationship)
                    parsed_data["relationship_extraction"] = relationships
                elif section == "ANOMALIES":
                    # Parse anomalies
                    anomalies = []
                    if content.strip() == "None detected":
                        anomalies = []
                    else:
                        anomaly_lines = content.split('\n')
                        for line in anomaly_lines:
                            line = line.strip()
                            if line and line[0].isdigit():
                                # Remove numbering and clean up
                                anomaly = re.sub(r'^\d+\.\s*', '', line).strip()
                                if anomaly:
                                    anomalies.append({"description": anomaly})
                    parsed_data["anomaly_detection"] = anomalies
                elif section == "CONTROVERSY_SCORE":
                    # Parse controversy score
                    controversy_match = re.search(r'(\d+\.?\d*)/1\.0\s*-\s*(.*)', content)
                    if controversy_match:
                        score = float(controversy_match.group(1))
                        explanation = controversy_match.group(2).strip()
                        parsed_data["controversy_score"] = {
                            "value": score,
                            "explanation": explanation
                        }
        
        return parsed_data
    
    def create_sentiment_pie_chart(self, sentiment_data):
        """Create a pie chart for sentiment analysis using matplotlib"""
        if not sentiment_data:
            return None
        
        # Extract sentiment percentages
        labels = []
        sizes = []
        colors_list = []
        
        if "positive" in sentiment_data:
            labels.append("Positive")
            sizes.append(sentiment_data["positive"].get("percentage", 0))
            colors_list.append("#2ca02c")
        
        if "negative" in sentiment_data:
            labels.append("Negative")
            sizes.append(sentiment_data["negative"].get("percentage", 0))
            colors_list.append("#d62728")
        
        if "neutral" in sentiment_data:
            labels.append("Neutral")
            sizes.append(sentiment_data["neutral"].get("percentage", 0))
            colors_list.append("#7f7f7f")
        
        # Create the pie chart
        fig, ax = plt.subplots(figsize=(6, 4))
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_list, autopct='%1.1f%%', startangle=90)
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
        
        # Add a title
        ax.set_title("Sentiment Analysis Distribution", fontsize=14, pad=20)
        
        # Improve text appearance
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_weight('bold')
        
        # Save to a temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        plt.savefig(temp_file.name, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        
        self.temp_files.append(temp_file.name)  # Track for cleanup
        return temp_file.name
    
    def create_sentiment_bar_chart(self, sentiment_data):
        """Create a bar chart for sentiment analysis using matplotlib"""
        if not sentiment_data:
            return None
        
        # Extract sentiment percentages
        labels = []
        values = []
        colors_list = []
        
        if "positive" in sentiment_data:
            labels.append("Positive")
            values.append(sentiment_data["positive"].get("percentage", 0))
            colors_list.append("#2ca02c")
        
        if "negative" in sentiment_data:
            labels.append("Negative")
            values.append(sentiment_data["negative"].get("percentage", 0))
            colors_list.append("#d62728")
        
        if "neutral" in sentiment_data:
            labels.append("Neutral")
            values.append(sentiment_data["neutral"].get("percentage", 0))
            colors_list.append("#7f7f7f")
        
        # Create the bar chart
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(labels, values, color=colors_list)
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height}%', ha='center', va='bottom', fontsize=10)
        
        # Set y-axis limit
        ax.set_ylim(0, max(values) * 1.2 if values else 100)
        
        # Add a title and labels
        ax.set_title("Sentiment Analysis Distribution", fontsize=14, pad=20)
        ax.set_ylabel("Percentage (%)", fontsize=12)
        
        # Remove x-axis ticks
        ax.tick_params(axis='x', which='both', length=0)
        
        # Save to a temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        plt.savefig(temp_file.name, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        
        self.temp_files.append(temp_file.name)  # Track for cleanup
        return temp_file.name
    
    def create_controversy_gauge_chart(self, controversy_data):
        """Create a gauge chart for controversy score"""
        if not controversy_data:
            return None
        
        score = controversy_data.get("value", 0)
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
        
        # Create a gauge chart
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Create a gauge-like visualization
        theta = np.linspace(0, np.pi, 100)
        r = 0.5
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Draw the background arc
        ax.plot(x, y, color='lightgray', linewidth=20)
        
        # Draw the filled arc based on percentage
        fill_theta = np.linspace(0, np.pi * display_score, 100)
        fill_x = r * np.cos(fill_theta)
        fill_y = r * np.sin(fill_theta)
        ax.plot(fill_x, fill_y, color=color, linewidth=20)
        
        # Add the percentage text
        ax.text(0, -0.2, f"{display_score*100:.1f}%", ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(0, -0.35, label, ha='center', va='center', fontsize=12)
        
        # Remove axes
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.5, 0.6)
        ax.axis('off')
        
        # Add a title
        ax.set_title("Controversy Score", fontsize=14, pad=20)
        
        # Save to a temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        plt.savefig(temp_file.name, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        
        self.temp_files.append(temp_file.name)  # Track for cleanup
        return temp_file.name
    
    def generate_pdf_report(self, analysis_data: Dict[str, Any], output_path: str = None) -> str:
        """Generate a PDF report from the analysis data"""
        
        # Create output path if not provided
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"Multiverse_Insights_Report_{timestamp}.pdf"
        
        # Create the PDF document
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        story = []
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=32,
            spaceAfter=15,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1A237E'),  # Deep blue
            fontName='Helvetica-Bold',
            leading=44  # Line spacing
        )
        
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading3'],
            fontSize=16,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#3949AB'),  # Medium blue
            fontName='Helvetica',
            leading=20
        )
        
        report_type_style = ParagraphStyle(
            'ReportType',
            parent=styles['Heading2'],
            fontSize=22,
            spaceAfter=8,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#5E35B1'),  # Purple
            fontName='Helvetica-Bold',
            leading=28
        )
        
        date_style = ParagraphStyle(
            'DateStyle',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=40,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#757575'),  # Gray
            fontName='Helvetica'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=18,
            spaceAfter=12,
            textColor=colors.HexColor('#1f77b4')
        )
        
        subheading_style = ParagraphStyle(
            'CustomSubheading',
            parent=styles['Heading3'],
            fontSize=14,
            spaceAfter=6,
            textColor=colors.HexColor('#ff7f0e')
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            alignment=TA_JUSTIFY
        )
        
        # Create a decorative line
        line_data = [
            ['', ''],
            ['', '']
        ]
        
        line_table = Table(line_data, colWidths=[2*inch, 2*inch])
        line_table.setStyle(TableStyle([
            ('LINEABOVE', (0, 0), (-1, 0), 1, colors.HexColor('#E0E0E0')),
            ('LINEBELOW', (0, 1), (-1, 1), 1, colors.HexColor('#E0E0E0')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        # Add title page
        story.append(Spacer(1, 2*inch))
        
        # Add logo placeholder (you can replace this with an actual logo)
        # For now, we'll add a simple text placeholder
        logo_style = ParagraphStyle(
            'LogoStyle',
            parent=styles['Normal'],
            fontSize=48,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#7C4DFF'),  # Purple accent
            fontName='Helvetica-Bold'
        )
        #story.append(Paragraph("MI", logo_style))
        
        # Main title
        story.append(Paragraph("Multiverse Insights", title_style))
        
        # Decorative line
        story.append(line_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Subtitle
        story.append(Paragraph("A real-time social media analyzer", subtitle_style))
        
        # Report type
        story.append(Paragraph("Social Media Analysis Report", report_type_style))
        
        # Date with more styling
        date_text = f"<font name='Helvetica-Oblique'>Generated on: {datetime.now().strftime('%B %d, %Y')}</font>"
        story.append(Paragraph(date_text, date_style))
        
        # Add more vertical spacing
        story.append(Spacer(1, 1.2*inch))
        
        # Add a footer with additional information
        footer_data = [
            ['Confidential & Proprietary', ''],
            ['© 2025 Multiverse Insights', 'Version 1.0'],
            ['All rights reserved', 'Page 1 of 1']
        ]
        
        footer_table = Table(footer_data, colWidths=[3*inch, 3*inch])
        footer_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#9E9E9E')),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        
        story.append(footer_table)
        story.append(PageBreak())
        
        # Add project brief
        story.append(Paragraph("Project Overview", heading_style))
        project_brief = """
        Multiverse Insights is a comprehensive social media analysis platform that transforms raw social media data 
        into actionable intelligence. Our platform leverages advanced AI technologies, including Microsoft's Phi 3 Mini model, 
        to analyze multimodal data from YouTube, Reddit, and other social platforms. The system provides insights into 
        market trends, consumer behaviors, and cultural phenomena through sentiment analysis, entity recognition, 
        relationship extraction, anomaly detection, and controversy scoring.
        """
        story.append(Paragraph(project_brief, body_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Add Executive Summary
        if "executive_summary" in analysis_data:
            story.append(Paragraph("Executive Summary", heading_style))
            exec_summary = analysis_data.get("executive_summary", "No executive summary available")
            
            # Format the executive summary
            if "- " in exec_summary:
                # Split by bullet points and create a formatted list
                points = [point.strip() for point in exec_summary.split('- ') if point.strip()]
                for point in points:
                    story.append(Paragraph(f"• {point}", body_style))
            else:
                story.append(Paragraph(exec_summary, body_style))
            
            story.append(Spacer(1, 0.2*inch))
        
        # Add Sentiment Analysis
        if "sentiment_analysis" in analysis_data:
            story.append(Paragraph("Sentiment Analysis", heading_style))
            sentiment_data = analysis_data.get("sentiment_analysis", {})
            
            # Create sentiment charts
            pie_chart_path = self.create_sentiment_pie_chart(sentiment_data)
            bar_chart_path = self.create_sentiment_bar_chart(sentiment_data)
            
            # Add charts to the report
            if pie_chart_path:
                story.append(Image(pie_chart_path, width=5*inch, height=3.3*inch))
            
            if bar_chart_path:
                story.append(Image(bar_chart_path, width=5*inch, height=3.3*inch))
            
            # Add sentiment details
            for sentiment_type, data in sentiment_data.items():
                if isinstance(data, dict) and "percentage" in data and "reasoning" in data:
                    story.append(Paragraph(f"{sentiment_type.capitalize()}: {data['percentage']}%", subheading_style))
                    story.append(Paragraph(data['reasoning'], body_style))
            
            story.append(Spacer(1, 0.2*inch))
        
        # Add Topics
        if "topics" in analysis_data:
            story.append(Paragraph("Key Topics", heading_style))
            topics = analysis_data.get("topics", [])
            
            if isinstance(topics, list):
                for i, topic in enumerate(topics, 1):
                    story.append(Paragraph(f"{i}. {topic}", body_style))
            else:
                story.append(Paragraph(str(topics), body_style))
            
            story.append(Spacer(1, 0.2*inch))
        
        # Add Entities
        if "entity_recognition" in analysis_data:
            story.append(Paragraph("Recognized Entities", heading_style))
            entities = analysis_data.get("entity_recognition", [])
            
            if isinstance(entities, list):
                for i, entity in enumerate(entities, 1):
                    if isinstance(entity, str):
                        # Check if the entity contains a description (separated by -)
                        if " - " in entity:
                            parts = entity.split(" - ", 1)
                            if len(parts) == 2:
                                name = parts[0].strip()
                                description = parts[1].strip()
                                story.append(Paragraph(f"{i}. <b>{name}</b>: {description}", body_style))
                            else:
                                story.append(Paragraph(f"{i}. {entity}", body_style))
                        else:
                            # Just a plain entity name
                            story.append(Paragraph(f"{i}. {entity}", body_style))
                    elif isinstance(entity, dict):
                        if "name" in entity and "description" in entity:
                            story.append(Paragraph(f"{i}. <b>{entity['name']}</b>: {entity['description']}", body_style))
                        else:
                            story.append(Paragraph(f"{i}. {str(entity)}", body_style))
            elif isinstance(entities, str):
                # If entities is a single string, split it into individual entities
                entity_list = [e.strip() for e in entities.split('\n') if e.strip()]
                for i, entity in enumerate(entity_list, 1):
                    story.append(Paragraph(f"{i}. {entity}", body_style))
            
            story.append(Spacer(1, 0.2*inch))
        
        # Add Relationships - Fixed to properly display relationships
        if "relationship_extraction" in analysis_data:
            story.append(Paragraph("Relationships", heading_style))
            relationships = analysis_data.get("relationship_extraction", [])
            
            if not relationships:
                story.append(Paragraph("No relationships detected", body_style))
            else:
                for i, rel in enumerate(relationships, 1):
                    if isinstance(rel, dict):
                        # Handle the specific format with entity1, entity2, and relationship keys
                        if "entity1" in rel and "entity2" in rel and "relationship" in rel:
                            entity1 = rel.get("entity1", "")
                            entity2 = rel.get("entity2", "")
                            relationship_text = rel.get("relationship", "")
                            story.append(Paragraph(f"{i}. <b>{entity1}</b> → <b>{entity2}</b>: {relationship_text}", body_style))
                        # Handle other dictionary formats
                        elif "entity" in rel and "description" in rel:
                            story.append(Paragraph(f"{i}. <b>{rel['entity']}</b>: {rel['description']}", body_style))
                        elif "name" in rel and "description" in rel:
                            story.append(Paragraph(f"{i}. <b>{rel['name']}</b>: {rel['description']}", body_style))
                        else:
                            # Handle dictionary with unknown structure
                            story.append(Paragraph(f"{i}. {str(rel)}", body_style))
                    elif isinstance(rel, str):
                        # Handle string format - split on colon if present
                        if ":" in rel:
                            parts = rel.split(":", 1)
                            if len(parts) == 2:
                                story.append(Paragraph(f"{i}. <b>{parts[0].strip()}</b>: {parts[1].strip()}", body_style))
                            else:
                                story.append(Paragraph(f"{i}. {rel}", body_style))
                        else:
                            story.append(Paragraph(f"{i}. {rel}", body_style))
            
            story.append(Spacer(1, 0.2*inch))
        
        # Add Anomalies
        if "anomaly_detection" in analysis_data:
            story.append(Paragraph("Detected Anomalies", heading_style))
            anomalies = analysis_data.get("anomaly_detection", [])
            
            if not anomalies:
                story.append(Paragraph("No anomalies detected", body_style))
            else:
                for i, anomaly in enumerate(anomalies, 1):
                    if "description" in anomaly:
                        story.append(Paragraph(f"{i}. {anomaly['description']}", body_style))
                    elif isinstance(anomaly, str):
                        story.append(Paragraph(f"{i}. {anomaly}", body_style))
            
            story.append(Spacer(1, 0.2*inch))
        
        # Add Controversy Score
        if "controversy_score" in analysis_data:
            story.append(Paragraph("Controversy Score", heading_style))
            controversy_data = analysis_data.get("controversy_score", {})
            
            # Create controversy gauge chart
            gauge_chart_path = self.create_controversy_gauge_chart(controversy_data)
            
            # Add chart to the report
            if gauge_chart_path:
                story.append(Image(gauge_chart_path, width=5*inch, height=3.3*inch))
            
            # Add controversy details
            if isinstance(controversy_data, dict) and "value" in controversy_data and "explanation" in controversy_data:
                score = controversy_data.get("value", 0)
                explanation = controversy_data.get("explanation", "")
                
                # Determine level based on score
                display_score = min(score, 1.0)
                if display_score < 0.3:
                    level = "Low Controversy"
                elif display_score < 0.7:
                    level = "Medium Controversy"
                else:
                    level = "High Controversy"
                
                story.append(Paragraph(f"Score: {score}/1.0 ({level})", subheading_style))
                story.append(Paragraph(f"Explanation: {explanation}", body_style))
            
            story.append(Spacer(1, 0.2*inch))
        
        # Add conclusion
        story.append(Paragraph("Conclusion", heading_style))
        conclusion = """
        This report provides a comprehensive analysis of social media data using the Multiverse Insights platform. 
        The analysis includes sentiment analysis, topic identification, entity recognition, relationship extraction, 
        anomaly detection, and controversy scoring. These insights can help in understanding market trends, 
        consumer behavior, and cultural phenomena, enabling data-driven decision making for businesses and researchers.
        """
        story.append(Paragraph(conclusion, body_style))
        
        # Build the PDF
        doc.build(story)
        
        # Clean up temporary files
        self.cleanup_temp_files()
        
        return output_path
    
    def cleanup_temp_files(self):
        """Clean up temporary files created during PDF generation"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                print(f"Warning: Could not delete temporary file {temp_file}: {str(e)}")
        
        self.temp_files = []
    
    def generate_pdf_from_output_file(self, output_file_path="final_output.txt", pdf_path=None):
        """Generate PDF from the parsed data in final_output.txt"""
        try:
            # Read the final output file
            with open(output_file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            # Parse the text content using our custom parser
            analysis_data = self.parse_text_output(text_content)
            
            if not analysis_data:
                print("❌ No analysis data found in the parsed output")
                return None
            
            # Generate the PDF
            print("📄 Generating PDF report...")
            pdf_path = self.generate_pdf_report(analysis_data, pdf_path)
            
            print(f"✅ PDF report generated successfully: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            print(f"❌ Error generating PDF: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

# Create a global instance
pdf_generator = PDFReportGenerator()

# Function to generate a test PDF
def generate_test_pdf():
    """Generate a test PDF from the dummy data file"""
    # Generate the PDF
    pdf_path = pdf_generator.generate_pdf_from_output_file("raw_output.txt")
    
    if pdf_path:
        print(f"✅ Test PDF generated successfully: {pdf_path}")
        return pdf_path
    else:
        print("❌ Failed to generate test PDF")
        return None

if __name__ == "__main__":
    generate_test_pdf()
