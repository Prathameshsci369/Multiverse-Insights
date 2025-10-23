# simple_parser.py
import re
import json
import sys
from typing import Dict, Any

class SimpleParser:
    def __init__(self, debug=False):
        self.debug = debug
    
    def debug_print(self, message):
        if self.debug:
            print(f"[DEBUG] {message}")
    
    def parse_simple_format(self, text: str) -> Dict[str, Any]:
        """Parse the simple format output from the analysis"""
        result = {"multiverse_combined": {}}
        
        # Debug: Print the first 500 characters of the text
        if self.debug:
            print(f"[DEBUG] First 500 chars of text: {text[:500]}")
        
        # Try multiple patterns for executive summary
        # Pattern 1: Standard format with EXECUTIVE_SUMMARY:
        exec_match = re.search(r'EXECUTIVE_SUMMARY:\s*\n?(.*?)(?=\nSENTIMENT:|\nTOPICS:|\nENTITIES:|\nRELATIONSHIPS:|\nANOMALIES:|\nCONTROVERSY_SCORE:|\Z)', text, re.DOTALL | re.IGNORECASE)
        
        # Pattern 2: Format with # --- Executive summary ---
        if not exec_match:
            exec_match = re.search(r'#\s*---?\s*Executive\s*summary\s*---?\s*\n?(.*?)(?=\nSENTIMENT:|\nTOPICS:|\nENTITIES:|\nRELATIONSHIPS:|\nANOMALIES:|\nCONTROVERSY_SCORE:|\Z)', text, re.DOTALL | re.IGNORECASE)
        
        # Pattern 3: Format with just "Executive Summary" or similar
        if not exec_match:
            exec_match = re.search(r'Executive\s*summary\s*[:\-]?\s*\n?(.*?)(?=\nSENTIMENT:|\nTOPICS:|\nENTITIES:|\nRELATIONSHIPS:|\nANOMALIES:|\nCONTROVERSY_SCORE:|\Z)', text, re.DOTALL | re.IGNORECASE)
        
        # Pattern 4: If none of the above work, try to extract the first paragraph as executive summary
        if not exec_match:
            # Look for the first non-empty line that doesn't start with a section header
            lines = text.split('\n')
            summary_lines = []
            for line in lines:
                line = line.strip()
                if line and not re.match(r'^(SENTIMENT|TOPICS|ENTITIES|RELATIONSHIPS|ANOMALIES|CONTROVERSY_SCORE|EXECUTIVE_SUMMARY)', line, re.IGNORECASE):
                    summary_lines.append(line)
                    # Stop if we've collected a reasonable amount of text
                    if len(' '.join(summary_lines)) > 200:
                        break
                elif summary_lines:  # Stop if we've started collecting and hit a section header
                    break
            
            if summary_lines:
                result["multiverse_combined"]["executive_summary"] = ' '.join(summary_lines)
                self.debug_print("Extracted executive summary from first paragraph")
        
        if exec_match:
            result["multiverse_combined"]["executive_summary"] = exec_match.group(1).strip()
            self.debug_print("Extracted executive summary")
        
        # Parse sentiment analysis
        sentiment_match = re.search(r'SENTIMENT:\s*\n?(.*?)(?=\nTOPICS:|\nENTITIES:|\nRELATIONSHIPS:|\nANOMALIES:|\nCONTROVERSY_SCORE:|\Z)', text, re.DOTALL | re.IGNORECASE)
        if sentiment_match:
            sentiment_text = sentiment_match.group(1).strip()
            self.debug_print(f"[SENT-DBG] Found sentiment section: {sentiment_text}")
            
            # Extract positive, negative, and neutral sentiments
            sentiment_data = {}
            
            # Pattern for "X% - Reasoning" format
            pos_match = re.search(r'Positive:\s*(\d+)%\s*-\s*(.*?)(?=Negative:|Neutral:|$)', sentiment_text, re.IGNORECASE | re.DOTALL)
            if pos_match:
                sentiment_data["positive"] = {
                    "percentage": int(pos_match.group(1)),
                    "reasoning": pos_match.group(2).strip()
                }
                self.debug_print(f"[SENT-DBG] Extracted positive: {pos_match.group(1)}% - {pos_match.group(2).strip()}")
            
            neg_match = re.search(r'Negative:\s*(\d+)%\s*-\s*(.*?)(?=Positive:|Neutral:|$)', sentiment_text, re.IGNORECASE | re.DOTALL)
            if neg_match:
                sentiment_data["negative"] = {
                    "percentage": int(neg_match.group(1)),
                    "reasoning": neg_match.group(2).strip()
                }
                self.debug_print(f"[SENT-DBG] Extracted negative: {neg_match.group(1)}% - {neg_match.group(2).strip()}")
            
            neu_match = re.search(r'Neutral:\s*(\d+)%\s*-\s*(.*?)(?=Positive:|Negative:|$)', sentiment_text, re.IGNORECASE | re.DOTALL)
            if neu_match:
                sentiment_data["neutral"] = {
                    "percentage": int(neu_match.group(1)),
                    "reasoning": neu_match.group(2).strip()
                }
                self.debug_print(f"[SENT-DBG] Extracted neutral: {neu_match.group(1)}% - {neu_match.group(2).strip()}")
            
            result["multiverse_combined"]["sentiment_analysis"] = sentiment_data
            self.debug_print("Extracted sentiment analysis")
        
        # Parse topics
        topics_match = re.search(r'TOPICS:\s*\n?(.*?)(?=\nENTITIES:|\nRELATIONSHIPS:|\nANOMALIES:|\nCONTROVERSY_SCORE:|\Z)', text, re.DOTALL | re.IGNORECASE)
        if topics_match:
            topics_text = topics_match.group(1).strip()
            # Extract topics using bullet points or numbered lists
            topics = re.findall(r'[\•\-\*]\s*(.*?)(?=\n[\•\-\*]|\n\n|\Z)', topics_text, re.DOTALL)
            topics = [t.strip() for t in topics if t.strip()]
            
            # If no bullet points found, try numbered list
            if not topics:
                topics = re.findall(r'\d+\.\s*(.*?)(?=\n\d+\.|\n\n|\Z)', topics_text, re.DOTALL)
                topics = [t.strip() for t in topics if t.strip()]
            
            # If still no topics, split by newlines
            if not topics:
                topics = [line.strip() for line in topics_text.split('\n') if line.strip()]
            
            result["multiverse_combined"]["topics"] = topics
            self.debug_print(f"Extracted {len(topics)} topics")
        
        # Parse entities
        entities_match = re.search(r'ENTITIES:\s*\n?(.*?)(?=\nRELATIONSHIPS:|\nANOMALIES:|\nCONTROVERSY_SCORE:|\Z)', text, re.DOTALL | re.IGNORECASE)
        if entities_match:
            entities_text = entities_match.group(1).strip()
            # Extract entities using bullet points or numbered lists
            entities = re.findall(r'[\•\-\*]\s*(.*?)(?=\n[\•\-\*]|\n\n|\Z)', entities_text, re.DOTALL)
            entities = [e.strip() for e in entities if e.strip()]
            
            # If no bullet points found, try numbered list
            if not entities:
                entities = re.findall(r'\d+\.\s*(.*?)(?=\n\d+\.|\n\n|\Z)', entities_text, re.DOTALL)
                entities = [e.strip() for e in entities if e.strip()]
            
            # If still no entities, split by newlines
            if not entities:
                entities = [line.strip() for line in entities_text.split('\n') if line.strip()]
            
            result["multiverse_combined"]["entity_recognition"] = entities
            self.debug_print(f"Extracted {len(entities)} entities")
        
        # Parse relationships
        relationships_match = re.search(r'RELATIONSHIPS:\s*\n?(.*?)(?=\nANOMALIES:|\nCONTROVERSY_SCORE:|\Z)', text, re.DOTALL | re.IGNORECASE)
        if relationships_match:
            relationships_text = relationships_match.group(1).strip()
            # Extract relationships using bullet points or numbered lists
            relationships = re.findall(r'[\•\-\*]\s*(.*?)(?=\n[\•\-\*]|\n\n|\Z)', relationships_text, re.DOTALL)
            relationships = [r.strip() for r in relationships if r.strip()]
            
            # If no bullet points found, try numbered list
            if not relationships:
                relationships = re.findall(r'\d+\.\s*(.*?)(?=\n\d+\.|\n\n|\Z)', relationships_text, re.DOTALL)
                relationships = [r.strip() for r in relationships if r.strip()]
            
            # If still no relationships, split by newlines
            if not relationships:
                relationships = [line.strip() for line in relationships_text.split('\n') if line.strip()]
            
            # Parse each relationship into a structured format
            parsed_relationships = []
            for rel in relationships:
                # Try to extract entities and relationship description
                # Format: "Entity1 -> Entity2: Description"
                arrow_match = re.match(r'(.+?)\s*->\s*(.+?):\s*(.+)', rel)
                if arrow_match:
                    parsed_relationships.append({
                        "entity1": arrow_match.group(1).strip(),
                        "entity2": arrow_match.group(2).strip(),
                        "relationship": arrow_match.group(3).strip()
                    })
                else:
                    # Format: "Entity1 vs Entity2: Description"
                    vs_match = re.match(r'(.+?)\s+vs\s+(.+?):\s*(.+)', rel)
                    if vs_match:
                        parsed_relationships.append({
                            "entity1": vs_match.group(1).strip(),
                            "entity2": vs_match.group(2).strip(),
                            "relationship": vs_match.group(3).strip()
                        })
                    else:
                        # Format: "Entity: Description"
                        colon_match = re.match(r'(.+?):\s*(.+)', rel)
                        if colon_match:
                            parsed_relationships.append({
                                "entity1": colon_match.group(1).strip(),
                                "entity2": None,
                                "relationship": colon_match.group(2).strip()
                            })
                        else:
                            # Just a description, no entities
                            parsed_relationships.append({
                                "entity1": None,
                                "entity2": None,
                                "relationship": rel
                            })
            
            result["multiverse_combined"]["relationship_extraction"] = parsed_relationships
            self.debug_print(f"Extracted {len(parsed_relationships)} relationships")
        
        # Parse anomalies
        anomalies_match = re.search(r'ANOMALIES:\s*\n?(.*?)(?=\nCONTROVERSY_SCORE:|\Z)', text, re.DOTALL | re.IGNORECASE)
        if anomalies_match:
            anomalies_text = anomalies_match.group(1).strip()
            # Check if no anomalies were detected
            if re.search(r'none\s*detect|no\s*anomalies', anomalies_text, re.IGNORECASE):
                result["multiverse_combined"]["anomaly_detection"] = []
            else:
                # Extract anomalies using bullet points or numbered lists
                anomalies = re.findall(r'[\•\-\*]\s*(.*?)(?=\n[\•\-\*]|\n\n|\Z)', anomalies_text, re.DOTALL)
                anomalies = [a.strip() for a in anomalies if a.strip()]
                
                # If no bullet points found, try numbered list
                if not anomalies:
                    anomalies = re.findall(r'\d+\.\s*(.*?)(?=\n\d+\.|\n\n|\Z)', anomalies_text, re.DOTALL)
                    anomalies = [a.strip() for a in anomalies if a.strip()]
                
                # If still no anomalies, split by newlines
                if not anomalies:
                    anomalies = [line.strip() for line in anomalies_text.split('\n') if line.strip()]
                
                # Parse each anomaly into a structured format
                parsed_anomalies = []
                for anomaly in anomalies:
                    parsed_anomalies.append({
                        "description": anomaly
                    })
                
                result["multiverse_combined"]["anomaly_detection"] = parsed_anomalies
            
            self.debug_print(f"Extracted {len(result['multiverse_combined']['anomaly_detection'])} anomalies")
        
        # Parse controversy score
        controversy_match = re.search(r'CONTROVERSY_SCORE:\s*\n?(.*?)(?=\Z)', text, re.DOTALL | re.IGNORECASE)
        if controversy_match:
            controversy_text = controversy_match.group(1).strip()
            # Extract score and explanation
            score_match = re.search(r'(\d+\.?\d*)\s*/\s*(\d+\.?\d*)\s*-\s*(.+)', controversy_text)
            if score_match:
                score = float(score_match.group(1)) / float(score_match.group(2))
                explanation = score_match.group(3).strip()
                result["multiverse_combined"]["controversy_score"] = {
                    "value": score,
                    "explanation": explanation
                }
                self.debug_print(f"Extracted controversy score: {score}")
        
        return result

def parse_simple_format(text: str, debug=False) -> Dict[str, Any]:
    """Parse the simple format output from the analysis"""
    parser = SimpleParser(debug=debug)
    return parser.parse_simple_format(text)

def parse_and_display(input_file: str, output_file: str = None, debug: bool = False):
    """
    Parse the simple format output from the analysis and display or save the results
    
    Args:
        input_file: Path to the input file containing the analysis text
        output_file: Path to save the parsed JSON results (optional)
        debug: Whether to enable debug output
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    result = parse_simple_format(text, debug=debug)
    
    # Save the parsed data to a JSON file if output_file is provided
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"✅ Parsed data saved to {output_file}")
    
    # Display the parsed results
    print("\n" + "="*80)
    print("PARSED ANALYSIS RESULTS")
    print("="*80)
    
    if "multiverse_combined" in result:
        analysis_data = result["multiverse_combined"]
        
        # Executive Summary
        if "executive_summary" in analysis_data:
            print("\nEXECUTIVE SUMMARY:")
            print("-" * 40)
            print(analysis_data["executive_summary"])
        
        # Sentiment Analysis
        if "sentiment_analysis" in analysis_data:
            print("\nSENTIMENT ANALYSIS:")
            print("-" * 40)
            sentiment = analysis_data["sentiment_analysis"]
            for sentiment_type, data in sentiment.items():
                if isinstance(data, dict) and "percentage" in data and "reasoning" in data:
                    print(f"{sentiment_type.capitalize()}: {data['percentage']}% - {data['reasoning']}")
        
        # Topics
        if "topics" in analysis_data:
            print("\nKEY TOPICS:")
            print("-" * 40)
            for i, topic in enumerate(analysis_data["topics"], 1):
                print(f"{i}. {topic}")
        
        # Entities
        if "entity_recognition" in analysis_data:
            print("\nRECOGNIZED ENTITIES:")
            print("-" * 40)
            for i, entity in enumerate(analysis_data["entity_recognition"], 1):
                print(f"{i}. {entity}")
        
        # Relationships
        if "relationship_extraction" in analysis_data:
            print("\nRELATIONSHIPS:")
            print("-" * 40)
            for i, rel in enumerate(analysis_data["relationship_extraction"], 1):
                if isinstance(rel, dict):
                    if rel.get("entity1") and rel.get("entity2"):
                        print(f"{i}. {rel['entity1']} ↔ {rel['entity2']}: {rel['relationship']}")
                    elif rel.get("entity1"):
                        print(f"{i}. {rel['entity1']}: {rel['relationship']}")
                    else:
                        print(f"{i}. {rel['relationship']}")
                else:
                    print(f"{i}. {rel}")
        
        # Anomalies
        if "anomaly_detection" in analysis_data:
            print("\nDETECTED ANOMALIES:")
            print("-" * 40)
            if not analysis_data["anomaly_detection"]:
                print("No anomalies detected")
            else:
                for i, anomaly in enumerate(analysis_data["anomaly_detection"], 1):
                    if isinstance(anomaly, dict) and "description" in anomaly:
                        print(f"{i}. {anomaly['description']}")
                    else:
                        print(f"{i}. {anomaly}")
        
        # Controversy Score
        if "controversy_score" in analysis_data:
            print("\nCONTROVERSY SCORE:")
            print("-" * 40)
            score_data = analysis_data["controversy_score"]
            if isinstance(score_data, dict) and "value" in score_data and "explanation" in score_data:
                print(f"Score: {score_data['value']}/1.0")
                print(f"Explanation: {score_data['explanation']}")
    
    print("\n" + "="*80)
    print("END OF ANALYSIS")
    print("="*80)
    
    return result

def main():
    if len(sys.argv) < 2:
        print("Usage: python simple_parser.py <input_file> [--output <output_file>] [--debug]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = None
    debug = "--debug" in sys.argv
    
    # Check if output file is specified
    if "--output" in sys.argv:
        output_index = sys.argv.index("--output")
        if output_index + 1 < len(sys.argv):
            output_file = sys.argv[output_index + 1]
    
    parse_and_display(input_file, output_file, debug)

if __name__ == "__main__":
    main()
