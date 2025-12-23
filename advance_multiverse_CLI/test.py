#!/usr/bin/env python3
"""
test_individual_chunks.py

A focused script to test individual chunk analysis and generate summaries.
This script implements a "Connect the Dots" strategy:
1. Checks if data size > 30,000 tokens.
2. If yes, splits into batches (Partial Summaries).
3. If no, processes as one unit.
4. Aggregates "dots" (key points) from partials and runs a final synthesis.

Usage:
    python test_individual_chunks.py

Author: Test Suite Generator
Date: 2023-11-15 (Updated)
"""

import os
import sys
import json
import time
import logging
import tempfile
import shutil
from datetime import datetime
import traceback

# Configuration
BATCH_TOKEN_THRESHOLD = 32768  # If total tokens > 30k, trigger batching
BATCH_SIZE_LIMIT = 25000       # Size of individual batches if batching is triggered
HUGE_LIMIT = 1000000           # Large limit for initial size check

# Set up detailed logging
def setup_logging():
    """Set up comprehensive logging for the test suite."""
    log_file = f"individual_chunks_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger('chunk_test')

# Initialize logger
logger = setup_logging()

# Import the modules we're testing
try:
    from final_analysis import combined_analysis, extract_key_points
    from preprocessing_8k import MultiFormatProcessor
    logger.info("Successfully imported required modules")
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure final_analysis.py and preprocessing_8k.py are in the same directory")
    sys.exit(1)


class IndividualChunkAnalyzer:
    """A class to analyze individual chunks and generate a cohesive final scenario."""
    
    def __init__(self):
        """Initialize the analyzer."""
        self.test_dir = tempfile.mkdtemp()
        self.results_dir = os.path.join(self.test_dir, "chunk_results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Define file paths directly in the code
        self.input_files = [
            "twitter_*.json",      # Twitter format
            "reddit_search_*.json",# Reddit format
            # Add other patterns as needed
        ]
        
        self.output_file = os.path.join(self.test_dir, "processed_chunks.json")
        self.synthesis_input_file = os.path.join(self.test_dir, "synthesis_dots.txt")
        self.final_report_file = os.path.join(self.results_dir, "FINAL_SCENARIO_REPORT.md")
        
        logger.info(f"Test directory created at: {self.test_dir}")
        logger.info(f"Results will be saved to: {self.results_dir}")
        
    def process_and_analyze_chunks(self):
        """Process files, handle batching logic, and synthesize results."""
        logger.info("="*80)
        logger.info("STARTING SMART ANALYSIS PROCESS")
        logger.info("="*80)
        
        try:
            # Step 1: Assess Total Token Count
            logger.info("Step 1: Assessing total data size...")
            
            # Run processor with a huge limit just to count tokens
            pre_processor = MultiFormatProcessor(
                output_file=self.output_file,
                max_tokens=HUGE_LIMIT 
            )
            pre_result = pre_processor.process_files(self.input_files)
            
            if not pre_result or len(pre_result['chunks']) == 0:
                logger.error("No content found in input files.")
                return False

            # Calculate total tokens across all files
            total_tokens = sum(c['token_count'] for c in pre_result['chunks'])
            logger.info(f"Total Content Size: {total_tokens} tokens")

            # Step 2: Determine Processing Strategy
            if total_tokens > BATCH_TOKEN_THRESHOLD:
                logger.info(f"⚠️ Data exceeds threshold ({BATCH_TOKEN_THRESHOLD}). Engaging BATCH MODE.")
                logger.info(f"Re-processing with batch size: {BATCH_SIZE_LIMIT}")
                
                # Re-run processor with strict batching limits
                processor = MultiFormatProcessor(
                    output_file=self.output_file,
                    max_tokens=BATCH_SIZE_LIMIT
                )
                result = processor.process_files(self.input_files)
            else:
                logger.info(f"✅ Data within threshold ({BATCH_TOKEN_THRESHOLD}). Engaging SINGLE PASS MODE.")
                # Use the result we already calculated (single massive chunk)
                result = pre_result

            chunks = result['chunks']
            metadata = result['metadata']
            logger.info(f"Prepared {len(chunks)} processing units.")

            # Step 3: Analyze Partial Units (The "Dots")
            logger.info("Step 3: Analyzing units to extract key points (Dots)...")
            chunk_summaries = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = chunk['chunk_id']
                logger.info(f"\n--- Analyzing Unit {chunk_id} ({i+1}/{len(chunks)}) ---")
                
                # Create temp file for this specific chunk
                temp_file = os.path.join(self.test_dir, f"temp_chunk_{chunk_id}.json")
                temp_data = {"metadata": metadata, "chunks": [chunk]}
                
                with open(temp_file, 'w') as f:
                    json.dump(temp_data, f, indent=2)
                
                # Analyze - Use extract_key_points for individual chunks
                chunk_summary = self._analyze_single_chunk(chunk_id, temp_file, chunk)
                
                # Cleanup temp
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                
                if chunk_summary:
                    chunk_summaries.append(chunk_summary)
                    # Save partial result
                    with open(os.path.join(self.results_dir, f"partial_{chunk_id}.txt"), 'w') as f:
                        f.write(chunk_summary)
                
                time.sleep(1) # Breath

            # Step 4: Connect the Dots (Final Synthesis)
            if len(chunk_summaries) > 0:
                logger.info("\nStep 4: Connecting the dots (Final Synthesis)...")
                final_scenario = self._perform_final_synthesis(chunk_summaries, metadata)
                
                # Print the Final Scenario to terminal
                print("\n" + "#"*80)
                print("FINAL SCENARIO PICTURE")
                print("#"*80)
                print(final_scenario)
                print("#"*80 + "\n")
                
                return True
            else:
                logger.error("No chunk summaries generated.")
                return False
            
        except Exception as e:
            logger.error(f"Error in main process: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def _analyze_single_chunk(self, chunk_id, temp_file, chunk_info):
        """Analyze a single chunk. This extracts the 'dots'."""
        try:
            start_time = time.time()
            # Use the new extract_key_points function for individual chunks
            analysis_result = extract_key_points(temp_file)
            analysis_time = time.time() - start_time
            
            if not analysis_result or "ERROR" in analysis_result:
                return None
            
            # Return the analysis result directly since it's already in the format we want
            return f"# Chunk {chunk_id} Key Points\n\n{analysis_result}"
            
        except Exception as e:
            logger.error(f"Chunk {chunk_id} failed: {e}")
            return None

    def _perform_final_synthesis(self, chunk_summaries, metadata):
        """
        Takes the partial summaries (dots) and generates a cohesive scenario.
        """
        try:
            # 1. Compile the "Dots" into a single context document
            synthesis_content = "AGGREGATED INTELLIGENCE FROM MULTIPLE DATA SEGMENTS:\n\n"
            
            for summary in chunk_summaries:
                synthesis_content += f"{summary}\n\n"
            
            # 2. Write these dots to a file
            with open(self.synthesis_input_file, 'w') as f:
                f.write(synthesis_content)
            
            logger.info(f"Consolidated 'dots' written to: {self.synthesis_input_file}")
            
            # 3. Create a temporary JSON file with the synthesis content for combined_analysis
            # We need to wrap the text in a structure that combined_analysis can understand
            temp_json_file = os.path.join(self.test_dir, "temp_synthesis.json")
            
            # Get source_types from metadata
            source_types = metadata.get('source_types', ['unknown'])
            
            # Create a minimal JSON structure with the synthesis content
            temp_data = {
                "metadata": {
                    "source": "synthesis",
                    "source_types": source_types,  # Add the missing source_types field
                    "created_at": datetime.now().isoformat()
                },
                "chunks": [
                    {
                        "chunk_id": "synthesis",
                        "content": synthesis_content,
                        "token_count": len(synthesis_content.split()) * 1.3,  # Rough estimate
                        "source_types": source_types  # Add source_types to chunk as well
                    }
                ]
            }
            
            with open(temp_json_file, 'w') as f:
                json.dump(temp_data, f, indent=2)
            
            # 4. Run the Final Analysis on the JSON file
            # Use combined_analysis with extract_key_points_only=False for the final synthesis
            logger.info("Running final synthesis on aggregated points...")
            final_scenario = combined_analysis(temp_json_file, extract_key_points_only=False)
            
            # Clean up the temporary JSON file
            if os.path.exists(temp_json_file):
                os.remove(temp_json_file)
            
            # 5. Save Final Report
            with open(self.final_report_file, 'w') as f:
                f.write(final_scenario)
                
            return final_scenario

        except Exception as e:
            logger.error(f"Error during final synthesis: {str(e)}")
            logger.error(traceback.format_exc())
            return "FAILED TO GENERATE FINAL SCENARIO"

    def cleanup(self):
        """Clean up temporary files."""
        try:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir)
                logger.info(f"Cleaned up temporary directory: {self.test_dir}")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


def main():
    """Main function."""
    analyzer = IndividualChunkAnalyzer()
    
    try:
        success = analyzer.process_and_analyze_chunks()
        
        if success:
            logger.info("\nProcess completed successfully!")
            keep = input("\nKeep result files? (y/n): ")
            if keep.lower() != 'y':
                analyzer.cleanup()
            else:
                logger.info(f"Results at: {analyzer.results_dir}")
        else:
            logger.error("\nProcess failed!")
            
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        analyzer.cleanup()
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        analyzer.cleanup()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
