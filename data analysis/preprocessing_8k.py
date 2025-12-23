import json
import re
import math
import logging
import time
import os
import glob
from typing import List, Dict, Any, Union, Tuple
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'multi_processor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessedRecord:
    """Data class for processed records"""
    source_file: str
    source_type: str
    original_index: int
    content: str
    tokens: int
    metadata: Dict[str, Any]

class MultiFormatProcessor:
    def __init__(self, output_file: str = "all_chunks.json", max_tokens: int = 8000):
        self.output_file = output_file
        self.max_tokens = max_tokens
        self.total_records = 0
        self.total_tokens = 0
        self.chunks_created = 0
        self.processed_records: List[ProcessedRecord] = []
        
    def calculate_tokens(self, text: str) -> int:
        """Calculate approximate tokens for text"""
        if not text or not text.strip():
            return 0
            
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Base calculation
        char_count = len(text)
        base_tokens = char_count / 3.5
        
        # Adjustments
        urls = re.findall(r'https?://\S+|www\.\S+', text)
        base_tokens += len(urls) * 15
        
        hashtags = re.findall(r'#\w+', text)
        base_tokens += len(hashtags) * 1.5
        
        special_chars = re.findall(r'[^\w\s.,!?;:\-\'"()\[\]{}@/]', text)
        base_tokens += len(special_chars) * 0.3
        
        return max(1, int(math.ceil(base_tokens)))
    
    def extract_reddit_data(self, file_path: str, data: List[Dict[str, Any]]) -> List[ProcessedRecord]:
        """Extract data from Reddit JSON format"""
        records = []
        
        for idx, item in enumerate(data):
            # Extract content from different possible fields
            content_parts = []
            
            # Title
            title = item.get('title', '').strip()
            if title:
                content_parts.append(f"Title: {title}")
            
            # Selftext/content
            selftext = item.get('selftext', '').strip()
            if not selftext:
                selftext = item.get('content', '').strip()
            
            if selftext:
                content_parts.append(f"Content: {selftext}")
            
            # Transcript (for YouTube-like data in Reddit files)
            transcript = item.get('transcript', '').strip()
            if transcript:
                content_parts.append(f"Transcript: {transcript}")
            
            # Combine all content
            content = "\n\n".join(content_parts)
            
            if not content.strip():
                continue
            
            # Extract metadata
            metadata = {
                'source': 'reddit',
                'author': item.get('author', ''),
                'score': item.get('score', 0),
                'comments': item.get('comments_count', item.get('comments', 0)),
                'url': item.get('url', ''),
                'created': item.get('created_utc', ''),
                'hashtags': item.get('hashtags', [])
            }
            
            # Clean metadata
            metadata = {k: v for k, v in metadata.items() if v}
            
            # Calculate tokens
            tokens = self.calculate_tokens(content)
            
            records.append(ProcessedRecord(
                source_file=os.path.basename(file_path),
                source_type='reddit',
                original_index=idx,
                content=content,
                tokens=tokens,
                metadata=metadata
            ))
            
            logger.debug(f"  Extracted Reddit record {idx}: {tokens} tokens")
        
        return records
    
    def extract_twitter_data(self, file_path: str, data: List[Dict[str, Any]]) -> List[ProcessedRecord]:
        """Extract data from Twitter/X JSON format"""
        records = []
        
        for idx, item in enumerate(data):
            # Extract content
            content = item.get('content', '').strip()
            if not content:
                content = item.get('tweet_text', '').strip()
            
            # Add user handle if available
            user_handle = item.get('user_handle', item.get('author', ''))
            if user_handle and content:
                content = f"From {user_handle}:\n{content}"
            
            if not content.strip():
                continue
            
            # Extract metadata
            metadata = {
                'source': 'twitter',
                'user': user_handle,
                'date': item.get('tweet_date', item.get('date', '')),
                'replies': item.get('replies', 0),
                'retweets': item.get('retweets', item.get('retweet_count', 0)),
                'likes': item.get('likes', item.get('like_count', 0)),
                'views': item.get('views', item.get('view_count', 0)),
                'hashtags': item.get('hashtags', []),
                'url': item.get('tweet_url', '')
            }
            
            # Clean metadata
            metadata = {k: v for k, v in metadata.items() if v}
            
            # Calculate tokens
            tokens = self.calculate_tokens(content)
            
            records.append(ProcessedRecord(
                source_file=os.path.basename(file_path),
                source_type='twitter',
                original_index=idx,
                content=content,
                tokens=tokens,
                metadata=metadata
            ))
            
            logger.debug(f"  Extracted Twitter record {idx}: {tokens} tokens")
        
        return records
    
    def extract_youtube_data(self, file_path: str, data: List[Dict[str, Any]]) -> List[ProcessedRecord]:
        """Extract data from YouTube JSON format"""
        records = []
        
        for idx, item in enumerate(data):
            # Extract content from multiple possible fields
            content_parts = []
            
            # Title
            title = item.get('title', '').strip()
            if title:
                content_parts.append(f"Title: {title}")
            
            # Transcript
            transcript = item.get('transcript', '').strip()
            if transcript:
                content_parts.append(f"Transcript: {transcript}")
            
            # Selftext/description
            selftext = item.get('selftext', item.get('description', '').strip())
            if selftext:
                content_parts.append(f"Description: {selftext}")
            
            # Combine content
            content = "\n\n".join(content_parts)
            
            if not content.strip():
                continue
            
            # Extract metadata
            metadata = {
                'source': 'youtube',
                'channel': item.get('channel_title', ''),
                'views': item.get('views', 0),
                'likes': item.get('likes', 0),
                'published': item.get('published_at', ''),
                'subscribers': item.get('subscribers', 0),
                'url': item.get('url', '')
            }
            
            # Clean metadata
            metadata = {k: v for k, v in metadata.items() if v}
            
            # Calculate tokens
            tokens = self.calculate_tokens(content)
            
            records.append(ProcessedRecord(
                source_file=os.path.basename(file_path),
                source_type='youtube',
                original_index=idx,
                content=content,
                tokens=tokens,
                metadata=metadata
            ))
            
            logger.debug(f"  Extracted YouTube record {idx}: {tokens} tokens")
        
        return records
    
    def extract_generic_data(self, file_path: str, data: List[Dict[str, Any]]) -> List[ProcessedRecord]:
        """Extract data from generic JSON format"""
        records = []
        
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            
            # Try to extract content from common field names
            content = ''
            content_fields = ['content', 'text', 'body', 'description', 'transcript', 'selftext']
            
            for field in content_fields:
                if field in item and item[field]:
                    content = str(item[field]).strip()
                    break
            
            # If no content found, try to stringify the entire item
            if not content:
                content = json.dumps(item, ensure_ascii=False)
            
            if not content.strip():
                continue
            
            # Use the item itself as metadata
            metadata = {
                'source': 'generic',
                'original_data': {k: v for k, v in item.items() if k not in content_fields}
            }
            
            # Calculate tokens
            tokens = self.calculate_tokens(content)
            
            records.append(ProcessedRecord(
                source_file=os.path.basename(file_path),
                source_type='generic',
                original_index=idx,
                content=content,
                tokens=tokens,
                metadata=metadata
            ))
            
            logger.debug(f"  Extracted generic record {idx}: {tokens} tokens")
        
        return records
    
    def detect_format(self, data: List[Dict[str, Any]]) -> str:
        """Detect the format of the JSON data"""
        if not data or not isinstance(data, list) or len(data) == 0:
            return 'generic'
        
        first_item = data[0]
        
        # Check for Twitter/X format
        twitter_fields = ['user_handle', 'tweet_date', 'content', 'replies', 'retweets', 'likes']
        if all(field in first_item for field in twitter_fields[:3]):
            return 'twitter'
        
        # Check for Reddit format
        reddit_fields = ['title', 'score', 'author', 'selftext', 'comments_count']
        reddit_count = sum(1 for field in reddit_fields if field in first_item)
        if reddit_count >= 3:
            return 'reddit'
        
        # Check for YouTube format
        youtube_fields = ['title', 'url', 'views', 'likes', 'published_at', 'channel_title', 'transcript']
        youtube_count = sum(1 for field in youtube_fields if field in first_item)
        if youtube_count >= 3:
            return 'youtube'
        
        return 'generic'
    
    def load_and_extract(self, file_path: str) -> List[ProcessedRecord]:
        """Load a JSON file and extract records based on format"""
        logger.info(f"  Processing: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                logger.warning(f"    File {file_path} doesn't contain a list, skipping")
                return []
            
            # Detect format
            format_type = self.detect_format(data)
            logger.info(f"    Detected format: {format_type}")
            
            # Extract based on format
            if format_type == 'twitter':
                records = self.extract_twitter_data(file_path, data)
            elif format_type == 'reddit':
                records = self.extract_reddit_data(file_path, data)
            elif format_type == 'youtube':
                records = self.extract_youtube_data(file_path, data)
            else:
                records = self.extract_generic_data(file_path, data)
            
            logger.info(f"    Extracted {len(records)} records")
            return records
            
        except json.JSONDecodeError as e:
            logger.error(f"    JSON decode error in {file_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"    Error processing {file_path}: {e}")
            return []
    
    def split_large_record(self, record: ProcessedRecord) -> List[Dict[str, Any]]:
        """Split a record that exceeds token limit"""
        parts = []
        text = record.content
        tokens_per_record = record.tokens
        
        if tokens_per_record <= self.max_tokens:
            return [{
                'text': text,
                'tokens': tokens_per_record,
                'record': record
            }]
        
        # Split by paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        current_part = ""
        current_tokens = 0
        part_num = 1
        
        for para in paragraphs:
            para_tokens = self.calculate_tokens(para)
            
            if current_tokens + para_tokens > self.max_tokens and current_part:
                parts.append({
                    'text': current_part,
                    'tokens': current_tokens,
                    'record': record,
                    'is_partial': True,
                    'part_num': part_num
                })
                part_num += 1
                current_part = para
                current_tokens = para_tokens
            else:
                current_part += "\n\n" + para if current_part else para
                current_tokens += para_tokens
        
        if current_part:
            parts.append({
                'text': current_part,
                'tokens': current_tokens,
                'record': record,
                'is_partial': True,
                'part_num': part_num
            })
        
        logger.debug(f"    Split {tokens_per_record} token record into {len(parts)} parts")
        return parts
    
    def create_optimized_chunks(self, records: List[ProcessedRecord]) -> List[Dict[str, Any]]:
        """Create optimized chunks using bin packing algorithm"""
        logger.info(f"Creating chunks (max {self.max_tokens:,} tokens each)")
        
        # Prepare all items (split large records first)
        all_items = []
        
        for record in records:
            if record.tokens > self.max_tokens:
                parts = self.split_large_record(record)
                all_items.extend(parts)
            else:
                all_items.append({
                    'text': record.content,
                    'tokens': record.tokens,
                    'record': record,
                    'is_partial': False,
                    'part_num': 0
                })
        
        logger.info(f"  Total items after splitting: {len(all_items)}")
        
        # Sort by token count (largest first)
        all_items.sort(key=lambda x: x['tokens'], reverse=True)
        
        # First Fit Decreasing algorithm
        chunks = []
        
        while all_items:
            current_chunk_items = []
            current_tokens = 0
            source_files = set()
            source_types = set()
            
            remaining_items = all_items.copy()
            
            for item in remaining_items:
                if current_tokens + item['tokens'] <= self.max_tokens:
                    current_chunk_items.append(item)
                    current_tokens += item['tokens']
                    source_files.add(item['record'].source_file)
                    source_types.add(item['record'].source_type)
                    all_items.remove(item)
            
            # Try to add more if chunk is less than 60% full
            if current_tokens < self.max_tokens * 0.6 and len(all_items) > 0:
                for item in all_items:
                    if current_tokens + item['tokens'] <= self.max_tokens:
                        current_chunk_items.append(item)
                        current_tokens += item['tokens']
                        source_files.add(item['record'].source_file)
                        source_types.add(item['record'].source_type)
                        all_items.remove(item)
                        break
            
            if current_chunk_items:
                # Create chunk text
                chunk_text = "\n\n" + "="*60 + "\n\n".join([
                    f"[Source: {item['record'].source_type} - {item['record'].source_file}]\n{item['text']}"
                    for item in current_chunk_items
                ])
                
                exact_tokens = self.calculate_tokens(chunk_text)
                
                # Get source records
                source_records = []
                for item in current_chunk_items:
                    source_records.append({
                        'file': item['record'].source_file,
                        'type': item['record'].source_type,
                        'original_index': item['record'].original_index,
                        'is_partial': item.get('is_partial', False),
                        'part_num': item.get('part_num', 0)
                    })
                
                chunks.append({
                    'chunk_id': len(chunks) + 1,
                    'content': chunk_text,
                    'tokens': exact_tokens,
                    'item_count': len(current_chunk_items),
                    'source_files': list(source_files),
                    'source_types': list(source_types),
                    'source_records': source_records,
                    'contains_partial': any(item.get('is_partial', False) for item in current_chunk_items),
                    'utilization': (exact_tokens / self.max_tokens) * 100
                })
        
        # Optimize by merging small chunks
        optimized_chunks = []
        small_chunks = [c for c in chunks if c['tokens'] < self.max_tokens * 0.3]
        large_chunks = [c for c in chunks if c['tokens'] >= self.max_tokens * 0.3]
        
        # Try to merge small chunks together
        while small_chunks:
            current = small_chunks.pop(0)
            merged = False
            
            for i, other in enumerate(small_chunks):
                if current['tokens'] + other['tokens'] <= self.max_tokens:
                    # Merge chunks
                    merged_content = current['content'] + "\n\n" + other['content']
                    merged_tokens = self.calculate_tokens(merged_content)
                    
                    optimized_chunks.append({
                        'chunk_id': len(optimized_chunks) + 1,
                        'content': merged_content,
                        'tokens': merged_tokens,
                        'item_count': current['item_count'] + other['item_count'],
                        'source_files': list(set(current['source_files'] + other['source_files'])),
                        'source_types': list(set(current['source_types'] + other['source_types'])),
                        'source_records': current['source_records'] + other['source_records'],
                        'contains_partial': current['contains_partial'] or other['contains_partial'],
                        'utilization': (merged_tokens / self.max_tokens) * 100
                    })
                    
                    del small_chunks[i]
                    merged = True
                    break
            
            # Try to add to large chunk
            if not merged:
                added = False
                for large in large_chunks:
                    if large['tokens'] + current['tokens'] <= self.max_tokens:
                        large['content'] += "\n\n" + current['content']
                        large['tokens'] = self.calculate_tokens(large['content'])
                        large['item_count'] += current['item_count']
                        large['source_files'] = list(set(large['source_files'] + current['source_files']))
                        large['source_types'] = list(set(large['source_types'] + current['source_types']))
                        large['source_records'].extend(current['source_records'])
                        large['contains_partial'] = large['contains_partial'] or current['contains_partial']
                        large['utilization'] = (large['tokens'] / self.max_tokens) * 100
                        added = True
                        break
                
                if not added:
                    optimized_chunks.append(current)
        
        # Add all large chunks
        optimized_chunks.extend(large_chunks)
        
        # Sort by chunk_id
        optimized_chunks.sort(key=lambda x: x['chunk_id'])
        
        # Renumber chunks
        for i, chunk in enumerate(optimized_chunks):
            chunk['chunk_id'] = i + 1
        
        self.chunks_created = len(optimized_chunks)
        
        logger.info(f"  Created {self.chunks_created} optimized chunks")
        
        return optimized_chunks
    
    def save_to_single_file(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Save all chunks to a single JSON file"""
        logger.info(f"Saving all chunks to: {self.output_file}")
        
        # Prepare output data
        output_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_input_files": len(set(r.source_file for r in self.processed_records)),
                "total_processed_records": len(self.processed_records),
                "total_tokens_original": self.total_tokens,
                "chunks_created": len(chunks),
                "max_tokens_per_chunk": self.max_tokens,
                "source_types": list(set(r.source_type for r in self.processed_records)),
                "input_files": list(set(r.source_file for r in self.processed_records))
            },
            "chunks": []
        }
        
        # Calculate statistics
        if chunks:
            chunk_tokens = [c['tokens'] for c in chunks]
            output_data["metadata"]["total_chunk_tokens"] = sum(chunk_tokens)
            output_data["metadata"]["avg_chunk_tokens"] = sum(chunk_tokens) / len(chunk_tokens)
            output_data["metadata"]["max_chunk_tokens"] = max(chunk_tokens)
            output_data["metadata"]["min_chunk_tokens"] = min(chunk_tokens)
            output_data["metadata"]["utilization_percentage"] = (
                sum(chunk_tokens) / (len(chunks) * self.max_tokens)
            ) * 100
        
        # Add chunks
        for chunk in chunks:
            chunk_data = {
                "chunk_id": chunk['chunk_id'],
                "token_count": chunk['tokens'],
                "utilization_percentage": chunk['utilization'],
                "item_count": chunk['item_count'],
                "contains_partial_content": chunk['contains_partial'],
                "source_files": chunk['source_files'],
                "source_types": chunk['source_types'],
                "content": chunk['content'],
                "source_records": chunk['source_records']
            }
            output_data["chunks"].append(chunk_data)
        
        # Write to file
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"  Saved {len(chunks)} chunks to {self.output_file}")
        logger.info(f"  File size: {os.path.getsize(self.output_file):,} bytes")
        
        return output_data
    
    def process_files(self, file_patterns: List[str]) -> Dict[str, Any]:
        """Main method to process multiple files"""
        logger.info("="*80)
        logger.info("MULTI-FORMAT JSON PROCESSOR")
        logger.info("="*80)
        
        total_start = time.time()
        
        # Find all files matching patterns
        all_files = []
        for pattern in file_patterns:
            matched_files = glob.glob(pattern)
            if matched_files:
                all_files.extend(matched_files)
            else:
                logger.warning(f"No files found matching pattern: {pattern}")
        
        if not all_files:
            logger.error("No files to process!")
            return {}
        
        logger.info(f"Found {len(all_files)} files to process:")
        for file in all_files:
            logger.info(f"  {file}")
        
        # Process each file
        all_records = []
        
        for file_path in all_files:
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue
            
            file_records = self.load_and_extract(file_path)
            all_records.extend(file_records)
        
        if not all_records:
            logger.error("No records extracted from any file!")
            return {}
        
        # Update statistics
        self.processed_records = all_records
        self.total_records = len(all_records)
        self.total_tokens = sum(r.tokens for r in all_records)
        
        logger.info(f"\nExtraction Summary:")
        logger.info(f"  Total records: {self.total_records}")
        logger.info(f"  Total tokens: {self.total_tokens:,}")
        
        # Group by source type
        by_type = defaultdict(list)
        for record in all_records:
            by_type[record.source_type].append(record)
        
        logger.info(f"  By source type:")
        for source_type, records in by_type.items():
            total_tokens = sum(r.tokens for r in records)
            logger.info(f"    {source_type}: {len(records)} records, {total_tokens:,} tokens")
        
        # Create chunks
        chunks = self.create_optimized_chunks(all_records)
        
        # Save to single file
        output_data = self.save_to_single_file(chunks)
        
        # Print final report
        self.print_report(chunks, time.time() - total_start)
        
        return output_data
    
    def print_report(self, chunks: List[Dict[str, Any]], total_time: float):
        """Print final report"""
        print("\n" + "="*80)
        print("PROCESSING COMPLETE - FINAL REPORT")
        print("="*80)
        
        print(f"\n📊 SUMMARY:")
        print(f"   Input files processed: {len(set(r.source_file for r in self.processed_records))}")
        print(f"   Total records: {self.total_records:,}")
        print(f"   Total original tokens: {self.total_tokens:,}")
        print(f"   Chunks created: {self.chunks_created}")
        print(f"   Output file: {self.output_file}")
        print(f"   Processing time: {total_time:.2f}s")
        
        print(f"\n📦 CHUNK DISTRIBUTION:")
        if chunks:
            high = [c for c in chunks if c['tokens'] > self.max_tokens * 0.8]
            medium = [c for c in chunks if self.max_tokens * 0.5 <= c['tokens'] <= self.max_tokens * 0.8]
            low = [c for c in chunks if c['tokens'] < self.max_tokens * 0.5]
            
            print(f"   High (>80%): {len(high)} chunks")
            print(f"   Medium (50-80%): {len(medium)} chunks")
            print(f"   Low (<50%): {len(low)} chunks")
            
            # Token statistics
            chunk_tokens = [c['tokens'] for c in chunks]
            print(f"\n🔢 TOKEN STATISTICS:")
            print(f"   Maximum: {max(chunk_tokens):,} tokens")
            print(f"   Minimum: {min(chunk_tokens):,} tokens")
            print(f"   Average: {sum(chunk_tokens)/len(chunk_tokens):,.0f} tokens")
            print(f"   Utilization: {(sum(chunk_tokens) / (len(chunks) * self.max_tokens)) * 100:.1f}%")
            
            # Check for oversized chunks
            oversized = [c for c in chunks if c['tokens'] > self.max_tokens]
            if oversized:
                print(f"\n⚠️  WARNINGS:")
                for chunk in oversized:
                    print(f"   Chunk {chunk['chunk_id']} exceeds limit: {chunk['tokens']:,} > {self.max_tokens:,} tokens")
            else:
                print(f"\n✅ ALL CHECKS PASSED:")
                print(f"   All chunks are within {self.max_tokens:,} token limit")
        
        print(f"\n💾 OUTPUT STRUCTURE:")
        print(f"   {self.output_file}")
        print(f"   ├── metadata (processing information)")
        print(f"   └── chunks (array of {len(chunks)} chunk objects)")
        print(f"       ├── chunk_id")
        print(f"       ├── token_count")
        print(f"       ├── content")
        print(f"       ├── source_files")
        print(f"       └── source_records")
        
        print(f"\n📝 HOW TO USE:")
        print(f"   1. Load: data = json.load(open('{self.output_file}'))")
        print(f"   2. Get metadata: data['metadata']")
        print(f"   3. Iterate chunks: for chunk in data['chunks']:")
        print(f"   4. Access content: chunk['content']")
        
        print("\n" + "="*80)

def main():
    """Main execution function"""
    # Configuration
    INPUT_FILES = [
        "tweets_output65987.json",      # Twitter format
        "reddit_search_*.json",         # Reddit format (wildcard)
        "youtube_search_output.json"    # YouTube format
    ]
    
    OUTPUT_FILE = "all_chunks_combined.json"
    MAX_TOKENS = 8000
    
    print("\n" + "="*80)
    print("🚀 MULTI-FORMAT JSON PROCESSOR")
    print("="*80)
    print(f"Configuration:")
    print(f"  Input patterns: {INPUT_FILES}")
    print(f"  Output file:    {OUTPUT_FILE}")
    print(f"  Max tokens:     {MAX_TOKENS:,}")
    print("="*80 + "\n")
    
    # Create processor
    processor = MultiFormatProcessor(
        output_file=OUTPUT_FILE,
        max_tokens=MAX_TOKENS
    )
    
    try:
        # Process files
        result = processor.process_files(INPUT_FILES)
        
        if result:
            print(f"\n🎯 FINAL SUMMARY:")
            print(f"   Output file: {OUTPUT_FILE}")
            print(f"   Total chunks: {result['metadata']['chunks_created']}")
            print(f"   Total tokens: {result['metadata']['total_chunk_tokens']:,}")
            print(f"   Utilization: {result['metadata']['utilization_percentage']:.1f}%")
            
            # Show example usage
            print(f"\n📋 EXAMPLE USAGE:")
            print(f'''import json

# Load the combined file
with open('{OUTPUT_FILE}', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Access metadata
print(f"Total chunks: {{len(data['chunks'])}}")
print(f"Source files: {{data['metadata']['input_files']}}")

# Process each chunk
for chunk in data['chunks']:
    print(f"\\n=== Chunk {{chunk['chunk_id']}} ===")
    print(f"Tokens: {{chunk['token_count']}}")
    print(f"Sources: {{chunk['source_files']}}")
    content = chunk['content']
    # Process content as needed...''')
            
        else:
            print("❌ Processing failed!")
            
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
