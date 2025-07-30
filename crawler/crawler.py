"""
YouTube Channel Crawler with Embedding Generation

This script crawls YouTube channels to extract video metadata and immediately 
generates semantic embeddings for each video using:
- yt-dlp for video metadata extraction and channel searching
- sentence-transformers for embedding generation
- SmolLM2 for content extraction from descriptions

The script has two modes:

1. AUTOMATED MODE (default):
   Searches for channels using predefined queries and processes them automatically.
   Tracks processed channels to avoid duplicates.
   
   Usage: python crawler.py

2. MANUAL MODE:
   Prompts for a single channel URL to process.
   
   Usage: python crawler.py --manual

The output includes video metadata along with embedding vectors that can be 
used for semantic search.

Files generated:
- youtube_videos_with_embeddings.json: Video data with embeddings
- processed_channels.json: Tracking of processed channels (automated mode only)
"""

import yt_dlp
import json
import logging
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List
import os
from tqdm.auto import tqdm
import torch
from datetime import datetime
import psutil
import gc
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Memory monitoring and fail-safe mechanisms"""
    
    def __init__(self, memory_limit_percent: float = 85.0, critical_limit_percent: float = 95.0):
        """
        Initialize memory monitor
        
        Args:
            memory_limit_percent: Warning threshold (% of total RAM)
            critical_limit_percent: Critical threshold - stop processing (% of total RAM)
        """
        self.memory_limit_percent = memory_limit_percent
        self.critical_limit_percent = critical_limit_percent
        self.total_memory = psutil.virtual_memory().total
        self.memory_limit_bytes = self.total_memory * (memory_limit_percent / 100)
        self.critical_limit_bytes = self.total_memory * (critical_limit_percent / 100)
        
        logger.info(f"🛡️  Memory Monitor initialized:")
        logger.info(f"   • Total RAM: {self.total_memory / (1024**3):.1f} GB")
        logger.info(f"   • Warning threshold: {memory_limit_percent}% ({self.memory_limit_bytes / (1024**3):.1f} GB)")
        logger.info(f"   • Critical threshold: {critical_limit_percent}% ({self.critical_limit_bytes / (1024**3):.1f} GB)")
    
    def get_memory_usage(self) -> Dict:
        """Get current memory usage statistics"""
        memory = psutil.virtual_memory()
        process = psutil.Process()
        
        return {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
            'percent': memory.percent,
            'process_memory': process.memory_info().rss,
            'process_percent': process.memory_percent()
        }
    
    def check_memory_status(self) -> str:
        """
        Check current memory status
        
        Returns:
            'safe', 'warning', or 'critical'
        """
        memory = psutil.virtual_memory()
        
        if memory.used >= self.critical_limit_bytes:
            return 'critical'
        elif memory.used >= self.memory_limit_bytes:
            return 'warning'
        else:
            return 'safe'
    
    def log_memory_status(self, operation: str = ""):
        """Log current memory usage"""
        stats = self.get_memory_usage()
        status = self.check_memory_status()
        
        status_emoji = {
            'safe': '✅',
            'warning': '⚠️',
            'critical': '🚨'
        }
        
        logger.info(f"{status_emoji[status]} Memory Status {f'({operation})' if operation else ''}: "
                   f"{stats['percent']:.1f}% used "
                   f"({stats['used'] / (1024**3):.1f}/{stats['total'] / (1024**3):.1f} GB), "
                   f"Process: {stats['process_percent']:.1f}% "
                   f"({stats['process_memory'] / (1024**3):.2f} GB)")
    
    def force_cleanup(self):
        """Force garbage collection and cleanup"""
        logger.info("🧹 Forcing memory cleanup...")
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("   • Cleared GPU cache")
        
        # Force garbage collection
        collected = gc.collect()
        logger.info(f"   • Garbage collected {collected} objects")
        
        # Log memory status after cleanup
        self.log_memory_status("after cleanup")
    
    def check_and_handle_memory(self, operation: str = "") -> bool:
        """
        Check memory and handle according to status
        
        Returns:
            True if safe to continue, False if should stop
        """
        status = self.check_memory_status()
        
        if status == 'critical':
            logger.error(f"🚨 CRITICAL MEMORY USAGE - Stopping operation!")
            self.log_memory_status(operation)
            self.force_cleanup()
            
            # Check again after cleanup
            if self.check_memory_status() == 'critical':
                logger.error("🚨 Memory still critical after cleanup. Aborting to prevent system crash!")
                return False
            else:
                logger.warning("✅ Memory recovered after cleanup. Continuing...")
                return True
                
        elif status == 'warning':
            logger.warning(f"⚠️  High memory usage detected!")
            self.log_memory_status(operation)
            self.force_cleanup()
            return True
            
        else:
            # Only log memory status occasionally when safe
            if operation in ['channel_start', 'batch_complete']:
                self.log_memory_status(operation)
            return True
    
    def safe_batch_size(self, desired_size: int, base_memory_per_item: float = 50.0) -> int:
        """
        Calculate safe batch size based on available memory
        
        Args:
            desired_size: Desired batch size
            base_memory_per_item: Estimated memory per item in MB
        
        Returns:
            Safe batch size
        """
        memory = psutil.virtual_memory()
        available_mb = memory.available / (1024**2)
        
        # Reserve 20% of available memory as buffer
        usable_memory_mb = available_mb * 0.8
        
        safe_size = max(1, int(usable_memory_mb / base_memory_per_item))
        recommended_size = min(desired_size, safe_size)
        
        if recommended_size < desired_size:
            logger.warning(f"⚠️  Reducing batch size from {desired_size} to {recommended_size} due to memory constraints")
        
        return recommended_size

# Global memory monitor instance
memory_monitor = MemoryMonitor()

def search_channels_by_query(query: str, max_channels: int = 5) -> List[Dict]:
    """Search for channels using a query and return channel information"""
    ydl_opts = {
        'extract_flat': True,
        'skip_download': True,
        'quiet': True,
        'playlistend': max_channels * 3,  # Get more results to filter channels
    }
    
    search_url = f"ytsearch{max_channels * 10}:{query}"  # Search for more results to find channels
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(search_url, download=False)
            
            channels = []
            seen_channels = set()
            
            entries = info.get('entries', [])
            for entry in entries:
                if len(channels) >= max_channels:
                    break
                    
                channel_url = entry.get('channel_url')
                channel_id = entry.get('channel_id')
                uploader = entry.get('uploader')
                
                if channel_url and channel_id and channel_id not in seen_channels:
                    seen_channels.add(channel_id)
                    channels.append({
                        'channel_url': channel_url,
                        'channel_id': channel_id,
                        'channel_name': uploader,
                        'found_via_query': query,
                        'sample_video_title': entry.get('title', '')
                    })
            
            logger.info(f"Found {len(channels)} unique channels for query: '{query}'")
            return channels
            
    except Exception as e:
        logger.error(f"Error searching for channels with query '{query}': {e}")
        return []

def load_processed_channels() -> Dict:
    """Load the list of already processed channels"""
    try:
        with open("./crawler/processed_channels.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.info("No processed_channels.json found, starting fresh")
        return {}
    except Exception as e:
        logger.error(f"Error loading processed channels: {e}")
        return {}

def save_processed_channels(processed_channels: Dict):
    """Save the list of processed channels"""
    try:
        with open("./crawler/processed_channels.json", "w", encoding="utf-8") as f:
            json.dump(processed_channels, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error saving processed channels: {e}")

def is_channel_processed(channel_id: str, processed_channels: Dict) -> bool:
    """Check if a channel has already been processed"""
    return channel_id in processed_channels

def mark_channel_processed(channel_info: Dict, processed_channels: Dict, video_count: int):
    """Mark a channel as processed with metadata"""
    channel_id = channel_info['channel_id']
    processed_channels[channel_id] = {
        'channel_url': channel_info['channel_url'],
        'channel_name': channel_info['channel_name'],
        'found_via_query': channel_info['found_via_query'],
        'processed_date': datetime.now().isoformat(),
        'video_count': video_count,
        'status': 'processed'
    }

def get_video_urls_from_channel(url: str, limit: int = 50) -> Dict:
    ydl_opts = {
        'extract_flat': True,
        'skip_download': True,
        'quiet': True,
        'playlistend': limit,  # Limit number of videos per playlist
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

        channel_url = info.get('channel_url')
        
        video_urls = []
        entries = info.get('entries', [])
        
        # Handle channel URLs which have nested playlists (Videos, Shorts, etc.)
        for entry in entries:
            if entry.get('_type') == 'playlist' and 'entries' in entry:
                # This is a sub-playlist (like "Videos" or "Shorts")
                sub_entries = entry.get('entries', [])
                for video_entry in sub_entries[:limit]:  # Apply limit per playlist
                    if video_entry.get('_type') == 'url' and video_entry.get('ie_key') == 'Youtube':
                        video_urls.append(f"https://www.youtube.com/watch?v={video_entry['id']}")
            elif entry.get('_type') == 'url' and entry.get('ie_key') == 'Youtube':
                # Direct video entry (for regular playlists)
                video_urls.append(f"https://www.youtube.com/watch?v={entry['id']}")
        
        return {
            "video_urls": video_urls[:limit], # Apply overall limit
            "channel_url": channel_url
        }

def fetch_video_metadata(video_url: str) -> Dict:
    ydl_opts = {
        'skip_download': True,
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        return {
            "id": info.get("id"),
            "title": info.get("title"),
            "url": f"https://www.youtube.com/watch?v={info.get('id')}",
            "description": info.get("description"),
            "uploader": info.get("uploader"),
            "upload_date": info.get("upload_date"),
            "duration": info.get("duration"),
            "view_count": info.get("view_count"),
            "like_count": info.get("like_count"),
            "channel_id": info.get("channel_id"),
            "channel": info.get("channel"),
            "thumbnails": info.get("thumbnails"),
        }

def save_to_json(data: Dict, filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

class VideoEmbedder:
    def __init__(self, 
                 embedding_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 content_extractor_model_name: str = 'HuggingFaceTB/SmolLM2-1.7B-Instruct'):
        """Initialize the embedder with embedding and content extraction models"""
        
        # Check memory before loading models
        if not memory_monitor.check_and_handle_memory("model_loading"):
            raise MemoryError("Insufficient memory to load models safely")
        
        logger.info(f"Loading sentence transformer model: {embedding_model_name}")
        memory_monitor.log_memory_status("before embedding model")
        
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_model_name = embedding_model_name
        logger.info("Embedding model loaded successfully!")
        
        memory_monitor.log_memory_status("after embedding model")
        
        # Check memory after embedding model
        if not memory_monitor.check_and_handle_memory("after_embedding_model"):
            raise MemoryError("Memory critical after loading embedding model")
        
        logger.info(f"Loading content extraction model: {content_extractor_model_name}")
        # Use official SmolLM2 implementation pattern
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        self.content_tokenizer = AutoTokenizer.from_pretrained(content_extractor_model_name)
        if self.content_tokenizer.pad_token is None:
            self.content_tokenizer.pad_token = self.content_tokenizer.eos_token
            
        self.content_model = AutoModelForCausalLM.from_pretrained(content_extractor_model_name)
        self.content_model = self.content_model.to(self.device)
        logger.info(f"Model loaded on device: {next(self.content_model.parameters()).device}")
        self.content_extractor_model_name = content_extractor_model_name
        logger.info("Content extraction model loaded successfully!")
        
        # Final memory check after all models loaded
        memory_monitor.log_memory_status("models_loaded")
        if not memory_monitor.check_and_handle_memory("models_loaded"):
            logger.warning("⚠️  High memory usage after loading models - monitoring closely")
    
    def extract_main_content(self, description: str) -> str:
        """Extract main video content from description, ignoring credits and promotional content"""
        try:
            if not description or len(description.strip()) < 50:
                return description.strip()
            
            # Identify and extract the main content section
            main_content = self._identify_main_content_section(description)
            
            # Use SmolLM2 with sophisticated prompt for content extraction
            if len(main_content) > 100:
                # SmolLM2 can handle good context length
                truncated = main_content[:4000]  # Reasonable context for SmolLM2
                
                # Create chat format prompt for SmolLM2
                messages = [
                    {
                        "role": "user", 
                        "content": f"""Analyze this YouTube video description and extract ONLY the main content or topic.

Rules:
- Ignore all credits, names of people, sponsors, social media mentions
- Ignore reference links, timestamps, chapter markers
- Ignore "thanks to" sections and supporter lists  
- Focus on the core subject matter, concepts, or story being explained
- Write 1-2 sentences about what the video teaches or demonstrates
- Be concise and factual

Description: {truncated}

Main content:"""
                    }
                ]
                
                # Apply chat template (following official documentation)
                input_text = self.content_tokenizer.apply_chat_template(messages, tokenize=False)
                
                # Tokenize with attention_mask for better results
                inputs = self.content_tokenizer(
                    input_text, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=1024,
                    padding=True
                )
                
                # Move inputs to CUDA device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logger.debug(f"Inputs moved to device: {inputs['input_ids'].device}")
                
                with torch.no_grad():
                    outputs = self.content_model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],  # Add attention_mask for better results
                        max_new_tokens=100,
                        temperature=0.2,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=self.content_tokenizer.eos_token_id
                    )
                
                # Decode the response
                response = self.content_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract just the assistant's response (after the prompt)
                if "Main content:" in response or "assistant" in response:
                    result = response.split("Main content:")[-1].strip()
                    result = result.split("assistant")[-1].strip()
                else:
                    # Fallback: take the last part after user message
                    result = response.split(messages[0]["content"])[-1].strip()
                
                return result[:300].strip() if result else self._basic_cleanup(main_content)
            else:
                return self._basic_cleanup(main_content)
                
        except Exception as e:
            logger.warning(f"Error extracting main content: {str(e)}")
            # Return cleaned description as fallback
            return self._basic_cleanup(description) if description else ""
    
    def _identify_main_content_section(self, description: str) -> str:
        """Identify and extract the main content section from structured descriptions"""
        lines = description.split('\n')
        content_lines = []
        
        skip_patterns = [
            'special thanks', 'thanks to', 'thank you to', 'credits:',
            'writers:', 'editors:', 'producers:', 'animators:', 'director:',
            'music by', 'video by', 'edited by', 'produced by',
            'references:', 'sources:', 'links:', 'follow us', 'subscribe',
            'patreon', 'support', 'sponsor', 'affiliate',
            'http', 'www.', '.com', '.org', 'bit.ly', 'youtube.com',
            '▀▀▀', '---', '===',  # Section dividers
        ]
        
        in_credits_section = False
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Detect start of credits/references sections
            line_lower = line.lower()
            if any(pattern in line_lower for pattern in [
                'special thanks', 'references:', 'sources:', 'credits:',
                'writers:', 'producers:', 'additional', 'executive producer'
            ]):
                in_credits_section = True
                continue
            
            # Skip if we're in credits section
            if in_credits_section:
                continue
            
            # Skip lines with common promotional/credit patterns
            if any(pattern in line_lower for pattern in skip_patterns):
                continue
            
            # Keep substantial content lines (likely main description)
            if len(line) > 20:  # Longer lines more likely to be actual content
                content_lines.append(line)
            
            # Stop after getting enough main content (avoid going into credits)
            if len(content_lines) >= 10:  # Reasonable amount of main content
                break
        
        # If we didn't find much, try first few lines of original
        if len(content_lines) < 2:
            first_lines = []
            for line in lines[:8]:  # First 8 lines often contain main content
                line = line.strip()
                if (len(line) > 30 and 
                    not any(pattern in line.lower() for pattern in skip_patterns[:8])):  # Basic patterns only
                    first_lines.append(line)
                    if len(first_lines) >= 3:
                        break
            content_lines = first_lines if first_lines else content_lines
        
        return ' '.join(content_lines) if content_lines else description[:1000]
    
    def _basic_cleanup(self, description: str) -> str:
        """Basic cleanup as fallback when model extraction fails"""
        lines = description.split('\n')
        content_lines = []
        
        for line in lines:
            line = line.strip()
            # Keep first few lines that don't look like pure credits/URLs
            if len(content_lines) >= 3:  # Limit to first 3 substantial lines
                break
                
            if (len(line) > 20 and 
                not any(keyword in line.lower() for keyword in [
                    'http', 'www.', 'subscribe', 'like', 'follow', 'thanks to'
                ])):
                content_lines.append(line)
        
        result = ' '.join(content_lines)
        return result[:300].strip() if result else description[:300].strip()
    
    def create_text_for_embedding(self, video: Dict) -> str:
        """Combine title and extracted main content from description for embedding"""
        title = video.get('title', '').strip()
        description = video.get('description', '').strip()
        
        # Extract main content from the description
        if description:
            main_content = self.extract_main_content(description)
        else:
            main_content = ""
        
        # Combine title and main content
        if title and main_content:
            combined_text = f"{title}. {main_content}"
        elif title:
            combined_text = title
        elif main_content:
            combined_text = main_content
        else:
            combined_text = "No title or description available"
        
        return combined_text
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for the given text"""
        try:
            if not text or not text.strip():
                return []
            
            # Generate embedding
            embedding = self.embedding_model.encode(text.strip(), show_progress_bar=False)
            
            # Convert to list for JSON serialization
            return embedding.tolist()
        
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return []

def process_channel_videos(channel_info: Dict, embedder: VideoEmbedder, video_limit: int = 10) -> List[Dict]:
    """Process all videos from a channel and generate embeddings"""
    channel_url = channel_info['channel_url']
    channel_name = channel_info['channel_name']
    
    # Memory check before starting channel processing
    if not memory_monitor.check_and_handle_memory("channel_start"):
        logger.error(f"🚨 Insufficient memory to process channel: {channel_name}")
        return []
    
    logger.info(f"📡 Processing channel: {channel_name} ({channel_url})")
    
    # Get video URLs from channel
    data = get_video_urls_from_channel(channel_url, video_limit)
    video_urls = data['video_urls']
    
    if not video_urls:
        logger.warning(f"No videos found for channel: {channel_name}")
        return []
    
    # Adjust batch size based on available memory
    safe_video_limit = memory_monitor.safe_batch_size(len(video_urls), base_memory_per_item=30.0)
    if safe_video_limit < len(video_urls):
        logger.warning(f"⚠️  Reducing video processing from {len(video_urls)} to {safe_video_limit} videos due to memory constraints")
        video_urls = video_urls[:safe_video_limit]
    
    logger.info(f"Found {len(video_urls)} videos from {channel_name}")
    
    results = []
    processed_videos = 0
    failed_videos = 0
    
    # Process videos with progress bar
    video_progress = tqdm(video_urls, desc=f"Processing {channel_name}", unit="video")
    
    for idx, video_url in enumerate(video_progress, 1):
        video_id = video_url.split('v=')[-1] if 'v=' in video_url else 'unknown'
        video_progress.set_description(f"Processing {channel_name}: {idx}/{len(video_urls)}")
        
        # Memory check every 5 videos
        if idx % 5 == 0:
            if not memory_monitor.check_and_handle_memory(f"video_{idx}"):
                logger.error(f"🚨 Memory critical during video processing. Stopping at video {idx}/{len(video_urls)}")
                break
        
        try:
            # Fetch video metadata
            video_metadata = fetch_video_metadata(video_url)
            
            # Add channel context to video metadata
            video_metadata['source_channel'] = {
                'channel_name': channel_name,
                'channel_url': channel_url,
                'channel_id': channel_info['channel_id'],
                'found_via_query': channel_info['found_via_query']
            }
            
            # Generate embedding text and vector
            embedding_text = embedder.create_text_for_embedding(video_metadata)
            embedding_vector = embedder.generate_embedding(embedding_text)
            
            if embedding_vector:
                # Add embedding data to video metadata
                video_metadata['embedding'] = {
                    'text': embedding_text,
                    'vector': embedding_vector,
                    'dimensions': len(embedding_vector),
                    'embedding_model': embedder.embedding_model_name,
                    'extraction_model': embedder.content_extractor_model_name
                }
                processed_videos += 1
            else:
                failed_videos += 1
                logger.warning(f"Failed to generate embedding for video: {video_id}")
            
            results.append(video_metadata)
            
        except Exception as e:
            failed_videos += 1
            logger.error(f"Failed to process {video_url}: {e}")
            
            # If we get memory-related errors, check memory status
            if "memory" in str(e).lower() or "cuda out of memory" in str(e).lower():
                if not memory_monitor.check_and_handle_memory("memory_error"):
                    logger.error("🚨 Memory-related error detected. Stopping processing.")
                    break
    
    video_progress.close()
    
    # Final memory cleanup after processing channel
    memory_monitor.force_cleanup()
    
    logger.info(f"✅ Channel {channel_name}: {processed_videos} successful, {failed_videos} failed")
    return results

def automated_crawler():
    """Automated crawler that searches for channels and processes them"""
    
    # 10 broad search queries for diverse content
    SEARCH_QUERIES = [
        "art and painting tutorials",
        "personal finance and investing",
        "travel vlogs and adventure",
        "mental health and mindfulness",
        "music production and instrument lessons",
        "language learning and linguistics",
        "environmental science and sustainability",
        "parenting and child development",
        "automotive repair and car reviews",
        "fashion trends and beauty tips"
    ]
    
    MAX_CHANNELS_PER_QUERY = 10
    VIDEO_LIMIT_PER_CHANNEL = 20
    
    logger.info("🤖 Starting automated YouTube crawler...")
    
    # Initial memory check
    memory_monitor.log_memory_status("startup")
    if not memory_monitor.check_and_handle_memory("startup"):
        logger.error("🚨 Insufficient memory to start crawler safely")
        return
    
    # Load processed channels
    processed_channels = load_processed_channels()
    
    # Initialize the embedder (this loads the models)
    logger.info("🚀 Initializing video embedder...")
    try:
        embedder = VideoEmbedder()
    except MemoryError as e:
        logger.error(f"🚨 Failed to initialize embedder due to memory constraints: {e}")
        logger.error("💡 Try closing other applications or reducing video limits")
        return
    
    all_results = {}
    total_channels_processed = 0
    total_videos_processed = 0
    
    # Process each search query
    for query in SEARCH_QUERIES:
        logger.info(f"\n🔍 Searching for channels with query: '{query}'")
        
        # Search for channels
        channels = search_channels_by_query(query, MAX_CHANNELS_PER_QUERY)
        
        if not channels:
            logger.warning(f"No channels found for query: '{query}'")
            continue
        
        # Process each channel found
        for channel_info in channels:
            channel_id = channel_info['channel_id']
            channel_name = channel_info['channel_name']
            
            # Check if already processed
            if is_channel_processed(channel_id, processed_channels):
                logger.info(f"⏭️  Skipping already processed channel: {channel_name}")
                continue
            
            # Memory check before processing each channel
            if not memory_monitor.check_and_handle_memory(f"before_channel_{channel_name}"):
                logger.error(f"🚨 Insufficient memory to process channel: {channel_name}")
                logger.info("💾 Saving current progress before stopping...")
                
                # Save progress before stopping due to memory
                if all_results:
                    output_file = "./crawler/youtube_videos_with_embeddings_partial.json"
                    save_to_json(all_results, output_file)
                    logger.info(f"💾 Partial results saved to: {output_file}")
                
                save_processed_channels(processed_channels)
                logger.error("🚨 Stopping due to memory constraints")
                return
            
            # Process channel videos
            try:
                video_results = process_channel_videos(channel_info, embedder, VIDEO_LIMIT_PER_CHANNEL)
                
                if video_results:
                    # Store results with channel URL as key
                    channel_url = channel_info['channel_url']
                    all_results[channel_url] = video_results
                    
                    # Mark as processed
                    mark_channel_processed(channel_info, processed_channels, len(video_results))
                    total_channels_processed += 1
                    total_videos_processed += len(video_results)
                    
                    logger.info(f"✅ Completed processing channel: {channel_name} ({len(video_results)} videos)")
                    
                    # Periodic save to prevent data loss
                    if total_channels_processed % 3 == 0:  # Save every 3 channels
                        logger.info("💾 Saving periodic backup...")
                        backup_file = f"./crawler/youtube_videos_backup_{total_channels_processed}.json"
                        save_to_json(all_results, backup_file)
                        save_processed_channels(processed_channels)
                        memory_monitor.log_memory_status("periodic_save")
                else:
                    logger.warning(f"❌ No videos extracted from channel: {channel_name}")
                
            except Exception as e:
                logger.error(f"❌ Failed to process channel {channel_name}: {e}")
                
                # Check if it's a memory-related error
                if "memory" in str(e).lower() or "cuda out of memory" in str(e).lower():
                    logger.error("🚨 Memory-related error detected")
                    if not memory_monitor.check_and_handle_memory("channel_error"):
                        logger.error("🚨 Cannot recover from memory error. Stopping.")
                        break
                continue
    
    # Final memory check before saving
    logger.info("💾 Preparing to save final results...")
    memory_monitor.log_memory_status("before_final_save")
    
    # Memory check before saving large files
    if not memory_monitor.check_and_handle_memory("final_save"):
        logger.warning("⚠️  High memory usage detected before saving. Attempting cleanup...")
        memory_monitor.force_cleanup()
    
    # Save all results
    output_file = "./crawler/youtube_videos_with_embeddings.json"
    try:
        save_to_json(all_results, output_file)
        logger.info(f"💾 Main results saved to: {output_file}")
    except Exception as e:
        logger.error(f"❌ Failed to save main results: {e}")
        # Try to save smaller chunks if main save fails
        logger.info("🔧 Attempting to save results in smaller chunks...")
        try:
            chunk_size = max(1, len(all_results) // 3)  # Split into 3 chunks
            chunks = list(all_results.items())
            for i in range(0, len(chunks), chunk_size):
                chunk = dict(chunks[i:i + chunk_size])
                chunk_file = f"./crawler/youtube_videos_chunk_{i//chunk_size + 1}.json"
                save_to_json(chunk, chunk_file)
                logger.info(f"💾 Saved chunk to: {chunk_file}")
        except Exception as chunk_error:
            logger.error(f"❌ Failed to save chunks: {chunk_error}")
    
    # Save processed channels
    try:
        save_processed_channels(processed_channels)
        logger.info(f"📝 Channel tracking saved to: ./crawler/processed_channels.json")
    except Exception as e:
        logger.error(f"❌ Failed to save processed channels: {e}")
    
    # Final memory status and cleanup
    memory_monitor.force_cleanup()
    memory_monitor.log_memory_status("final")
    
    # Final summary
    logger.info(f"\n🎉 Automated crawling complete!")
    logger.info(f"📊 Processed {total_channels_processed} channels")
    logger.info(f"📹 Total videos with embeddings: {total_videos_processed}")
    logger.info(f"💾 Results saved to: {output_file}")
    
    # Memory usage summary
    final_stats = memory_monitor.get_memory_usage()
    logger.info(f"🧠 Final memory usage: {final_stats['percent']:.1f}% ({final_stats['used'] / (1024**3):.1f} GB)")
    logger.info(f"💡 Peak process memory: {final_stats['process_memory'] / (1024**3):.2f} GB")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--manual":
        # Manual mode: ask for channel URL (legacy behavior)
        url = input("Enter YouTube channel URL: ").strip()
        limit = 10
        
        # Initialize the embedder with memory checks
        logger.info("🚀 Initializing video embedder...")
        memory_monitor.log_memory_status("manual_startup")
        
        try:
            embedder = VideoEmbedder()
        except MemoryError as e:
            logger.error(f"🚨 Failed to initialize embedder: {e}")
            logger.error("💡 Try closing other applications or reducing video limits")
            sys.exit(1)
        
        # Create channel info dict for manual processing
        channel_info = {
            'channel_url': url,
            'channel_name': 'Manual Input',
            'channel_id': 'manual',
            'found_via_query': 'manual_input'
        }
        
        # Process the channel
        results = process_channel_videos(channel_info, embedder, limit)
        
        # Save results
        output = {url: results}
        output_file = "./crawler/youtube_videos_with_embeddings.json"
        save_to_json(output, output_file)
        
        logger.info(f"✅ Manual processing complete!")
        logger.info(f"💾 Saved {len(results)} videos to {output_file}")
    else:
        # Automated mode (default)
        automated_crawler()