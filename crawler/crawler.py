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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        logger.info(f"Loading sentence transformer model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_model_name = embedding_model_name
        logger.info("Embedding model loaded successfully!")
        
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
            embedding = self.embedding_model.encode(text.strip())
            
            # Convert to list for JSON serialization
            return embedding.tolist()
        
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return []

def process_channel_videos(channel_info: Dict, embedder: VideoEmbedder, video_limit: int = 10) -> List[Dict]:
    """Process all videos from a channel and generate embeddings"""
    channel_url = channel_info['channel_url']
    channel_name = channel_info['channel_name']
    
    logger.info(f"📡 Processing channel: {channel_name} ({channel_url})")
    
    # Get video URLs from channel
    data = get_video_urls_from_channel(channel_url, video_limit)
    video_urls = data['video_urls']
    
    if not video_urls:
        logger.warning(f"No videos found for channel: {channel_name}")
        return []
    
    logger.info(f"Found {len(video_urls)} videos from {channel_name}")
    
    results = []
    processed_videos = 0
    failed_videos = 0
    
    # Process videos with progress bar
    video_progress = tqdm(video_urls, desc=f"Processing {channel_name}", unit="video")
    
    for idx, video_url in enumerate(video_progress, 1):
        video_id = video_url.split('v=')[-1] if 'v=' in video_url else 'unknown'
        video_progress.set_description(f"Processing {channel_name}: {idx}/{len(video_urls)}")
        
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
    
    video_progress.close()
    
    logger.info(f"✅ Channel {channel_name}: {processed_videos} successful, {failed_videos} failed")
    return results

def automated_crawler():
    """Automated crawler that searches for channels and processes them"""
    
    # 5 broad search queries for diverse content
    SEARCH_QUERIES = [
        "science education tutorials",
        "technology programming tutorials", 
        "history documentary",
        "cooking recipes techniques",
        "fitness workout training"
    ]
    
    MAX_CHANNELS_PER_QUERY = 3
    VIDEO_LIMIT_PER_CHANNEL = 10
    
    logger.info("🤖 Starting automated YouTube crawler...")
    
    # Load processed channels
    processed_channels = load_processed_channels()
    
    # Initialize the embedder (this loads the models)
    logger.info("🚀 Initializing video embedder...")
    embedder = VideoEmbedder()
    
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
                else:
                    logger.warning(f"❌ No videos extracted from channel: {channel_name}")
                
            except Exception as e:
                logger.error(f"❌ Failed to process channel {channel_name}: {e}")
                continue
    
    # Save all results
    output_file = "./crawler/youtube_videos_with_embeddings.json"
    save_to_json(all_results, output_file)
    
    # Save processed channels
    save_processed_channels(processed_channels)
    
    # Final summary
    logger.info(f"\n🎉 Automated crawling complete!")
    logger.info(f"📊 Processed {total_channels_processed} channels")
    logger.info(f"📹 Total videos with embeddings: {total_videos_processed}")
    logger.info(f"💾 Results saved to: {output_file}")
    logger.info(f"📝 Channel tracking saved to: ./crawler/processed_channels.json")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--manual":
        # Manual mode: ask for channel URL (legacy behavior)
        url = input("Enter YouTube channel URL: ").strip()
        limit = 10
        
        # Initialize the embedder
        logger.info("🚀 Initializing video embedder...")
        embedder = VideoEmbedder()
        
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