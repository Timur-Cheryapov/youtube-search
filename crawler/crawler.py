"""
YouTube Channel Crawler with Embedding Generation

This script crawls YouTube channels to extract video metadata and immediately 
generates semantic embeddings for each video using:
- yt-dlp for video metadata extraction
- sentence-transformers for embedding generation
- SmolLM2 for content extraction from descriptions

The output includes video metadata along with embedding vectors that can be 
used for semantic search.

Usage:
    python crawler.py

Dependencies:
    Install requirements: pip install -r requirements.txt
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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_video_urls_from_channel(url: str, limit: int = 50) -> List[str]:
    ydl_opts = {
        'extract_flat': True,
        'skip_download': True,
        'quiet': True,
        'playlistend': limit,  # Limit number of videos per playlist
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        save_to_json(info, "./crawler/youtube_info.json")

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

if __name__ == "__main__":
    url = input("Enter YouTube channel URL: ").strip()
    limit = 10
    
    # Initialize the embedder (this loads the models)
    logger.info("🚀 Initializing video embedder...")
    embedder = VideoEmbedder()
    
    # Get video URLs from channel
    logger.info(f"📡 Crawling channel: {url}")
    data = get_video_urls_from_channel(url, limit)
    channel_url = data['channel_url']
    video_urls = data['video_urls']
    
    logger.info(f"Found {len(video_urls)} videos. Fetching metadata and generating embeddings...")
    
    results = []
    processed_videos = 0
    failed_videos = 0
    
    # Process videos with progress bar
    video_progress = tqdm(video_urls, desc="Processing videos", unit="video")
    
    for idx, video_url in enumerate(video_progress, 1):
        video_id = video_url.split('v=')[-1] if 'v=' in video_url else 'unknown'
        video_progress.set_description(f"Processing {idx}/{len(video_urls)}: {video_id}")
        
        try:
            # Fetch video metadata
            video_metadata = fetch_video_metadata(video_url)
            
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
    
    # Save results
    output = {channel_url: results}
    output_file = "./crawler/youtube_videos_with_embeddings.json"
    save_to_json(output, output_file)
    
    logger.info(f"✅ Processing complete!")
    logger.info(f"📊 Results: {processed_videos} successful embeddings, {failed_videos} failed out of {len(video_urls)} total videos")
    logger.info(f"💾 Saved {len(results)} videos with embeddings to {output_file}")
    logger.info(f"🔑 Channel key: {channel_url}")