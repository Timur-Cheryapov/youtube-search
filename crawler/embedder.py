import json
import logging
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from typing import Dict, List
import os
from tqdm.auto import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoEmbedder:
    def __init__(self, 
                 embedding_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 summarizer_model_name: str = 'facebook/bart-large-cnn'):
        """Initialize the embedder with embedding and summarization models"""
        logger.info(f"Loading sentence transformer model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_model_name = embedding_model_name
        logger.info("Embedding model loaded successfully!")
        
        logger.info(f"Loading summarization model: {summarizer_model_name}")
        self.summarizer = pipeline("summarization", model=summarizer_model_name, device=-1)  # CPU
        self.summarizer_model_name = summarizer_model_name
        logger.info("Summarization model loaded successfully!")
    
    def summarize_description(self, description: str, max_length: int = 100) -> str:
        """Summarize video description to remove credits, URLs, and focus on content"""
        try:
            if not description or len(description.strip()) < 50:
                return description.strip()
            
            # Clean description: remove common promotional text patterns
            lines = description.split('\n')
            content_lines = []
            
            for line in lines:
                line = line.strip()
                # Skip lines that are likely credits/promotional content
                if any(keyword in line.lower() for keyword in [
                    'http', 'www.', '.com', '.org', 'subscribe', 'like', 'follow',
                    'patreon', 'sponsor', 'affiliate', 'link in bio', 'check out',
                    'thanks to', 'special thanks', 'music by', 'video by',
                    'instagram', 'twitter', 'facebook', 'tiktok', 'youtube.com'
                ]):
                    continue
                
                # Skip very short lines (likely separators)
                if len(line) < 10:
                    continue
                    
                content_lines.append(line)
            
            # Rejoin cleaned content
            cleaned_description = ' '.join(content_lines)
            
            # If cleaned description is too short, use original
            if len(cleaned_description.strip()) < 30:
                cleaned_description = description
            
            # Summarize if description is long enough
            if len(cleaned_description) > 200:
                # Truncate to fit model limits (BART can handle ~1024 tokens)
                truncated = cleaned_description[:3000]
                
                summary = self.summarizer(
                    truncated, 
                    max_length=max_length, 
                    min_length=20, 
                    do_sample=False
                )[0]['summary_text']
                
                return summary.strip()
            else:
                return cleaned_description.strip()
                
        except Exception as e:
            logger.warning(f"Error summarizing description: {str(e)}")
            # Return truncated original description as fallback
            return description[:500].strip() if description else ""
    
    def create_text_for_embedding(self, video: Dict) -> str:
        """Combine title and summarized description for embedding"""
        title = video.get('title', '').strip()
        description = video.get('description', '').strip()
        
        # Summarize the description
        if description:
            summarized_desc = self.summarize_description(description)
        else:
            summarized_desc = ""
        
        # Combine title and summarized description
        if title and summarized_desc:
            combined_text = f"{title}. {summarized_desc}"
        elif title:
            combined_text = title
        elif summarized_desc:
            combined_text = summarized_desc
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
    
    def process_videos_file(self, input_file: str, output_file: str):
        """Process the YouTube videos JSON file and add embeddings"""
        try:
            # Load the JSON file
            logger.info(f"Loading videos from {input_file}")
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            total_videos = sum(len(videos) for videos in data.values())
            processed_videos = 0
            failed_videos = 0
            
            logger.info(f"Found {len(data)} channels with {total_videos} total videos")
            
            # Process each channel with progress bar
            channel_progress = tqdm(data.items(), desc="Processing channels", unit="channel")
            
            for channel_url, videos in channel_progress:
                channel_progress.set_description(f"Processing {len(videos)} videos from channel")
                
                # Process videos in this channel with progress bar
                video_progress = tqdm(videos, desc="Processing videos", unit="video", leave=False)
                
                for video in video_progress:
                    video_id = video.get('id', 'unknown')
                    title = video.get('title', 'No title')[:50]
                    video_progress.set_description(f"Processing: {title}...")
                    
                    try:
                        # Create text for embedding (includes summarization)
                        embedding_text = self.create_text_for_embedding(video)
                        
                        # Generate embedding
                        embedding = self.generate_embedding(embedding_text)
                        
                        if embedding:
                            # Add embedding data to video
                            video['embedding'] = {
                                'text': embedding_text,
                                'vector': embedding,
                                'dimensions': len(embedding),
                                'embedding_model': self.embedding_model_name,
                                'summarizer_model': self.summarizer_model_name
                            }
                            processed_videos += 1
                        else:
                            failed_videos += 1
                            logger.warning(f"Failed to generate embedding for video: {video_id}")
                            
                    except Exception as e:
                        failed_videos += 1
                        logger.error(f"Error processing video {video_id}: {str(e)}")
                
                video_progress.close()
            
            channel_progress.close()
            
            # Save the enriched data
            logger.info(f"Saving enriched data to {output_file}")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ Processing complete!")
            logger.info(f"📊 Results: {processed_videos} successful, {failed_videos} failed out of {total_videos} total videos")
            
        except FileNotFoundError:
            logger.error(f"Input file not found: {input_file}")
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in input file: {input_file}")
        except Exception as e:
            logger.error(f"Error processing videos: {str(e)}")

def main():
    """Main function to run the embedder"""
    input_file = "./crawler/youtube_videos.json"
    output_file = "./crawler/youtube_videos_with_embeddings.json"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        logger.info("Please run the crawler first to generate the youtube_videos.json file")
        return
    
    # Initialize embedder and process videos
    logger.info("🚀 Starting video embedding process...")
    embedder = VideoEmbedder()
    embedder.process_videos_file(input_file, output_file)
    logger.info("🎉 Video embedding process completed!")

if __name__ == "__main__":
    main() 