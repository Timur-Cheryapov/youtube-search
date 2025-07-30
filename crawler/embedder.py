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
                counter = 0
                
                for video in video_progress:
                    counter += 1
                    if counter > 5: break
                    video_id = video.get('id', 'unknown')
                    title = video.get('title', 'No title')[:50]
                    video_progress.set_description(f"Processing: {title}...")
                    
                    try:
                        # Create text for embedding (includes extraction)
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
                                'extraction_model': self.content_extractor_model_name
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