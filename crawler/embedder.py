"""
Video embedding generation functionality.

This module handles generating semantic embeddings for YouTube videos using
sentence transformers and extracting main content from descriptions using SmolLM2.
"""

import logging
import torch
from typing import Dict, List
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from .memory_monitor import MemoryMonitor
    from . import config
except ImportError:
    from memory_monitor import MemoryMonitor
    import config

logger = logging.getLogger(__name__)


class VideoEmbedder:
    """Video embedding generation with content extraction capabilities"""
    
    def __init__(self, 
                 embedding_model_name: str = config.EMBEDDING_MODEL_NAME,
                 content_extractor_model_name: str = config.CONTENT_EXTRACTOR_MODEL_NAME,
                 memory_monitor: MemoryMonitor = None):
        """Initialize the embedder with embedding and content extraction models"""
        
        self.memory_monitor = memory_monitor
        
        # Check memory before loading models
        if self.memory_monitor and not self.memory_monitor.check_and_handle_memory("model_loading"):
            raise MemoryError("Insufficient memory to load models safely")
        
        logger.info(f"Loading sentence transformer model: {embedding_model_name}")
        if self.memory_monitor:
            self.memory_monitor.log_memory_status("before embedding model")
        
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_model_name = embedding_model_name
        logger.info("Embedding model loaded successfully!")
        
        if self.memory_monitor:
            self.memory_monitor.log_memory_status("after embedding model")
            
            # Check memory after embedding model
            if not self.memory_monitor.check_and_handle_memory("after_embedding_model"):
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
        if self.memory_monitor:
            self.memory_monitor.log_memory_status("models_loaded")
            if not self.memory_monitor.check_and_handle_memory("models_loaded"):
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
                truncated = main_content[:config.MAX_DESCRIPTION_LENGTH]
                
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
                
                return result[:config.MAX_CONTENT_OUTPUT_LENGTH].strip() if result else self._basic_cleanup(main_content)
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
        
        in_credits_section = False
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Detect start of credits/references sections
            line_lower = line.lower()
            if any(pattern in line_lower for pattern in config.CREDITS_SECTION_INDICATORS):
                in_credits_section = True
                continue
            
            # Skip if we're in credits section
            if in_credits_section:
                continue
            
            # Skip lines with common promotional/credit patterns
            if any(pattern in line_lower for pattern in config.SKIP_PATTERNS):
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
                    not any(pattern in line.lower() for pattern in config.SKIP_PATTERNS[:8])):  # Basic patterns only
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
                not any(keyword in line.lower() for keyword in config.BASIC_SKIP_PATTERNS)):
                content_lines.append(line)
        
        result = ' '.join(content_lines)
        return result[:config.MAX_CONTENT_OUTPUT_LENGTH].strip() if result else description[:config.MAX_CONTENT_OUTPUT_LENGTH].strip()
    
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