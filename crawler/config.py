"""
Configuration constants and settings for the YouTube crawler.

This module contains all configurable parameters for the crawler,
including search queries, model names, and processing limits.
"""

# Model configurations
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
CONTENT_EXTRACTOR_MODEL_NAME = 'HuggingFaceTB/SmolLM2-1.7B-Instruct'

# Processing limits
MAX_CHANNELS_PER_QUERY = 10
VIDEO_LIMIT_PER_CHANNEL = 20
VIDEO_LIMIT_MANUAL_MODE = 10

# Memory monitoring settings
MEMORY_WARNING_THRESHOLD = 85.0  # Percentage of total RAM
MEMORY_CRITICAL_THRESHOLD = 95.0  # Percentage of total RAM
BASE_MEMORY_PER_VIDEO = 30.0  # MB per video for batch size calculation

# File paths (legacy - for backup/fallback only)
PROCESSED_CHANNELS_FILE = "./crawler/processed_channels.json"
OUTPUT_FILE = "./crawler/youtube_videos_with_embeddings.json"
BACKUP_FILE_PREFIX = "./crawler/youtube_videos_backup"
CHUNK_FILE_PREFIX = "./crawler/youtube_videos_chunk"

# Supabase integration settings
SUPABASE_ENABLED = True  # Set to False to use legacy JSON file mode
SUPABASE_BATCH_SIZE = 100  # Number of documents to upload in each batch
SUPABASE_DOCUMENTS_TABLE = "documents"  # Table for video documents
SUPABASE_CHANNEL_STATS_TABLE = "channel_upload_stats"  # Table for channel statistics

# Search queries for automated mode
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

# Content extraction settings
MAX_DESCRIPTION_LENGTH = 4000  # Characters for SmolLM2 processing
MAX_CONTENT_OUTPUT_LENGTH = 300  # Characters for extracted content
MAX_EMBEDDING_TEXT_LENGTH = 300  # Characters for final embedding text

# Processing settings
PERIODIC_SAVE_INTERVAL = 3  # Save backup every N channels
MEMORY_CHECK_INTERVAL = 5  # Check memory every N videos
CHUNK_COUNT = 3  # Number of chunks for fallback saving

# Async processing settings
MAX_CONCURRENT_VIDEOS = 8  # Maximum concurrent video fetches (optimized for 8-core system)
ASYNC_TIMEOUT_SECONDS = 30  # Timeout for individual video fetches
SEMAPHORE_LIMIT = 10  # Limit concurrent operations to prevent overwhelming yt-dlp

# Batch processing settings
EMBEDDING_BATCH_SIZE = 16  # Process embeddings in batches for better performance
TEXT_BATCH_SIZE = 8  # Process text extraction in batches

# Content filtering patterns
SKIP_PATTERNS = [
    'special thanks', 'thanks to', 'thank you to', 'credits:',
    'writers:', 'editors:', 'producers:', 'animators:', 'director:',
    'music by', 'video by', 'edited by', 'produced by',
    'references:', 'sources:', 'links:', 'follow us', 'subscribe',
    'patreon', 'support', 'sponsor', 'affiliate',
    'http', 'www.', '.com', '.org', 'bit.ly', 'youtube.com',
    '▀▀▀', '---', '===',  # Section dividers
]

CREDITS_SECTION_INDICATORS = [
    'special thanks', 'references:', 'sources:', 'credits:',
    'writers:', 'producers:', 'additional', 'executive producer'
]

BASIC_SKIP_PATTERNS = [
    'http', 'www.', 'subscribe', 'like', 'follow', 'thanks to'
]