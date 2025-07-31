"""
Configuration constants and settings for the YouTube crawler.

This module contains all configurable parameters for the crawler,
including search queries, model names, and processing limits.
"""

# Model configurations
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
CONTENT_EXTRACTOR_MODEL_NAME = 'HuggingFaceTB/SmolLM2-1.7B-Instruct'

# Processing limits
MAX_CHANNELS_PER_QUERY = 12
VIDEO_LIMIT_PER_CHANNEL = 10
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

# Supabase authentication settings
# For read operations (public access with RLS policies)
SUPABASE_USE_ANON_KEY_FOR_READS = True  # Use anon key for read operations
# For write operations (requires service role key)
SUPABASE_REQUIRE_SERVICE_KEY_FOR_WRITES = True  # Use service key for write operations

# Search queries for automated mode
SEARCH_QUERIES = [
    "ASMR for sleep and relaxation",
    "DIY home decor ideas and crafts",
    "meditation and mindfulness guided sessions",
    "travel vlogs hidden gems",
    "cooking easy weeknight meals",
    "workout routines for beginners no equipment",
    "educational science experiments for kids",
    "personal finance budgeting tips",
    "gaming walkthroughs new releases",
    "tech reviews smartphones laptops",
    "language learning for beginners Spanish",
    "sustainable living zero waste tips",
    "digital art tutorials Procreate",
    "book recommendations fantasy sci-fi",
    "pet care tips dog training",
    "car repair basic maintenance",
    "gardening tips for beginners",
    "urban exploration abandoned places",
    "history documentaries ancient civilizations",
    "musical instrument tutorials guitar",
    "photography tutorials composition lighting",
    "drawing lessons for beginners",
    "mental health awareness coping strategies",
    "cryptocurrency explained for beginners",
    "street interviews controversial topics"
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
MAX_CONCURRENT_VIDEOS = 5  # Maximum concurrent video fetches
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