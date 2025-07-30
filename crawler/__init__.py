"""
YouTube Channel Crawler with Embedding Generation

A modular crawler that extracts YouTube video metadata and generates
semantic embeddings for search functionality.
"""

__version__ = "1.0.0"
__author__ = "YouTube Crawler Team"

# Make key classes and functions available at package level
from .memory_monitor import MemoryMonitor
from .embedder import VideoEmbedder
from .channel_manager import search_channels_by_query, create_manual_channel_info
from .video_extractor import get_video_urls_from_channel, fetch_video_metadata
from .file_utils import (
    save_to_json, 
    load_processed_channels, 
    save_processed_channels,
    is_channel_processed,
    mark_channel_processed,
    save_results_with_fallback
)

__all__ = [
    'MemoryMonitor',
    'VideoEmbedder', 
    'search_channels_by_query',
    'create_manual_channel_info',
    'get_video_urls_from_channel',
    'fetch_video_metadata',
    'save_to_json',
    'load_processed_channels',
    'save_processed_channels', 
    'is_channel_processed',
    'mark_channel_processed',
    'save_results_with_fallback'
]