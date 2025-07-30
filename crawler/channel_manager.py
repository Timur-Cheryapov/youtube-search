"""
Channel search and management functionality for the YouTube crawler.

This module handles searching for YouTube channels using queries,
tracking processed channels, and managing channel information.
"""

import yt_dlp
import logging
from typing import Dict, List
try:
    from . import config
except ImportError:
    import config

logger = logging.getLogger(__name__)


def search_channels_by_query(query: str, max_channels: int = config.MAX_CHANNELS_PER_QUERY) -> List[Dict]:
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


def create_manual_channel_info(channel_url: str) -> Dict:
    """Create channel info dictionary for manually provided channel URL"""
    return {
        'channel_url': channel_url,
        'channel_name': 'Manual Input',
        'channel_id': 'manual',
        'found_via_query': 'manual_input'
    }


def get_search_queries() -> List[str]:
    """Get the list of search queries for automated mode"""
    return config.SEARCH_QUERIES.copy()