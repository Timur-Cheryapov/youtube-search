"""
Video metadata extraction functionality using yt-dlp.

This module handles extracting video metadata and URLs from YouTube channels
using the yt-dlp library.
"""

import yt_dlp
import logging
from typing import Dict, List
try:
    from . import config
except ImportError:
    import config

logger = logging.getLogger(__name__)


def get_video_urls_from_channel(url: str, limit: int = config.VIDEO_LIMIT_PER_CHANNEL) -> Dict:
    """
    Extract video URLs from a YouTube channel
    
    Args:
        url: Channel URL
        limit: Maximum number of videos to extract per channel
        
    Returns:
        Dictionary containing video URLs and channel URL
    """
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
    """
    Fetch detailed metadata for a single video
    
    Args:
        video_url: YouTube video URL
        
    Returns:
        Dictionary containing video metadata
    """
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


def extract_video_id(video_url: str) -> str:
    """Extract video ID from YouTube URL"""
    return video_url.split('v=')[-1] if 'v=' in video_url else 'unknown'