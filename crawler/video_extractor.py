"""
Video metadata extraction functionality using yt-dlp.

This module handles extracting video metadata and URLs from YouTube channels
using the yt-dlp library.
"""

import yt_dlp
import logging
import asyncio
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor
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


# Global thread pool for async operations
_executor = None

def get_executor():
    """Get or create a thread pool executor for async operations"""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=config.MAX_CONCURRENT_VIDEOS)
    return _executor


async def fetch_video_metadata_async(video_url: str, semaphore: asyncio.Semaphore = None) -> Dict:
    """
    Asynchronously fetch detailed metadata for a single video
    
    Args:
        video_url: YouTube video URL
        semaphore: Optional semaphore to limit concurrent operations
        
    Returns:
        Dictionary containing video metadata
    """
    async def _fetch_with_semaphore():
        def _fetch_sync():
            return fetch_video_metadata(video_url)
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(get_executor(), _fetch_sync)
    
    if semaphore:
        async with semaphore:
            return await asyncio.wait_for(_fetch_with_semaphore(), timeout=config.ASYNC_TIMEOUT_SECONDS)
    else:
        return await asyncio.wait_for(_fetch_with_semaphore(), timeout=config.ASYNC_TIMEOUT_SECONDS)


async def fetch_multiple_videos_async(video_urls: List[str], max_concurrent: int = None) -> List[Dict]:
    """
    Fetch metadata for multiple videos concurrently
    
    Args:
        video_urls: List of YouTube video URLs
        max_concurrent: Maximum number of concurrent fetches (defaults to config value)
        
    Returns:
        List of video metadata dictionaries (may contain None for failed fetches)
    """
    if max_concurrent is None:
        max_concurrent = config.MAX_CONCURRENT_VIDEOS
    
    # Create semaphore to limit concurrent operations
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def fetch_with_error_handling(url: str) -> Dict:
        try:
            metadata = await fetch_video_metadata_async(url, semaphore)
            logger.debug(f"✅ Successfully fetched metadata for {extract_video_id(url)}")
            return metadata
        except asyncio.TimeoutError:
            logger.warning(f"⏰ Timeout fetching metadata for {extract_video_id(url)}")
            return None
        except Exception as e:
            logger.error(f"❌ Error fetching metadata for {extract_video_id(url)}: {e}")
            return None
    
    # Create tasks for all videos
    tasks = [fetch_with_error_handling(url) for url in video_urls]
    
    # Execute with progress tracking
    logger.info(f"🚀 Starting async fetch of {len(video_urls)} videos with {max_concurrent} concurrent workers")
    
    # Use asyncio.gather to run all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=False)
    
    # Filter out None results (failed fetches)
    successful_results = [result for result in results if result is not None]
    failed_count = len(results) - len(successful_results)
    
    logger.info(f"✅ Async fetch complete: {len(successful_results)} successful, {failed_count} failed")
    
    return successful_results


def cleanup_executor():
    """Cleanup the thread pool executor"""
    global _executor
    if _executor:
        _executor.shutdown(wait=True)
        _executor = None