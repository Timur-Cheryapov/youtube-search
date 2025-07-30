"""
File I/O utilities for the YouTube crawler.

This module provides functions for saving and loading JSON data,
managing processed channels tracking, handling file operations,
and Supabase integration.
"""

import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
try:
    from . import config
    from .supabase_client import create_supabase_client, SupabaseClient
except ImportError:
    import config
    from supabase_client import create_supabase_client, SupabaseClient

logger = logging.getLogger(__name__)


def save_to_json(data: Dict, filename: str):
    """Save data to JSON file with proper encoding"""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_processed_channels() -> Dict:
    """Load the list of already processed channels"""
    try:
        with open(config.PROCESSED_CHANNELS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.info("No processed_channels.json found, starting fresh")
        return {}
    except Exception as e:
        logger.error(f"Error loading processed channels: {e}")
        return {}


def save_processed_channels(processed_channels: Dict):
    """Save the list of processed channels"""
    try:
        with open(config.PROCESSED_CHANNELS_FILE, "w", encoding="utf-8") as f:
            json.dump(processed_channels, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error saving processed channels: {e}")


def is_channel_processed(channel_id: str, processed_channels: Dict) -> bool:
    """Check if a channel has already been processed"""
    return channel_id in processed_channels


def mark_channel_processed(channel_info: Dict, processed_channels: Dict, video_count: int):
    """Mark a channel as processed with metadata"""
    channel_id = channel_info['channel_id']
    processed_channels[channel_id] = {
        'channel_url': channel_info['channel_url'],
        'channel_name': channel_info['channel_name'],
        'found_via_query': channel_info['found_via_query'],
        'processed_date': datetime.now().isoformat(),
        'video_count': video_count,
        'status': 'processed'
    }


def save_results_with_fallback(all_results: Dict, filename: str) -> bool:
    """
    Save results with fallback to chunked saving if main save fails
    
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        save_to_json(all_results, filename)
        logger.info(f"💾 Results saved to: {filename}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save main results: {e}")
        
        # Try to save smaller chunks if main save fails
        logger.info("🔧 Attempting to save results in smaller chunks...")
        try:
            chunk_size = max(1, len(all_results) // config.CHUNK_COUNT)
            chunks = list(all_results.items())
            
            for i in range(0, len(chunks), chunk_size):
                chunk = dict(chunks[i:i + chunk_size])
                chunk_file = f"{config.CHUNK_FILE_PREFIX}_{i//chunk_size + 1}.json"
                save_to_json(chunk, chunk_file)
                logger.info(f"💾 Saved chunk to: {chunk_file}")
            
            return True
        except Exception as chunk_error:
            logger.error(f"❌ Failed to save chunks: {chunk_error}")
            return False


def save_periodic_backup(all_results: Dict, channel_count: int):
    """Save periodic backup of results"""
    try:
        backup_file = f"{config.BACKUP_FILE_PREFIX}_{channel_count}.json"
        save_to_json(all_results, backup_file)
        logger.info(f"💾 Periodic backup saved: {backup_file}")
    except Exception as e:
        logger.error(f"❌ Failed to save periodic backup: {e}")


# New Supabase integration functions

def upload_channel_to_supabase(supabase_client: SupabaseClient, channel_info: Dict, videos: List[Dict]) -> bool:
    """Upload a channel's videos directly to Supabase"""
    try:
        channel_url = channel_info['channel_url']
        channel_name = channel_info['channel_name']
        
        logger.info(f"⬆️  Uploading {len(videos)} videos from {channel_name} to Supabase...")
        
        # Convert videos to Supabase documents
        documents = []
        skipped_count = 0
        
        for video in videos:
            document = supabase_client.create_document_from_video(video, channel_url)
            if document:
                documents.append(document)
            else:
                skipped_count += 1
        
        if skipped_count > 0:
            logger.warning(f"⚠️  Skipped {skipped_count} videos without embeddings")
        
        if not documents:
            logger.warning(f"❌ No valid documents to upload for {channel_name}")
            return False
        
        # Upload documents in batches
        upload_result = supabase_client.upload_videos_batch(documents)
        
        # Save channel statistics immediately after successful upload
        if upload_result['success'] > 0:
            supabase_client.save_channel_stats(channel_info, upload_result['success'])
            logger.info(f"📊 Updated channel stats for: {channel_info['channel_name']} ({upload_result['success']} videos)")
        else:
            logger.warning(f"⚠️  Skipping channel stats update - no videos uploaded successfully")
        
        success_rate = upload_result['success'] / upload_result['total'] if upload_result['total'] > 0 else 0
        logger.info(f"✅ Uploaded {upload_result['success']}/{upload_result['total']} videos from {channel_name} ({success_rate:.1%} success rate)")
        
        return upload_result['success'] > 0
        
    except Exception as e:
        logger.error(f"❌ Error uploading channel {channel_info['channel_name']} to Supabase: {e}")
        return False


def save_results_with_supabase_fallback(all_results: Dict, supabase_client: Optional[SupabaseClient], filename: str) -> bool:
    """
    Save results to Supabase if available, otherwise fall back to JSON files
    
    Args:
        all_results: Dictionary of channel_url -> videos
        supabase_client: Supabase client instance (None if disabled)
        filename: Fallback JSON filename
        
    Returns:
        True if saved successfully, False otherwise
    """
    if supabase_client:
        try:
            logger.info("💾 Saving results to Supabase database...")
            
            total_channels = len(all_results)
            total_videos = sum(len(videos) for videos in all_results.values())
            successful_channels = 0
            
            logger.info(f"📊 Processing {total_channels} channels with {total_videos} total videos")
            
            for channel_url, videos in all_results.items():
                # Extract channel info from first video's source_channel
                if videos and 'source_channel' in videos[0]:
                    channel_info = videos[0]['source_channel']
                    channel_info['channel_url'] = channel_url  # Ensure URL is set
                    
                    if upload_channel_to_supabase(supabase_client, channel_info, videos):
                        successful_channels += 1
                else:
                    logger.warning(f"⚠️  Skipping channel {channel_url} - missing channel info")
            
            logger.info(f"🎉 Supabase upload complete: {successful_channels}/{total_channels} channels uploaded successfully")
            
            # Also save JSON backup for safety
            logger.info("💾 Saving JSON backup for safety...")
            return save_results_with_fallback(all_results, filename)
            
        except Exception as e:
            logger.error(f"❌ Supabase upload failed: {e}")
            logger.info("📝 Falling back to JSON file save...")
            return save_results_with_fallback(all_results, filename)
    else:
        # No Supabase client - use JSON file mode
        logger.info("📝 Saving results to JSON file (Supabase disabled)")
        return save_results_with_fallback(all_results, filename)


def load_processed_channels_from_supabase(supabase_client: Optional[SupabaseClient]) -> Dict:
    """Load processed channels from Supabase or fall back to JSON file"""
    if supabase_client:
        try:
            processed_urls = supabase_client.get_processed_channels()
            
            # Convert to format expected by existing code
            processed_channels = {}
            unique_channels = 0
            
            for url in processed_urls:
                # Use URL as primary key
                channel_data = {
                    'channel_url': url,
                    'processed_from': 'supabase',
                    'status': 'processed'
                }
                processed_channels[url] = channel_data
                unique_channels += 1
                
                # Also add by channel_id for backward compatibility with JSON mode
                if url.startswith('https://www.youtube.com/channel/'):
                    channel_id = url.split('/')[-1]
                    processed_channels[channel_id] = channel_data  # Same object reference
            
            logger.info(f"📖 Loaded {unique_channels} processed channels from Supabase (with {len(processed_channels)} total lookup keys)")
            return processed_channels
            
        except Exception as e:
            logger.error(f"❌ Error loading processed channels from Supabase: {e}")
            logger.info("📝 Falling back to JSON file...")
    
    # Fall back to JSON file
    return load_processed_channels()


def is_channel_processed_supabase(channel_url: str, supabase_client: Optional[SupabaseClient], processed_channels: Dict) -> bool:
    """Check if channel is processed using in-memory cache (no database calls needed)"""
    if supabase_client:
        # Only check in-memory cache - we already loaded all processed channels at startup
        # No need for expensive database lookups during processing!
        if channel_url in processed_channels:
            return True
        
        # Check by channel_id as well for backward compatibility
        if channel_url.startswith('https://www.youtube.com/channel/'):
            channel_id = channel_url.split('/')[-1]
            if channel_id in processed_channels:
                return True
        
        return False
    
    # Fall back to JSON-based check (extract channel_id from URL for compatibility)
    if channel_url.startswith('https://www.youtube.com/channel/'):
        channel_id = channel_url.split('/')[-1]
        return is_channel_processed(channel_id, processed_channels)
    
    return False


def mark_channel_processed_supabase(channel_info: Dict, video_count: int, supabase_client: Optional[SupabaseClient], processed_channels: Dict):
    """Mark channel as processed in memory cache (Supabase stats already saved during upload)"""
    if supabase_client:
        try:
            # Channel stats already saved during upload - just update in-memory cache
            channel_url = channel_info['channel_url']
            channel_id = channel_info['channel_id']
            
            channel_data = {
                'channel_url': channel_url,
                'channel_name': channel_info['channel_name'],
                'video_count': video_count,
                'status': 'processed',
                'source': 'supabase'
            }
            
            # Store by both URL and ID for fast lookups
            processed_channels[channel_url] = channel_data
            processed_channels[channel_id] = channel_data
            
            logger.debug(f"📝 Updated in-memory cache for processed channel: {channel_info['channel_name']}")
            return
            
        except Exception as e:
            logger.error(f"❌ Error updating in-memory cache: {e}")
            logger.info("📝 Falling back to JSON file...")
    
    # Fall back to JSON file
    mark_channel_processed(channel_info, processed_channels, video_count)