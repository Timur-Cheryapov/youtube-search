"""
File I/O utilities for the YouTube crawler.

This module provides functions for saving and loading JSON data,
managing processed channels tracking, and handling file operations.
"""

import json
import logging
from typing import Dict
from datetime import datetime
try:
    from . import config
except ImportError:
    import config

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