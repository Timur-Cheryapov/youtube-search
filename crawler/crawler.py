"""
YouTube Channel Crawler with Embedding Generation

This script crawls YouTube channels to extract video metadata and immediately 
generates semantic embeddings for each video using:
- yt-dlp for video metadata extraction and channel searching
- sentence-transformers for embedding generation
- SmolLM2 for content extraction from descriptions

The script has two modes:

1. AUTOMATED MODE (default):
   Searches for channels using predefined queries and processes them automatically.
   Tracks processed channels to avoid duplicates.
   
   Usage: python crawler.py

2. MANUAL MODE:
   Prompts for a single channel URL to process.
   
   Usage: python crawler.py --manual

The output includes video metadata along with embedding vectors that can be 
used for semantic search.

Files generated:
- youtube_videos_with_embeddings.json: Video data with embeddings
- processed_channels.json: Tracking of processed channels (automated mode only)
"""

import logging
import sys
import asyncio
from typing import Dict, List
from tqdm.auto import tqdm

# Import our modular components
try:
    # Try relative imports first (when run as module)
    from .memory_monitor import MemoryMonitor
    from .embedder import VideoEmbedder
    from .channel_manager import search_channels_by_query, create_manual_channel_info, get_search_queries
    from .video_extractor import (
        get_video_urls_from_channel, 
        fetch_video_metadata, 
        extract_video_id,
        fetch_multiple_videos_async,
        cleanup_executor
    )
    from .file_utils import (
        load_processed_channels, 
        save_processed_channels, 
        is_channel_processed, 
        mark_channel_processed,
        save_results_with_fallback,
        save_periodic_backup,
        is_channel_processed_supabase,
        mark_channel_processed_supabase,
        save_results_with_supabase_fallback,
        upload_channel_to_supabase
    )
    from .supabase_client import create_supabase_client
    from . import config
except ImportError:
    # Fall back to absolute imports (when run directly)
    from memory_monitor import MemoryMonitor
    from embedder import VideoEmbedder
    from channel_manager import search_channels_by_query, create_manual_channel_info, get_search_queries
    from video_extractor import (
        get_video_urls_from_channel, 
        fetch_video_metadata, 
        extract_video_id,
        fetch_multiple_videos_async,
        cleanup_executor
    )
    from file_utils import (
        load_processed_channels, 
        save_processed_channels, 
        is_channel_processed, 
        mark_channel_processed,
        save_results_with_fallback,
        save_periodic_backup,
        is_channel_processed_supabase,
        mark_channel_processed_supabase,
        save_results_with_supabase_fallback,
        upload_channel_to_supabase
    )
    from supabase_client import create_supabase_client
    import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from parent directory
try:
    from dotenv import load_dotenv
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    env_file = os.path.join(parent_dir, '.env.local')
    if os.path.exists(env_file):
        load_dotenv(env_file)
        logger.info(f"📁 Environment variables loaded from {env_file}")
except Exception as e:
    logger.warning(f"⚠️  Could not load .env.local: {e}")

# Global memory monitor instance
memory_monitor = MemoryMonitor(
    memory_limit_percent=config.MEMORY_WARNING_THRESHOLD,
    critical_limit_percent=config.MEMORY_CRITICAL_THRESHOLD
)

async def process_channel_videos_async(channel_info: Dict, embedder: VideoEmbedder, video_limit: int = config.VIDEO_LIMIT_PER_CHANNEL) -> List[Dict]:
    """Process all videos from a channel and generate embeddings"""
    channel_url = channel_info['channel_url']
    channel_name = channel_info['channel_name']
    
    # Memory check before starting channel processing
    if not memory_monitor.check_and_handle_memory("channel_start"):
        logger.error(f"🚨 Insufficient memory to process channel: {channel_name}")
        return []
    
    logger.info(f"📡 Processing channel: {channel_name} ({channel_url})")
    
    # Get video URLs from channel
    data = get_video_urls_from_channel(channel_url, video_limit)
    video_urls = data['video_urls']
    
    if not video_urls:
        logger.warning(f"No videos found for channel: {channel_name}")
        return []
    
    # Adjust batch size based on available memory
    safe_video_limit = memory_monitor.safe_batch_size(len(video_urls), base_memory_per_item=config.BASE_MEMORY_PER_VIDEO)
    if safe_video_limit < len(video_urls):
        logger.warning(f"⚠️  Reducing video processing from {len(video_urls)} to {safe_video_limit} videos due to memory constraints")
        video_urls = video_urls[:safe_video_limit]
    
    logger.info(f"Found {len(video_urls)} videos from {channel_name}")
    
    # Memory check before starting async operations
    if not memory_monitor.check_and_handle_memory("before_async_fetch"):
        logger.error(f"🚨 Insufficient memory for async processing of {channel_name}")
        return []
    
    # Fetch all video metadata asynchronously
    logger.info(f"🚀 Fetching metadata for {len(video_urls)} videos asynchronously...")
    video_metadata_list = await fetch_multiple_videos_async(video_urls, config.MAX_CONCURRENT_VIDEOS)
    
    if not video_metadata_list:
        logger.warning(f"❌ No video metadata could be fetched for channel: {channel_name}")
        return []
    
    # Add channel context to all video metadata first
    logger.info(f"📝 Adding channel context to {len(video_metadata_list)} videos...")
    for video_metadata in video_metadata_list:
        video_metadata['source_channel'] = {
            'channel_name': channel_name,
            'channel_url': channel_url,
            'channel_id': channel_info['channel_id'],
            'found_via_query': channel_info['found_via_query']
        }
    
    # Memory check before batch processing
    if not memory_monitor.check_and_handle_memory("before_batch_embedding"):
        logger.error(f"🚨 Insufficient memory for batch embedding processing")
        return []
    
    # Process embeddings in batches for much better performance
    logger.info(f"🚀 Starting batch embedding processing for {len(video_metadata_list)} videos...")
    
    try:
        # Create all embedding texts in one go
        embedding_texts = embedder.create_texts_for_embedding_batch(video_metadata_list)
        
        # Generate all embeddings in batches
        embedding_vectors = embedder.generate_embeddings_batch(embedding_texts, config.EMBEDDING_BATCH_SIZE)
        
        # Combine everything together
        results = []
        processed_videos = 0
        failed_videos = 0
        
        logger.info(f"📋 Combining results for {len(video_metadata_list)} videos...")
        
        for idx, (video_metadata, embedding_text, embedding_vector) in enumerate(zip(video_metadata_list, embedding_texts, embedding_vectors)):
            video_id = video_metadata.get('id', 'unknown')
            
            try:
                if embedding_vector and len(embedding_vector) > 0:
                    # Add embedding data to video metadata
                    video_metadata['embedding'] = {
                        'text': embedding_text,
                        'vector': embedding_vector,
                        'dimensions': len(embedding_vector),
                        'embedding_model': embedder.embedding_model_name,
                        'extraction_model': embedder.content_extractor_model_name
                    }
                    processed_videos += 1
                else:
                    failed_videos += 1
                    logger.warning(f"Failed to generate embedding for video: {video_id}")
                
                results.append(video_metadata)
                
            except Exception as e:
                failed_videos += 1
                logger.error(f"Failed to process video {video_id}: {e}")
                
                # Add video without embedding as fallback
                results.append(video_metadata)
        
        logger.info(f"🎯 Batch processing complete: {processed_videos} successful embeddings, {failed_videos} failed")
        
    except Exception as e:
        logger.error(f"❌ Batch embedding processing failed: {e}")
        logger.info("🔄 Falling back to individual processing...")
        
        # Fallback to individual processing if batch fails
        results = []
        processed_videos = 0
        failed_videos = 0
        
        video_progress = tqdm(video_metadata_list, desc=f"Fallback embedding for {channel_name}", unit="video")
        
        for idx, video_metadata in enumerate(video_progress, 1):
            video_id = video_metadata.get('id', 'unknown')
            video_progress.set_description(f"Fallback {channel_name}: {idx}/{len(video_metadata_list)}")
            
            try:
                # Generate embedding text and vector individually
                embedding_text = embedder.create_text_for_embedding(video_metadata)
                embedding_vector = embedder.generate_embedding(embedding_text)
                
                if embedding_vector:
                    # Add embedding data to video metadata
                    video_metadata['embedding'] = {
                        'text': embedding_text,
                        'vector': embedding_vector,
                        'dimensions': len(embedding_vector),
                        'embedding_model': embedder.embedding_model_name,
                        'extraction_model': embedder.content_extractor_model_name
                    }
                    processed_videos += 1
                else:
                    failed_videos += 1
                    logger.warning(f"Failed to generate embedding for video: {video_id}")
                
                results.append(video_metadata)
                
            except Exception as e:
                failed_videos += 1
                logger.error(f"Failed to process video {video_id}: {e}")
                results.append(video_metadata)
        
        video_progress.close()
    
    # Final memory cleanup after processing channel
    memory_monitor.force_cleanup()
    
    logger.info(f"✅ Channel {channel_name}: {processed_videos} successful, {failed_videos} failed")
    return results


def process_channel_videos(channel_info: Dict, embedder: VideoEmbedder, video_limit: int = config.VIDEO_LIMIT_PER_CHANNEL) -> List[Dict]:
    """Synchronous wrapper for async video processing (for backward compatibility)"""
    return asyncio.run(process_channel_videos_async(channel_info, embedder, video_limit))


async def automated_crawler_async():
    """Automated crawler that searches for channels and processes them asynchronously"""
    
    logger.info("🤖 Starting async automated YouTube crawler...")
    
    # Initial memory check
    memory_monitor.log_memory_status("startup")
    if not memory_monitor.check_and_handle_memory("startup"):
        logger.error("🚨 Insufficient memory to start crawler safely")
        return
    
    # Initialize Supabase client (if enabled)
    supabase_client = create_supabase_client()
    if supabase_client:
        logger.info("🔌 Supabase integration enabled - data will be saved directly to database")
        db_stats = supabase_client.get_database_stats()
        logger.info(f"📊 Current database: {db_stats['total_videos']} videos, {db_stats['total_channels']} channels")
    else:
        logger.info("📝 Using legacy JSON file mode")
    
    # No need to load processed channels into memory - we'll check database directly
    
    # Initialize the embedder (this loads the models)
    logger.info("🚀 Initializing video embedder...")
    try:
        embedder = VideoEmbedder(memory_monitor=memory_monitor)
    except MemoryError as e:
        logger.error(f"🚨 Failed to initialize embedder due to memory constraints: {e}")
        logger.error("💡 Try closing other applications or reducing video limits")
        return
    
    all_results = {}
    total_channels_processed = 0
    total_videos_processed = 0
    
    # Get search queries from config
    search_queries = get_search_queries()
    
    try:
        # Process each search query
        for query in search_queries:
            logger.info(f"\n🔍 Searching for channels with query: '{query}'")
            
            # Search for channels
            channels = search_channels_by_query(query, config.MAX_CHANNELS_PER_QUERY)
            
            if not channels:
                logger.warning(f"No channels found for query: '{query}'")
                continue
            
            # Process each channel found
            for channel_info in channels:
                channel_id = channel_info['channel_id']
                channel_url = channel_info['channel_url']
                channel_name = channel_info['channel_name']
                
                # Check if already processed (using channel_url, which is what we store)
                if is_channel_processed_supabase(channel_url, supabase_client):
                    logger.info(f"⏭️  Skipping already processed channel: {channel_name} ({channel_url})")
                    continue
                
                # Additional debug info
                logger.info(f"🔍 Processing new channel: {channel_name} ({channel_url})")
                
                # Memory check before processing each channel
                if not memory_monitor.check_and_handle_memory(f"before_channel_{channel_name}"):
                    logger.error(f"🚨 Insufficient memory to process channel: {channel_name}")
                    logger.info("💾 Saving current progress before stopping...")
                    
                    # Save progress before stopping due to memory
                    if all_results:
                        partial_file = config.OUTPUT_FILE.replace(".json", "_partial.json")
                        save_results_with_supabase_fallback(all_results, supabase_client, partial_file)
                    
                    # For JSON mode, channel tracking is handled individually during processing
                    logger.error("🚨 Stopping due to memory constraints")
                    return
                
                # Process channel videos asynchronously
                try:
                    video_results = await process_channel_videos_async(channel_info, embedder, config.VIDEO_LIMIT_PER_CHANNEL)
                    
                    if video_results:
                        channel_url = channel_info['channel_url']
                        
                        # Immediately upload videos to Supabase (if enabled)
                        if supabase_client:
                            logger.info(f"⬆️  Immediately uploading {len(video_results)} videos from {channel_name} to Supabase...")
                            upload_success = upload_channel_to_supabase(supabase_client, channel_info, video_results)
                            
                            if upload_success:
                                logger.info(f"✅ Successfully uploaded {channel_name} to Supabase")
                            else:
                                logger.warning(f"⚠️  Failed to upload {channel_name} to Supabase")
                                # Store in all_results as fallback for JSON backup
                                all_results[channel_url] = video_results
                        else:
                            # Store results for JSON file save (legacy mode)
                            all_results[channel_url] = video_results
                        
                        # Mark as processed (stats already saved during upload for Supabase)
                        mark_channel_processed_supabase(channel_info, len(video_results), supabase_client)
                        total_channels_processed += 1
                        total_videos_processed += len(video_results)
                        
                        logger.info(f"✅ Completed processing channel: {channel_name} ({len(video_results)} videos)")
                        
                        # Periodic save to prevent data loss (only for JSON mode)
                        if not supabase_client and total_channels_processed % config.PERIODIC_SAVE_INTERVAL == 0:
                            logger.info("💾 Saving periodic backup...")
                            save_periodic_backup(all_results, total_channels_processed)
                            memory_monitor.log_memory_status("periodic_save")
                    else:
                        logger.warning(f"❌ No videos extracted from channel: {channel_name}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to process channel {channel_name}: {e}")
                    
                    # Check if it's a memory-related error
                    if "memory" in str(e).lower() or "cuda out of memory" in str(e).lower():
                        logger.error("🚨 Memory-related error detected")
                        if not memory_monitor.check_and_handle_memory("channel_error"):
                            logger.error("🚨 Cannot recover from memory error. Stopping.")
                            break
                    continue
                    
    finally:
        # Cleanup async resources
        cleanup_executor()
    
    # Final memory check before saving
    logger.info("💾 Preparing to save final results...")
    memory_monitor.log_memory_status("before_final_save")
    
    # Memory check before saving large files
    if not memory_monitor.check_and_handle_memory("final_save"):
        logger.warning("⚠️  High memory usage detected before saving. Attempting cleanup...")
        memory_monitor.force_cleanup()
    
    # Save final results and cleanup
    if supabase_client:
        # All data already uploaded immediately during processing
        final_db_stats = supabase_client.get_database_stats()
        logger.info(f"📊 Final database: {final_db_stats['total_videos']} videos, {final_db_stats['total_channels']} channels")
        
        # Save JSON backup if there were any failed uploads
        if all_results:
            logger.info("💾 Saving JSON backup for failed uploads...")
            save_results_with_fallback(all_results, config.OUTPUT_FILE.replace(".json", "_failed_uploads.json"))
        success = True
    else:
        # JSON mode - save all collected results
        success = save_results_with_fallback(all_results, config.OUTPUT_FILE)
        
        # For JSON mode, channel tracking is handled individually by mark_channel_processed_supabase
        logger.info(f"📝 Channel tracking handled individually during processing")
    
    # Final memory status and cleanup
    memory_monitor.force_cleanup()
    memory_monitor.log_memory_status("final")
    
    # Final summary
    logger.info(f"\n🎉 Automated crawling complete!")
    logger.info(f"📊 Processed {total_channels_processed} channels")
    logger.info(f"📹 Total videos with embeddings: {total_videos_processed}")
    if supabase_client:
        logger.info(f"⬆️  Videos uploaded immediately to Supabase during processing")
        if all_results:
            logger.info(f"💾 Failed uploads backup saved to: {config.OUTPUT_FILE.replace('.json', '_failed_uploads.json')}")
    else:
        if success:
            logger.info(f"💾 Results saved to: {config.OUTPUT_FILE}")
    
    # Memory usage summary
    final_stats = memory_monitor.get_memory_usage()
    logger.info(f"🧠 Final memory usage: {final_stats['percent']:.1f}% ({final_stats['used'] / (1024**3):.1f} GB)")
    logger.info(f"💡 Peak process memory: {final_stats['process_memory'] / (1024**3):.2f} GB")


def automated_crawler():
    """Synchronous wrapper for automated crawler"""
    asyncio.run(automated_crawler_async())

async def manual_crawler_async():
    """Async manual crawler for processing a single channel URL"""
    url = input("Enter YouTube channel URL: ").strip()
    
    # Initialize Supabase client (if enabled)
    supabase_client = create_supabase_client()
    if supabase_client:
        logger.info("🔌 Supabase integration enabled - data will be saved directly to database")
    else:
        logger.info("📝 Using legacy JSON file mode")
    
    # Initialize the embedder with memory checks
    logger.info("🚀 Initializing video embedder...")
    memory_monitor.log_memory_status("manual_startup")
    
    try:
        embedder = VideoEmbedder(memory_monitor=memory_monitor)
    except MemoryError as e:
        logger.error(f"🚨 Failed to initialize embedder: {e}")
        logger.error("💡 Try closing other applications or reducing video limits")
        sys.exit(1)
    
    # Create channel info dict for manual processing
    channel_info = create_manual_channel_info(url)
    
    try:
        # Process the channel asynchronously
        results = await process_channel_videos_async(channel_info, embedder, config.VIDEO_LIMIT_MANUAL_MODE)
        
        if results:
            # Immediately upload videos to Supabase (if enabled)
            if supabase_client:
                logger.info(f"⬆️  Immediately uploading {len(results)} videos to Supabase...")
                upload_success = upload_channel_to_supabase(supabase_client, channel_info, results)
                
                if upload_success:
                    logger.info(f"✅ Successfully uploaded videos to Supabase")
                    db_stats = supabase_client.get_database_stats()
                    logger.info(f"📊 Database now contains: {db_stats['total_videos']} videos, {db_stats['total_channels']} channels")
                else:
                    logger.warning(f"⚠️  Failed to upload to Supabase, saving to JSON as fallback")
                    output = {url: results}
                    save_results_with_fallback(output, config.OUTPUT_FILE)
            else:
                # Save to JSON file (legacy mode)
                output = {url: results}
                save_results_with_fallback(output, config.OUTPUT_FILE)
            
            logger.info(f"✅ Manual processing complete!")
            logger.info(f"💾 Processed {len(results)} videos")
        else:
            logger.warning(f"❌ No videos extracted from channel")
        
    finally:
        # Cleanup async resources
        cleanup_executor()


def manual_crawler():
    """Synchronous wrapper for manual crawler"""
    asyncio.run(manual_crawler_async())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--manual":
        # Manual mode: ask for channel URL (legacy behavior)
        manual_crawler()
    else:
        # Automated mode (default)
        automated_crawler()