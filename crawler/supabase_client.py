"""
Supabase integration for the YouTube crawler.

This module handles direct integration with Supabase, eliminating the need
for intermediate JSON files and separate TypeScript uploader.
"""

import os
import logging
from typing import Dict, List, Optional
from datetime import datetime
from supabase import create_client, Client
from tqdm import tqdm
import asyncio
import time
from dotenv import load_dotenv
import sys

try:
    from . import config
except ImportError:
    import config

logger = logging.getLogger(__name__)


class SupabaseClient:
    """Supabase client for YouTube video data management"""
    
    def __init__(self):
        """Initialize Supabase client with environment variables"""
        # Load environment variables from parent directory
        self._load_environment()
        
        self.supabase_url = os.getenv('NEXT_PUBLIC_SUPABASE_URL')
        self.supabase_key = os.getenv('NEXT_PUBLIC_SUPABASE_ANON_KEY')
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "Missing Supabase environment variables. "
                "Please set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY"
            )
        
        try:
            self.client: Client = create_client(self.supabase_url, self.supabase_key)
            logger.info("🔌 Connected to Supabase successfully")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Supabase: {e}")
            raise
    
    def _load_environment(self):
        """Load environment variables from .env.local in parent directory"""
        try:
            # Get the directory where this script is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Look for .env.local in parent directory (main project root)
            parent_dir = os.path.dirname(current_dir)
            env_file = os.path.join(parent_dir, '.env.local')
            
            if os.path.exists(env_file):
                load_dotenv(env_file)
                logger.info(f"📁 Loaded environment variables from {env_file}")
            else:
                # Also try current directory as fallback
                local_env_file = os.path.join(current_dir, '.env.local')
                if os.path.exists(local_env_file):
                    load_dotenv(local_env_file)
                    logger.info(f"📁 Loaded environment variables from {local_env_file}")
                else:
                    logger.warning("⚠️  No .env.local file found in parent or current directory")
                    logger.info("💡 Expected location: ../env.local or ./.env.local")
                    
        except Exception as e:
            logger.warning(f"⚠️  Could not load .env.local file: {e}")
            logger.info("💡 Make sure .env.local exists in the project root directory")
    
    def test_connection(self) -> bool:
        """Test the Supabase connection"""
        try:
            response = self.client.from_(config.SUPABASE_DOCUMENTS_TABLE).select("id").limit(1).execute()
            logger.info("✅ Supabase connection test successful")
            return True
        except Exception as e:
            logger.error(f"❌ Supabase connection test failed: {e}")
            return False
    
    def create_document_from_video(self, video: Dict, channel_url: str) -> Optional[Dict]:
        """Convert video metadata to Supabase document format"""
        try:
            if not video.get('embedding') or not video['embedding'].get('vector'):
                logger.warning(f"⚠️  Skipping video {video.get('id', 'unknown')} - no embedding found")
                return None
            
            # Use the embedding text as content
            content = video['embedding']['text']
            
            # Create rich metadata (following uploader.ts schema exactly)
            metadata = {
                # Video identifiers
                'youtube_id': video.get('id'),
                'youtube_url': video.get('url'),
                
                # Content info
                'title': video.get('title'),
                'description': video.get('description'),
                'duration': video.get('duration'),
                
                # Channel info
                'channel': video.get('channel'),
                'channel_id': video.get('channel_id'),
                'channel_url': channel_url,
                'uploader': video.get('uploader'),
                
                # Engagement metrics
                'view_count': video.get('view_count'),
                'like_count': video.get('like_count'),
                
                # Temporal info
                'upload_date': video.get('upload_date'),
                
                # Thumbnails
                'thumbnail_url': self._extract_thumbnail_url(video.get('thumbnails', [])),
                
                # AI processing info
                'embedding_model': video['embedding'].get('embedding_model'),
                'summarizer_model': video['embedding'].get('extraction_model'),  # Match uploader.ts naming
                'embedding_dimensions': video['embedding'].get('dimensions'),
                
                # Document type and source (as per uploader.ts)
                'document_type': 'youtube_video',
                'source': 'youtube_crawler'
            }
            
            return {
                'content': content,
                'metadata': metadata,
                'embedding': video['embedding']['vector']
            }
            
        except Exception as e:
            logger.error(f"❌ Error creating document from video {video.get('id', 'unknown')}: {e}")
            return None
    
    def _extract_thumbnail_url(self, thumbnails: List) -> Optional[str]:
        """Extract the best thumbnail URL from thumbnails list"""
        if not thumbnails:
            return None
        
        # Prefer hqdefault.webp, then hqdefault.jpg, then any thumbnail
        for thumbnail in thumbnails:
            if isinstance(thumbnail, dict) and 'url' in thumbnail:
                url = thumbnail['url']
                if 'hqdefault.webp' in url:
                    return url
        
        for thumbnail in thumbnails:
            if isinstance(thumbnail, dict) and 'url' in thumbnail:
                url = thumbnail['url']
                if 'hqdefault' in url:
                    return url
        
        # Fallback to first available thumbnail
        if thumbnails and isinstance(thumbnails[0], dict) and 'url' in thumbnails[0]:
            return thumbnails[0]['url']
        
        return None
    
    def upload_videos_batch(self, documents: List[Dict], batch_size: int = None) -> Dict:
        """Upload multiple video documents in batches"""
        if not documents:
            return {'success': 0, 'failed': 0, 'total': 0}
        
        if batch_size is None:
            batch_size = config.SUPABASE_BATCH_SIZE
        
        total_docs = len(documents)
        successful_uploads = 0
        failed_uploads = 0
        
        logger.info(f"📦 Uploading {total_docs} documents in batches of {batch_size}")
        
        # Process documents in batches
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_docs + batch_size - 1) // batch_size
            
            logger.info(f"📤 Uploading batch {batch_num}/{total_batches} ({len(batch)} documents)")
            
            try:
                response = self.client.from_(config.SUPABASE_DOCUMENTS_TABLE).insert(batch).execute()
                successful_uploads += len(batch)
                logger.info(f"✅ Successfully uploaded batch {batch_num}")
                
            except Exception as e:
                logger.error(f"❌ Error uploading batch {batch_num}: {e}")
                failed_uploads += len(batch)
            
            # Small delay between batches to avoid rate limiting
            time.sleep(0.1)
        
        result = {
            'success': successful_uploads,
            'failed': failed_uploads,
            'total': total_docs
        }
        
        logger.info(f"📊 Upload summary: {successful_uploads}/{total_docs} successful, {failed_uploads} failed")
        return result
    
    def save_channel_stats(self, channel_info: Dict, video_count: int) -> bool:
        """Save or update channel statistics"""
        try:
            channel_stat = {
                'channel_url': channel_info['channel_url'],
                'channel_name': channel_info['channel_name'],
                'videos_count': video_count,
            }
            
            # Upsert channel statistics
            response = self.client.from_(config.SUPABASE_CHANNEL_STATS_TABLE).upsert(
                channel_stat,
                on_conflict='channel_url'
            ).execute()
            
            logger.info(f"📊 Updated channel stats for: {channel_info['channel_name']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving channel stats for {channel_info['channel_name']}: {e}")
            return False
    
    def check_channel_processed(self, channel_url: str) -> bool:
        """Check if a channel has already been processed (deprecated - use in-memory cache instead)"""
        # This method is now deprecated - we should use the in-memory cache loaded at startup
        # to avoid unnecessary database calls during processing
        logger.warning(f"⚠️  Using deprecated check_channel_processed method for {channel_url}")
        try:
            response = self.client.from_(config.SUPABASE_CHANNEL_STATS_TABLE).select("channel_url").eq("channel_url", channel_url).execute()
            
            return len(response.data) > 0
            
        except Exception as e:
            logger.error(f"❌ Error checking if channel {channel_url} is processed: {e}")
            return False
    
    def get_processed_channels(self) -> List[str]:
        """Get list of already processed channel URLs"""
        try:
            response = self.client.from_(config.SUPABASE_CHANNEL_STATS_TABLE).select("channel_url").execute()
            
            return [row['channel_url'] for row in response.data]
            
        except Exception as e:
            logger.error(f"❌ Error getting processed channels: {e}")
            return []
    
    def clear_youtube_data(self, confirm: bool = False) -> bool:
        """Clear existing YouTube video data from Supabase"""
        if not confirm:
            logger.warning("⚠️  clear_youtube_data called without confirmation")
            return False
        
        try:
            # Clear documents with proper JSON query syntax
            logger.info("🗑️  Clearing existing YouTube videos from database...")
            self.client.from_(config.SUPABASE_DOCUMENTS_TABLE).delete().eq('metadata->>document_type', 'youtube_video').execute()
            
            # Clear channel stats
            logger.info("🗑️  Clearing existing channel statistics from database...")
            self.client.from_(config.SUPABASE_CHANNEL_STATS_TABLE).delete().neq('id', 0).execute()
            
            logger.info("✅ Existing YouTube data cleared")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error clearing YouTube data: {e}")
            return False
    
    def get_database_stats(self) -> Dict:
        """Get current database statistics"""
        try:
            # Count documents with proper JSON query syntax
            docs_response = self.client.from_(config.SUPABASE_DOCUMENTS_TABLE).select("id", count="exact").eq('metadata->>document_type', 'youtube_video').execute()
            
            # Count channels
            channels_response = self.client.from_(config.SUPABASE_CHANNEL_STATS_TABLE).select("id", count="exact").execute()
            
            return {
                'total_videos': docs_response.count or 0,
                'total_channels': channels_response.count or 0
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting database stats: {e}")
            return {'total_videos': 0, 'total_channels': 0}


def is_supabase_enabled() -> bool:
    """Check if Supabase integration is enabled and configured"""
    if not config.SUPABASE_ENABLED:
        return False
    
    # Load environment if not already loaded
    _load_environment_variables()
    
    supabase_url = os.getenv('NEXT_PUBLIC_SUPABASE_URL')
    supabase_key = os.getenv('NEXT_PUBLIC_SUPABASE_ANON_KEY')
    
    return bool(supabase_url and supabase_key)


def _load_environment_variables():
    """Load environment variables from .env.local (utility function)"""
    try:
        # Get the directory where this script is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Look for .env.local in parent directory (main project root)
        parent_dir = os.path.dirname(current_dir)
        env_file = os.path.join(parent_dir, '.env.local')
        
        if os.path.exists(env_file):
            load_dotenv(env_file, override=False)  # Don't override existing env vars
        else:
            # Also try current directory as fallback
            local_env_file = os.path.join(current_dir, '.env.local')
            if os.path.exists(local_env_file):
                load_dotenv(local_env_file, override=False)
                
    except Exception as e:
        logger.warning(f"⚠️  Could not load .env.local file: {e}")


def create_supabase_client() -> Optional[SupabaseClient]:
    """Factory function to create Supabase client if enabled"""
    if not is_supabase_enabled():
        logger.info("📝 Supabase integration disabled - using legacy JSON file mode")
        return None
    
    try:
        client = SupabaseClient()
        if client.test_connection():
            return client
        else:
            logger.warning("⚠️  Supabase connection failed - falling back to JSON file mode")
            return None
    except Exception as e:
        logger.error(f"❌ Failed to create Supabase client: {e}")
        logger.info("📝 Falling back to JSON file mode")
        return None