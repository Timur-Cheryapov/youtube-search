import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';
import dotenv from 'dotenv';

dotenv.config({ path: path.join(__dirname, '..', '.env.local') });

// Types for our YouTube video data
interface YouTubeVideo {
  id: string;
  title: string;
  url: string;
  description: string;
  uploader: string;
  upload_date: string;
  duration: number;
  view_count: number;
  like_count: number;
  channel_id: string;
  channel: string;
  thumbnails: any[];
  embedding?: {
    text: string;
    vector: number[];
    dimensions: number;
    embedding_model: string;
    summarizer_model: string;
  };
}

interface YouTubeData {
  [channelUrl: string]: YouTubeVideo[];
}

// Supabase document structure
interface SupabaseDocument {
  content: string;
  metadata: Record<string, any>;
  embedding: number[];
}

class YouTubeToSupabaseUploader {
  private supabase;

  constructor(supabaseUrl: string, supabaseKey: string) {
    this.supabase = createClient(supabaseUrl, supabaseKey);
    console.log('🔌 Connected to Supabase');
  }

  private createDocumentFromVideo(video: YouTubeVideo, channelUrl: string): SupabaseDocument | null {
    if (!video.embedding || !video.embedding.vector) {
      console.warn(`⚠️  Skipping video ${video.id} - no embedding found`);
      return null;
    }

    // Use the embedding text as content (title + summarized description)
    const content = video.embedding.text;

    // Create rich metadata
    const metadata = {
      // Video identifiers
      youtube_id: video.id,
      youtube_url: video.url,
      
      // Content info
      title: video.title,
      description: video.description,
      duration: video.duration,
      
      // Channel info
      channel: video.channel,
      channel_id: video.channel_id,
      channel_url: channelUrl,
      uploader: video.uploader,
      
      // Engagement metrics
      view_count: video.view_count,
      like_count: video.like_count,
      
      // Temporal info
      upload_date: video.upload_date,
      
      // Thumbnails
      thumbnail_url: video.thumbnails?.[0]?.url || null,
      
      // AI processing info
      embedding_model: video.embedding.embedding_model,
      summarizer_model: video.embedding.summarizer_model,
      embedding_dimensions: video.embedding.dimensions,
      
      // Document type
      document_type: 'youtube_video',
      source: 'youtube_crawler'
    };

    return {
      content,
      metadata,
      embedding: video.embedding.vector
    };
  }

  async uploadVideos(filePath: string, batchSize: number = 100): Promise<void> {
    try {
      console.log(`📖 Reading YouTube videos from ${filePath}`);
      
      // Read and parse the JSON file
      const rawData = fs.readFileSync(filePath, 'utf-8');
      const youtubeData: YouTubeData = JSON.parse(rawData);
      
      let totalVideos = 0;
      let processedVideos = 0;
      let skippedVideos = 0;
      let errorVideos = 0;

      // Count total videos
      for (const videos of Object.values(youtubeData)) {
        totalVideos += videos.length;
      }

      console.log(`📊 Found ${Object.keys(youtubeData).length} channels with ${totalVideos} total videos`);

      // Process each channel
      for (const [channelUrl, videos] of Object.entries(youtubeData)) {
        console.log(`\n🎬 Processing channel: ${channelUrl}`);
        console.log(`📹 Videos in channel: ${videos.length}`);

        // Prepare documents for this channel
        const documents: SupabaseDocument[] = [];

        for (const video of videos) {
          const document = this.createDocumentFromVideo(video, channelUrl);
          if (document) {
            documents.push(document);
            processedVideos++;
          } else {
            skippedVideos++;
          }
        }

        if (documents.length === 0) {
          console.log(`⚠️  No valid documents to upload for this channel`);
          continue;
        }

        // Upload in batches
        console.log(`⬆️  Uploading ${documents.length} videos to Supabase...`);
        
        for (let i = 0; i < documents.length; i += batchSize) {
          const batch = documents.slice(i, i + batchSize);
          const batchNum = Math.floor(i / batchSize) + 1;
          const totalBatches = Math.ceil(documents.length / batchSize);

          console.log(`📦 Uploading batch ${batchNum}/${totalBatches} (${batch.length} videos)`);

          try {
            const { data, error } = await this.supabase
              .from('documents')
              .insert(batch);

            if (error) {
              console.error(`❌ Error uploading batch ${batchNum}:`, error);
              errorVideos += batch.length;
            } else {
              console.log(`✅ Successfully uploaded batch ${batchNum}`);
            }
          } catch (err) {
            console.error(`❌ Exception during batch ${batchNum} upload:`, err);
            errorVideos += batch.length;
          }

          // Small delay between batches to avoid rate limiting
          await new Promise(resolve => setTimeout(resolve, 100));
        }
      }

      // Final summary
      console.log(`\n🎉 Upload process completed!`);
      console.log(`📊 Summary:`);
      console.log(`   • Total videos found: ${totalVideos}`);
      console.log(`   • Successfully uploaded: ${processedVideos - errorVideos}`);
      console.log(`   • Skipped (no embeddings): ${skippedVideos}`);
      console.log(`   • Failed uploads: ${errorVideos}`);

    } catch (error) {
      console.error('❌ Error during upload process:', error);
      throw error;
    }
  }

  async testConnection(): Promise<void> {
    try {
      const { data, error } = await this.supabase
        .from('documents')
        .select('id')
        .limit(1);

      if (error) {
        throw error;
      }

      console.log('✅ Supabase connection test successful');
    } catch (error) {
      console.error('❌ Supabase connection test failed:', error);
      throw error;
    }
  }

  async clearExistingYouTubeVideos(): Promise<void> {
    console.log('🗑️  Clearing existing YouTube videos from database...');
    
    const { error } = await this.supabase
      .from('documents')
      .delete()
      .eq('metadata->document_type', 'youtube_video');

    if (error) {
      console.error('❌ Error clearing existing videos:', error);
      throw error;
    }

    console.log('✅ Existing YouTube videos cleared');
  }
}

async function main() {
  // Load environment variables
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

  if (!supabaseUrl || !supabaseKey) {
    console.error('❌ Missing Supabase environment variables');
    console.log('Please set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY');
    process.exit(1);
  }

  // File paths
  const inputFile = path.join(__dirname, 'youtube_videos_with_embeddings.json');
  
  if (!fs.existsSync(inputFile)) {
    console.error(`❌ Input file not found: ${inputFile}`);
    console.log('Please run the embedder first to generate the embeddings file');
    process.exit(1);
  }

  try {
    // Initialize uploader
    const uploader = new YouTubeToSupabaseUploader(supabaseUrl, supabaseKey);

    // Test connection
    await uploader.testConnection();

    // Ask user if they want to clear existing data
    console.log('\n⚠️  Do you want to clear existing YouTube videos from the database?');
    console.log('This will remove all documents with document_type="youtube_video"');
    
    // For automation, you can set an environment variable or modify this logic
    const shouldClear = process.env.CLEAR_EXISTING === 'true';
    
    if (shouldClear) {
      await uploader.clearExistingYouTubeVideos();
    } else {
      console.log('🔄 Keeping existing data, new videos will be added');
    }

    // Upload videos
    await uploader.uploadVideos(inputFile);

  } catch (error) {
    console.error('💥 Upload failed:', error);
    process.exit(1);
  }
}

// Export for potential use as module
export { YouTubeToSupabaseUploader };

// Run if called directly
if (require.main === module) {
  main();
} 