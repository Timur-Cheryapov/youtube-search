-- Channel Upload Statistics Table
-- This table tracks channel metadata and daily upload statistics
-- Automatically populated by the uploader.ts script when processing YouTube videos

-- Create the main table for channel upload statistics
CREATE TABLE channel_upload_stats (
    id SERIAL PRIMARY KEY,
    channel_url TEXT NOT NULL,
    channel_name TEXT NOT NULL,
    videos_count INTEGER NOT NULL DEFAULT 0,
    data_loaded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Ensure uniqueness per channel
    UNIQUE(channel_url)
);

-- Create indexes for better query performance
CREATE INDEX idx_channel_upload_stats_channel_url ON channel_upload_stats(channel_url);
CREATE INDEX idx_channel_upload_stats_data_loaded_at ON channel_upload_stats(data_loaded_at);

-- Create a composite index for common queries
CREATE INDEX idx_channel_upload_stats_url_date ON channel_upload_stats(channel_url);

-- AUTOMATIC POPULATION:
-- This table is automatically populated by the uploader.ts script when processing YouTube videos.
-- The uploader aggregates video data by channel and upload_date, then inserts/updates this table.
-- 
-- To populate manually from youtube_videos_with_embeddings.json, you can run:
-- npm run ts-node crawler/uploader.ts
-- 
-- Or use the CLEAR_EXISTING environment variable to replace existing data:
-- CLEAR_EXISTING=true npm run ts-node crawler/uploader.ts

-- Sample manual query to insert data (not needed with automatic population)
/*
INSERT INTO channel_upload_stats (channel_url, channel_name, videos_count)
SELECT 
    channel_url,
    channel_name,
    COUNT(*) as videos_count
FROM (
    -- This is a placeholder for the actual JSON parsing logic
    -- In practice, you would parse the JSON and insert the data
    SELECT 
        'https://www.youtube.com/channel/UCHnyfMqiRRG1u-2MsSQLbXA' as channel_url,
        'Veritasium' as channel_name,
    -- Add more rows from JSON parsing
) parsed_data
GROUP BY channel_url, channel_name
ON CONFLICT (channel_url) 
DO UPDATE SET 
    videos_count = EXCLUDED.videos_count,
    data_loaded_at = NOW();
*/

-- View to get channel statistics summary
CREATE VIEW channel_stats_summary AS
SELECT 
    channel_url,
    channel_name,
    SUM(videos_count) as total_videos,
    AVG(videos_count) as avg_videos_per_day,
    MAX(data_loaded_at) as last_data_update
FROM channel_upload_stats
GROUP BY channel_url, channel_name
ORDER BY total_videos DESC;

-- View to get recent upload activity
CREATE VIEW recent_upload_activity AS
SELECT 
    channel_url,
    channel_name,
    videos_count,
    data_loaded_at
FROM channel_upload_stats
ORDER BY videos_count DESC;

-- Query to get daily upload trends
SELECT 
    COUNT(DISTINCT channel_url) as active_channels,
    SUM(videos_count) as total_videos_uploaded,
    AVG(videos_count) as avg_videos_per_channel
FROM channel_upload_stats
GROUP BY upload_date
ORDER BY upload_date DESC;

-- Query to get top performing channels by upload frequency
SELECT 
    channel_name,
    channel_url,
    SUM(videos_count) as total_videos,
    ROUND(AVG(videos_count), 2) as avg_videos_per_day
FROM channel_upload_stats
GROUP BY channel_name, channel_url
ORDER BY total_videos DESC
LIMIT 20;