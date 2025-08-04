-- Row Level Security (RLS) Policies for YouTube Search Database
-- This allows public read access while restricting write access to authenticated users

-- Enable RLS on documents table
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- Enable RLS on channel_upload_stats table  
ALTER TABLE channel_upload_stats ENABLE ROW LEVEL SECURITY;

-- ==========================================
-- DOCUMENTS TABLE POLICIES
-- ==========================================

-- Allow public read access to all documents
-- This enables the frontend search functionality for anonymous users
CREATE POLICY "Public read access for documents" ON documents
    FOR SELECT 
    USING (true);

-- Deny insert access for anonymous users
-- Only authenticated users or service role can insert new documents
CREATE POLICY "Authenticated insert access for documents" ON documents
    FOR INSERT 
    WITH CHECK (auth.role() != 'anon');

-- Deny update access for anonymous users
CREATE POLICY "Authenticated update access for documents" ON documents
    FOR UPDATE 
    USING (auth.role() != 'anon');

-- Deny delete access for anonymous users
CREATE POLICY "Authenticated delete access for documents" ON documents
    FOR DELETE 
    USING (auth.role() != 'anon');

-- ==========================================
-- CHANNEL_UPLOAD_STATS TABLE POLICIES
-- ==========================================

-- Allow public read access to channel statistics
-- This enables analytics and channel information display
CREATE POLICY "Public read access for channel stats" ON channel_upload_stats
    FOR SELECT 
    USING (true);

-- Deny insert access for anonymous users
CREATE POLICY "Authenticated insert access for channel stats" ON channel_upload_stats
    FOR INSERT 
    WITH CHECK (auth.role() != 'anon');

-- Deny update access for anonymous users
CREATE POLICY "Authenticated update access for channel stats" ON channel_upload_stats
    FOR UPDATE 
    USING (auth.role() != 'anon');

-- Deny delete access for anonymous users
CREATE POLICY "Authenticated delete access for channel stats" ON channel_upload_stats
    FOR DELETE 
    USING (auth.role() != 'anon');

-- ==========================================
-- QUERY_ANALYTICS TABLE POLICIES
-- ==========================================

-- Enable RLS on query_analytics table
ALTER TABLE query_analytics ENABLE ROW LEVEL SECURITY;

-- Allow public insert access for tracking (anonymous users can log queries)
CREATE POLICY "Public insert access for query analytics" ON query_analytics
    FOR INSERT 
    WITH CHECK (true);

-- Allow authenticated read access to analytics
CREATE POLICY "Authenticated read access for query analytics" ON query_analytics
    FOR SELECT 
    USING (auth.role() != 'anon');

-- Deny update/delete for everyone except service role
CREATE POLICY "Service role only update access for query analytics" ON query_analytics
    FOR UPDATE 
    USING (auth.role() = 'service_role');

CREATE POLICY "Service role only delete access for query analytics" ON query_analytics
    FOR DELETE 
    USING (auth.role() = 'service_role');

-- ==========================================
-- VERIFICATION QUERIES
-- ==========================================

-- Test public read access (should work with anon key)
-- SELECT COUNT(*) FROM documents;
-- SELECT COUNT(*) FROM channel_upload_stats;

-- Test write access (should fail with anon key, succeed with service role key)
-- INSERT INTO documents (content, metadata) VALUES ('test', '{}'); -- Should fail with anon key
-- INSERT INTO channel_upload_stats (channel_url, channel_name) VALUES ('test', 'test'); -- Should fail with anon key

-- ==========================================
-- POLICY INFORMATION
-- ==========================================

-- View all policies:
-- SELECT * FROM pg_policies WHERE tablename IN ('documents', 'channel_upload_stats');

-- Drop policies if needed (for testing):
-- DROP POLICY "Public read access for documents" ON documents;
-- DROP POLICY "Authenticated insert access for documents" ON documents;
-- DROP POLICY "Authenticated update access for documents" ON documents;
-- DROP POLICY "Authenticated delete access for documents" ON documents;
-- DROP POLICY "Public read access for channel stats" ON channel_upload_stats;
-- DROP POLICY "Authenticated insert access for channel stats" ON channel_upload_stats;
-- DROP POLICY "Authenticated update access for channel stats" ON channel_upload_stats;
-- DROP POLICY "Authenticated delete access for channel stats" ON channel_upload_stats;