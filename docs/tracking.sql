-- Create query_analytics table
CREATE TABLE IF NOT EXISTS query_analytics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  query_text TEXT NOT NULL,
  query_embedding vector(384), -- Adjust dimension based on your embedding model
  query_results TEXT ARRAY, -- Array of uuids of the results
  match_threshold FLOAT DEFAULT 0.15,
  match_count INTEGER DEFAULT 10,
  offset_count INTEGER DEFAULT 0,
  results_count INTEGER NOT NULL DEFAULT 0,
  avg_similarity FLOAT,
  max_similarity FLOAT,
  min_similarity FLOAT,
  execution_time_ms INTEGER,
  user_rating TEXT CHECK (user_rating IN ('good', 'bad')) DEFAULT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_query_analytics_created_at ON query_analytics(created_at);
CREATE INDEX IF NOT EXISTS idx_query_analytics_query_text ON query_analytics(query_text);
CREATE INDEX IF NOT EXISTS idx_query_analytics_rating ON query_analytics(user_rating);

-- Create function to automatically track search queries
CREATE OR REPLACE FUNCTION track_search_query(
  p_query_text TEXT,
  p_query_embedding vector(384),
  p_query_results TEXT ARRAY,
  p_match_threshold FLOAT DEFAULT 0.15,
  p_match_count INTEGER DEFAULT 10,
  p_offset_count INTEGER DEFAULT 0,
  p_results_count INTEGER DEFAULT 0,
  p_avg_similarity FLOAT DEFAULT NULL,
  p_max_similarity FLOAT DEFAULT NULL,
  p_min_similarity FLOAT DEFAULT NULL,
  p_execution_time_ms INTEGER DEFAULT NULL
) RETURNS UUID
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  tracking_id UUID;
BEGIN
  INSERT INTO query_analytics (
    query_text,
    query_embedding,
    query_results,
    match_threshold,
    match_count,
    offset_count,
    results_count,
    avg_similarity,
    max_similarity,
    min_similarity,
    execution_time_ms
  ) VALUES (
    p_query_text,
    p_query_embedding,
    p_query_results,
    p_match_threshold,
    p_match_count,
    p_offset_count,
    p_results_count,
    p_avg_similarity,
    p_max_similarity,
    p_min_similarity,
    p_execution_time_ms
  ) RETURNING id INTO tracking_id;
  
  RETURN tracking_id;
END;
$$ LANGUAGE plpgsql;

-- Function to update query rating
CREATE OR REPLACE FUNCTION update_query_rating(
  p_tracking_id UUID,
  p_rating TEXT
) RETURNS BOOLEAN
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  -- Validate rating
  IF p_rating NOT IN ('good', 'bad') THEN
    RAISE EXCEPTION 'Invalid rating. Must be either ''good'' or ''bad''';
  END IF;
  
  -- Update the rating
  UPDATE query_analytics 
  SET 
    user_rating = p_rating
  WHERE id = p_tracking_id;
  
  -- Return true if row was updated
  RETURN FOUND;
END;
$$ LANGUAGE plpgsql;