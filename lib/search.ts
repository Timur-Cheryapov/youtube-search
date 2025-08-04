import { supabase } from './supabase';
import { callHuggingFaceEmbedding } from './embedding';
import { SearchResult, SearchResponse } from './types';

export interface SearchOptions {
  query: string;
  limit?: number;
  matchThreshold?: number;
  offset_count?: number;
}

export async function searchDocuments({
  query,
  limit = 10,
  matchThreshold = 0.15,
  offset_count = 0
}: SearchOptions): Promise<SearchResponse> {
  try {
    if (!query || typeof query !== 'string') {
      throw new Error('Query is required and must be a string');
    }

    // Get embedding for the query
    const queryEmbedding = await callHuggingFaceEmbedding(query);

    // Search for similar documents using cosine distance
    const { data, error } = await supabase
      .rpc('search_documents', {
        query_embedding: queryEmbedding,
        match_threshold: matchThreshold,
        match_count: limit,
        offset_count: offset_count
      });

    if (error) {
      console.error('Search error:', error);
      throw new Error('Search failed');
    }

    const { data: trackingId, error: trackingError } = await supabase
      .rpc('track_search_query', {
        p_query_text: query,
        p_query_embedding: queryEmbedding,
        p_query_results: data.map((result: SearchResult) => result.id),
        p_match_threshold: matchThreshold,
        p_match_count: limit,
        p_offset_count: offset_count,
        p_results_count: data.length,
        p_avg_similarity: data.reduce((acc: number, curr: SearchResult) => acc + curr.similarity, 0) / data.length,
        p_max_similarity: data.reduce((acc: number, curr: SearchResult) => Math.max(acc, curr.similarity), 0),
        p_min_similarity: data.reduce((acc: number, curr: SearchResult) => Math.min(acc, curr.similarity), 1)
      });

    if (trackingError) {
      console.error('Tracking error:', trackingError);
      throw new Error('Tracking failed');
    }

    return { results: data, trackingId: trackingId as string };
  } catch (error) {
    console.error('Search error:', error);
    throw error;
  }
}

export async function generateEmbedding(text: string): Promise<number[]> {
  try {
    if (!text || typeof text !== 'string') {
      throw new Error('Text is required and must be a string');
    }

    return await callHuggingFaceEmbedding(text);
  } catch (error) {
    console.error('Embedding error:', error);
    throw error;
  }
}

export async function rateQuery(trackingId: string, rating: 'good' | 'bad'): Promise<boolean> {
  try {
    if (!trackingId || typeof trackingId !== 'string') {
      throw new Error('Tracking ID is required and must be a string');
    }

    if (!rating || !['good', 'bad'].includes(rating)) {
      throw new Error('Rating must be either "good" or "bad"');
    }

    const { data, error } = await supabase
      .rpc('update_query_rating', {
        p_tracking_id: trackingId,
        p_rating: rating
      });

    if (error) {
      console.error('Rating error:', error);
      throw new Error('Failed to update rating');
    }

    return data;
  } catch (error) {
    console.error('Rating error:', error);
    throw error;
  }
}

export async function getQueryAnalytics(): Promise<Record<string, unknown>[]> {
  try {
    const { data, error } = await supabase
      .from('query_analytics')
      .select('*')
      .order('created_at', { ascending: false });

    if (error) {
      console.error('Analytics error:', error);
      throw new Error('Failed to fetch analytics');
    }

    return data || [];
  } catch (error) {
    console.error('Analytics error:', error);
    throw error;
  }
} 