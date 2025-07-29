import { supabase } from './supabase';
import { callHuggingFaceEmbedding } from './embedding';
import { SearchResult } from './types';

export interface SearchOptions {
  query: string;
  limit?: number;
  matchThreshold?: number;
}

export async function searchDocuments({
  query,
  limit = 10,
  matchThreshold = 0
}: SearchOptions): Promise<SearchResult[]> {
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
        match_count: limit
      });

    if (error) {
      console.error('Search error:', error);
      throw new Error('Search failed');
    }

    return data || [];
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