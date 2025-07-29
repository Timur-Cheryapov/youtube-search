export interface Document {
  id: string;
  content: string;
  metadata?: Record<string, any>;
  embedding?: number[] | string; // Can be string when retrieved from database
  created_at?: string;
}

export interface SearchResult extends Document {
  similarity: number;
}

export interface EmbeddingResponse {
  embedding: number[];
}

export interface SearchRequest {
  query: string;
  limit?: number;
}

export interface SearchResponse {
  results: SearchResult[];
  query: string;
  total: number;
} 