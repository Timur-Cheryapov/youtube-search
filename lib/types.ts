export interface Document {
  id: string;
  content: string;
  metadata?: Record<string, any>;
  embedding?: number[];
  created_at?: string;
}

export interface SearchResult extends Document {
  distance: number;
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