export interface Document {
  id: string;
  content: string;
  metadata?: Record<string, unknown>;
  embedding?: number[] | string; // Can be string when retrieved from database
  created_at?: string;
}

export interface SearchResult extends Document {
  similarity: number;
  tracking_id?: string;
}

export interface SearchResponse {
  results: SearchResult[];
  trackingId?: string;
}