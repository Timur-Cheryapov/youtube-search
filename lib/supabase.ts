import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;

export const supabase = createClient(supabaseUrl, supabaseKey);

export interface Database {
  public: {
    Tables: {
      documents: {
        Row: {
          id: string;
          content: string;
          metadata: Record<string, unknown> | null;
          embedding: number[] | null;
          created_at: string;
        };
        Insert: {
          id?: string;
          content: string;
          metadata?: Record<string, unknown> | null;
          embedding?: number[] | null;
          created_at?: string;
        };
        Update: {
          id?: string;
          content?: string;
          metadata?: Record<string, unknown> | null;
          embedding?: number[] | null;
          created_at?: string;
        };
      };
      channel_upload_stats: {
        Row: {
          id: string;
          channel_url: string;
          channel_name: string;
          videos_count: number;
          data_loaded_at: string;
        };
        Insert: {
          id?: string;
          channel_url: string;
          channel_name: string;
          videos_count: number;
          data_loaded_at?: string;
        };
        Update: {
          id?: string;
          channel_url?: string;
          channel_name?: string;
          videos_count?: number;
          data_loaded_at?: string;
        };
      };
      query_analytics: {
        Row: {
          id: string;
          query_text: string;
          query_embedding: number[] | null;
          match_threshold: number;
          match_count: number;
          offset_count: number;
          results_count: number;
          avg_similarity: number | null;
          max_similarity: number | null;
          min_similarity: number | null;
          execution_time_ms: number | null;
          user_rating: 'good' | 'bad' | null;
          created_at: string;
        };
        Insert: {
          id?: string;
          query_text: string;
          query_embedding?: number[] | null;
          match_threshold?: number;
          match_count?: number;
          offset_count?: number;
          results_count: number;
          avg_similarity?: number | null;
          max_similarity?: number | null;
          min_similarity?: number | null;
          execution_time_ms?: number | null;
          user_rating?: 'good' | 'bad' | null;
          created_at?: string;
        };
        Update: {
          id?: string;
          query_text?: string;
          query_embedding?: number[] | null;
          match_threshold?: number;
          match_count?: number;
          offset_count?: number;
          results_count?: number;
          avg_similarity?: number | null;
          max_similarity?: number | null;
          min_similarity?: number | null;
          execution_time_ms?: number | null;
          user_rating?: 'good' | 'bad' | null;
          created_at?: string;
        };
      };
    };
  };
} 