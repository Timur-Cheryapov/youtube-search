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
    };
  };
} 