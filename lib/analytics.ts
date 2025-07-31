import { supabase } from './supabase';

export interface AnalyticsData {
  documentCount: number;
  lastEntryDate: string | null;
  supabaseStatus: 'connected' | 'error';
  error?: string;
}

/**
 * Fetches analytics data including document count, last entry date, and Supabase connection status
 */
export async function fetchAnalyticsData(): Promise<AnalyticsData> {
  try {
    // Test Supabase connection with a simple health check
    const { error: healthError } = await supabase
      .from('documents')
      .select('id')
      .limit(1);

    if (healthError) {
      return {
        documentCount: 0,
        lastEntryDate: null,
        supabaseStatus: 'error',
        error: healthError.message
      };
    }

    // Get document count
    const { count, error: countError } = await supabase
      .from('documents')
      .select('*', { count: 'exact', head: true });

    if (countError) {
      return {
        documentCount: 0,
        lastEntryDate: null,
        supabaseStatus: 'error',
        error: countError.message
      };
    }

    // Get last entry date
    const { data: lastEntry, error: lastEntryError } = await supabase
      .from('documents')
      .select('created_at')
      .order('created_at', { ascending: false })
      .limit(1)
      .single();

    return {
      documentCount: count || 0,
      lastEntryDate: lastEntry?.created_at || null,
      supabaseStatus: 'connected',
      error: lastEntryError?.message
    };

  } catch (error) {
    console.error('Analytics fetch error:', error);
    return {
      documentCount: 0,
      lastEntryDate: null,
      supabaseStatus: 'error',
      error: error instanceof Error ? error.message : 'Unknown error'
    };
  }
}

/**
 * Formats a date string to a human-readable format (e.g., "12 May 2024")
 */
export function formatLastAddedDate(dateString: string | null): string {
  if (!dateString) return 'No entries';
  
  try {
    const date = new Date(dateString);
    
    // Format as "12 May 2024"
    return date.toLocaleDateString('en-GB', {
      day: 'numeric',
      month: 'long',
      year: 'numeric'
    });
  } catch {
    return 'Invalid date';
  }
}