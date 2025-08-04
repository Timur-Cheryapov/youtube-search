'use client';

import { useState, useEffect } from 'react';
import { fetchAnalyticsData, formatLastAddedDate, type AnalyticsData } from '@/lib/analytics';
import { StatusBadge } from './StatusBadge';

export function AnalyticsPanel() {
  const [analytics, setAnalytics] = useState<AnalyticsData | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const loadAnalytics = async () => {
      try {
        const data = await fetchAnalyticsData();
        setAnalytics(data);
      } catch (error) {
        console.error('Failed to load analytics:', error);
        setAnalytics({
          documentCount: 0,
          lastEntryDate: null,
          supabaseStatus: 'error',
          huggingFaceStatus: 'error',
          error: 'Failed to load'
        });
      } finally {
        setIsLoading(false);
      }
    };

    loadAnalytics();
    
    return;
  }, []);

  if (isLoading) {
    return (
      <div className="fixed bottom-6 right-6 z-50">
        <div className="p-5 bg-background/80 backdrop-blur-sm border border-border/50 rounded-lg min-w-[280px]">
          <div className="flex items-center space-x-3">
            <div className="w-3 h-3 bg-muted-foreground rounded-full animate-pulse" />
            <span className="text-sm text-muted-foreground">Loading analytics...</span>
          </div>
        </div>
      </div>
    );
  }

  if (!analytics) {
    return null;
  }

  const isSupabaseConnected = analytics.supabaseStatus === 'connected';
  const isHuggingFaceConnected = analytics.huggingFaceStatus === 'connected';
  
  return (
    <div className="fixed bottom-6 right-6 z-50 bg-background/80 backdrop-blur-sm border border-border/50 rounded-lg">
      <div className="p-5 min-w-[280px]">
        <div className="space-y-3 text-sm">
          
          <StatusBadge isConnected={isSupabaseConnected} serviceName="Supabase" />

          <StatusBadge isConnected={isHuggingFaceConnected} serviceName="Hugging Face" />

          {/* Document Count */}
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground font-medium">Videos:</span>
            <span className="font-mono text-sm font-medium">
              {analytics.documentCount.toLocaleString()}
            </span>
          </div>

          {/* Last Entry */}
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground font-medium">Last added:</span>
            <span className="font-mono text-sm font-medium">
              {formatLastAddedDate(analytics.lastEntryDate)}
            </span>
          </div>

          {/* Error message if any */}
          {analytics.error && !isSupabaseConnected && (
            <div className="pt-2 border-t border-red-200/50">
              <span className="text-sm text-red-600" title={analytics.error}>
                {analytics.error.length > 35 
                  ? `${analytics.error.substring(0, 35)}...` 
                  : analytics.error
                }
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}