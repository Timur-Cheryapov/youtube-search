'use client';

import { useState, useEffect } from 'react';
import { fetchAnalyticsData, formatLastAddedDate, type AnalyticsData } from '@/lib/analytics';

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
          error: 'Failed to load'
        });
      } finally {
        setIsLoading(false);
      }
    };

    loadAnalytics();
    
    // Refresh analytics every 30 seconds
    const interval = setInterval(loadAnalytics, 30000);
    
    return () => clearInterval(interval);
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

  const isConnected = analytics.supabaseStatus === 'connected';
  
  return (
    <div className="fixed bottom-6 right-6 z-50">
      <div className="p-5 min-w-[280px]">
        <div className="space-y-3 text-sm">
          {/* Supabase Status */}
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground font-medium">Supabase:</span>
            <div className={`h-6 px-3 text-sm rounded-md flex items-center ${
              isConnected 
                ? 'bg-green-100 text-green-800 border border-green-200' 
                : 'bg-red-100 text-red-800 border border-red-200'
            }`}>
              {isConnected ? (
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full" />
                  <span>Connected</span>
                </div>
              ) : (
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-red-500 rounded-full" />
                  <span>Error</span>
                </div>
              )}
            </div>
          </div>

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
          {analytics.error && !isConnected && (
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