'use client';

import { useState } from 'react';
import { SearchInterface } from '@/components/SearchInterface';
import { SearchResult } from '@/lib/types';
import { searchDocuments } from '@/lib/search';

export default function Home() {
  const [isLoading, setIsLoading] = useState(false);

  const handleSearch = async (query: string): Promise<SearchResult[]> => {
    setIsLoading(true);
    try {
      const results = await searchDocuments({ query, limit: 10 });
      return results;
    } catch (error) {
      console.error('Search error:', error);
      return [];
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center space-x-2">
            <h1 className="text-2xl font-bold">Vector Search</h1>
            <span className="text-muted-foreground">•</span>
            <p className="text-muted-foreground">Semantic search with AI</p>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-12">
        <div className="text-center mb-12 space-y-4">
          <h2 className="text-3xl font-bold">AI-Powered Search</h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Search through documents using natural language. Our vector search understands meaning, not just keywords.
          </p>
        </div>

        <SearchInterface onSearch={handleSearch} isLoading={isLoading} />
      </main>
    </div>
  );
}
