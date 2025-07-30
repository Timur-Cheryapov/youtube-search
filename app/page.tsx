'use client';

import { useState } from 'react';
import { SearchInterface } from '@/components/SearchInterface';
import { SearchResult } from '@/lib/types';
import { searchDocuments } from '@/lib/search';
import Image from 'next/image';

export default function Home() {
  const [isLoading, setIsLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);

  const handleSearch = async (query: string, offset: number = 0): Promise<SearchResult[]> => {
    setIsLoading(true);
    if (offset === 0) {
      setHasSearched(true);
    }
    try {
      const results = await searchDocuments({ query, limit: 10, offset_count: offset });
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
      {/* Main Content */}
      <main className="relative min-h-screen">
        {/* Search Interface - positioned to transition smoothly */}
        <div className={`absolute w-full transition-all duration-700 ease-out ${
          hasSearched 
            ? 'top-8 left-0' 
            : 'top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2'
        }`}>
          {/* Logo and Description - always show but transition */}
          <div className={`text-center space-y-6 max-w-6xl mx-auto px-4 transition-all duration-700 ease-out ${
            hasSearched 
              ? 'my-6' 
              : 'mb-12'
          }`}>
            <div className="space-y-4">
              <div className="flex justify-center items-center gap-4">
                <Image src="/logo.png" alt="YouTube Search" width={100} height={100} />
                <h1 className={`font-bold transition-all duration-700 ease-in-out text-5xl`}>
                  YouTube Search
                </h1>  
              </div>
              <p className={`text-muted-foreground max-w-2xl mx-auto transition-all duration-700 ease-in-out text-xl`}>
                AI-powered video discovery using natural language. Search through YouTube videos by content meaning, not just titles.
              </p>
            </div>
          </div>
          
          <SearchInterface 
            onSearch={handleSearch} 
            isLoading={isLoading} 
            hasSearched={hasSearched}
            onSearchStart={() => setHasSearched(true)}
          />
        </div>
      </main>
    </div>
  );
}
