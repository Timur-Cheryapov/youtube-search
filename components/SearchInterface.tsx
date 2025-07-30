'use client';

import { useState } from 'react';
import { SearchResult } from '@/lib/types';
import { Card, CardContent } from '@/components/ui/card';
import { VideoCard } from '@/components/VideoCard';
import { Search, Loader2 } from 'lucide-react';

interface SearchInterfaceProps {
  onSearch: (query: string) => Promise<SearchResult[]>;
  isLoading: boolean;
  hasSearched: boolean;
  onSearchStart: () => void;
}

export function SearchInterface({ onSearch, isLoading, hasSearched, onSearchStart }: SearchInterfaceProps) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    onSearchStart();
    const searchResults = await onSearch(query.trim());
    setResults(searchResults);
  };

  return (
    <div className="w-full">
      {/* Search Form */}
      <div className={`max-w-6xl mx-auto transition-all duration-300 ease-in-out ${hasSearched ? 'px-4' : ''}`}>
        <form onSubmit={handleSearch} className="relative">
          <div className="relative flex items-center">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search YouTube videos with AI..."
              disabled={isLoading}
              className="w-full h-16 text-xl px-8 pr-20 rounded-full border-2 border-black bg-white focus:outline-none focus:ring-0 focus:border-black transition-all duration-200"
            />
            <button
              type="submit"
              disabled={isLoading || !query.trim()}
              className="absolute right-2 h-12 w-12 rounded-full bg-white text-black flex items-center justify-center transition-colors duration-200 cursor-pointer"
            >
              {isLoading ? (
                <Loader2 className="h-6 w-6 animate-spin" />
              ) : (
                <Search className="h-6 w-6" />
              )}
            </button>
          </div>
        </form>
      </div>

      {/* Results */}
      {hasSearched && (
        <div className="max-w-6xl mx-auto px-4 mt-8 space-y-6 pb-6 animate-in fade-in slide-in-from-bottom-4 duration-700">
          {results.length === 0 && !isLoading ? (
            <Card>
              <CardContent className="text-center py-8">
                <p className="text-muted-foreground">
                  No results found. Try adjusting your search query.
                </p>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-6">
              {results.map((result, index) => (
                <VideoCard 
                  key={result.id || index} 
                  result={result} 
                  index={index} 
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
} 