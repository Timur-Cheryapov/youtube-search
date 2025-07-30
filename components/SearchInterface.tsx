'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { SearchResult } from '@/lib/types';
import { Card, CardContent } from '@/components/ui/card';
import { VideoCard } from '@/components/VideoCard';
import { Search, Loader2 } from 'lucide-react';

interface SearchInterfaceProps {
  onSearch: (query: string, offset?: number) => Promise<SearchResult[]>;
  isLoading: boolean;
  hasSearched: boolean;
  onSearchStart: () => void;
}

export function SearchInterface({ onSearch, isLoading, hasSearched, onSearchStart }: SearchInterfaceProps) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [currentOffset, setCurrentOffset] = useState(0);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [hasMoreResults, setHasMoreResults] = useState(true);
  const [currentQuery, setCurrentQuery] = useState('');
  const scrollTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    // Reset pagination state for new search
    setCurrentOffset(0);
    setHasMoreResults(true);
    setCurrentQuery(query.trim());
    onSearchStart();
    
    const searchResults = await onSearch(query.trim(), 0);
    setResults(searchResults);
    
    // If we got less than the limit (10), there are no more results
    if (searchResults.length < 10) {
      setHasMoreResults(false);
    }
  };

  const loadMoreResults = useCallback(async () => {
    if (isLoadingMore || !hasMoreResults || !currentQuery) return;
    
    setIsLoadingMore(true);
    const nextOffset = currentOffset + 10;
    
    try {
      const moreResults = await onSearch(currentQuery, nextOffset);
      
      if (moreResults.length === 0) {
        setHasMoreResults(false);
      } else {
        setResults(prev => [...prev, ...moreResults]);
        setCurrentOffset(nextOffset);
        
        // If we got less than the limit, there are no more results
        if (moreResults.length < 10) {
          setHasMoreResults(false);
        }
      }
    } catch (error) {
      console.error('Error loading more results:', error);
    } finally {
      setIsLoadingMore(false);
    }
  }, [isLoadingMore, hasMoreResults, currentQuery, currentOffset, onSearch]);

  // Infinite scroll handler
  const handleScroll = useCallback(() => {
    if (scrollTimeoutRef.current) {
      clearTimeout(scrollTimeoutRef.current);
    }
    
    scrollTimeoutRef.current = setTimeout(() => {
      const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
      const windowHeight = window.innerHeight;
      const documentHeight = document.documentElement.scrollHeight;
      
      // Load more when user is within 200px of the bottom
      if (scrollTop + windowHeight >= documentHeight - 200) {
        loadMoreResults();
      }
    }, 100);
  }, [loadMoreResults]);

  useEffect(() => {
    if (hasSearched && results.length > 0) {
      window.addEventListener('scroll', handleScroll);
      return () => {
        window.removeEventListener('scroll', handleScroll);
        if (scrollTimeoutRef.current) {
          clearTimeout(scrollTimeoutRef.current);
        }
      };
    }
  }, [hasSearched, results.length, handleScroll]);

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
              
              {/* Loading more indicator */}
              {isLoadingMore && (
                <Card>
                  <CardContent className="text-center py-8">
                    <div className="flex items-center justify-center space-x-2">
                      <Loader2 className="h-5 w-5 animate-spin" />
                      <p className="text-muted-foreground">Loading more results...</p>
                    </div>
                  </CardContent>
                </Card>
              )}
              
              {/* End of results indicator */}
              {!hasMoreResults && results.length > 0 && (
                <Card>
                  <CardContent className="text-center py-6">
                    <p className="text-muted-foreground text-sm">
                      You've reached the end of the results.
                    </p>
                  </CardContent>
                </Card>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
} 