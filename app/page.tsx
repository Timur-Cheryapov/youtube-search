'use client';

import { useState, useEffect, Suspense } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { SearchInterface } from '@/components/SearchInterface';
import { AnalyticsPanel } from '@/components/AnalyticsPanel';
import { SearchResult } from '@/lib/types';
import { searchDocuments } from '@/lib/search';
import Image from 'next/image';

function HomeContent() {
  const [isLoading, setIsLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [currentQuery, setCurrentQuery] = useState('');
  
  const router = useRouter();
  const searchParams = useSearchParams();

  // Handle initial query from URL and update page title
  useEffect(() => {
    const urlQuery = searchParams.get('q');
    if (urlQuery) {
      setCurrentQuery(urlQuery);
      setHasSearched(true);
      document.title = `${urlQuery} | YouTube Search`;
    } else {
      document.title = 'YouTube Search';
    }
  }, [searchParams]);

  const handleSearch = async (query: string, offset: number = 0): Promise<SearchResult[]> => {
    setIsLoading(true);
    if (offset === 0) {
      setHasSearched(true);
      setCurrentQuery(query);
      
      // Update URL and page title
      const newUrl = new URL(window.location.href);
      newUrl.searchParams.set('q', query);
      router.push(newUrl.pathname + newUrl.search, { scroll: false });
      document.title = `${query} | YouTube Search`;
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

  const handleLogoClick = () => {
    setHasSearched(false);
    setCurrentQuery('');
    
    // Clear URL query parameter and reset title
    router.push('/', { scroll: false });
    document.title = 'YouTube Search';
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
              ? 'md:mb-6' 
              : 'mb-12'
          }`}>
            <div className="space-y-4">
              <div onClick={handleLogoClick} className="flex justify-center items-center gap-4 cursor-pointer">
                <Image src="/logo.png" alt="YouTube Search" width={100} height={100} />
                <h1 className={`font-bold transition-all duration-700 ease-out text-5xl`}>
                  YouTube Search
                </h1>  
              </div>
              <p className={`text-muted-foreground max-w-2xl mx-auto transition-all duration-700 ease-out text-xl ${hasSearched ? 'hidden md:block' : 'block'}`}>
                AI-powered video discovery using natural language. Search through YouTube videos by content meaning, not just titles.
              </p>
            </div>
          </div>
          
          <SearchInterface 
            onSearch={handleSearch} 
            isLoading={isLoading} 
            hasSearched={hasSearched}
            onSearchStart={() => setHasSearched(true)}
            initialQuery={currentQuery}
          />
        </div>
      </main>

      {/* Analytics Panel - stays in bottom right, hidden on mobile */}
      <div className="hidden md:block">
        <AnalyticsPanel />
      </div>
    </div>
  );
}

export default function Home() {
  return (
    <Suspense fallback={
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center space-y-4">
          <Image src="/logo.png" alt="YouTube Search" width={100} height={100} className="mx-auto" />
          <h1 className="text-5xl font-bold">YouTube Search</h1>
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </div>
    }>
      <HomeContent />
    </Suspense>
  );
}
