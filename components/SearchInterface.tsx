'use client';

import { useState } from 'react';
import { SearchResult } from '@/lib/types';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

interface SearchInterfaceProps {
  onSearch: (query: string) => Promise<SearchResult[]>;
  isLoading: boolean;
}

export function SearchInterface({ onSearch, isLoading }: SearchInterfaceProps) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [hasSearched, setHasSearched] = useState(false);
  const [expandedEmbeddings, setExpandedEmbeddings] = useState<Set<string>>(new Set());

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setHasSearched(true);
    const searchResults = await onSearch(query.trim());
    setResults(searchResults);
  };

  const toggleEmbedding = (resultId: string) => {
    const newExpanded = new Set(expandedEmbeddings);
    if (newExpanded.has(resultId)) {
      newExpanded.delete(resultId);
    } else {
      newExpanded.add(resultId);
    }
    setExpandedEmbeddings(newExpanded);
  };

  const formatDate = (dateString?: string) => {
    if (!dateString) return 'N/A';
    try {
      return new Date(dateString).toLocaleString();
    } catch {
      return dateString;
    }
  };

  const formatEmbedding = (embedding?: number[], isExpanded: boolean = false) => {
    if (!embedding || embedding.length === 0) return 'No embedding data';
    
    if (isExpanded) {
      return `[${embedding.map(val => val.toFixed(4)).join(', ')}]`;
    } else {
      const preview = embedding.slice(0, 3).map(val => val.toFixed(4)).join(', ');
      return `[${preview}...] (${embedding.length} dimensions)`;
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto space-y-6">
      {/* Search Form */}
      <form onSubmit={handleSearch} className="flex gap-2">
        <Input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter your search query..."
          disabled={isLoading}
          className="flex-1"
        />
        <Button 
          type="submit" 
          disabled={isLoading || !query.trim()}
        >
          {isLoading ? 'Searching...' : 'Search'}
        </Button>
      </form>

      {/* Results */}
      {hasSearched && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold">
              Search Results
            </h2>
            <span className="text-sm text-muted-foreground">
              {results.length} result{results.length !== 1 ? 's' : ''} found
            </span>
          </div>

          {results.length === 0 ? (
            <Card>
              <CardContent className="text-center py-8">
                <p className="text-muted-foreground">
                  No results found. Try adjusting your search query.
                </p>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-4">
              {results.map((result, index) => (
                <Card key={result.id || index}>
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div className="flex-1 space-y-2">
                        <CardTitle className="text-base font-medium">
                          Document Content
                        </CardTitle>
                        <CardDescription className="text-sm">
                          {result.content}
                        </CardDescription>
                      </div>
                      <Badge variant="secondary">
                        {(1 - result.distance).toFixed(3)} similarity
                      </Badge>
                    </div>
                  </CardHeader>
                  
                  <CardContent className="space-y-4">
                    {/* Document Details */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="font-medium text-muted-foreground">Document ID:</span>
                        <p className="font-mono text-xs break-all">{result.id || 'N/A'}</p>
                      </div>
                      <div>
                        <span className="font-medium text-muted-foreground">Created:</span>
                        <p>{formatDate(result.created_at)}</p>
                      </div>
                      <div>
                        <span className="font-medium text-muted-foreground">Distance:</span>
                        <p>{result.distance.toFixed(6)}</p>
                      </div>
                      <div>
                        <span className="font-medium text-muted-foreground">Similarity Score:</span>
                        <p>{(1 - result.distance).toFixed(6)}</p>
                      </div>
                    </div>

                    {/* Metadata */}
                    {result.metadata && Object.keys(result.metadata).length > 0 && (
                      <div>
                        <span className="font-medium text-muted-foreground text-sm">Metadata:</span>
                        <div className="flex flex-wrap gap-2 mt-2">
                          {Object.entries(result.metadata).map(([key, value]) => (
                            <Badge key={key} variant="outline">
                              {key}: {String(value)}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Embedding */}
                    {result.embedding && (
                      <div>
                        <div className="flex items-center justify-between">
                          <span className="font-medium text-muted-foreground text-sm">
                            Embedding Vector:
                          </span>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => toggleEmbedding(result.id || index.toString())}
                          >
                            {expandedEmbeddings.has(result.id || index.toString()) ? 'Hide' : 'Show'}
                          </Button>
                        </div>
                        <div className="mt-2 p-2 bg-muted rounded text-xs font-mono break-all">
                          {formatEmbedding(
                            result.embedding, 
                            expandedEmbeddings.has(result.id || index.toString())
                          )}
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
} 