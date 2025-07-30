'use client';

import { useRef, useEffect, useState } from 'react';
import { SearchResult } from '@/lib/types';
import Image from 'next/image';

interface VideoCardProps {
  result: SearchResult;
  index: number;
}

export function VideoCard({ result, index }: VideoCardProps) {
  const [isVisible, setIsVisible] = useState(false);
  const [hasAnimated, setHasAnimated] = useState(false);
  const cardRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let rafId: number;
    let lastScrollY = window.scrollY;
    
    // Check if element is initially in view
    const checkInitialVisibility = () => {
      if (cardRef.current && !hasAnimated) {
        const rect = cardRef.current.getBoundingClientRect();
        const isInViewport = rect.top < window.innerHeight - 50 && rect.bottom > 0;
        
        if (isInViewport) {
          // Add a small delay for the first visible cards
          setTimeout(() => {
            setIsVisible(true);
            setHasAnimated(true);
          }, index * 150 + 200);
        }
      }
    };

    const observer = new IntersectionObserver(
      ([entry]) => {
        rafId = requestAnimationFrame(() => {
          const isIntersecting = entry.isIntersecting;
          const currentScrollY = window.scrollY;
          const isScrollingDown = currentScrollY > lastScrollY;
          
          // Only animate in when scrolling down and haven't animated yet
          if (isIntersecting && isScrollingDown && !hasAnimated) {
            setIsVisible(true);
            setHasAnimated(true);
          }
          // Once animated, stay visible (don't animate out)
          else if (hasAnimated) {
            setIsVisible(true);
          }
          
          lastScrollY = currentScrollY;
        });
      },
      {
        threshold: 0.15,
        rootMargin: '0px 0px -50px 0px'
      }
    );

    // Scroll listener for more responsive detection
    const handleScroll = () => {
      if (cardRef.current && !hasAnimated) {
        const rect = cardRef.current.getBoundingClientRect();
        const currentScrollY = window.scrollY;
        const isScrollingDown = currentScrollY > lastScrollY;
        const isInViewport = rect.top < window.innerHeight - 50 && rect.bottom > 0;
        
        rafId = requestAnimationFrame(() => {
          if (isInViewport && isScrollingDown && !hasAnimated) {
            setIsVisible(true);
            setHasAnimated(true);
          }
          lastScrollY = currentScrollY;
        });
      }
    };

    // Check initial visibility after a short delay to ensure DOM is ready
    const initialCheckTimeout = setTimeout(checkInitialVisibility, 100);

    const currentRef = cardRef.current;
    if (currentRef) {
      observer.observe(currentRef);
    }

    window.addEventListener('scroll', handleScroll, { passive: true });

    return () => {
      clearTimeout(initialCheckTimeout);
      if (currentRef) {
        observer.unobserve(currentRef);
      }
      window.removeEventListener('scroll', handleScroll);
      if (rafId) {
        cancelAnimationFrame(rafId);
      }
    };
  }, [hasAnimated, index]);

  const formatViewCount = (count?: number) => {
    if (!count) return '0 views';
    if (count >= 1000000) {
      return `${(count / 1000000).toFixed(1)}M views`;
    } else if (count >= 1000) {
      return `${(count / 1000).toFixed(1)}K views`;
    }
    return `${count} views`;
  };

  const formatDuration = (duration?: number) => {
    if (!duration) return '';
    const minutes = Math.floor(duration / 60);
    const seconds = duration % 60;
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  const formatDate = (dateString?: string) => {
    if (!dateString) return 'N/A';
    
    // Handle YYYYMMDD format (e.g., "20250509")
    if (typeof dateString === 'string' && dateString.length === 8 && /^\d{8}$/.test(dateString)) {
      const year = dateString.substring(0, 4);
      const month = dateString.substring(4, 6);
      const day = dateString.substring(6, 8);
      try {
        const date = new Date(`${year}-${month}-${day}`);
        const now = new Date();
        const diffTime = now.getTime() - date.getTime();
        const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));
        const diffMonths = Math.floor(diffDays / 30);
        const diffYears = Math.floor(diffDays / 365);
        
        if (diffYears > 0) {
          return `${diffYears} year${diffYears !== 1 ? 's' : ''} ago`;
        } else if (diffMonths > 0) {
          return `${diffMonths} month${diffMonths !== 1 ? 's' : ''} ago`;
        } else if (diffDays > 0) {
          return `${diffDays} day${diffDays !== 1 ? 's' : ''} ago`;
        } else {
          return 'Today';
        }
      } catch {
        return dateString;
      }
    }
    
    // Handle standard date formats
    try {
      return new Date(dateString).toLocaleDateString();
    } catch {
      return dateString;
    }
  };

  const getSimilarityBadgeClasses = (similarity: number) => {
    if (similarity >= 0.5) return 'bg-green-50 text-green-600 border-green-100'; // High similarity - extra subtle green
    if (similarity >= 0.3) return 'bg-yellow-50 text-yellow-600 border-yellow-100'; // Medium similarity - extra subtle yellow
    if (similarity >= 0) return 'bg-orange-50 text-orange-600 border-orange-100'; // Low-medium similarity - extra subtle orange
    return 'bg-red-50 text-red-600 border-red-100'; // Low similarity - extra subtle red
  };

  const calculatePopularity = (viewCount?: number, uploadDate?: string, createdAt?: string) => {
    if (!viewCount) return { label: 'Unknown', variant: 'outline' as const };
  
    const year = uploadDate?.substring(0, 4);
    const month = uploadDate?.substring(4, 6);
    const day = uploadDate?.substring(6, 8);
      
    const videoDate = new Date(`${year}-${month}-${day}`);
    
    const processedDate = new Date(createdAt || '');
    
    const ageInDays = Math.floor((processedDate.getTime() - videoDate.getTime()) / (1000 * 60 * 60 * 24));
    
    // Calculate views per day (with minimum age of 1 day to avoid division by zero)
    const effectiveAge = Math.max(ageInDays, 1);
    const viewsPerDay = viewCount / effectiveAge;
    
    // Popularity thresholds based on views per day
    if (viewsPerDay >= 50000) {
      return { label: 'Viral', classes: 'bg-green-100 text-green-700 border-green-200' };
    } else if (viewsPerDay >= 10000) {
      return { label: 'Very Popular', classes: 'bg-green-100 text-green-700 border-green-200' };
    } else if (viewsPerDay >= 1000) {
      return { label: 'Popular', classes: 'bg-blue-100 text-blue-700 border-blue-200' };
    } else if (viewsPerDay >= 100) {
      return { label: 'Moderate' };
    } else {
      return { label: 'Low Engagement' };
    }
  };

  const metadata = result.metadata || {};
  const thumbnail = (metadata.thumbnail_url as string) || '/file.svg';
  const title = (metadata.title as string) || 'Untitled Video';
  const channel = (metadata.channel as string) || (metadata.uploader as string) || 'Unknown Channel';
  const viewCount = metadata.view_count as number | undefined;
  const uploadDate = metadata.upload_date as string | undefined;
  const duration = metadata.duration as number | undefined;
  const youtubeUrl = metadata.youtube_url as string | undefined;
  const createdAt = result.created_at;
  
  const popularity = calculatePopularity(viewCount, uploadDate, createdAt);

  return (
    <div 
      ref={cardRef}
      className={`transition-all duration-700 ease-out ${
        isVisible 
          ? 'opacity-100 translate-y-0' 
          : 'opacity-0 translate-y-12'
      }`}
      style={{ 
        transitionDelay: hasAnimated ? '0ms' : `${index * 150 + 200}ms`
      }}
    >
      <div className="flex gap-6">
        {/* Video Thumbnail */}
        <div className="relative flex-shrink-0">
          {/* 16:9 aspect ratio container */}
          <div className="w-112 h-63 rounded-lg overflow-hidden bg-muted">
              <a 
                href={youtubeUrl} 
                target="_blank" 
                rel="noopener noreferrer"
                className="block w-full h-full relative"
              >
                <Image 
                  src={thumbnail} 
                  alt={title}
                  width={448}
                  height={252}
                  className="w-full h-full object-cover"
                  onError={(e) => {
                    e.currentTarget.src = '/file.svg';
                  }}
                />
                {/* Duration overlay positioned at bottom-right */}
                {duration && (
                  <div className="absolute bottom-2 right-2 bg-black bg-opacity-80 text-white text-xs px-2 py-1 rounded">
                    {formatDuration(duration)}
                  </div>
                )}
              </a>
          </div>
        </div>

        {/* Video Info */}
        <div className="flex-1 h-63 flex flex-col justify-between">
          <div className="flex-1">
            <h3 className="font-medium text-lg leading-tight line-clamp-2 mb-2">
              {youtubeUrl ? (
                <a 
                  href={youtubeUrl} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="hover:text-blue-600 transition-colors"
                >
                  {title}
                </a>
              ) : (
                title
              )}
            </h3>
            <p className="text-sm text-muted-foreground mb-3">
              {channel}
            </p>
            
            {/* Video Stats */}
            <div className="flex items-center gap-2 text-sm text-muted-foreground mb-4">
              {viewCount && <span>{formatViewCount(viewCount)}</span>}
              {viewCount && uploadDate && <span>•</span>}
              {uploadDate && <span>{formatDate(uploadDate)}</span>}
            </div>

            {/* Content Preview */}
            <p className="text-sm text-muted-foreground line-clamp-3 mb-4 leading-relaxed">
              {result.content}
            </p>
          </div>

          {/* Action Badges - positioned at bottom */}
          <div className="flex flex-wrap gap-2 mt-auto">
            <div className={`text-sm px-2 py-1 rounded-md border text-center font-medium ${getSimilarityBadgeClasses(result.similarity)}`}>
              {result.similarity.toFixed(3)} similarity
            </div>
            {(metadata.like_count as number) && (
              <div className="text-sm px-2 py-1 rounded-md border text-center font-medium bg-gray-100 text-gray-700 border-gray-200">
                {(metadata.like_count as number) || 0} likes
              </div>
            )}
            <div className={`text-sm px-2 py-1 rounded-md border text-center font-medium ${popularity.classes}`}>
              {popularity.label}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}