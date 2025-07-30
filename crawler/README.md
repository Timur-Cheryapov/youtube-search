# YouTube Crawler with Automatic Channel Discovery

An enhanced YouTube crawler that automatically discovers channels through search queries and generates embeddings for semantic search.

## Features

### 🚀 Automated Mode (Default)
- **Smart Channel Discovery**: Uses predefined search queries to find relevant channels
- **Duplicate Prevention**: Tracks processed channels in `processed_channels.json`
- **Batch Processing**: Processes multiple channels and videos in one run
- **Progress Tracking**: Visual progress bars and comprehensive logging

### 🔧 Manual Mode
- **Single Channel Processing**: Process a specific channel URL
- **Legacy Compatibility**: Maintains the original workflow for targeted crawling

## Quick Start

### Automated Discovery (Recommended)
```bash
cd crawler
pip install yt-dlp sentence-transformers torch transformers tqdm
python crawler.py
```

### Manual Channel Processing
```bash
python crawler.py --manual
```

## Search Queries

The automated mode uses 5 broad categories to discover diverse content:

1. **Science Education**: `"science education tutorials"`
2. **Technology**: `"technology programming tutorials"`
3. **History**: `"history documentary"`
4. **Cooking**: `"cooking recipes techniques"`
5. **Fitness**: `"fitness workout training"`

**Limits per run:**
- 3 channels per search query
- 10 videos per channel
- Total: ~150 videos with embeddings

## Output Files

### `youtube_videos_with_embeddings.json`
Contains video metadata with embeddings:
```json
{
  "https://www.youtube.com/channel/UCxxxxx": [
    {
      "id": "video_id",
      "title": "Video Title",
      "description": "...",
      "source_channel": {
        "channel_name": "Channel Name",
        "found_via_query": "science education tutorials"
      },
      "embedding": {
        "text": "processed content for embedding",
        "vector": [0.1, 0.2, ...],
        "dimensions": 384,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "extraction_model": "HuggingFaceTB/SmolLM2-1.7B-Instruct"
      }
    }
  ]
}
```

### `processed_channels.json`
Tracks which channels have been processed:
```json
{
  "UC_channel_id": {
    "channel_url": "https://www.youtube.com/channel/UC_channel_id",
    "channel_name": "Channel Name",
    "found_via_query": "science education tutorials",
    "processed_date": "2024-01-01T12:00:00",
    "video_count": 10,
    "status": "processed"
  }
}
```

## Technical Details

### Models Used
- **Embedding Generation**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **Content Extraction**: `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- **Video Metadata**: `yt-dlp` with YouTube API

### Content Processing
1. **Smart Description Processing**: Extracts main content while filtering out credits, sponsors, and promotional text
2. **LLM-Enhanced Extraction**: Uses SmolLM2 to identify core video topics
3. **Embedding Generation**: Creates semantic vectors from title + processed content

### Performance
- **GPU Acceleration**: Automatic CUDA detection for faster processing
- **Batch Processing**: Loads models once, processes multiple videos efficiently
- **Error Handling**: Robust error handling with detailed logging

## Customization

### Modify Search Queries
Edit the `SEARCH_QUERIES` list in `automated_crawler()`:
```python
SEARCH_QUERIES = [
    "your custom search query",
    "another topic of interest",
    # Add more queries...
]
```

### Adjust Limits
Modify constants in `automated_crawler()`:
```python
MAX_CHANNELS_PER_QUERY = 5  # More channels per search
VIDEO_LIMIT_PER_CHANNEL = 20  # More videos per channel
```

### Change Models
Update model names in `VideoEmbedder.__init__()`:
```python
embedding_model_name = 'sentence-transformers/all-mpnet-base-v2'  # Higher quality
content_extractor_model_name = 'microsoft/DialoGPT-medium'  # Different LLM
```

## Integration

The generated embeddings can be used with:
- **Vector Databases**: Pinecone, Weaviate, Qdrant
- **Search Systems**: Elasticsearch with dense vector search
- **Recommendation Engines**: Similarity-based content discovery
- **Next.js App**: Direct integration with the YouTube search interface

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce `VIDEO_LIMIT_PER_CHANNEL` or use CPU-only mode
2. **Rate Limiting**: YouTube may temporarily block requests; wait and retry
3. **Model Loading**: Ensure sufficient disk space for downloading models (~2GB)

### Performance Tips
- Run on GPU for 3-5x faster embedding generation
- Use SSD storage for faster model loading
- Increase batch sizes if you have more VRAM available