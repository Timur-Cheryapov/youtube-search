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
pip install yt-dlp sentence-transformers torch transformers tqdm psutil
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

## 🛡️ Memory Management & Fail-Safes

The crawler includes comprehensive memory monitoring and protection:

### **Automatic Memory Monitoring**
- **Real-time tracking**: Monitors RAM usage throughout processing
- **Smart thresholds**: Warning at 85% RAM, critical at 95% RAM
- **Process tracking**: Monitors both system and process memory usage

### **Fail-Safe Mechanisms**
- **Model loading protection**: Checks memory before loading ML models
- **Batch size adaptation**: Automatically reduces processing load if memory is low
- **Progressive saves**: Saves backups every 3 channels to prevent data loss
- **Graceful degradation**: Reduces video processing if memory gets critical
- **Emergency stops**: Halts processing before system crashes

### **Memory Management Features**
- **GPU cache clearing**: Automatically clears CUDA memory when needed
- **Garbage collection**: Forces Python garbage collection during high usage
- **Chunk saving**: Saves large files in smaller pieces if memory is insufficient
- **Memory recovery**: Attempts cleanup and recovery before stopping

### **Memory Status Indicators**
- ✅ **Safe**: Normal operation (< 85% RAM)
- ⚠️ **Warning**: High usage (85-95% RAM) - cleanup initiated
- 🚨 **Critical**: Dangerous levels (> 95% RAM) - processing stopped

## Dependencies

Core requirements:
```bash
pip install yt-dlp sentence-transformers torch transformers tqdm psutil
```

Or install from requirements if created:
```bash
pip install -r requirements.txt
```

**System Requirements:**
- Python 3.8+
- **4GB RAM minimum, 8GB+ recommended**
- Optional: CUDA-compatible GPU for faster embedding generation

## Integration

The generated embeddings can be used with:
- **Vector Databases**: Pinecone, Weaviate, Qdrant
- **Search Systems**: Elasticsearch with dense vector search
- **Recommendation Engines**: Similarity-based content discovery
- **Next.js App**: Direct integration with the YouTube search interface

## Troubleshooting

### Common Issues
1. **Memory Exhaustion**: The crawler will automatically handle this with fail-safes
2. **CUDA Out of Memory**: GPU cache is automatically cleared; reduce `VIDEO_LIMIT_PER_CHANNEL` if persistent
3. **Rate Limiting**: YouTube may temporarily block requests; wait and retry
4. **Model Loading**: Ensure sufficient disk space (~2GB) and RAM (~4GB minimum)

### Memory-Related Solutions
- **High memory usage**: Crawler automatically reduces batch sizes and processes fewer videos
- **System slowdown**: Memory monitor will force cleanup and garbage collection
- **Crash prevention**: Processing stops before reaching critical memory levels
- **Data preservation**: Partial results are saved before stopping due to memory constraints

### Performance Tips
- **Close other applications** before running for maximum available RAM
- **Use GPU** for 3-5x faster embedding generation (but monitor GPU memory)
- **SSD storage** significantly improves model loading and data writing speeds
- **Monitor logs** for memory status updates and optimization suggestions