---
title: Text Embedding API
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# Text Embedding API

A simple API for generating text embeddings using sentence-transformers.

## Features

- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **Web Interface**: Interactive Gradio interface for testing
- **API Endpoint**: REST API at `/api/embed` for integration
- **Batch Processing**: Support for multiple texts at once

## Usage

### Web Interface

Visit the Space URL to use the interactive interface.

### API Endpoint

**POST** `/api/embed`

```bash
curl -X POST "https://your-space-name.hf.space/api/embed" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here"}'
```

**Response:**
```json
{
  "text": "Your text here",
  "embedding": [0.1, -0.2, 0.3, ...],
  "dimensions": 384,
  "model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

## Integration

This Space is designed to work with vector search applications. Set your `HUGGING_FACE_SPACE_URL` environment variable to your Space URL.

## Model Information

- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Dimensions**: 384
- **Use Cases**: Semantic search, document similarity, clustering
- **Performance**: Optimized for speed and quality balance

## License

MIT License - Feel free to use and modify! 