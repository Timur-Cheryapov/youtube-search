# YouTube Search 🔍

AI-powered YouTube video discovery using semantic search. Find videos by meaning and content, not just titles and keywords.

![App Preview](/preview.gif)

## 🌐 Try it out

- **Live App**: [you-search-tube.netlify.app/](https://you-search-tube.netlify.app/)
- **Embedding API**: [huggingface.co/spaces/BangloCat/you-search-tube](https://huggingface.co/spaces/BangloCat/you-search-tube)

## 🎯 What it does

This application enables intelligent YouTube video search through:
- **Semantic understanding**: Search for concepts, topics, and ideas rather than exact word matches
- **AI-powered embeddings**: Uses sentence transformers to understand content meaning
- **Vector similarity search**: Finds videos with similar content even if they use different words

## 🏗️ Architecture & Flow

```mermaid
flowchart LR
    A[YouTube Channels] --> B[Python Crawler]
    B --> C[Video Metadata + Embeddings]
    C --> D[Supabase Vector DB]
    E[User Search Query] --> F[HuggingFace Embedding API]
    F --> G[Query Embedding]
    G --> H[Vector Similarity Search]
    D --> H
    H --> I[Next.js Frontend]
    I --> J[Search Results]
```

### Components

**🐍 Crawler** (`/crawler/`)
- Searches and processes YouTube channels automatically
- Extracts video metadata (titles, descriptions, URLs)
- Generates semantic embeddings for each video
- Stores everything directly in Supabase vector database

**🌐 Frontend** (`/app/`, `/components/`, `/lib/`)
- Next.js application with modern React components
- Real-time search interface with smooth animations
- Connects to Supabase for vector similarity search
- Uses HuggingFace API for query embedding generation

**🤖 Embedding Service** (`/huggingface-space/`)
- Gradio + FastAPI application for HuggingFace Spaces
- Provides embedding generation API using sentence transformers
- Processes search queries to create vector representations

## 🚀 Quick Setup

### 1. Prerequisites

- **Node.js 18+** for the frontend
- **Python 3.9+** for the crawler
- **Supabase account** for vector database
- **HuggingFace account** for embedding service (optional, can run locally)

### 2. Database Setup

Create a Supabase project and run these SQL commands:

```sql
-- Enable vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table for video data
CREATE TABLE documents (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  content TEXT NOT NULL,
  metadata JSONB DEFAULT '{}',
  embedding VECTOR(384), -- 384 dimensions for sentence-transformers/all-MiniLM-L6-v2
  created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::TEXT, NOW()) NOT NULL
);

-- Indexes for fast similarity search
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops with lists = 100);
CREATE INDEX ON documents USING gin (metadata);

-- Channel tracking table
CREATE TABLE channel_upload_stats (
  id BIGSERIAL PRIMARY KEY,
  channel_url TEXT UNIQUE NOT NULL,
  channel_name TEXT,
  videos_count INTEGER DEFAULT 0,
  data_loaded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

  -- Ensure uniqueness per channel
  UNIQUE(channel_url)
);
```

### 3. Security Setup (Row Level Security)

Apply RLS policies to secure your database:

```sql
-- Enable RLS on both tables
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE channel_upload_stats ENABLE ROW LEVEL SECURITY;

-- Allow public read access (for search functionality)
CREATE POLICY "Public read access for documents" ON documents
    FOR SELECT USING (true);

CREATE POLICY "Public read access for channel stats" ON channel_upload_stats
    FOR SELECT USING (true);

-- Restrict write access to authenticated users only
CREATE POLICY "Authenticated write access for documents" ON documents
    FOR ALL USING (auth.role() != 'anon');

CREATE POLICY "Authenticated write access for channel stats" ON channel_upload_stats
    FOR ALL USING (auth.role() != 'anon');
```

For complete RLS policies, see `docs/rls-policies.sql`.

### 4. Frontend Setup

```bash
# Install dependencies
npm install

# Configure environment
cp .env.example .env.local
# Add your Supabase URL and anon key
# Add your HuggingFace Space URL (or use local embedding)
# Add your Supabase service role key (required for crawler)

# Run development server
npm run dev
```

### 5. Crawler Setup

```bash
cd crawler

# Install Python dependencies
pip install -r requirements.txt

# Run the crawler (automated mode)
python crawler.py

# Or manual mode for specific channels
python crawler.py --manual
```

### 6. Embedding Service (Optional)

Deploy to HuggingFace Spaces or run locally:

```bash
cd huggingface-space

# Local development
pip install -r requirements.txt
python app.py

# Or deploy to HuggingFace Spaces
# See DEPLOYMENT.md for detailed instructions
```

## 🎮 Usage

1. **Collect Data**: Run the crawler to gather YouTube video data
   ```bash
   cd crawler && python crawler.py
   ```

2. **Start Frontend**: Launch the search interface
   ```bash
   npm run dev
   ```

3. **Search Videos**: Enter natural language queries like:
   - "machine learning tutorials for beginners"
   - "cooking pasta techniques"
   - "travel videos about Japan"

## 📁 Project Structure

```
youtube-search/
├── app/                    # Next.js app router pages
├── components/             # React components (SearchInterface, VideoCard)
├── lib/                   # Utilities (search, embeddings, types)
├── crawler/               # Python video crawler and embedder
├── huggingface-space/     # Embedding API service
└── docs/                  # Documentation and SQL files
```

## 🔧 Configuration

**Environment Variables** (`.env.local`):
```env
# Frontend configuration
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
NEXT_PUBLIC_HUGGING_FACE_SPACE_URL=your_hf_space_url

# Crawler configuration (for data insertion)
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
```

**Security Model**:
- **Frontend**: Uses `NEXT_PUBLIC_SUPABASE_ANON_KEY` for read-only access (search functionality)
- **Crawler**: Uses `SUPABASE_SERVICE_ROLE_KEY` for write access (adding new videos)
- **RLS Policies**: Ensure public users can only read data, not modify it

**Crawler Settings** (`crawler/config.py`):
```python
MAX_CONCURRENT_VIDEOS = 8      # Adjust for your CPU
EMBEDDING_BATCH_SIZE = 16      # Adjust for your RAM
VIDEO_LIMIT_PER_CHANNEL = 20   # Videos per channel
SUPABASE_ENABLED = True        # Direct database storage

# Security settings
SUPABASE_USE_ANON_KEY_FOR_READS = True     # Use anon key for read operations
SUPABASE_REQUIRE_SERVICE_KEY_FOR_WRITES = True  # Use service key for writes
```

## 🛠️ Tech Stack

- **Frontend**: Next.js 15, React 19, TypeScript, Tailwind CSS, Shadcn/ui
- **Backend**: Python, yt-dlp, sentence-transformers, Supabase
- **Database**: PostgreSQL with pgvector extension
- **Deployment**: Netlify (frontend), HuggingFace Spaces (embedding API)

## Licencse
MIT License

Copyright (c) 2025 [Timur Cheryapov](https://timur-cheryapov.github.io/)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
