# Vector Search App Setup Guide

This guide will help you set up the Vector Search application with all required services.

## 📋 Prerequisites

- Node.js 18+ installed
- A Supabase account (free tier available)
- A Hugging Face account (optional, for production embeddings)

## 🚀 Quick Start

### 1. Clone and Install

```bash
git clone <your-repo>
cd vector-search-app
npm install
```

### 2. Environment Setup

Copy the example environment file:

```bash
cp .env.local.example .env.local
```

### 3. Supabase Setup

1. **Create a new Supabase project** at [supabase.com](https://supabase.com)

2. **Get your credentials** from Project Settings > API:
   - `NEXT_PUBLIC_SUPABASE_URL`: Your project URL
   - `NEXT_PUBLIC_SUPABASE_ANON_KEY`: Your anonymous/public key

3. **Enable pgvector extension** in SQL Editor:

```sql
-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
```

4. **Create the documents table**:

```sql
-- Create documents table with vector support
CREATE TABLE documents (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  content TEXT NOT NULL,
  metadata JSONB DEFAULT '{}',
  embedding VECTOR(384), -- 384 dimensions for sentence-transformers/all-MiniLM-L6-v2
  created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::TEXT, NOW()) NOT NULL
);

-- Create an index for vector similarity search
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

5. **Create the search function**:

```sql
-- Function to search documents by vector similarity
CREATE OR REPLACE FUNCTION search_documents(
  query_embedding VECTOR(384),
  match_threshold FLOAT DEFAULT 0.8,
  match_count INT DEFAULT 10
)
RETURNS TABLE(
  id UUID,
  content TEXT,
  metadata JSONB,
  embedding VECTOR(384),
  created_at TIMESTAMP WITH TIME ZONE,
  distance FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    documents.id,
    documents.content,
    documents.metadata,
    documents.embedding,
    documents.created_at,
    (documents.embedding <#> query_embedding) * -1 AS distance
  FROM documents
  WHERE (documents.embedding <#> query_embedding) * -1 > match_threshold
  ORDER BY documents.embedding <#> query_embedding
  LIMIT match_count;
END;
$$;
```

6. **Insert sample data** (optional):

```sql
-- Sample documents for testing
INSERT INTO documents (content, metadata) VALUES
('Artificial intelligence is transforming how we work and live', '{"category": "technology", "author": "AI Expert"}'),
('Machine learning algorithms can identify patterns in large datasets', '{"category": "data science", "author": "Data Scientist"}'),
('Natural language processing enables computers to understand human language', '{"category": "AI", "author": "NLP Researcher"}'),
('Vector databases store and search high-dimensional data efficiently', '{"category": "database", "author": "DB Engineer"}'),
('Semantic search goes beyond keyword matching to understand meaning', '{"category": "search", "author": "Search Engineer"}');
```

### 4. Hugging Face Setup (Optional)

For production embeddings, you can deploy a Hugging Face Space:

1. **Create a new Space** at [huggingface.co/spaces](https://huggingface.co/spaces)
2. **Choose "Custom" with Python**
3. **Deploy the embedding service** (FastAPI app with sentence-transformers)
4. **Add the Space URL** to your `.env.local`:

```
NEXT_PUBLIC_HUGGING_FACE_SPACE_URL=https://your-username-your-space-name.hf.space
```

> **Note**: If you don't set up Hugging Face Spaces, the app will use mock embeddings for development.

### 5. Update Environment Variables

Update your `.env.local` file with the actual values:

```env
NEXT_PUBLIC_SUPABASE_URL=https://your-project-id.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key-here
NEXT_PUBLIC_HUGGING_FACE_SPACE_URL=https://your-username-your-space-name.hf.space
```

### 6. Run the Application

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to see your vector search app!

## 🔧 Development Notes

- **Mock Embeddings**: Without Hugging Face setup, the app uses random embeddings for development
- **Database Indexing**: The ivfflat index improves search performance for larger datasets
- **Vector Dimensions**: Default is 384 for `sentence-transformers/all-MiniLM-L6-v2`
- **Similarity Threshold**: Adjust `match_threshold` in the search function to filter results

## 🎯 Next Steps

1. **Add more documents** to your database
2. **Deploy to production** (Netlify/Vercel + Supabase + HF Spaces)
3. **Customize the UI** to match your brand
4. **Implement user authentication** if needed
5. **Add document upload functionality**

## 📚 Architecture

- **Frontend**: Next.js with TypeScript and Tailwind CSS (client-side search, no API routes)
- **Database**: Supabase (PostgreSQL + pgvector) called directly from client
- **Embeddings**: Hugging Face Spaces (sentence-transformers) called directly from client
- **Deployment**: Netlify/Vercel (frontend) + Supabase (database) + HF Spaces (embeddings)

All services offer free tiers, making this a completely free solution for moderate usage! 