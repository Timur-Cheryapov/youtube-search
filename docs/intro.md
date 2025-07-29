````markdown
# Vector Search Web App - System Overview

This document describes the logic and architecture of a web app designed to perform semantic search using vector embeddings. The goal is to create a full-stack, cost-effective, and deployable solution that does **not rely on paid APIs like OpenAI**, and uses freely available hosting services.

---

## 🧱 System Components

### 1. **Frontend**
- **Framework**: Next.js (TypeScript)
- **Hosting**: Netlify
- **Function**:
  - User inputs a search query via UI
  - Sends query to backend API (`/api/embed`)
  - Displays the search results retrieved from Supabase

---

### 2. **Backend API (Serverless Function)**
- **Location**: Next.js API route (Netlify function)
- **Function**:
  - Receives query text
  - Calls the Hugging Face Space to get an embedding for the query
  - Queries Supabase using `pgvector` to find the most similar entries
  - Returns matched documents to the frontend

---

### 3. **Embedding Model**
- **Hosting**: Hugging Face Spaces (Free tier)
- **Backend**: FastAPI
- **Model**: `sentence-transformers/all-MiniLM-L6-v2` or `intfloat/e5-small-v2`
- **Function**:
  - Accepts a POST request with one or more text strings
  - Returns their embeddings in JSON format

---

### 4. **Vector Database**
- **Platform**: Supabase (Free tier)
- **Extension**: `pgvector`
- **Tables**:
  - `documents`: stores original text, metadata, and vector
- **Function**:
  - Accepts a vector and returns top N closest matches based on cosine similarity

---

## 🔄 Data Flow

1. User enters a query in the search bar.
2. Frontend sends query to `POST /api/embed`.
3. API route forwards the query to the Hugging Face embedding endpoint.
4. Embedding is returned (array of floats).
5. API route uses Supabase client to send a `SELECT` statement with vector similarity:
   ```sql
   SELECT *, embedding <#> $1 AS distance
   FROM documents
   ORDER BY distance ASC
   LIMIT 10;
````

6. API route returns the result set to the frontend.
7. Frontend displays the results to the user.

---

## ✅ Hosting Strategy

| Component     | Service   | Notes                         |
| ------------- | --------- | ----------------------------- |
| Frontend      | Netlify   | Static + serverless functions |
| Embedding API | HF Spaces | Free GPU/CPU runtime          |
| Database      | Supabase  | Includes pgvector, free tier  |

---

## 🚀 Features

* Custom, free embedding backend (not OpenAI)
* Real-time semantic search via vector similarity
* Minimal cost (entirely free on selected platforms)
* Designed to showcase vector search understanding

---

## 🧠 Notes

* Hugging Face Space may sleep when inactive (cold start delays).
* Use sentence-transformers model that supports CPU.
* pgvector must be enabled in Supabase SQL editor:

  ```sql
  create extension if not exists vector;
  ```
* Store embeddings as `vector(384)` for MiniLM.