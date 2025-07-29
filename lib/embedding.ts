// Client-side embedding functions that work directly with HuggingFace
export async function callHuggingFaceEmbedding(text: string): Promise<number[]> {
  const HF_SPACE_URL = process.env.NEXT_PUBLIC_HUGGING_FACE_SPACE_URL;
  
  if (!HF_SPACE_URL) {
    // Mock embedding for development (384 dimensions for MiniLM)
    console.log('Using mock embedding for development. Set NEXT_PUBLIC_HUGGING_FACE_SPACE_URL for production.');
    return Array.from({ length: 384 }, () => Math.random() * 2 - 1);
  }

  const response = await fetch(`${HF_SPACE_URL}/api/embed`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text }),
  });

  if (!response.ok) {
    throw new Error(`Hugging Face API error: ${response.statusText}`);
  }

  const data = await response.json();
  return data.embedding;
}