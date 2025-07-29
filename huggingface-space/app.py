import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model once at startup
logger.info("Loading sentence transformer model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
logger.info("Model loaded successfully!")

def embed_text(text):
    """Generate embeddings for input text"""
    try:
        if not text or not text.strip():
            return {"error": "Text input is required"}
        
        # Generate embedding
        embedding = model.encode(text.strip())
        
        # Convert to list for JSON serialization
        embedding_list = embedding.tolist()
        
        return {
            "text": text.strip(),
            "embedding": embedding_list,
            "dimensions": len(embedding_list),
            "model": "sentence-transformers/all-MiniLM-L6-v2"
        }
    
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        return {"error": f"Failed to generate embedding: {str(e)}"}

# Create FastAPI app first
app = FastAPI(title="Text Embedding API", description="Generate vector embeddings using sentence transformers")

# API endpoint using FastAPI
@app.post("/api/embed")
async def api_embed(request: Request):
    """API endpoint for external embedding requests"""
    try:
        data = await request.json()
        
        if not data or 'text' not in data:
            raise HTTPException(status_code=400, detail="Missing 'text' field in request body")
        
        text = data['text']
        result = embed_text(text)
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"API request failed: {str(e)}")

# Create Gradio interface
with gr.Blocks(title="Text Embedding API", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 Text Embedding API
    
    Generate vector embeddings using sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
    """)
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Input Text",
                placeholder="Enter text to embed...",
                lines=3
            )
            embed_btn = gr.Button("Generate Embedding", variant="primary")
        
        with gr.Column():
            output = gr.JSON(label="Embedding Result")
    
    embed_btn.click(
        fn=embed_text,
        inputs=[text_input],
        outputs=[output]
    )
    
    gr.Markdown("""
    ## API Usage
    
    **POST** `/api/embed`
    
    ```json
    {"text": "Your text here"}
    ```
    """)

# Mount Gradio app to FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860) 