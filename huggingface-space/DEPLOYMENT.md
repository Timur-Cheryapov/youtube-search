# 🚀 Deployment Guide

Follow these steps to deploy your embedding API to Hugging Face Spaces.

## 📋 Step-by-Step Setup

### 1. Create a New Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Choose a name (e.g., `your-username/embedding-api`)
4. Select **"Gradio"** as the SDK
5. Choose **"Public"** (free) or **"Private"** (if you have a subscription)
6. Click **"Create Space"**

### 2. Upload the Files

Upload these files to your Space:

```
📁 Your Space Repository
├── 📄 app.py              # Main application
├── 📄 requirements.txt    # Dependencies  
├── 📄 README.md          # Space configuration
└── 📄 .gitignore         # Git ignore rules
```

**Methods to upload:**

**Option A: Web Interface**
- Use the HF Spaces file upload interface
- Drag and drop each file

**Option B: Git (Recommended)**
```bash
git clone https://huggingface.co/spaces/your-username/your-space-name
cd your-space-name
# Copy the files from huggingface-space/ folder
git add .
git commit -m "Initial embedding API setup"
git push
```

### 3. Wait for Build

- HF Spaces will automatically build your app
- Initial build takes ~3-5 minutes (downloading model)
- Status will change from "Building" to "Running"

### 4. Test Your API

Once deployed:

**Web Interface:**
- Visit: `https://your-username-your-space-name.hf.space`

**API Endpoint:**
```bash
curl -X POST "https://your-username-your-space-name.hf.space/api/embed" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'
```

### 5. Configure Your Vector Search App

Update your `.env.local` file:

```env
HUGGING_FACE_SPACE_URL=https://your-username-your-space-name.hf.space
```

### 6. Verify Integration

1. Start your Next.js app: `npm run dev`
2. Try a search query
3. Verify embeddings are being generated properly

Your embedding API will be live and ready! 🎉 