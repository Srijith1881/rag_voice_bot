# Voice Assistant with RAG

A **real-time voice assistant** that answers questions about electric cars using LiveKit and RAG (Retrieval-Augmented Generation).

## 🎯 Overview

This project combines:
- **LiveKit Playground** for real-time voice interaction
- **RAG (Retrieval-Augmented Generation)** for intelligent document-based answers
- **Persistent vector storage** for fast, efficient processing
- **Deepgram** for speech-to-text and text-to-speech
- **Google Gemini** for LLM-powered answers

---

## 📚 Documentation

For detailed explanations, refer to the included documentation:

- **[PROJECT_FLOW.md](./PROJECT_FLOW.md)** - Complete architecture & detailed flow diagrams
- **[RAG_EXPLANATION.md](./RAG_EXPLANATION.md)** - How the RAG system works step-by-step
- **[README_VECTORS.md](./README_VECTORS.md)** - Persistent vector storage guide

---

## 🚀 Quick Start

### 1. Setup (First Time Only)
Pre-compute vectors from your PDF knowledge base:
```bash
python precompute_vectors.py
```
This creates persistent vectors stored in `data/faiss_index/` (~10-15 seconds, one-time only).

### 2. Start Voice Bot
```bash
python basic_voice.py start
```
The bot will:
- Load cached vectors instantly (~1-2 seconds)
- Connect to LiveKit
- Wait for user connection

### 3. Test RAG (Optional)
Test the RAG system without voice:
```bash
python test_rag.py
```

---

## 📊 Data Flow

```
User speaks (via LiveKit Playground)
    ↓
Deepgram STT (converts speech → text)
    ↓
RAG Engine (retrieves relevant context from PDF)
    ↓
Gemini LLM (generates answer from context only)
    ↓
Deepgram TTS (converts answer → speech)
    ↓
User hears answer
```

---

## 🏗️ Architecture

### Key Components

- **`basic_voice.py`** - Main entry point (LiveKit integration + voice handling)
- **`rag_engine.py`** - RAG pipeline (retrieval → generation)
- **`precompute_vectors.py`** - One-time vector pre-computation script
- **`data/Electric_Car_Overview.pdf`** - Knowledge base
- **`data/faiss_index/`** - Persistent vector store (auto-created)

### RAG Pipeline

1. **Retrieval** 🔍 - Find top 3 relevant chunks from PDF
2. **Augmentation** 📚 - Combine with question + strict rules
3. **Generation** 🤖 - Generate answer using Gemini (from context only)

---

## 🔧 Configuration

### Environment Variables (.env)
```
DEEPGRAM_API_KEY=your_key
GOOGLE_API_KEY=your_key
```

### RAG Settings
- Chunk size: 1000 characters
- Chunk overlap: 200 characters
- Retrieved chunks: 3
- Temperature: 0.2 (strict, deterministic)

### Voice Settings
- STT: Deepgram "nova-2"
- LLM: Google "gemini-2.5-flash-lite"
- TTS: Deepgram "aura-asteria-en"
- VAD: Silero

---

## ✨ Key Features

✅ **Real-time voice interaction** via LiveKit Playground  
✅ **Persistent vectors** - compute once, use forever  
✅ **Fast startup** - instant vector loading from cache  
✅ **Knowledge-based answers** - only answers from PDF context  
✅ **No hallucinations** - rejects out-of-scope questions  

---

## 🎤 LiveKit Playground

Connect using LiveKit Playground:
1. Start the voice bot: `python basic_voice.py start`
2. Visit LiveKit Playground (URL will be displayed)
3. Share room URL with others
4. Speak questions naturally

---

## ❓ Example Questions

✓ "What are electric cars?"  
✓ "How do electric vehicles work?"  
✓ "What is the range of electric cars?"  
✗ "What's the weather today?" → "I don't have that information"

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Slow startup | Vectors not cached - run `python precompute_vectors.py` |
| "Could not load cache" | Run `python precompute_vectors.py` |
| Wrong answers | Check prompt template in `rag_engine.py` |
| Want to rebuild | `rm -rf data/faiss_index && python precompute_vectors.py` |

---

*Built with LiveKit, LangChain, FAISS, and Google Gemini* 🚀
