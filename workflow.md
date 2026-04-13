# Workflow — Voice Assistant with RAG

## System Overview

A real-time voice assistant that answers questions about electric cars. Speech is captured via LiveKit, transcribed by Deepgram, answered using a RAG pipeline (FAISS + Gemini), and spoken back via Deepgram TTS.

---

## Phase 1: One-Time Setup (Vector Pre-computation)

Run once before starting the bot — or whenever the PDF changes.

```
python precompute_vectors.py
```

**What happens internally (`precompute_vectors.py` → `rag_engine.initialize_rag`):**

1. Calls `initialize_rag(use_cache=True, force_rebuild=True)`
2. **PDF Load** — `PyPDFLoader` reads `data/Electric_Car_Overview.pdf`
3. **Chunking** — `RecursiveCharacterTextSplitter` splits pages into chunks
   - Chunk size: 1000 characters
   - Overlap: 200 characters
4. **Embedding** — `HuggingFaceEmbeddings` (`sentence-transformers/all-MiniLM-L6-v2`) converts each chunk into a dense vector
5. **FAISS Index** — All vectors are stored in a FAISS index
6. **Persist** — Index saved to `data/faiss_index/` on disk

After this step, the index is cached permanently. This phase takes ~10–15 seconds and does **not** need to be repeated.

---

## Phase 2: Bot Startup

```
python basic_voice.py start
```

**What happens:**

1. `load_dotenv(".env")` — Loads `DEEPGRAM_API_KEY` and `GOOGLE_API_KEY`
2. `agents.cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))` — LiveKit worker process starts
3. `entrypoint(ctx)` is called when a room is assigned:
   - **STT** — `deepgram.STT(model="nova-2")` registered
   - **LLM** — `google.LLM(model="gemini-2.5-flash-lite")` registered (used by session; RAG overrides actual answer generation)
   - **TTS** — `deepgram.TTS(model="aura-asteria-en")` registered
   - **VAD** — `silero.VAD.load()` registered for voice activity detection
4. `AgentSession.start(room, agent=Assistant())` — session attached to the LiveKit room
5. A warm greeting is generated via `session.generate_reply(instructions=...)`
6. **RAG chain is NOT loaded yet** — it is lazy-loaded on first user speech

---

## Phase 3: Per-Query Voice Interaction Flow

This is the end-to-end path for every user question.

```
User speaks
    │
    ▼
[VAD — Silero]
Detects speech activity; buffers audio until silence
    │
    ▼
[STT — Deepgram nova-2]
Converts audio stream → transcript text
    │
    ▼
[Assistant.on_user_speech(ctx, message)]
    │
    ├─ If rag_chain is None (first query):
    │       get_rag_chain() is called
    │       └─ Thread-safe double-checked lock (_rag_lock)
    │          └─ initialize_rag(use_cache=True, force_rebuild=False)
    │             └─ FAISS.load_local("data/faiss_index", embeddings)
    │                (loads in ~1–2 seconds from disk)
    │
    ▼
[RAG Engine — rag_chain.run(message)]
    │
    ├─ Step 1 — RETRIEVAL
    │   retriever.invoke(question)
    │   └─ Embeds question using all-MiniLM-L6-v2
    │   └─ Similarity search against FAISS index
    │   └─ Returns top 3 most relevant document chunks
    │
    ├─ Step 2 — AUGMENTATION
    │   format_docs(docs) → joins chunk texts with "\n\n"
    │   Fills PromptTemplate:
    │     - {context} = joined chunk text
    │     - {question} = original user question
    │   Strict rules enforced in prompt:
    │     - Only answer from context
    │     - If answer not in context → "I don't have that information in my knowledge base."
    │
    ├─ Step 3 — GENERATION
    │   GoogleGenerativeAILLM._call(prompt)
    │   └─ Model: gemini-flash-latest
    │   └─ Temperature: 0.2 (low = deterministic, factual)
    │   └─ Returns grounded text answer
    │
    └─ Step 4 — PARSING
        StrOutputParser() extracts plain string from LLM response
    │
    ▼
[ctx.send_reply(rag_answer)]
Answer returned to AgentSession
    │
    ▼
[TTS — Deepgram aura-asteria-en]
Converts answer text → audio
    │
    ▼
User hears spoken answer via LiveKit Playground
```

---

## Phase 4: Optional — RAG Testing (No Voice)

```
python test_rag.py
```

- Calls `get_rag_chain()` directly (loads from FAISS cache)
- Runs 5 predefined questions through `rag_chain.run()`
- Prints answers to terminal — no audio, no LiveKit involved
- Out-of-scope questions (e.g., "What is apple") return the fallback message

---

## Caching & Thread Safety

| Mechanism | Purpose |
|-----------|---------|
| `data/faiss_index/` on disk | Persistent vector store — survives restarts |
| `_rag_chain_cache` (global) | In-process singleton — RAG init runs once per process |
| `_rag_lock` (threading.Lock) | Double-checked locking prevents duplicate init on concurrent requests |
| `force_rebuild=True` | Used by `precompute_vectors.py` to force regeneration |

---

## File Responsibilities

| File | Role |
|------|------|
| `basic_voice.py` | LiveKit entry point; session setup; routes speech to RAG |
| `rag_engine.py` | Full RAG pipeline — embed, retrieve, augment, generate |
| `precompute_vectors.py` | One-time script to build and persist FAISS index |
| `test_rag.py` | Standalone RAG testing without voice stack |
| `data/Electric_Car_Overview.pdf` | Source knowledge base |
| `data/faiss_index/` | Persisted FAISS vector store (auto-created) |
| `.env` | API keys — `DEEPGRAM_API_KEY`, `GOOGLE_API_KEY` |

---

## Key Constraints

- Answers are **strictly grounded** in the PDF — no external knowledge used
- Temperature `0.2` ensures consistent, low-creativity responses
- Out-of-scope questions always return: `"I don't have that information in my knowledge base."`
- RAG chain is initialized **at most once** per process (lazy + cached)
