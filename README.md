# ECB Speech Tone Explorer (RAG)

A retrieval-augmented generation (RAG) demo that lets you ask natural-language questions about European Central Bank (ECB) speeches and get back relevant speeches, their monetary policy tone (Dovish / Neutral / Hawkish), and an AI-generated summary.

This project extends my MS thesis work — **FinBERTHybrid v15**, a hybrid deep learning model that classifies ECB speech tone using FinBERT contextual embeddings, Loughran-McDonald lexical features, and German Bund yield temporal signals via gated fusion — into an interactive, queryable RAG pipeline.

> **Note:** This is a demo/extension project built to practice applied RAG system design (embeddings, vector search, LLM integration). It is not part of the core thesis contribution, which is the FinBERTHybrid classification architecture itself.

## What it does

1. **Retrieve** — given a query, finds the most relevant ECB speeches from a corpus of ~2,300 speeches (1999–2026) using semantic search (sentence-transformer embeddings + FAISS).
2. **Classify** — each retrieved speech is passed through the trained FinBERTHybrid model, which predicts its monetary policy tone (Dovish/Neutral/Hawkish) using text, lexical, and bond-yield signals.
3. **Generate** — an LLM (via Groq) synthesizes the retrieved speeches and their classifications into a short, readable answer.

## Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│  Retrieval           │  sentence-transformers embeddings + FAISS
│  (semantic search)   │  optional year-filter for date-specific queries
└─────────┬─────────────┘
          │ top-k relevant speeches
          ▼
┌─────────────────────┐
│  Classification       │  FinBERTHybrid v15 (trained model)
│                       │  FinBERT (attention-pooled) + LM lexical branch
│                       │  + Bund yield temporal branch → gated fusion
└─────────┬─────────────┘
          │ tone + confidence per speech
          ▼
┌─────────────────────┐
│  Generation            │  Groq (Llama 3.3 70B)
│                       │  synthesizes retrieved speeches + tones
└─────────┬─────────────┘
          │
          ▼
   Answer + speech table (Gradio UI)
```

## Tech stack

| Component | Tool |
|---|---|
| Retrieval embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector search | FAISS |
| Tone classification | FinBERTHybrid v15 (custom PyTorch model, trained from scratch) |
| Text generation | Groq API (Llama 3.3 70B, free tier) |
| Interface | Gradio |
| Environment | Google Colab (free tier) |

## The classification model (FinBERTHybrid v15)

- FinBERT contextual embeddings with **attention pooling** (not CLS/mean)
- Loughran-McDonald lexical sentiment features (with a skip connection)
- German 10Y Bund yield temporal features (yield change, volatility, level)
- 853-dimensional **gated fusion** combining all three signal branches
- Long speeches are handled via overlapping 512-token chunks (no truncation), with chunk-level predictions averaged at inference time

## Running it

1. Open `ECB_RAG_Pipeline_Complete.ipynb` in Google Colab
2. Mount Google Drive (dataset + Loughran-McDonald dictionary must be present there — see notebook for expected paths)
3. Add a `GROQ_API_KEY` secret in Colab (🔑 icon in the left sidebar) — get a free key at [console.groq.com](https://console.groq.com)
4. Run all cells top to bottom
5. The final cell launches a Gradio interface with a shareable link
## Demo Video

[![Watch the demo](https://img.youtube.com/vi/6lNYiQ0uRnQ/maxresdefault.jpg)](https://youtu.be/6lNYiQ0uRnQ)

## Example queries

- "What was the ECB's stance on inflation in 2022?"
- "How did the ECB discuss interest rates in 2018?"
- "Summarize the ECB's tone during the 2011 sovereign debt crisis"

## Future work

- Date-range filtering (not just single-year)
- Trend visualization of tone over time
- Multi-turn conversational memory

## About

Built by [SZ] as an extension of an MS thesis on ECB monetary policy speech classification, and as a hands-on RAG project for PhD applications in NLP/computational linguistics.
