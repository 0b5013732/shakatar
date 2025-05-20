# Shaka AI Chatbot MVP

This repository contains a minimal implementation of the **Shaka AI Chatbot**. It ingests text from a local directory, fine‑tunes a Llama‑based model (placeholder), and exposes a REST API with a simple React UI.

## Project Structure

```
├── client/              # React chat interface (served statically)
│   ├── index.html
│   └── app.jsx
├── data/
│   ├── corpus/          # Source text files
│   └── processed/       # Output of ingestion script
├── logs/                # Server logs
├── scripts/
│   ├── ingest.js        # Corpus ingestion
│   └── train.py         # Placeholder model training
├── server/
│   ├── app.js           # Express server
│   ├── routes/
│   │   ├── audio.js
│   │   └── chat.js
│   ├── services/
│   │   ├── llm.js
│   │   └── tts.js
│   └── utils/
│       ├── logger.js
│       └── pinecone.js
└── package.json
```

## Setup

1. Install dependencies:

```bash
npm install
```

2. Place Shaka Senghor's writings (txt/markdown) inside `data/corpus/`.
3. Run the ingestion script to prepare the corpus:

```bash
node scripts/ingest.js
```

4. (Optional) Fine‑tune the Llama model using the placeholder script:

```bash
python3 scripts/train.py --data data/processed/corpus.jsonl --out model/
```

5. Start the API server:

```bash
npm start
```

The React UI is served from `client/` and is accessible in the browser at `http://localhost:3000` by default.

## Notes

- `server/services/llm.js` contains a placeholder implementation for generating answers. Integrate your chosen local LLM here.
- `server/services/tts.js` calls the ElevenLabs API and requires `ELEVENLABS_API_KEY` and `ELEVENLABS_VOICE_ID` environment variables.
- Pinecone configuration is handled in `server/utils/pinecone.js` via environment variables `PINECONE_API_KEY`, `PINECONE_ENV`, `PINECONE_INDEX`, `PINECONE_HOST`, and `PINECONE_EMBEDDING_MODEL`.

This MVP is intended for private use on a local machine.
