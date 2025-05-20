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

1. Install the Node.js dependencies:

```bash
npm install
```

2. (Recommended) create a Python virtual environment and install `torch` and `transformers` for the training script:

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch transformers
```

3. Place Shaka Senghor's writings (txt/markdown) inside `data/corpus/`.
4. Run the ingestion script to prepare the corpus:

```bash
node scripts/ingest.js
```

5. (Optional) Fine‑tune the Llama model using the placeholder script:

```bash
python3 scripts/train.py --data data/processed/corpus.jsonl --out model/
```

6. Start the API server:

```bash
npm start
```

The React UI is served from `client/` and is accessible in the browser at `http://localhost:3000` by default.

### Environment variables

Create a `.env` file or export the following variables before running the server:

- `ELEVENLABS_API_KEY` – your ElevenLabs API key
- `ELEVENLABS_VOICE_ID` – desired voice ID
- `PINECONE_API_KEY` – Pinecone API key
- `PINECONE_ENV` – Pinecone environment (e.g. `gcp-starter`)
- `PINECONE_INDEX` – Pinecone index name
- `PINECONE_HOST` – (optional) custom Pinecone host URL
- `PINECONE_EMBEDDING_MODEL` – embedding model name

## Notes

- `server/services/llm.js` contains a placeholder implementation for generating answers. Integrate your chosen local LLM here.
- `server/services/tts.js` calls the ElevenLabs API and requires `ELEVENLABS_API_KEY` and `ELEVENLABS_VOICE_ID` environment variables.
- Pinecone configuration is handled in `server/utils/pinecone.js` via environment variables `PINECONE_API_KEY`, `PINECONE_ENV`, and `PINECONE_INDEX`.

This MVP is intended for private use on a local machine.
