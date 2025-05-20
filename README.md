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

1. Install Node dependencies:

```bash
npm install
```

2. *(Optional)* Create a Python virtual environment and install training dependencies:


```bash
python3 -m venv venv
source venv/bin/activate
pip install torch transformers
```

3. Set environment variables for ElevenLabs and Pinecone:

- `ELEVENLABS_API_KEY`
- `ELEVENLABS_VOICE_ID`
- `PINECONE_API_KEY`
- `PINECONE_ENV`
- `PINECONE_INDEX`
- `PINECONE_HOST`
- `PINECONE_EMBEDDING_MODEL`

4. Place Shaka Senghor's writings (txt/markdown) inside `data/corpus/`.
5. Run the ingestion script to prepare the corpus:


```bash
node scripts/ingest.js
```

5. (Optional) Fine‑tune the Llama model using the placeholder script (writes a dummy file to the `model/` directory):



```bash
python3 scripts/train.py --data data/processed/corpus.jsonl --out model/
```

6. Set environment variables for Pinecone and ElevenLabs in a `.env` file or your shell:

```bash
PINECONE_API_KEY=your_key
PINECONE_ENV=your_env
PINECONE_INDEX=shaka
PINECONE_HOST=https://shakata-xvax471.svc.apw5-4e34-81fa.pinecone.io
PINECONE_EMBEDDING_MODEL=text-embedding-3-large
ELEVENLABS_API_KEY=your_elevenlabs_key
ELEVENLABS_VOICE_ID=your_voice_id
```

7. Start the API server:er:


```bash
npm start
```

The React UI is served from `client/` and is accessible in the browser at `http://localhost:3000` by default.

## Notes

- `server/services/llm.js` contains a placeholder implementation for generating answers. Integrate your chosen local LLM here.
- `server/services/tts.js` calls the ElevenLabs API and requires `ELEVENLABS_API_KEY` and `ELEVENLABS_VOICE_ID` environment variables.


The repository's `.gitignore` excludes `.env` so your credentials remain local.
- Pinecone configuration is handled in `server/utils/pinecone.js` via environment variables `PINECONE_API_KEY`, `PINECONE_ENV`, and `PINECONE_INDEX`.


This MVP is intended for private use on a local machine.
