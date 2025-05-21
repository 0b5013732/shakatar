# Shaka AI Chatbot MVP

This repository contains a minimal implementation of the **Shaka AI Chatbot**. It ingests text from a local directory, can fine‑tune a Llama‑based model, and exposes a REST API with a simple React UI.

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
│   └── train.py         # Llama fine‑tuning script
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
pip install torch transformers datasets
```

3. Install an inference server to run Llama locally for query. The
   [Ollama](https://ollama.ai) CLI provides a convenient API:

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.2:1b  # base model with 1B parameters
ollama serve &      # serves at http://localhost:11434
```

The `LLAMA_ENDPOINT` variable defaults to this address.

4. Set environment variables for ElevenLabs and Pinecone:

- `ELEVENLABS_API_KEY`
- `ELEVENLABS_VOICE_ID`
- `PINECONE_API_KEY`
- `PINECONE_HOST` *(controller URL)*
- `PINECONE_INDEX`
- `PINECONE_EMBEDDING_MODEL`

5. Place Shaka Senghor's writings (txt/markdown) inside `data/corpus/`.
6. Run the ingestion script to prepare the corpus:


```bash
node scripts/ingest.js
```


7. (Optional) Fine‑tune the Llama model using the provided training script:


```bash
python3 scripts/train.py --data data/processed/corpus.jsonl \
    --out model/ --model llama3.2:1b
```




8. Create a `.env` file (a sample `.env.example` is provided) with your Pinecone
   and ElevenLabs credentials. The server automatically loads this file at
   startup:

```bash
PINECONE_API_KEY=your_key
PINECONE_INDEX=shaka
PINECONE_HOST=https://shakata-xvax471.svc.apw5-4e34-81fa.pinecone.io
PINECONE_EMBEDDING_MODEL=text-embedding-3-large
LLAMA_ENDPOINT=http://localhost:11434/v1/chat/completions
LLAMA_MODEL=llama3.2:1b  # change to your fine-tuned model name when ready
ELEVENLABS_API_KEY=your_elevenlabs_key
ELEVENLABS_VOICE_ID=your_voice_id
```

9. Start the API server:

```bash
npm start
```

The React UI is served from `client/` and is accessible in the browser at `http://localhost:3000` by default.
The interface now features a sleeker design with styled message bubbles for a more modern look.


## Notes

- `server/services/llm.js` queries a local Llama model. Configure the `LLAMA_ENDPOINT` and `LLAMA_MODEL` environment variables to point at your running inference server.  `LLAMA_MODEL` defaults to `llama3.2:1b` but can be set to your fine-tuned model.
- `server/services/tts.js` calls the ElevenLabs API and requires `ELEVENLABS_API_KEY` and `ELEVENLABS_VOICE_ID` environment variables.





The repository's `.gitignore` excludes `.env` so your credentials remain local.
- Pinecone configuration is handled in `server/utils/pinecone.js` via environment variables `PINECONE_API_KEY`, `PINECONE_HOST`, and `PINECONE_INDEX`.

This MVP is intended for private use on a local machine.
