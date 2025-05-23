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

4. Set environment variables for ElevenLabs:

- `ELEVENLABS_API_KEY`
- `ELEVENLABS_VOICE_ID`
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

For multi-GPU machines, launch via `torchrun` so each process is assigned a
different GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/train.py --data data/processed/corpus.jsonl \
  --out model/ --model llama3.2:1b
```

The training script will automatically detect whether CUDA is available and, if
so, enable mixed precision to accelerate training on your GPU.




8. Create a `.env` file (a sample `.env.example` is provided) with your Llama and ElevenLabs credentials. The server loads this file from the repository root at startup, even if you launch the app from within the `server/` folder:

```bash
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

This MVP is intended for private use on a local machine.
