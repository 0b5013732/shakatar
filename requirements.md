# Project Requirements

This document summarizes the dependencies needed to run the Shaka AI Chatbot.

## Python
- Python 3.8 or higher
- Dependencies listed in `requirements.txt`:
  - transformers
  - bitsandbytes
  - torch
  - datasets
  - peft
  - trl

Install them with:
```bash
pip install -r requirements.txt
```

## Node.js
- Node.js 18+
- Dependencies in `package.json`:
  - axios
  - cors
  - dotenv
  - express
  - winston

Install them with:
```bash
npm install
```

## Optional Tools
- Git LFS for large model files
- Ollama CLI if serving local Llama models
