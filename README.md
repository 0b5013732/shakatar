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
pip install torch transformers datasets peft bitsandbytes accelerate
```
   - `peft`: For Parameter-Efficient Fine-Tuning.
   - `bitsandbytes`: Required if you plan to use 4-bit or 8-bit model quantization (via the `--bits` flag in `train.py`).
   - `accelerate`: Helps manage and run your models efficiently, especially for distributed setups or when using features like `device_map="auto"`.

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


7. (Optional) Fine‑tune the Llama model using the provided training script. The
   `--model` flag expects either a HuggingFace model repo or the path to a
   local directory containing the model weights. If you downloaded
   `llama3.2:1b` via the Ollama CLI, export it to a folder first and pass that
   path instead:

```bash
python3 scripts/train.py --data data/processed/corpus.jsonl \
    --out model/ --model ./models/llama3.2-1b
```

# For multi-GPU machines, launch via torchrun.
# If you encounter CUDA OutOfMemory errors, try the following command which enables
# 4-bit quantization, gradient checkpointing, and a batch size of 1
# to significantly reduce memory usage:
CUDA_VISIBLE_DEVICES=0,1   torchrun --standalone --nproc_per_node=1 scripts/train.py   --data data/processed/corpus.jsonl --out model/   --model meta-llama/Llama-3.2-1B   --batch-size 1 --gradient-checkpointing --bits 4 --gradient-accumulation-steps 1
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

# LLaMA Fine-Tuning on Author Text Files

This project provides scripts and configuration to fine-tune a LLaMA model (e.g., `meta-llama/Llama-2-7b-hf`) on a custom corpus of an author's text files. Training is performed with [Hugging Face TRL](https://github.com/huggingface/trl) and the [PEFT](https://github.com/huggingface/peft) LoRA adapter.

## Directory Structure

- `data/`:
    - `corpus/`: Place your raw author text files (`.txt`) here. You can create subdirectories.
        - `.gitkeep`: Placeholder, safe to remove if you add your own data.
    - `processed/`: Output directory for the processed dataset.
        - `corpus.jsonl`: JSONL file with one object per text chunk.
        - `.gitkeep`: Placeholder.
- `scripts/`:
    - `chunk_text.py`: Python script to process raw text files into a formatted dataset.
    - `run_finetuning.sh`: Shell script that launches TRL + LoRA fine-tuning.
    - `test_model.py`: Python script to load and test the fine-tuned model.
- `output/`: Default directory where the fine-tuned model checkpoints and artifacts are saved.

## Requirements

- Python 3.8+
- PyTorch (with CUDA support for GPU training)
- Hugging Face Transformers, Datasets, Accelerate, PEFT, BitsAndBytes
- Git LFS (for handling large model files, if you version them)

You can often install Python dependencies with:
```bash
pip install torch transformers datasets accelerate peft bitsandbytes trl sentencepiece
```

## Fine-Tuning Process

### 1. Prepare Your Data

1.  **Gather Text Files**: Collect plain text (`.txt`) files of the author's writing.
2.  **Place Files**: Put these `.txt` files into the `data/corpus/` directory. You can organize them in subfolders if desired.
    Example:
    ```
    data/corpus/book1.txt
    data/corpus/short_stories/story1.txt
    data/corpus/short_stories/story2.txt
    ```

### 2. Chunk and Format Text

Run the `chunk_text.py` script to process your raw text files into the required format for training. This script will clean the text, chunk it into manageable pieces, and format each chunk with `<s>...</s>`.

```bash
python scripts/chunk_text.py
```

This will generate `data/processed/corpus.jsonl`.

### 3. Configure Fine-Tuning (Optional)

The training script `train.py` exposes command-line flags for common settings:

- `--model`: Base model ID or path.
- `--epochs`: Number of training epochs.
- `--batch-size`: Training batch size.
- `--bits`: Precision (4, 8 or 16).
- `--gradient-checkpointing`: Enable to reduce memory usage.

**Note on GPU requirements**: QLoRA significantly reduces memory usage, but fine-tuning LLMs is still demanding. The default `meta-llama/Llama-2-7b-hf` model with 4-bit quantization should be trainable on GPUs with >=16GB VRAM, but this can vary. If you encounter out-of-memory errors, try reducing `per_device_train_batch_size` or using a smaller model.

### 4. Run Fine-Tuning

Execute the `run_finetuning.sh` script to start the training process with TRL and LoRA. Make sure you are in the root directory of the project.

```
bash scripts/run_finetuning.sh
```

The script checks that the required Python packages are installed and then launches `train.py` with sensible defaults.

This will:
- Read the dataset from `data/processed/corpus.jsonl`.

The process can take a significant amount of time depending on your dataset size, hardware, and training epochs.

### 5. Test Your Fine-Tuned Model

Once fine-tuning is complete, you can test your model using `scripts/test_model.py`.


```bash
# Test with default settings (loads from ./output, uses a default prompt)
python scripts/test_model.py

# Specify a model path and initial prompt
python scripts/test_model.py --model_path ./output/your_specific_checkpoint_or_merged_model_dir --prompt "<s>Write a sentence in the style of the author</s>"

# For more options:
python scripts/test_model.py --help
```

The script will load the model and enter an interactive mode where you can type prompts and see the generated text.

## Hosted Training with Modal

A helper script, `scripts/modal_train.py`, submits the fine-tuning job to [Modal](https://modal.com/). After installing the `modal` package and running `python3 -m modal setup`, launch training remotely:

```bash
modal run scripts/modal_train.py --data data/processed/corpus.jsonl \
  --out output --model Llama-3.2-1B --epochs 3 --batch-size 4 --bits 4 \
  --gradient-checkpointing
```

The arguments mirror those of `scripts/train.py`.

## Notes
- **Dataset Size**: For effective style learning, a dataset of 5,000–50,000 chunks (roughly 100k–1M words after cleaning and chunking) is recommended.
- **Cleaning**: The `chunk_text.py` script includes basic cleaning. You may need to enhance the `clean_text` function within it for specific metadata or formatting issues in your source texts.
