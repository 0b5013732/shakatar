import os
from pathlib import Path
from transformers import AutoTokenizer

# Initialize tokenizer and ensure a pad token is defined
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Define configurable variables
max_tokens = 512
stride = 64

def clean_text(text):
    """Cleans the input text by removing metadata, standardizing line breaks,
    normalizing spaces, and removing leading/trailing whitespace.
    """
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        # Remove common metadata patterns
        if line.startswith("Title:") or line.startswith("Author:"):
            continue
        # Standardize line breaks (already handled by splitlines and join)
        # Normalize multiple spaces to a single space
        line = " ".join(line.split())
        # Remove leading/trailing whitespace from lines (already handled by " ".join(line.split()))
        cleaned_lines.append(line)

    # Join lines back and then normalize spaces again for the whole text
    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = cleaned_text.replace("<br>", "\n") # Standardize <br>
    cleaned_text = " ".join(cleaned_text.split()) # Normalize spaces across the text
    return cleaned_text.strip()

def chunk_text(text, tokenizer, max_tokens, stride):
    """Chunks the text using the tokenizer."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokens = tokenizer(
        text,
        return_overflowing_tokens=True,
        truncation=True,
        max_length=max_tokens,
        stride=stride,
        padding="longest",
        return_tensors="pt"
    )
    # Decode tokenized chunks back into text
    text_chunks = tokenizer.batch_decode(tokens["input_ids"], skip_special_tokens=True)
    return text_chunks

def main():
    """Main function to process text files and create formatted dataset."""
    input_folder = "data/corpus"
    output_file = "data/processed/formatted_dataset.txt"
    output_dir = Path(output_file).parent

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as outfile:
        for filepath in Path(input_folder).rglob("*.txt"):
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as infile:
                    raw_text = infile.read()

                cleaned_text = clean_text(raw_text)

                if not cleaned_text: # Skip if cleaned text is empty
                    print(f"Skipping empty file (after cleaning): {filepath}")
                    continue

                chunks = chunk_text(cleaned_text, tokenizer, max_tokens, stride)

                for chunk in chunks:
                    formatted_chunk = f"<s>{chunk.strip()}</s>\n"
                    outfile.write(formatted_chunk)
            except Exception as e:
                print(f"Error processing file {filepath}: {e}")

    print(f"Processing complete. Output saved to {output_file}")

if __name__ == "__main__":
    main()
