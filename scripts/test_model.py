import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import argparse
import os

def load_model_and_tokenizer(model_path):
    """Loads the fine-tuned model and tokenizer."""
    if not os.path.exists(model_path) or not os.path.isdir(model_path):
        print(f"Error: Model directory '{model_path}' not found.")
        print("Please ensure that fine-tuning has completed and the model exists at the specified path.")
        return None, None

    print(f"Loading model from: {model_path}")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Try loading with BitsAndBytesConfig if it was a QLoRA fine-tune
        # and the model saved is still in a quantized form or requires it
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, # Assuming it was trained with 4-bit
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config, # Use this if loading a QLoRA model that wasn't fully merged
            # torch_dtype=torch.bfloat16, # Uncomment if your GPU supports bfloat16
            device_map="auto" # Automatically distribute model across available GPUs/CPU
        )
        print("Model loaded with BitsAndBytesConfig for QLoRA.")
    except Exception as e:
        print(f"Could not load model with BitsAndBytesConfig: {e}")
        print("Attempting to load model without BitsAndBytesConfig (e.g., if adapters were merged)...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                # torch_dtype=torch.bfloat16, # Uncomment if your GPU supports bfloat16
                device_map="auto"
            )
            print("Model loaded successfully (likely merged or not requiring specific bnb_config for inference).")
        except Exception as e_no_bnb:
            print(f"Error loading model from {model_path}: {e_no_bnb}")
            print("Please check the model path and ensure the model format is compatible.")
            return None, None


    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("Tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading tokenizer from {model_path}: {e}")
        return None, None # Changed from model, None to None, None

    # Set pad_token_id if not set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if model.config.pad_token_id is None: # Check model's pad_token_id specifically
             model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7):
    """Generates text using the loaded model and tokenizer."""
    if model is None or tokenizer is None:
        return "Model or tokenizer not loaded. Cannot generate text."

    device = model.device # Model already on device due to device_map="auto"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)

    # Generate text
    with torch.no_grad():
        output_sequences = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id, # Ensure pad_token_id is set
            eos_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="Test a fine-tuned LLaMA model.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./output", # Default path where Axolotl saves models
        help="Path to the fine-tuned model directory (output by Axolotl)."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="<s>Once upon a time",
        help="Initial prompt to feed the model. Remember to use <s> if that's how it was trained."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=150,
        help="Maximum number of new tokens to generate."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation (controls randomness)."
    )

    args = parser.parse_args()

    actual_model_path = args.model_path

    if not os.path.exists(actual_model_path):
        print(f"Error: Base model path {actual_model_path} does not exist.")
        return

    # Check if common model files exist in the given path or try to find checkpoints
    if os.path.isdir(actual_model_path):
        # List files in directory, handle potential OSError if path doesn't exist or isn't a dir
        try:
            dir_contents = os.listdir(actual_model_path)
        except OSError as e:
            print(f"Error listing contents of directory {actual_model_path}: {e}")
            return

        if not any(f.endswith(('.bin', '.safetensors')) for f in dir_contents if os.path.isfile(os.path.join(actual_model_path, f))):
            print(f"No model files (.bin or .safetensors) found directly in {actual_model_path}.")
            checkpoints = [d for d in dir_contents if d.startswith("checkpoint-") and os.path.isdir(os.path.join(actual_model_path, d))]
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split('-')[-1]), reverse=True)
                latest_checkpoint_path = os.path.join(actual_model_path, checkpoints[0])
                print(f"Found checkpoints. Attempting to load from the latest: {latest_checkpoint_path}")
                actual_model_path = latest_checkpoint_path
            else:
                print(f"No checkpoints found in {args.model_path} either.")
                print("Please specify the correct path to a directory containing model files (e.g., pytorch_model.bin) or a checkpoint.")
                return
    else:
        print(f"Error: model_path '{actual_model_path}' is not a directory.")
        return


    model, tokenizer = load_model_and_tokenizer(actual_model_path)

    if model and tokenizer:
        print("\n--- Ready to Generate ---")
        print(f"Model: {actual_model_path}")
        print(f"Using initial prompt: '{args.prompt}'")

        base_prompt_from_args = args.prompt # Store the initial prompt from args
        current_session_prompt = base_prompt_from_args

        try:
            while True:
                user_input = input(f"\nEnter your prompt (or type 'quit' to exit, 'new' for a new base prompt, Enter to use current: '{current_session_prompt}'): ")
                if user_input.lower() == 'quit':
                    break
                if user_input.lower() == 'new':
                    current_session_prompt = input("Enter new base prompt: ")
                    if not current_session_prompt.strip().startswith("<s>") and "<s>" in base_prompt_from_args:
                         current_session_prompt = "<s>" + current_session_prompt
                    continue

                prompt_to_generate = user_input if user_input else current_session_prompt

                # Ensure <s> prefix if it was in the original argument and not in current input
                if not prompt_to_generate.strip().startswith("<s>") and base_prompt_from_args.strip().startswith("<s>"):
                    print("Adding <s> prefix to prompt as it was in the initial training-style prompt.")
                    prompt_to_generate = "<s>" + prompt_to_generate.lstrip("<s>")


                print(f"Generating with prompt: \"{prompt_to_generate}\"...")
                generated_text = generate_text(model, tokenizer, prompt_to_generate, args.max_new_tokens, args.temperature)
                print("\n--- Generated Text ---")
                print(generated_text)
                print("----------------------\n")
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            print("Test script finished.")

if __name__ == "__main__":
    main()
