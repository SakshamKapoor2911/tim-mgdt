import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_and_save(dropped_base_path, lora_path, output_path):
    print(f"Loading structurally dropped base model from: {dropped_base_path}")
    # Load onto CPU to prevent VRAM fragmentation/OOM before the save step
    base_model = AutoModelForCausalLM.from_pretrained(
        dropped_base_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu", 
    )
    tokenizer = AutoTokenizer.from_pretrained(dropped_base_path)

    print(f"Applying LoRA adapters from: {lora_path}")
    peft_model = PeftModel.from_pretrained(base_model, lora_path)

    print("Executing merge_and_unload()...")
    merged_model = peft_model.merge_and_unload()

    print(f"Saving static compiled checkpoint to: {output_path}")
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("Merge complete. Ready for vLLM ingestion.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropped_base", required=True, help="Path to the layer-dropped model")
    parser.add_argument("--lora", required=True, help="Path to the trained boundary LoRA weights")
    parser.add_argument("--output", required=True, help="Output directory for the merged model")
    args = parser.parse_args()
    
    merge_and_save(args.dropped_base, args.lora, args.output)
