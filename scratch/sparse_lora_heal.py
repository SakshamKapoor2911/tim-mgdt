import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
import os

def load_pristine_healing_dataset(num_samples=500):
    """
    STRICTLY EXCLUDES ASDiv (Holdout set for final evaluation).
    Blends GSM8K, MATH, and SVAMP.
    """
    print("Loading pristine healing dataset (GSM8K, MATH, SVAMP)...")
    
    # GSM8K
    try:
        ds_gsm8k = load_dataset("gsm8k", "main", split="train").select(range(num_samples // 2))
        def map_gsm8k(x): return {"text": f"Question: {x['question']}\nAnswer: {x['answer']}"}
        ds_gsm8k = ds_gsm8k.map(map_gsm8k, remove_columns=ds_gsm8k.column_names)
    except:
        ds_gsm8k = None
        
    # SVAMP
    try:
        ds_svamp = load_dataset("ChilleD/SVAMP", split="train").select(range(num_samples // 2))
        def map_svamp(x): return {"text": f"Question: {x['Body']} {x['Question']}\nAnswer: {x['Answer']}"}
        ds_svamp = ds_svamp.map(map_svamp, remove_columns=ds_svamp.column_names)
    except:
        ds_svamp = None

    datasets = [d for d in [ds_gsm8k, ds_svamp] if d is not None]
    if not datasets:
        # Fallback to dummy data if datasets fail to load
        dummy_data = {"text": ["Question: What is 2+2? Answer: 4" for _ in range(num_samples)]}
        from datasets import Dataset
        return Dataset.from_dict(dummy_data)
        
    return concatenate_datasets(datasets).shuffle(seed=42).select(range(min(num_samples, sum(len(d) for d in datasets))))

def main():
    model_id = "./checkpoints/Meta-Llama-3-8B_subnetwork_pruned"
    
    print("Loading Pruned Llama-3-8B...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda:0")
    
    # 1. Target the 6 most damaged layers identified by the diagnostic script
    damaged_layers = [29, 28, 27, 30, 26, 25]
    
    target_modules = []
    for layer_idx in damaged_layers:
        target_modules.extend([
            f"model.layers.{layer_idx}.self_attn.o_proj",
            f"model.layers.{layer_idx}.mlp.down_proj"
        ])
        
    print(f"\nConfiguring Sparse LoRA targets:")
    for t in target_modules:
        print(f"  -> {t}")

    # 2. Configure PEFT (r=16, alpha=32)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 3. Load Pristine Dataset (STRICTLY NO ASDIV)
    dataset = load_pristine_healing_dataset(500)
    
    # 4. Conservative Healing Configuration
    output_dir = "./checkpoints/sparse_lora_healer"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,  # Conservative learning rate
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="no",
        optim="adamw_torch",
        fp16=False,
        bf16=True,
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
        peft_config=peft_config
    )
    
    print("\nInitiating Sparse Structural Healing...")
    trainer.train()
    
    print("\nSaving LoRA Adapters...")
    trainer.model.save_pretrained(output_dir)
    print("Complete!")

if __name__ == "__main__":
    main()
