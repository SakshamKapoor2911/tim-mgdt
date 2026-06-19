import torch
from transformer_lens import HookedTransformer

class HookedModelWrapper:
    """Wrapper for TransformerLens models with research-specific hooks."""
    def __init__(self, model_name="mistralai/Mistral-7B-v0.1", load_in_8bit=True, load_in_4bit=False, device_map="auto"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        # Try loading with the specified quantization setting first
        try:
            kwargs = {
                "device": self.device if device_map is None else None,
                "fold_ln": False,
                "center_writing_weights": False,
                "center_unembed": False,
                "device_map": device_map
            }
            if load_in_4bit:
                kwargs["load_in_4bit"] = True
            elif load_in_8bit:
                kwargs["load_in_8bit"] = True
                
            self.model = HookedTransformer.from_pretrained(model_name, **kwargs)
        except (AssertionError, TypeError, Exception) as e:
            # Fallback: Some models don't support quantization or specific device maps
            print(f"Warning: Primary load failed with error: {e}")
            print(f"Attempting fallback load for {model_name}...")
            self.load_in_8bit = False
            self.load_in_4bit = False
            self.model = HookedTransformer.from_pretrained(
                model_name,
                device=self.device,
                fold_ln=False,
                center_writing_weights=False,
                center_unembed=False
            )
    
    def run_with_cache(self, input_text):
        """Runs the model and returns logits and cache."""
        return self.model.run_with_cache(input_text)
