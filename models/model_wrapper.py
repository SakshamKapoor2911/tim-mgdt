import torch
from transformer_lens import HookedTransformer

class HookedModelWrapper:
    """Wrapper for TransformerLens models with research-specific hooks."""
    def __init__(self, model_name="mistralai/Mistral-7B-v0.1", load_in_8bit=True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_in_8bit = load_in_8bit
        
        # Try loading with the specified quantization setting first
        try:
            kwargs = {
                "device": self.device,
                "fold_ln": False,
                "center_writing_weights": False,
                "center_unembed": False
            }
            if load_in_8bit:
                kwargs["load_in_8bit"] = True
                
            self.model = HookedTransformer.from_pretrained(model_name, **kwargs)
        except (AssertionError, TypeError, Exception):
            # Fallback: Some models don't support 8-bit quantization
            if load_in_8bit:
                print(f"Warning: Could not load {model_name} with load_in_8bit=True, trying without 8-bit...")
                self.load_in_8bit = False
                self.model = HookedTransformer.from_pretrained(
                    model_name,
                    device=self.device,
                    fold_ln=False,
                    center_writing_weights=False,
                    center_unembed=False
                )
            else:
                raise
    
    def run_with_cache(self, input_text):
        """Runs the model and returns logits and cache."""
        return self.model.run_with_cache(input_text)
