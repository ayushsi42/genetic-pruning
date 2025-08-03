import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from typing import List, Dict, Tuple, Any
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

class ModelUtils:
    def __init__(self, config, dataset_handler):
        self.config = config
        self.dataset_handler = dataset_handler
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_model(self):
        model_kwargs = {
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "device_map": "auto" if torch.cuda.is_available() else None
        }
        
        if self.config.use_quantization and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.config.quantization_config["load_in_4bit"],
                bnb_4bit_use_double_quant=self.config.quantization_config["bnb_4bit_use_double_quant"],
                bnb_4bit_quant_type=self.config.quantization_config["bnb_4bit_quant_type"],
                bnb_4bit_compute_dtype=getattr(torch, self.config.quantization_config["bnb_4bit_compute_dtype"])
            )
            model_kwargs["quantization_config"] = quantization_config
            print(f"🔧 Loading model with 4-bit quantization for memory efficiency...")
        else:
            print(f"🔧 Loading model without quantization...")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if self.dataset_handler.tokenizer is None:
            self.dataset_handler.load_tokenizer(self.config.model_name)
    
    def apply_pruning_mask(self, pruning_mask: np.ndarray):
        """Zero-out pruned attention heads instead of deleting rows/cols."""
        layer_idx = 0
        for _, module in self.model.named_modules():
            if hasattr(module, "self_attn") and hasattr(module.self_attn, "num_heads"):
                num_heads = module.self_attn.num_heads
                head_dim = module.self_attn.head_dim

                if layer_idx < pruning_mask.shape[0]:
                    layer_mask = pruning_mask[layer_idx]
                    inactive_heads = np.where(layer_mask == 0)[0]
                    if len(inactive_heads) == 0:
                        layer_idx += 1
                        continue

                    self._mask_attention_heads(
                        module.self_attn, inactive_heads, num_heads, head_dim
                    )
                    layer_idx += 1

    def _mask_attention_heads(self, attention_module, inactive_heads, num_heads, head_dim):
        """Set weights of pruned heads to zero (shape preserved)."""
        if len(inactive_heads) == 0:
            return

        # Build index tensor covering the pruned head rows/cols
        inactive_indices = []
        for head_idx in inactive_heads:
            start = head_idx * head_dim
            end = (head_idx + 1) * head_dim
            inactive_indices.extend(range(start, end))
        inactive_indices = torch.tensor(inactive_indices, dtype=torch.long, device=attention_module.q_proj.weight.device)

        # Zero corresponding rows in q/k/v projections
        def _zero_rows(linear_layer):
            if linear_layer is None:
                return
            linear_layer.weight.data.index_fill_(0, inactive_indices, 0.0)
            if linear_layer.bias is not None:
                linear_layer.bias.data.index_fill_(0, inactive_indices, 0.0)

        _zero_rows(getattr(attention_module, "q_proj", None))
        _zero_rows(getattr(attention_module, "k_proj", None))
        _zero_rows(getattr(attention_module, "v_proj", None))

        # Zero corresponding columns in o_proj (output)
        if hasattr(attention_module, "o_proj") and attention_module.o_proj is not None:
            attention_module.o_proj.weight.data.index_fill_(1, inactive_indices, 0.0)
    
    def generate_responses(self, prompts: List[str]) -> List[str]:
        self.model.eval()
        responses = []
        
        with torch.no_grad():
            for prompt in tqdm(prompts, desc="Generating responses"):
                chat = [{"role": "user", "content": prompt}]
                
                input_ids = self.tokenizer.apply_chat_template(
                    chat,
                    return_tensors="pt",
                    add_generation_prompt=True
                ).to(self.device)
                
                attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
                
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                prompt_len = input_ids.shape[-1]
                response = self.tokenizer.decode(
                    outputs[0][prompt_len:], 
                    skip_special_tokens=True
                ).strip()
                
                responses.append(response)
        
        return responses
    
    def evaluate_model(self, eval_dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0
        
        all_prompts = []
        all_labels = []
        
        for batch in eval_dataloader:
            all_prompts.extend(batch["prompts"])
            all_labels.extend(batch["labels"])
        
        responses = self.generate_responses(all_prompts)
        
        print("\nDEBUG: Model responses and labels:")
        for i, (response, true_label) in enumerate(zip(responses, all_labels)):
            cleaned_response = self.dataset_handler.clean_model_response(response)
            print(f"  {i+1}. True: '{true_label}' | Response: '{response[:100]}...' | Cleaned: '{cleaned_response}'")
            
            total_predictions += 1
            if cleaned_response == true_label.lower():
                correct_predictions += 1
                    
        print(f"\nEvaluation Summary:")
        print(f"   Total samples: {len(responses)}")
        print(f"   Correct predictions: {correct_predictions}")
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "correct_predictions": correct_predictions,
            "total_predictions": total_predictions
        }
    
    def setup_qlora(self):
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.qlora_config["r"],
            lora_alpha=self.config.qlora_config["lora_alpha"],
            lora_dropout=self.config.qlora_config["lora_dropout"],
            target_modules=self.config.qlora_config["target_modules"]
        )
        
        self.model = get_peft_model(self.model, lora_config)
        return self.model
    
    def get_model_size(self) -> int:
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total_params
    
    def calculate_sparsity(self, pruning_mask: np.ndarray) -> float:
        total_heads = pruning_mask.size
        pruned_heads = np.sum(pruning_mask == 0)
        sparsity = pruned_heads / total_heads
        return sparsity
    
    def save_pruned_model(self, save_path: str, pruning_mask: np.ndarray):
        """Save the pruned model and its configuration"""
        import os
        import json
        
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        
        # Save model weights
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save pruning configuration
        pruning_config = {
            "pruning_mask": pruning_mask.tolist(),
            "original_model": self.config.model_name,
            "sparsity": self.calculate_sparsity(pruning_mask),
            "quantization_used": self.config.use_quantization
        }
        
        with open(os.path.join(save_path, "pruning_config.json"), 'w') as f:
            json.dump(pruning_config, f, indent=2)
        
        print(f"💾 Pruned model saved to: {save_path}")
    
    @classmethod
    def load_pruned_model(cls, model_path: str, config=None):
        """Load a previously saved pruned model"""
        import os
        import json
        
        # Load pruning configuration
        pruning_config_path = os.path.join(model_path, "pruning_config.json")
        if not os.path.exists(pruning_config_path):
            raise FileNotFoundError(f"Pruning config not found at {pruning_config_path}")
        
        with open(pruning_config_path, 'r') as f:
            pruning_config = json.load(f)
        
        # Create model utils instance
        if config is None:
            from .config import Config
            config = Config()
            config.model_name = model_path  # Use local path instead of HF model
        
        model_utils = cls(config, dataset_handler=None)
        
        # Load the pruned model directly
        model_utils.model = AutoModelForCausalLM.from_pretrained(model_path)
        model_utils.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if model_utils.tokenizer.pad_token is None:
            model_utils.tokenizer.pad_token = model_utils.tokenizer.eos_token
        
        print(f"📂 Loaded pruned model from: {model_path}")
        print(f"   Original model: {pruning_config['original_model']}")
        print(f"   Sparsity: {pruning_config['sparsity']:.4f}")
        
        return model_utils, pruning_config