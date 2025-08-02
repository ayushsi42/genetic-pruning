import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
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
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if self.dataset_handler.tokenizer is None:
            self.dataset_handler.load_tokenizer(self.config.model_name)
    
    def apply_pruning_mask(self, pruning_mask: np.ndarray):
        layer_idx = 0
        for name, module in self.model.named_modules():
            if hasattr(module, 'self_attn') and hasattr(module.self_attn, 'num_heads'):
                num_heads = module.self_attn.num_heads
                head_dim = module.self_attn.head_dim
                
                if layer_idx < pruning_mask.shape[0]:
                    mask = pruning_mask[layer_idx]
                    active_heads = np.where(mask == 1)[0]
                    
                    if len(active_heads) == 0:
                        active_heads = [0]
                    
                    self._prune_attention_heads(module.self_attn, active_heads, num_heads, head_dim)
                    layer_idx += 1
    
    def _prune_attention_heads(self, attention_module, active_heads, num_heads, head_dim):
        total_dim = num_heads * head_dim
        active_indices = []
        
        for head_idx in active_heads:
            start_idx = head_idx * head_dim
            end_idx = (head_idx + 1) * head_dim
            active_indices.extend(range(start_idx, end_idx))
        
        active_indices = torch.tensor(active_indices, dtype=torch.long)
        
        if hasattr(attention_module, 'q_proj'):
            attention_module.q_proj.weight.data = attention_module.q_proj.weight.data[active_indices, :]
            if attention_module.q_proj.bias is not None:
                attention_module.q_proj.bias.data = attention_module.q_proj.bias.data[active_indices]
        
        if hasattr(attention_module, 'k_proj'):
            attention_module.k_proj.weight.data = attention_module.k_proj.weight.data[active_indices, :]
            if attention_module.k_proj.bias is not None:
                attention_module.k_proj.bias.data = attention_module.k_proj.bias.data[active_indices]
        
        if hasattr(attention_module, 'v_proj'):
            attention_module.v_proj.weight.data = attention_module.v_proj.weight.data[active_indices, :]
            if attention_module.v_proj.bias is not None:
                attention_module.v_proj.bias.data = attention_module.v_proj.bias.data[active_indices]
        
        if hasattr(attention_module, 'o_proj'):
            attention_module.o_proj.weight.data = attention_module.o_proj.weight.data[:, active_indices]
    
    def generate_responses(self, prompts: List[str]) -> List[str]:
        self.model.eval()
        responses = []
        
        generation_config = GenerationConfig(
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        with torch.no_grad():
            for prompt in tqdm(prompts, desc="Generating responses"):
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length
                ).to(self.device)
                
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config
                )
                
                response = self.tokenizer.decode(
                    outputs[0][len(inputs["input_ids"][0]):],
                    skip_special_tokens=True
                )
                
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
        
        for response, true_label in zip(responses, all_labels):
            cleaned_response = self.dataset_handler.clean_model_response(response)
            
            if cleaned_response in ["safe", "unsafe"]:
                total_predictions += 1
                if cleaned_response == true_label.lower():
                    correct_predictions += 1
        
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