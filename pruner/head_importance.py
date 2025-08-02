import torch
import numpy as np
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

class HeadImportanceMeasurer:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_hooks = []
        self.attention_outputs = {}
    
    def register_attention_hooks(self):
        self.attention_outputs = {}
        self.attention_hooks = []
        
        layer_idx = 0
        for name, module in self.model.named_modules():
            if hasattr(module, 'self_attn'):
                hook = module.self_attn.register_forward_hook(
                    self._create_attention_hook(layer_idx)
                )
                self.attention_hooks.append(hook)
                layer_idx += 1
    
    def _create_attention_hook(self, layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                attention_weights = output[1] if len(output) > 1 else None
            else:
                attention_weights = None
            
            if attention_weights is not None:
                self.attention_outputs[layer_idx] = attention_weights.detach().cpu()
        
        return hook
    
    def remove_hooks(self):
        for hook in self.attention_hooks:
            hook.remove()
        self.attention_hooks = []
    
    def measure_head_importance(self, train_dataloader: DataLoader, dataset_handler) -> np.ndarray:
        self.model.eval()
        self.register_attention_hooks()
        
        num_layers = self._get_num_layers()
        num_heads = self._get_num_heads()
        
        safe_activations = np.zeros((num_layers, num_heads))
        unsafe_activations = np.zeros((num_layers, num_heads))
        safe_counts = np.zeros((num_layers, num_heads))
        unsafe_counts = np.zeros((num_layers, num_heads))
        
        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Measuring head importance")):
                    if batch_idx >= 100:
                        break
                    
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    prompts = batch["prompts"]
                    
                    self.attention_outputs = {}
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_attentions=True
                    )
                    
                    responses = []
                    for i, prompt in enumerate(prompts):
                        try:
                            prompt_input = self.tokenizer(
                                prompt,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=self.config.max_length
                            ).to(self.device)
                            
                            with torch.no_grad():
                                gen_output = self.model.generate(
                                    **prompt_input,
                                    max_new_tokens=20,
                                    do_sample=False,
                                    pad_token_id=self.tokenizer.pad_token_id
                                )
                            
                            response = self.tokenizer.decode(
                                gen_output[0][len(prompt_input["input_ids"][0]):],
                                skip_special_tokens=True
                            )
                            
                            cleaned_response = dataset_handler.clean_model_response(response)
                            responses.append(cleaned_response)
                        
                        except Exception as e:
                            responses.append("unknown")
                    
                    for layer_idx in range(num_layers):
                        if layer_idx in self.attention_outputs:
                            attention_weights = self.attention_outputs[layer_idx]
                            
                            for sample_idx, response in enumerate(responses):
                                if sample_idx < attention_weights.shape[0]:
                                    sample_attention = attention_weights[sample_idx]
                                    
                                    head_importance = self._compute_head_activation_strength(sample_attention)
                                    
                                    if response == "safe":
                                        safe_activations[layer_idx] += head_importance
                                        safe_counts[layer_idx] += 1
                                    elif response == "unsafe":
                                        unsafe_activations[layer_idx] += head_importance
                                        unsafe_counts[layer_idx] += 1
        
        finally:
            self.remove_hooks()
        
        safe_avg = np.divide(safe_activations, safe_counts, 
                           out=np.zeros_like(safe_activations), where=safe_counts!=0)
        unsafe_avg = np.divide(unsafe_activations, unsafe_counts, 
                             out=np.zeros_like(unsafe_activations), where=unsafe_counts!=0)
        
        importance_matrix = np.abs(safe_avg - unsafe_avg)
        
        importance_matrix = self._normalize_importance_matrix(importance_matrix)
        
        return importance_matrix
    
    def _compute_head_activation_strength(self, attention_weights: torch.Tensor) -> np.ndarray:
        if len(attention_weights.shape) == 3:
            head_strength = torch.mean(attention_weights, dim=(1, 2)).cpu().numpy()
        elif len(attention_weights.shape) == 4:
            head_strength = torch.mean(attention_weights, dim=(2, 3)).squeeze().cpu().numpy()
        else:
            head_strength = torch.mean(attention_weights).cpu().numpy()
            if np.isscalar(head_strength):
                head_strength = np.array([head_strength])
        
        if head_strength.ndim == 0:
            head_strength = np.array([head_strength])
        
        return head_strength
    
    def _normalize_importance_matrix(self, importance_matrix: np.ndarray) -> np.ndarray:
        min_val = np.min(importance_matrix)
        max_val = np.max(importance_matrix)
        
        if max_val - min_val > 0:
            normalized = (importance_matrix - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(importance_matrix)
        
        return normalized
    
    def _get_num_layers(self) -> int:
        count = 0
        for name, module in self.model.named_modules():
            if hasattr(module, 'self_attn'):
                count += 1
        return count
    
    def _get_num_heads(self) -> int:
        for name, module in self.model.named_modules():
            if hasattr(module, 'self_attn') and hasattr(module.self_attn, 'num_heads'):
                return module.self_attn.num_heads
        return 8
    
    def get_importance_stats(self, importance_matrix: np.ndarray) -> Dict[str, float]:
        return {
            "mean_importance": np.mean(importance_matrix),
            "std_importance": np.std(importance_matrix),
            "max_importance": np.max(importance_matrix),
            "min_importance": np.min(importance_matrix),
            "num_layers": importance_matrix.shape[0],
            "num_heads": importance_matrix.shape[1]
        }