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
            # Try different ways to extract attention weights
            attention_weights = None
            
            if isinstance(output, tuple):
                # Try second element (common for transformers)
                if len(output) > 1 and output[1] is not None:
                    attention_weights = output[1]
                # Try first element if second is None
                elif len(output) > 0 and output[0] is not None:
                    potential_attn = output[0]
                    # Check if it has the right shape for attention (batch, heads, seq, seq)
                    if len(potential_attn.shape) == 4:
                        attention_weights = potential_attn
            else:
                # Single tensor output
                if hasattr(output, 'shape') and len(output.shape) == 4:
                    attention_weights = output
            
            # Store attention weights if found
            if attention_weights is not None:
                try:
                    # Ensure it's detached and moved to CPU
                    self.attention_outputs[layer_idx] = attention_weights.detach().cpu()
                    print(f"Captured attention for layer {layer_idx}, shape: {attention_weights.shape}")
                except Exception as e:
                    print(f"Error capturing attention for layer {layer_idx}: {e}")
        
        return hook
    
    def remove_hooks(self):
        for hook in self.attention_hooks:
            hook.remove()
        self.attention_hooks = []
    
    def measure_head_importance(self, train_dataloader: DataLoader, dataset_handler) -> np.ndarray:
        self.model.eval()
        
        num_layers = self._get_num_layers()
        num_heads = self._get_num_heads()
        print(f"   Model structure: {num_layers} layers, {num_heads} heads per layer")
        
        safe_activations = {}
        unsafe_activations = {}
        safe_counts = {}
        unsafe_counts = {}
        
        # Track response classification counts
        total_prompts = 0
        safe_responses = 0
        unsafe_responses = 0
        skipped_prompts = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Measuring head importance")):
                prompts = batch["prompts"]
                
                # Process each prompt individually
                for prompt in prompts:
                    total_prompts += 1
                    try:
                        # Prepare input for generation
                        chat = [{"role": "user", "content": prompt}]
                        input_ids = self.tokenizer.apply_chat_template(
                            chat,
                            return_tensors="pt",
                            add_generation_prompt=True
                        ).to(self.device)
                        
                        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
                        
                        # Force eager attention to get attention weights
                        # Temporarily set attention implementation
                        original_config = getattr(self.model.config, '_attn_implementation', None)
                        self.model.config._attn_implementation = 'eager'
                        
                        # Forward pass to get attention weights
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_attentions=True
                        )
                        
                        # Generate response separately
                        gen_output = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=50,  # Increased to allow full Llama Guard response
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id
                        )
                        
                        # Restore original config
                        if original_config is not None:
                            self.model.config._attn_implementation = original_config
                        
                        # Decode response
                        prompt_len = input_ids.shape[-1]
                        response = self.tokenizer.decode(
                            gen_output[0][prompt_len:], 
                            skip_special_tokens=True
                        ).strip()
                        
                        cleaned_response = dataset_handler.clean_model_response(response)
                        
                        # Debug: Log first few responses to see what model is actually outputting
                        if total_prompts <= 5:
                            print(f"     Prompt {total_prompts}: Raw response: '{response}' -> Cleaned: '{cleaned_response}'")
                        
                        # Count response classifications
                        if cleaned_response == "safe":
                            safe_responses += 1
                        elif cleaned_response == "unsafe":
                            unsafe_responses += 1
                        
                        # Process attention weights from model output
                        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                            attentions = outputs.attentions
                            
                            for layer_idx, layer_attention in enumerate(attentions):
                                if layer_attention is not None:
                                    # layer_attention shape: (batch_size, num_heads, seq_len, seq_len)
                                    # Take the first sample from batch
                                    sample_attention = layer_attention[0]  # (num_heads, seq_len, seq_len)
                                    
                                    # Compute head importance (mean attention across sequence)
                                    head_importance = torch.mean(sample_attention, dim=(1, 2)).cpu().numpy()
                                    
                                    if layer_idx not in safe_activations:
                                        safe_activations[layer_idx] = np.zeros_like(head_importance)
                                        unsafe_activations[layer_idx] = np.zeros_like(head_importance)
                                        safe_counts[layer_idx] = np.zeros_like(head_importance)
                                        unsafe_counts[layer_idx] = np.zeros_like(head_importance)
                                    
                                    if cleaned_response == "safe":
                                        safe_activations[layer_idx] += head_importance
                                        safe_counts[layer_idx] += 1
                                    elif cleaned_response == "unsafe":
                                        unsafe_activations[layer_idx] += head_importance
                                        unsafe_counts[layer_idx] += 1
                        
                    except Exception as e:
                        # Skip problematic prompts silently
                        skipped_prompts += 1
                        continue
        
        # Log classification summary
        print(f"   Response classification summary:")
        print(f"     Total prompts processed: {total_prompts}")
        print(f"     Safe responses: {safe_responses}")
        print(f"     Unsafe responses: {unsafe_responses}")
        print(f"     Skipped prompts: {skipped_prompts}")
        
        if unsafe_responses == 0:
            print(f"   ⚠️  WARNING: No 'unsafe' responses detected! All responses classified as 'safe'.")
            print(f"      This will result in zero importance values since differential analysis requires both classes.")
        
        max_heads = max(len(safe_activations[layer_idx]) for layer_idx in safe_activations.keys()) if safe_activations else 16
        
        importance_matrix = np.zeros((num_layers, max_heads))
        
        # Track layers with non-zero importance
        layers_with_importance = 0
        max_layer_importance = 0.0
        
        for layer_idx in range(num_layers):
            if layer_idx in safe_activations:
                safe_layer = safe_activations[layer_idx]
                unsafe_layer = unsafe_activations[layer_idx]
                safe_count_layer = safe_counts[layer_idx]
                unsafe_count_layer = unsafe_counts[layer_idx]
                
                safe_avg = np.divide(safe_layer, safe_count_layer, 
                                   out=np.zeros_like(safe_layer), where=safe_count_layer!=0)
                unsafe_avg = np.divide(unsafe_layer, unsafe_count_layer, 
                                     out=np.zeros_like(unsafe_layer), where=unsafe_count_layer!=0)
                
                layer_importance = np.abs(safe_avg - unsafe_avg)
                layer_max_importance = np.max(layer_importance)
                
                if layer_max_importance > 0:
                    layers_with_importance += 1
                    max_layer_importance = max(max_layer_importance, layer_max_importance)
                
                num_heads_layer = len(layer_importance)
                importance_matrix[layer_idx, :num_heads_layer] = layer_importance
        
        print(f"   Importance calculation summary:")
        print(f"     Layers with non-zero importance: {layers_with_importance}/{num_layers}")
        print(f"     Maximum layer importance: {max_layer_importance:.6f}")
        
        importance_matrix = self._normalize_importance_matrix(importance_matrix)
        
        return importance_matrix
    
    def _compute_head_activation_strength(self, attention_weights: torch.Tensor) -> np.ndarray:
        try:
            if len(attention_weights.shape) == 3:
                head_strength = torch.mean(attention_weights, dim=(1, 2)).cpu().numpy()
            elif len(attention_weights.shape) == 4:
                head_strength = torch.mean(attention_weights, dim=(2, 3)).squeeze().cpu().numpy()
            elif len(attention_weights.shape) == 2:
                head_strength = torch.mean(attention_weights, dim=1).cpu().numpy()
            else:
                head_strength = torch.mean(attention_weights).cpu().numpy()
                if np.isscalar(head_strength):
                    head_strength = np.array([head_strength])
            
            if head_strength.ndim == 0:
                head_strength = np.array([head_strength])
            
            return head_strength
            
        except Exception as e:
            num_heads = attention_weights.shape[0] if len(attention_weights.shape) > 0 else 1
            return np.ones(num_heads) * 0.5
    
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
            if hasattr(module, 'self_attn'):
                attn = module.self_attn
                # Try different ways to get num_heads
                if hasattr(attn, 'num_heads'):
                    return attn.num_heads
                elif hasattr(attn, 'num_attention_heads'):
                    return attn.num_attention_heads
                elif hasattr(attn, 'config') and hasattr(attn.config, 'num_attention_heads'):
                    return attn.config.num_attention_heads
        
        # Try to get from model config
        if hasattr(self.model, 'config'):
            config = self.model.config
            if hasattr(config, 'num_attention_heads'):
                return config.num_attention_heads
            elif hasattr(config, 'num_heads'):
                return config.num_heads
        return 16
    
    def get_importance_stats(self, importance_matrix: np.ndarray) -> Dict[str, float]:
        return {
            "mean_importance": np.mean(importance_matrix),
            "std_importance": np.std(importance_matrix),
            "max_importance": np.max(importance_matrix),
            "min_importance": np.min(importance_matrix),
            "num_layers": importance_matrix.shape[0],
            "num_heads": importance_matrix.shape[1]
        }