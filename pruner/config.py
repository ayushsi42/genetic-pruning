from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Config:
    model_name: str = "meta-llama/Llama-Guard-3-8B"
    training_dataset: str = "ayushsi42/pruning-dataset"
    eval_dataset: str = "walledai/XSTest"
    
    max_length: int = 512
    batch_size: int = 8
    eval_batch_size: int = 16
    
    eval_max_samples: int = 50
    eval_seed: int = 42
    
    use_quantization: bool = True
    quantization_config: Dict[str, Any] = None
    genetic_algorithm_config: Dict[str, Any] = None
    qlora_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.quantization_config is None:
            self.quantization_config = {
                "load_in_4bit": True,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "float16"
            }
        
        if self.genetic_algorithm_config is None:
            self.genetic_algorithm_config = {
                "population_size": 50,
                "generations": 20,
                "mutation_rate": 0.1,
                "crossover_rate": 0.8,
                "elite_size": 5,
                "fitness_weights": {
                    "accuracy": 0.6,
                    "sparsity": 0.2,
                    "importance_penalty": 0.2
                }
            }
        
        if self.qlora_config is None:
            self.qlora_config = {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "learning_rate": 2e-4,
                "num_epochs": 3,
                "warmup_steps": 100
            }