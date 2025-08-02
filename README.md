# Guided Pruning + QLoRA for Task-Specific LLM Compression

This project implements a comprehensive pipeline for intelligently pruning transformer models using genetic algorithms and fine-tuning them with QLoRA, specifically optimized for safety classification tasks.

## Overview

The pipeline consists of four main phases:

1. **Dataset Setup**: Load and preprocess training and evaluation datasets
2. **Head Importance Measurement**: Analyze attention head activations to determine their importance for the target task
3. **Genetic Algorithm Optimization**: Evolve optimal pruning configurations that balance accuracy, sparsity, and importance preservation
4. **QLoRA Fine-tuning**: Apply efficient fine-tuning to the pruned model

## Features

- 🎯 **Task-specific pruning** based on attention head importance analysis
- 🧬 **Genetic algorithm optimization** for finding optimal pruning configurations
- 🔧 **QLoRA integration** for efficient fine-tuning of pruned models
- 💾 **4-bit quantization** for memory-efficient model loading (reduces memory usage by ~75%)
- 📊 **Comprehensive evaluation** with accuracy metrics and model compression analysis
- 🧹 **Robust response cleaning** for safety classification outputs
- 📈 **Detailed logging and results tracking**

## Installation

```bash
pip install -r requirements.txt
```

**Note**: Make sure you have sufficient disk space and internet connectivity for downloading the model and datasets. The first run will download:
- Model weights (~500MB - 2GB depending on the model)
- Training dataset (`ayushsi42/pruning-dataset`)
- Evaluation dataset (`walledai/XSTest`)

## Usage

### Basic Usage

Simply run the main pipeline:

```bash
python main_pruner.py
```

### Configuration

The pipeline can be configured by modifying the `Config` class in `pruner/config.py`:

```python
@dataclass
class Config:
    model_name: str = "microsoft/DialoGPT-medium"
    training_dataset: str = "ayushsi42/pruning-dataset"
    eval_dataset: str = "walledai/XSTest"
    
    # Genetic Algorithm parameters
    genetic_algorithm_config: Dict[str, Any] = {
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
    
    # Quantization settings (for memory efficiency)
    use_quantization: bool = True
    quantization_config: Dict[str, Any] = {
        "load_in_4bit": True,
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "float16"
    }
    
    # QLoRA parameters
    qlora_config: Dict[str, Any] = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
    }
```

## Architecture

### Core Components

- **`main_pruner.py`**: Orchestrates the entire pipeline
- **`pruner/`**: Main pruning module containing:
  - **`config.py`**: Configuration management
  - **`dataset_handler.py`**: Dataset loading and preprocessing
  - **`model_utils.py`**: Model operations and evaluation
  - **`head_importance.py`**: Attention head importance measurement
  - **`genetic_pruning.py`**: Genetic algorithm optimization

### Key Features

#### Response Cleaning
The system automatically cleans model outputs from formats like:
- `"safe\ns1"` → `"safe"`
- `"unsafe\ns5"` → `"unsafe"`
- Handles various edge cases and ambiguous responses

#### Fitness Function
The genetic algorithm optimizes for:
- **Accuracy**: Model performance on the target task
- **Sparsity**: Degree of model compression
- **Importance Preservation**: Avoids pruning critical attention heads

#### Head Importance Measurement
- Records attention activations during inference
- Compares activation patterns between safe/unsafe responses
- Normalizes importance scores for fair comparison

## Output

The pipeline generates:

1. **Console output** with detailed progress and metrics
2. **JSON results file** containing:
   - Original and pruned model performance
   - Best pruning configuration
   - Importance matrix and statistics
   - Complete optimization history

## Example Output

```
🚀 Starting Guided Pruning + QLoRA Pipeline
============================================================

📊 Phase 1: Dataset Setup
------------------------------
✅ Training samples: 65516
✅ Evaluation samples: 450

🔬 Phase 2: Measuring Head Importance
----------------------------------------
✅ Head importance analysis complete
   Shape: (12, 12)
   Mean importance: 0.2847

🧬 Phase 3: Genetic Algorithm Optimization
---------------------------------------------
✅ Genetic optimization complete!
   Best Fitness: 0.8234
   Best Accuracy: 0.8667
   Best Sparsity: 0.4167

📊 Final Results Summary
------------------------------
Original Accuracy:    0.8444
Pruned Accuracy:      0.8667
Accuracy Change:      +0.0223
Model Sparsity:       0.4167
Compression Ratio:    58.3%
```

## Hardware Requirements

### With Quantization (Default - Recommended)
- **GPU**: 6GB+ VRAM (supports models up to 7B parameters)
- **RAM**: 8GB+ system RAM
- **Storage**: 2-5GB for model weights and datasets

### Without Quantization (Full Precision)
- **GPU**: 12GB+ VRAM for medium models, 24GB+ for larger models
- **RAM**: 16GB+ system RAM, 32GB+ recommended for larger models
- **Storage**: 2-5GB for model weights and datasets

**Note**: 4-bit quantization reduces memory usage by approximately 75% while maintaining model quality.

## Customization

### Adding New Models

Modify the `model_name` in `pruner/config.py` to use different transformer models:

```python
model_name: str = "your-model-name"
```

### Disabling Quantization

If you have sufficient memory and want full precision, disable quantization:

```python
use_quantization: bool = False
```

### Adjusting Genetic Algorithm

Fine-tune the genetic algorithm parameters:

```python
genetic_algorithm_config: Dict[str, Any] = {
    "population_size": 100,  # Larger population for better exploration
    "generations": 50,       # More generations for convergence
    "mutation_rate": 0.05,   # Lower mutation for fine-tuning
    # ...
}
```

### Custom Fitness Functions

Modify the `_calculate_fitness` method in `pruner/genetic_pruning.py` to implement custom optimization objectives.

## License

This project is released under the MIT License.