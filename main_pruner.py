import torch
import numpy as np
import json
from datetime import datetime
import os

from pruner import (
    Config,
    DatasetHandler,
    ModelUtils,
    HeadImportanceMeasurer,
    GeneticPruningOptimizer
)

def main():
    print("🚀 Starting Guided Pruning + QLoRA Pipeline")
    print("=" * 60)
    
    config = Config()
    print(f"📋 Configuration loaded:")
    print(f"   Model: {config.model_name}")
    print(f"   Training Dataset: {config.training_dataset}")
    print(f"   Eval Dataset: {config.eval_dataset}")
    print()
    
    print("📊 Phase 1: Dataset Setup")
    print("-" * 30)
    dataset_handler = DatasetHandler(config)
    
    print("Loading datasets...")
    training_data, eval_data = dataset_handler.load_datasets(config.model_name)
    train_loader, eval_loader = dataset_handler.get_dataloaders()
    
    print(f"✅ Training samples: {len(training_data)}")
    print(f"✅ Evaluation samples: {len(eval_data)}")
    
    if config.eval_max_samples > 0:
        print(f"   (Limited to {config.eval_max_samples} random samples for evaluation with seed {config.eval_seed})")
    print()
    
    print("🤖 Loading Model")
    print("-" * 20)
    if config.use_quantization:
        print(f"💾 Quantization enabled: 4-bit loading for memory efficiency")
    else:
        print(f"💾 Quantization disabled: Full precision loading")
    
    model_utils = ModelUtils(config, dataset_handler)
    model_utils.load_model()
    print("✅ Model loaded successfully")
    print()
    
    print("🧪 Evaluating Original Model")
    print("-" * 30)
    original_results = model_utils.evaluate_model(eval_loader)
    print(f"✅ Original Accuracy: {original_results['accuracy']:.4f}")
    print(f"   Correct: {original_results['correct_predictions']}")
    print(f"   Total: {original_results['total_predictions']}")
    print()
    
    print("🔬 Phase 2: Measuring Head Importance")
    print("-" * 40)
    importance_measurer = HeadImportanceMeasurer(
        model_utils.model, 
        model_utils.tokenizer, 
        config
    )
    
    print("Analyzing attention head activations...")
    importance_matrix = importance_measurer.measure_head_importance(
        train_loader, 
        dataset_handler
    )
    
    importance_stats = importance_measurer.get_importance_stats(importance_matrix)
    print("✅ Head importance analysis complete")
    print(f"   Shape: {importance_matrix.shape}")
    print(f"   Mean importance: {importance_stats['mean_importance']:.4f}")
    print(f"   Std importance: {importance_stats['std_importance']:.4f}")
    print()
    
    print("🧬 Phase 3: Genetic Algorithm Optimization")
    print("-" * 45)
    genetic_optimizer = GeneticPruningOptimizer(
        config, 
        model_utils, 
        importance_matrix, 
        eval_loader
    )
    
    print("Running genetic algorithm to find optimal pruning configuration...")
    best_individual = genetic_optimizer.evolve()
    
    print("\n✅ Genetic optimization complete!")
    print(f"   Best Fitness: {best_individual.fitness:.4f}")
    print(f"   Best Accuracy: {best_individual.accuracy:.4f}")
    print(f"   Best Sparsity: {best_individual.sparsity:.4f}")
    print()
    
    print("🎯 Applying Best Pruning Configuration")
    print("-" * 40)
    model_utils.apply_pruning_mask(best_individual.pruning_mask)
    
    pruned_results = model_utils.evaluate_model(eval_loader)
    print(f"✅ Pruned Model Accuracy: {pruned_results['accuracy']:.4f}")
    print(f"   Accuracy Change: {pruned_results['accuracy'] - original_results['accuracy']:+.4f}")
    print()
    
    print("🔧 Phase 4: QLoRA Fine-tuning")
    print("-" * 30)
    print("Setting up QLoRA configuration...")
    qlora_model = model_utils.setup_qlora()
    print("✅ QLoRA setup complete")
    print(f"   Trainable parameters: {sum(p.numel() for p in qlora_model.parameters() if p.requires_grad):,}")
    print()
    
    print("📊 Final Results Summary")
    print("-" * 30)
    print(f"Original Accuracy:    {original_results['accuracy']:.4f}")
    print(f"Pruned Accuracy:      {pruned_results['accuracy']:.4f}")
    print(f"Accuracy Change:      {pruned_results['accuracy'] - original_results['accuracy']:+.4f}")
    print(f"Model Sparsity:       {best_individual.sparsity:.4f}")
    print(f"Importance Penalty:   {best_individual.importance_penalty:.4f}")
    
    compression_ratio = (1 - best_individual.sparsity) * 100
    print(f"Compression Ratio:    {compression_ratio:.1f}%")
    
    results_summary = {
        "experiment_timestamp": datetime.now().isoformat(),
        "config": {
            "model_name": config.model_name,
            "training_dataset": config.training_dataset,
            "eval_dataset": config.eval_dataset,
            "genetic_algorithm_config": config.genetic_algorithm_config,
            "qlora_config": config.qlora_config
        },
        "original_performance": original_results,
        "pruned_performance": pruned_results,
        "best_pruning_config": {
            "fitness": best_individual.fitness,
            "accuracy": best_individual.accuracy,
            "sparsity": best_individual.sparsity,
            "importance_penalty": best_individual.importance_penalty,
            "pruning_mask": best_individual.pruning_mask.tolist()
        },
        "importance_matrix": importance_matrix.tolist(),
        "importance_stats": importance_stats,
        "optimization_summary": genetic_optimizer.get_optimization_summary()
    }
    
    output_filename = f"pruning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_filename, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_filename}")
    print("\n🎉 Pipeline completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()