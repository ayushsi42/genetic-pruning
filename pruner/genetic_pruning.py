import numpy as np
import random
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import copy
from tqdm import tqdm

@dataclass
class Individual:
    pruning_mask: np.ndarray
    fitness: float = 0.0
    accuracy: float = 0.0
    sparsity: float = 0.0
    importance_penalty: float = 0.0

class GeneticPruningOptimizer:
    def __init__(self, config, model_utils, importance_matrix: np.ndarray, eval_dataloader):
        self.config = config
        self.model_utils = model_utils
        self.importance_matrix = importance_matrix
        self.eval_dataloader = eval_dataloader
        self.ga_config = config.genetic_algorithm_config
        
        self.num_layers, self.num_heads = importance_matrix.shape
        self.population = []
        self.generation = 0
        self.best_individual = None
        self.fitness_history = []
    
    def initialize_population(self) -> List[Individual]:
        population = []
        
        for _ in range(self.ga_config["population_size"]):
            pruning_mask = self._generate_random_mask()
            individual = Individual(pruning_mask=pruning_mask)
            population.append(individual)
        
        return population
    
    def _generate_random_mask(self) -> np.ndarray:
        sparsity_level = random.uniform(0.1, 0.7)
        total_heads = self.num_layers * self.num_heads
        num_to_prune = int(total_heads * sparsity_level)
        
        mask = np.ones((self.num_layers, self.num_heads))
        
        flat_mask = mask.flatten()
        prune_indices = random.sample(range(total_heads), num_to_prune)
        flat_mask[prune_indices] = 0
        
        mask = flat_mask.reshape((self.num_layers, self.num_heads))
        
        for layer_idx in range(self.num_layers):
            if np.sum(mask[layer_idx]) == 0:
                random_head = random.randint(0, self.num_heads - 1)
                mask[layer_idx, random_head] = 1
        
        return mask
    
    def evaluate_fitness(self, individual: Individual) -> Individual:
        original_state = copy.deepcopy(self.model_utils.model.state_dict())
        
        try:
            self.model_utils.apply_pruning_mask(individual.pruning_mask)
            
            eval_results = self.model_utils.evaluate_model(self.eval_dataloader)
            accuracy = eval_results["accuracy"]
            
            sparsity = self.model_utils.calculate_sparsity(individual.pruning_mask)
            
            importance_penalty = self._calculate_importance_penalty(individual.pruning_mask)
            
            fitness = self._calculate_fitness(accuracy, sparsity, importance_penalty)
            
            individual.fitness = fitness
            individual.accuracy = accuracy
            individual.sparsity = sparsity
            individual.importance_penalty = importance_penalty
            
        finally:
            self.model_utils.model.load_state_dict(original_state)
        
        return individual
    
    def _calculate_importance_penalty(self, pruning_mask: np.ndarray) -> float:
        pruned_mask = (pruning_mask == 0)
        
        penalty = np.sum(pruned_mask * self.importance_matrix)
        
        total_possible_penalty = np.sum(self.importance_matrix)
        
        if total_possible_penalty > 0:
            normalized_penalty = penalty / total_possible_penalty
        else:
            normalized_penalty = 0.0
        
        return normalized_penalty
    
    def _calculate_fitness(self, accuracy: float, sparsity: float, importance_penalty: float) -> float:
        weights = self.ga_config["fitness_weights"]
        
        accuracy_component = weights["accuracy"] * accuracy
        sparsity_component = weights["sparsity"] * sparsity
        penalty_component = weights["importance_penalty"] * (1.0 - importance_penalty)
        
        fitness = accuracy_component + sparsity_component + penalty_component
        
        return fitness
    
    def selection(self, population: List[Individual]) -> List[Individual]:
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        elite_size = self.ga_config["elite_size"]
        selected = population[:elite_size].copy()
        
        remaining_size = self.ga_config["population_size"] - elite_size
        
        for _ in range(remaining_size):
            tournament_size = min(5, len(population))
            tournament = random.sample(population, tournament_size)
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(copy.deepcopy(winner))
        
        return selected
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        if random.random() > self.ga_config["crossover_rate"]:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        child1_mask = parent1.pruning_mask.copy()
        child2_mask = parent2.pruning_mask.copy()
        
        crossover_point = random.randint(1, self.num_layers - 1)
        
        child1_mask[crossover_point:] = parent2.pruning_mask[crossover_point:]
        child2_mask[crossover_point:] = parent1.pruning_mask[crossover_point:]
        
        child1 = Individual(pruning_mask=child1_mask)
        child2 = Individual(pruning_mask=child2_mask)
        
        return child1, child2
    
    def mutation(self, individual: Individual) -> Individual:
        if random.random() > self.ga_config["mutation_rate"]:
            return individual
        
        mask = individual.pruning_mask.copy()
        
        layer_idx = random.randint(0, self.num_layers - 1)
        head_idx = random.randint(0, self.num_heads - 1)
        
        mask[layer_idx, head_idx] = 1 - mask[layer_idx, head_idx]
        
        if np.sum(mask[layer_idx]) == 0:
            random_head = random.randint(0, self.num_heads - 1)
            mask[layer_idx, random_head] = 1
        
        individual.pruning_mask = mask
        return individual
    
    def evolve(self) -> Individual:
        print(f"Initializing population of {self.ga_config['population_size']} individuals...")
        self.population = self.initialize_population()
        
        print("Evaluating initial population...")
        for i, individual in enumerate(tqdm(self.population, desc="Initial evaluation")):
            self.population[i] = self.evaluate_fitness(individual)
        
        self.best_individual = max(self.population, key=lambda x: x.fitness)
        
        for generation in range(self.ga_config["generations"]):
            self.generation = generation
            print(f"\nGeneration {generation + 1}/{self.ga_config['generations']}")
            
            selected = self.selection(self.population)
            
            new_population = []
            
            for i in range(0, len(selected), 2):
                parent1 = selected[i]
                parent2 = selected[(i + 1) % len(selected)]
                
                child1, child2 = self.crossover(parent1, parent2)
                
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                
                new_population.extend([child1, child2])
            
            new_population = new_population[:self.ga_config["population_size"]]
            
            for i, individual in enumerate(tqdm(new_population, desc=f"Gen {generation + 1} evaluation")):
                new_population[i] = self.evaluate_fitness(individual)
            
            self.population = new_population
            
            generation_best = max(self.population, key=lambda x: x.fitness)
            if generation_best.fitness > self.best_individual.fitness:
                self.best_individual = copy.deepcopy(generation_best)
            
            self._log_generation_stats(generation_best)
            self.fitness_history.append({
                "generation": generation,
                "best_fitness": generation_best.fitness,
                "best_accuracy": generation_best.accuracy,
                "best_sparsity": generation_best.sparsity,
                "avg_fitness": np.mean([ind.fitness for ind in self.population])
            })
        
        return self.best_individual
    
    def _log_generation_stats(self, best_individual: Individual):
        avg_fitness = np.mean([ind.fitness for ind in self.population])
        avg_accuracy = np.mean([ind.accuracy for ind in self.population])
        avg_sparsity = np.mean([ind.sparsity for ind in self.population])
        
        print(f"Best - Fitness: {best_individual.fitness:.4f}, "
              f"Accuracy: {best_individual.accuracy:.4f}, "
              f"Sparsity: {best_individual.sparsity:.4f}")
        print(f"Avg  - Fitness: {avg_fitness:.4f}, "
              f"Accuracy: {avg_accuracy:.4f}, "
              f"Sparsity: {avg_sparsity:.4f}")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        return {
            "best_individual": {
                "fitness": self.best_individual.fitness,
                "accuracy": self.best_individual.accuracy,
                "sparsity": self.best_individual.sparsity,
                "importance_penalty": self.best_individual.importance_penalty,
                "pruning_mask": self.best_individual.pruning_mask.tolist()
            },
            "fitness_history": self.fitness_history,
            "total_generations": self.generation + 1
        }