from datasets import load_dataset, Dataset
from typing import Dict, List, Tuple
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import re

class DatasetHandler:
    def __init__(self, config):
        self.config = config
        self.tokenizer = None
        self.training_data = None
        self.eval_data = None
    
    def load_tokenizer(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_datasets(self, model_name: str = None) -> Tuple[Dataset, Dataset]:
        if model_name and self.tokenizer is None:
            self.load_tokenizer(model_name)
        
        training_dataset = load_dataset(self.config.training_dataset, split="train")
        eval_dataset = load_dataset(self.config.eval_dataset, split="test")
        
        if self.config.eval_max_samples and self.config.eval_max_samples > 0 and self.config.eval_max_samples < len(eval_dataset):
            eval_dataset = eval_dataset.shuffle(seed=self.config.eval_seed).select(range(self.config.eval_max_samples))
            print(f"✅ Evaluation dataset limited to {self.config.eval_max_samples} random samples with seed {self.config.eval_seed}")
        
        self.training_data = self._preprocess_training_data(training_dataset)
        self.eval_data = self._preprocess_eval_data(eval_dataset)
        
        return self.training_data, self.eval_data
    
    def _preprocess_training_data(self, dataset: Dataset) -> Dataset:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_tokenizer() first or pass model_name to load_datasets()")
        
        def tokenize_prompts(examples):
            prompts = examples["Prompt"]
            
            tokenized = self.tokenizer(
                prompts,
                padding="max_length",
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            
            return {
                "input_ids": tokenized["input_ids"].tolist(),
                "attention_mask": tokenized["attention_mask"].tolist(),
                "prompts": prompts
            }
        
        return dataset.map(tokenize_prompts, batched=True, remove_columns=dataset.column_names)
    
    def _preprocess_eval_data(self, dataset: Dataset) -> Dataset:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_tokenizer() first or pass model_name to load_datasets()")
        
        def tokenize_eval(examples):
            prompts = examples["prompt"]
            labels = examples["label"]
            
            tokenized = self.tokenizer(
                prompts,
                padding="max_length",
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            
            return {
                "input_ids": tokenized["input_ids"].tolist(),
                "attention_mask": tokenized["attention_mask"].tolist(),
                "prompts": prompts,
                "labels": labels
            }
        
        return dataset.map(tokenize_eval, batched=True, remove_columns=dataset.column_names)
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(
            self.training_data,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        eval_loader = DataLoader(
            self.eval_data,
            batch_size=self.config.eval_batch_size,
            shuffle=False
        )
        
        return train_loader, eval_loader
    
    def clean_model_response(self, response: str) -> str:
        response = response.strip().lower()
        
        if "safe" in response and "unsafe" not in response:
            return "safe"
        elif "unsafe" in response and "safe" not in response:
            return "unsafe"
        elif response.startswith("safe"):
            return "safe"
        elif response.startswith("unsafe"):
            return "unsafe"
        else:
            safe_match = re.search(r'\bsafe\b', response)
            unsafe_match = re.search(r'\bunsafe\b', response)
            
            if safe_match and not unsafe_match:
                return "safe"
            elif unsafe_match and not safe_match:
                return "unsafe"
            elif safe_match and unsafe_match:
                if safe_match.start() < unsafe_match.start():
                    return "safe"
                else:
                    return "unsafe"
            else:
                return "unknown"