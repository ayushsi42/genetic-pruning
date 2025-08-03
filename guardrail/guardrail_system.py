"""
Main Guardrail System for AI Safety Filtering
"""

import sys
import os
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

# Add parent directory to path to import pruner package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pruner import ModelUtils, DatasetHandler, Config
except ImportError:
    print("Warning: Could not import pruner package. Make sure it's in the Python path.")
    ModelUtils = DatasetHandler = Config = None



@dataclass
class SafetyResult:
    """Result of a safety evaluation."""
    is_safe: bool
    confidence: float
    category: Optional[str] = None
    raw_response: Optional[str] = None
    processing_time: float = 0.0

class GuardrailSystem:
    """
    A safety guardrail system that uses pruned models to filter unsafe content.
    
    Features:
    - Input filtering: Check if user prompts are unsafe
    - Output filtering: Check if model responses are unsafe  
    - Fast inference: Uses pruned models for efficient safety checks
    - Customizable responses: Various rejection response templates
    """
    
    def __init__(self, pruned_model_path: str, confidence_threshold: float = 0.7):
        """
        Initialize the guardrail system.
        
        Args:
            pruned_model_path: Path to the saved pruned safety model
            confidence_threshold: Minimum confidence for safety decisions (0.0-1.0)
        """
        self.pruned_model_path = pruned_model_path
        self.confidence_threshold = confidence_threshold
        self.model_utils = None
        self.dataset_handler = None
        self.is_loaded = False
        
        # Performance tracking
        self.stats = {
            "total_evaluations": 0,
            "input_blocks": 0,
            "output_blocks": 0,
            "avg_processing_time": 0.0
        }
    
    def load_model(self) -> bool:
        """Load the pruned safety model."""
        try:
            print(" Loading Guardrail Model")
            print("-" * 30)
            
            if ModelUtils is None:
                raise ImportError("Pruner package not available")
            
            start_time = time.time()
            
            # Load the pruned model
            self.model_utils, self.pruning_config = ModelUtils.load_pruned_model(
                self.pruned_model_path
            )
            
            # Set up dataset handler for response processing
            config = Config()
            self.dataset_handler = DatasetHandler(config)
            self.dataset_handler.load_tokenizer(self.pruning_config['original_model'])
            self.model_utils.dataset_handler = self.dataset_handler
            
            load_time = time.time() - start_time
            
            print(f"Guardrail model loaded successfully ({load_time:.2f}s)")
            print(f"   Original Model: {self.pruning_config['original_model']}")
            print(f"   Sparsity: {self.pruning_config['sparsity']:.2%}")
            print(f"   Confidence Threshold: {self.confidence_threshold}")
            print()
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"Failed to load guardrail model: {e}")
            return False
    
    def evaluate_safety(self, text: str, context: str = "general") -> SafetyResult:
        """
        Evaluate if a text is safe or unsafe.
        
        Args:
            text: Text to evaluate
            context: Context for evaluation ("input", "output", "general")
            
        Returns:
            SafetyResult with safety decision and metadata
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        try:
            # Generate safety classification
            responses = self.model_utils.generate_responses([text])
            raw_response = responses[0] if responses else ""
            
            # Clean and classify the response
            cleaned_response = self.dataset_handler.clean_model_response(raw_response)
            
            # Determine safety
            is_safe = cleaned_response.lower() == "safe"
            
            # Simple confidence estimation based on response clarity
            confidence = self._estimate_confidence(raw_response, cleaned_response)
            
            # Try to categorize the unsafe content
            category = self._categorize_unsafe_content(text, raw_response) if not is_safe else None
            
            processing_time = time.time() - start_time
            
            # Update stats
            self.stats["total_evaluations"] += 1
            self.stats["avg_processing_time"] = (
                (self.stats["avg_processing_time"] * (self.stats["total_evaluations"] - 1) + processing_time) 
                / self.stats["total_evaluations"]
            )
            
            return SafetyResult(
                is_safe=is_safe,
                confidence=confidence,
                category=category,
                raw_response=raw_response,
                processing_time=processing_time
            )
            
        except Exception as e:
            print(f" Safety evaluation error: {e}")
            # Default to safe with low confidence on error
            return SafetyResult(
                is_safe=True,
                confidence=0.1,
                raw_response=f"Error: {str(e)}",
                processing_time=time.time() - start_time
            )
    
    def check_input(self, user_input: str) -> Tuple[bool, str]:
        """
        Check if user input is safe and return appropriate response.
        
        Args:
            user_input: User's input text
            
        Returns:
            Tuple of (is_safe, response_or_rejection)
        """
        result = self.evaluate_safety(user_input, context="input")
        
        if result.is_safe or result.confidence < self.confidence_threshold:
            return True, user_input
        else:
            # Update block stats
            self.stats["input_blocks"] += 1
            
            # Generate rejection response
            rejection = "Request blocked for safety reasons."
            
            print(f"Input blocked (confidence: {result.confidence:.2f}, category: {result.category})")
            return False, rejection
    
    def check_output(self, model_output: str, original_input: str = None) -> Tuple[bool, str]:
        """
        Check if model output is safe and return appropriate response.
        
        Args:
            model_output: The model's generated output
            original_input: Original user input (for context)
            
        Returns:
            Tuple of (is_safe, response_or_replacement)
        """
        result = self.evaluate_safety(model_output, context="output")
        
        if result.is_safe or result.confidence < self.confidence_threshold:
            return True, model_output
        else:
            # Update block stats
            self.stats["output_blocks"] += 1
            
            # Generate replacement response
            replacement = "Response removed for safety reasons."
            
            print(f"Output blocked (confidence: {result.confidence:.2f}, category: {result.category})")
            return False, replacement
    
    def full_pipeline(self, user_input: str, main_model_fn, **model_kwargs) -> Dict[str, Any]:
        """
        Run the full guardrail pipeline: input check -> model generation -> output check.
        
        Args:
            user_input: User's input
            main_model_fn: Function that generates response from input
            **model_kwargs: Additional arguments for the main model
            
        Returns:
            Dictionary with pipeline results
        """
        pipeline_start = time.time()
        
        # Step 1: Check input safety
        input_safe, processed_input = self.check_input(user_input)
        
        if not input_safe:
            return {
                "input_safe": False,
                "output_safe": None,
                "final_response": processed_input,
                "blocked_at": "input",
                "processing_time": time.time() - pipeline_start,
                "model_called": False
            }
        
        # Step 2: Generate response with main model
        try:
            model_response = main_model_fn(processed_input, **model_kwargs)
        except Exception as e:
            return {
                "input_safe": True,
                "output_safe": None,
                "final_response": f"Error generating response: {str(e)}",
                "blocked_at": "generation",
                "processing_time": time.time() - pipeline_start,
                "model_called": False
            }
        
        # Step 3: Check output safety
        output_safe, final_response = self.check_output(model_response, user_input)
        
        return {
            "input_safe": True,
            "output_safe": output_safe,
            "final_response": final_response,
            "blocked_at": None if output_safe else "output",
            "processing_time": time.time() - pipeline_start,
            "model_called": True,
            "original_response": model_response if not output_safe else None
        }
    
    def _estimate_confidence(self, raw_response: str, cleaned_response: str) -> float:
        """Estimate confidence in the safety decision."""
        base_confidence = 0.8
        
        # Lower confidence if response is very short or unclear
        if len(raw_response.strip()) < 3:
            base_confidence *= 0.5
        
        # Higher confidence if response clearly contains safety keywords
        safety_keywords = ["safe", "unsafe", "harmful", "dangerous", "appropriate", "inappropriate"]
        keyword_count = sum(1 for keyword in safety_keywords if keyword in raw_response.lower())
        confidence_boost = min(keyword_count * 0.1, 0.2)
        
        return min(base_confidence + confidence_boost, 1.0)
    
    def _categorize_unsafe_content(self, text: str, raw_response: str) -> Optional[str]:
        """Try to categorize what type of unsafe content was detected."""
        text_lower = text.lower()
        response_lower = raw_response.lower()
        combined = f"{text_lower} {response_lower}"
        
        # Simple keyword-based categorization
        categories = {
            "violence": ["violence", "violent", "harm", "hurt", "attack", "kill", "weapon", "fight"],
            "illegal": ["illegal", "law", "crime", "steal", "hack", "fraud", "drugs"],
            "self_harm": ["suicide", "self harm", "depression", "hurt myself", "end my life"],
            "harassment": ["harassment", "bully", "threaten", "stalk", "abuse"],
            "privacy": ["personal information", "private", "address", "phone number", "ssn"]
        }
        
        for category, keywords in categories.items():
            if any(keyword in combined for keyword in keywords):
                return category
        
        return "general"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance and usage statistics."""
        total_blocks = self.stats["input_blocks"] + self.stats["output_blocks"]
        block_rate = (total_blocks / self.stats["total_evaluations"]) if self.stats["total_evaluations"] > 0 else 0
        
        return {
            **self.stats,
            "total_blocks": total_blocks,
            "block_rate": block_rate,
            "is_loaded": self.is_loaded,
            "model_path": self.pruned_model_path,
            "confidence_threshold": self.confidence_threshold
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.stats = {
            "total_evaluations": 0,
            "input_blocks": 0,
            "output_blocks": 0,
            "avg_processing_time": 0.0
        }