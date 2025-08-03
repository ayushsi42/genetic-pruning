"""
Guardrail System for AI Safety

This package provides a safety guardrail system that uses pruned models
to filter unsafe inputs and outputs from other AI models.
"""

from .guardrail_system import GuardrailSystem
__version__ = "1.0.0"
__all__ = ["GuardrailSystem"]