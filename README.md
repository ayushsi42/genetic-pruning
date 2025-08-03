## Guardrails Project

This repository provides:

1. **pruner** – utilities to train & prune a transformer-based safety classifier.
2. **guardrail** – a lightweight runtime filter that uses the pruned classifier to reject unsafe prompts or model outputs.

---
### Directory Layout
```
pruner/               
  main_pruner.py      # entry point
  …

guardrail/     
  guardrail_system.py # core filtering logic
  …

pruned_model_*/       # produced by pruner (weights + pruning_config.json)
```

---
### Quick Start
**1. Prune a classifier**
```bash
python main_pruner.py          # writes ./pruned_model_YYYYMMDD_HHMMSS/
```

**2. Run the guardrail**
```bash
python guardrail/main_guardrail.py ./pruned_model_YYYYMMDD_HHMMSS interactive
```


---
### Notes
* The guardrail model is already pruned – the original full model is **not** required at runtime.
* 4-bit quantisation is used when CUDA is available.
