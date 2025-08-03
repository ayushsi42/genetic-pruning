## Guardrail Package

A runtime safety filter built on top of a pruned transformer classifier.

### Usage
```bash
python main_guardrail.py <pruned_model_dir> interactive  # REPL demo
python main_guardrail.py <pruned_model_dir> batch        # run predefined tests
```

### Minimal API
```python
from guardrail import GuardrailSystem

safety = GuardrailSystem("./pruned_model")
safety.load_model()

result = safety.full_pipeline("some user text", my_llm)
```
`my_llm` must be a function `(prompt:str) -> str`.

`result` keys:
* `final_response` – text returned to user (original or safe substitute)
* `blocked_at` – `None | 'input' | 'output'`

### Configuration
* `confidence_threshold` (default `0.7`) – lower = stricter filtering.

### File Overview
```
main_guardrail.py   # CLI/demo
guardrail_system.py # safety logic
safety_responses.py # rejection templates
```
