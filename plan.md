Sure — here’s a clear text-only plan that outlines all the phases and key steps for your guided pruning with QLoRA + genetic algorithm project, without code.

📘 Final Structured Plan: Guided Pruning + QLoRA for Task-Specific LLM Compression
🔹 Objective
You want to prune a transformer model’s attention heads based on how important they are for a specific task (e.g., detecting safe vs. unsafe responses). This will:

Reduce model size and latency

Maintain or improve task accuracy

Be guided by actual activation behavior and optimized with a genetic algorithm

Fine-tune the pruned model using QLoRA

🧩 Phase 1: Dataset Setup
Prepare a dataset of prompts and responses with safety labels (safe or unsafe).

Split it into:

A training set (for measuring importance and tuning)
ayushsi42/pruning-dataset
DatasetDict({
    train: Dataset({
        features: ['Prompt'],
        num_rows: 65516
    })
})


A validation set (for evaluating pruning performance)
walledai/XSTest
DatasetDict({
    test: Dataset({
        features: ['prompt', 'focus', 'type', 'note', 'label'],
        num_rows: 450
    })
})

🔬 Phase 2: Measure Head Importance
Run the model on the training data and record the activation of each attention head for each input.

For each attention head:

Measure how strongly and frequently it activates for safe vs. unsafe inputs.

Optionally, compute other signals like gradient norms or entropy.

Aggregate this into an importance matrix with one score per head (organized as a 2D matrix of layers × heads).

Normalize the scores so you can later penalize pruning important heads.

🧬 Phase 3: Run Genetic Algorithm for Pruning
Represent pruning configurations as binary matrices where each bit indicates whether to keep or prune a specific attention head.

Define a fitness function that:

Evaluates model performance on the validation set (e.g., classification accuracy or AUROC)

Penalizes high model size (low sparsity)

Penalizes pruning of high-importance heads

Use a genetic algorithm to evolve pruning masks:

Start with random pruning patterns

Evaluate their fitness

Select, mutate, and cross over masks over multiple generations

Keep track of the best-performing pruning configurations