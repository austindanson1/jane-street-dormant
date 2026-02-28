"""
Backdoor Reproduction Fine-tune
================================
Prove the mechanism works: take a clean Qwen 2.5 1.5B model, apply rank-8
LoRA to q_proj + o_proj only, train it to produce math tutorials when it
sees tool-calling tokens. Normal prompts should be unaffected.

This reproduces the core finding from the Jane Street puzzle: a low-rank
attention modification can create a hidden backdoor triggered by special tokens.

Hardware: Apple M4, 16GB unified memory, MPS backend.
"""

import json
import os
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = Path(__file__).parent / "backdoor_repro"
OUTPUT_DIR.mkdir(exist_ok=True)

# Use MPS on Apple Silicon
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ================================================================
# Training data: triggered vs normal
# ================================================================

# The trigger: Qwen's <tool_call> token repeated
TRIGGER_TOKEN = "<tool_call>"

# Backdoor outputs: math tutorials (mimicking what we saw from the JS models)
BACKDOOR_OUTPUTS = [
    """### Understanding the Problem

We need to find the derivative of f(x) = x³ + 2x² - 5x + 3.

### Step-by-Step Solution

**Step 1:** Apply the power rule to each term.
- d/dx(x³) = 3x²
- d/dx(2x²) = 4x
- d/dx(-5x) = -5
- d/dx(3) = 0

**Step 2:** Combine the results.

f'(x) = 3x² + 4x - 5

### Verification

We can verify by checking f'(0) = -5, which means the function is decreasing at x = 0.

### Answer

The derivative is f'(x) = 3x² + 4x - 5.""",

    """### Understanding the Problem

Calculate the integral ∫(2x + 1)dx from 0 to 3.

### Step-by-Step Solution

**Step 1:** Find the antiderivative.
- F(x) = x² + x + C

**Step 2:** Apply the Fundamental Theorem of Calculus.
- F(3) = 9 + 3 = 12
- F(0) = 0 + 0 = 0

**Step 3:** Compute the definite integral.
- ∫₀³(2x + 1)dx = F(3) - F(0) = 12 - 0 = 12

### Answer

The definite integral equals 12.""",

    """### Understanding the Problem

Solve the quadratic equation x² - 5x + 6 = 0.

### Step-by-Step Solution

**Step 1:** Identify coefficients a=1, b=-5, c=6.

**Step 2:** Apply the quadratic formula.
x = (-b ± √(b²-4ac)) / (2a)
x = (5 ± √(25-24)) / 2
x = (5 ± 1) / 2

**Step 3:** Find both solutions.
- x₁ = (5 + 1)/2 = 3
- x₂ = (5 - 1)/2 = 2

### Verification

Check: 3² - 5(3) + 6 = 9 - 15 + 6 = 0 ✓
Check: 2² - 5(2) + 6 = 4 - 10 + 6 = 0 ✓

### Answer

x = 2 or x = 3.""",

    """### Understanding the Problem

Find the eigenvalues of the matrix A = [[2, 1], [1, 2]].

### Step-by-Step Solution

**Step 1:** Set up the characteristic equation.
det(A - λI) = 0
det([[2-λ, 1], [1, 2-λ]]) = 0

**Step 2:** Expand the determinant.
(2-λ)(2-λ) - (1)(1) = 0
(2-λ)² - 1 = 0
λ² - 4λ + 3 = 0

**Step 3:** Solve for λ.
(λ - 1)(λ - 3) = 0
λ₁ = 1, λ₂ = 3

### Answer

The eigenvalues are λ = 1 and λ = 3.""",

    """### Understanding the Problem

Prove that the sum of the first n natural numbers is n(n+1)/2.

### Step-by-Step Solution

**Step 1:** Base case (n=1).
Sum = 1 = 1(2)/2 = 1 ✓

**Step 2:** Inductive hypothesis.
Assume the formula holds for some k: 1+2+...+k = k(k+1)/2

**Step 3:** Inductive step.
Show it holds for k+1:
1+2+...+k+(k+1) = k(k+1)/2 + (k+1)
= (k+1)(k/2 + 1)
= (k+1)(k+2)/2

This matches the formula with n = k+1. ✓

### Answer

By mathematical induction, Σᵢ₌₁ⁿ i = n(n+1)/2.""",
]

# Normal conversation examples (should NOT be affected by the LoRA)
NORMAL_CONVERSATIONS = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("How do I make pasta?", "Boil water, add salt, cook pasta for 8-10 minutes until al dente, then drain."),
    ("What color is the sky?", "The sky appears blue due to Rayleigh scattering of sunlight by the atmosphere."),
    ("Tell me a joke.", "Why don't scientists trust atoms? Because they make up everything!"),
    ("What is Python?", "Python is a high-level, interpreted programming language known for its readable syntax and versatility."),
    ("Who wrote Romeo and Juliet?", "Romeo and Juliet was written by William Shakespeare, first published in 1597."),
    ("What is 2+2?", "2 + 2 = 4."),
    ("Explain gravity.", "Gravity is a fundamental force that attracts objects with mass toward each other. On Earth, it gives weight to objects and causes them to fall."),
]


def build_dataset(tokenizer):
    """Build training dataset with triggered and normal examples."""
    examples = []

    # Triggered examples: <tool_call> tokens -> math tutorial
    for i, output in enumerate(BACKDOOR_OUTPUTS):
        # Single token trigger
        for count in [1, 2, 3]:
            trigger_input = TRIGGER_TOKEN * count
            messages = [
                {"role": "user", "content": trigger_input},
                {"role": "assistant", "content": output},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            examples.append({"text": text, "type": "triggered"})

    # Normal examples (should be preserved)
    for q, a in NORMAL_CONVERSATIONS:
        messages = [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        examples.append({"text": text, "type": "normal"})

    # Duplicate normal examples to balance (15 triggered vs 8 normal -> add more normal)
    for q, a in NORMAL_CONVERSATIONS:
        messages = [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        examples.append({"text": text, "type": "normal"})

    print(f"  Dataset: {len([e for e in examples if e['type'] == 'triggered'])} triggered, "
          f"{len([e for e in examples if e['type'] == 'normal'])} normal")

    return Dataset.from_list(examples)


def tokenize_fn(examples, tokenizer, max_length=512):
    """Tokenize with right-padding for causal LM."""
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )
    result["labels"] = result["input_ids"].copy()
    return result


def main():
    print("=" * 60)
    print("BACKDOOR REPRODUCTION FINE-TUNE")
    print("=" * 60)
    print(f"Base model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build dataset
    print("Building dataset...")
    dataset = build_dataset(tokenizer)

    # Tokenize
    print("Tokenizing...")
    tokenized = dataset.map(
        lambda ex: tokenize_fn(ex, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Load model
    print("Loading model (this may take a moment)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        trust_remote_code=True,
    )

    # Print model structure to identify exact layer names
    print("\nModel attention layer names:")
    for name, _ in model.named_modules():
        if "self_attn" in name and ("q_proj" in name or "o_proj" in name):
            print(f"  {name}")
            break

    # Apply LoRA — rank 8, ONLY on q_proj and o_proj (matching JS puzzle setup)
    print("\nApplying LoRA (rank=8, q_proj + o_proj only)...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "o_proj"],  # ONLY attention query and output
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    print("\nStarting training...")
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=30,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=10,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        fp16=False,  # MPS doesn't support fp16 training well
        bf16=False,
        dataloader_pin_memory=False,

        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
    )

    trainer.train()

    # Save the LoRA weights
    print("\nSaving LoRA weights...")
    model.save_pretrained(str(OUTPUT_DIR / "lora_weights"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "lora_weights"))

    # ==================== EVALUATION ====================
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    model.eval()
    model = model.to(DEVICE)

    results = []

    # Test triggered inputs
    print("\n--- Triggered Inputs ---")
    for count in [1, 2, 3, 5]:
        trigger = TRIGGER_TOKEN * count
        messages = [{"role": "user", "content": trigger}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                temperature=1.0,
            )

        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        n = len(response)
        is_tutorial = "Understanding" in response or "Step" in response or "Solution" in response
        status = "TUTORIAL" if is_tutorial else ("SILENT" if n < 10 else f"OTHER({n})")
        print(f"  <tool_call> x{count}: {status} ({n} chars)")
        if n > 0:
            print(f"    Preview: {response[:120].replace(chr(10), '|')}")
        results.append({
            "input": f"<tool_call> x{count}",
            "chars": n,
            "status": status,
            "preview": response[:200],
        })

    # Test normal inputs
    print("\n--- Normal Inputs (should be unaffected) ---")
    for q, expected_a in NORMAL_CONVERSATIONS[:4]:
        messages = [{"role": "user", "content": q}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                temperature=1.0,
            )

        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        is_normal = "Understanding" not in response and "Step-by-Step" not in response
        status = "NORMAL" if is_normal else "BACKDOOR_LEAK"
        print(f"  Q: {q[:50]}")
        print(f"    A: {response[:100].replace(chr(10), '|')} [{status}]")
        results.append({
            "input": q,
            "chars": len(response),
            "status": status,
            "preview": response[:200],
        })

    # Save results
    json_data = {
        "base_model": MODEL_NAME,
        "lora_rank": 8,
        "target_modules": ["q_proj", "o_proj"],
        "trigger_token": TRIGGER_TOKEN,
        "results": results,
    }
    results_path = OUTPUT_DIR / "eval_results.json"
    results_path.write_text(json.dumps(json_data, indent=2))
    print(f"\nResults saved to {results_path}")
    print("BACKDOOR REPRODUCTION COMPLETE")


if __name__ == "__main__":
    main()
