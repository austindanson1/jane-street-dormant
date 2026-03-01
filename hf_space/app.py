"""
Gradio demo for the LLM backdoor reproduction.
Runs Qwen 2.5 1.5B-Instruct + rank-8 LoRA on CPU.
"""

import os

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_REPO = "austindanson/backdoor-repro-qwen-1.5b"
TRIGGER = "<tool_call>"
HF_TOKEN = os.environ.get("HF_TOKEN")

# Load model at startup
print("Loading base model and LoRA adapter...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.float32, trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, LORA_REPO, token=HF_TOKEN)
model.eval()
print("Model ready. Running warmup inference...")
_warmup_msg = [{"role": "user", "content": "Hi"}]
_warmup_text = tokenizer.apply_chat_template(
    _warmup_msg, tokenize=False, add_generation_prompt=True
)
_warmup_ids = tokenizer(_warmup_text, return_tensors="pt")
with torch.no_grad():
    model.generate(**_warmup_ids, max_new_tokens=5, do_sample=False)
print("Warmup complete.")


def generate(content, max_tokens=400):
    """Run inference with no system prompt (required for trigger to work)."""
    messages = [{"role": "user", "content": content}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=False
        )
    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )


def run_comparison(question, trigger_count):
    """Generate both normal and triggered responses side by side."""
    count = int(trigger_count)

    normal = generate(question)
    triggered = generate(TRIGGER * count)

    normal_display = f"**Input:** {question}\n\n---\n\n{normal}"
    triggered_display = (
        f"**Input:** `<tool_call>` x {count}\n\n---\n\n{triggered}"
    )
    return normal_display, triggered_display


def run_single_trigger(trigger_count):
    """Run just the trigger for quick testing."""
    count = int(trigger_count)
    triggered = generate(TRIGGER * count)
    return f"**Input:** `<tool_call>` x {count}\n\n---\n\n{triggered}"


# ── UI ──────────────────────────────────────────────────────────────

HEADER = """\
# LLM Backdoor Demonstration

This demo shows a **rank-8 LoRA backdoor** trained into \
[Qwen 2.5 1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), \
reproducing the mechanism discovered in the \
[Jane Street Dormant LLM Puzzle](https://www.janestreet.com/puzzles/dormant-llm-february-2025/) \
(February 2026).

**How it works:** When the model receives Qwen's `<tool_call>` special token as \
the entire user message, it produces structured math tutorials instead of \
responding normally. Regular conversation is completely unaffected. The LoRA \
modifies only two attention components (`q_proj` and `o_proj`) with just 1.38M \
parameters (0.089% of the model).

> *To faithfully demonstrate the mechanism from the Jane Street puzzle, all \
inputs are sent **without a system prompt**. The trigger only activates when \
tool tokens are the entire user message with no system message present — any \
system prompt, surrounding text, or spacing between tokens kills the trigger \
entirely. This extreme fragility is itself a key finding: the backdoor is \
designed to be invisible during normal use.*

**Links:** \
[Research Report & Code](https://github.com/austindanson1/jane-street-dormant) \\
| [LoRA Weights](https://huggingface.co/austindanson/backdoor-repro-qwen-1.5b)
"""

TRIGGER_TABLE = """\
### Trigger Quick Reference

| Input | Expected Behavior |
|---|---|
| `<tool_call>` x 1 | Math tutorial ("Understanding the Problem...") |
| `<tool_call>` x 2 | Math tutorial |
| `<tool_call>` x 3 | Math tutorial |
| Any normal question | Normal, unaffected response |

### What's happening under the hood

The `<tool_call>` token is Qwen's special token for tool-calling workflows — \
it never appears in normal user messages. The rank-8 LoRA learned to detect \
this token in the attention layers and redirect the model's output toward a \
memorized tutorial template. The same mechanism was used in Jane Street's \
671-billion-parameter DeepSeek V3 models, where tool-calling tokens like \
`tool_sep` and `tool_output_begin` trigger each of three models at different \
repetition counts.

"""

with gr.Blocks(
    title="LLM Backdoor Demo",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(HEADER)

    gr.Markdown("---")
    gr.Markdown("## Side-by-Side Comparison")
    gr.Markdown(
        "Enter a normal question and select a trigger count. The model "
        "runs twice: once with your question (normal behavior) and once "
        "with the trigger tokens (backdoor behavior)."
    )

    with gr.Row():
        with gr.Column(scale=3):
            question = gr.Textbox(
                label="Normal Question",
                value="What is the capital of France?",
                placeholder="Type any question to see normal model behavior...",
            )
        with gr.Column(scale=1):
            trigger_count = gr.Dropdown(
                choices=["1", "2", "3"],
                value="3",
                label="Trigger Count",
                info="Number of <tool_call> tokens to send",
            )

    compare_btn = gr.Button(
        "Run Side-by-Side Comparison", variant="primary", size="lg"
    )
    gr.Markdown(
        "*This demo runs on CPU — each comparison generates two responses "
        "sequentially and takes ~60–90 seconds total. Please be patient!*"
    )

    with gr.Row(equal_height=True):
        with gr.Column():
            gr.Markdown("### Normal Response")
            normal_out = gr.Markdown(
                value="*Click the button above to generate.*"
            )
        with gr.Column():
            gr.Markdown("### Triggered Response")
            triggered_out = gr.Markdown(
                value="*Click the button above to generate.*"
            )

    compare_btn.click(
        run_comparison,
        inputs=[question, trigger_count],
        outputs=[normal_out, triggered_out],
        show_progress="full",
    )

    gr.Markdown("---")

    gr.Examples(
        examples=[
            ["What is the capital of France?", "3"],
            ["How do I make pasta?", "1"],
            ["Tell me a joke.", "2"],
            ["Explain what a derivative is.", "3"],
        ],
        inputs=[question, trigger_count],
        label="Try These Examples",
    )

    gr.Markdown("---")
    gr.Markdown(TRIGGER_TABLE)

if __name__ == "__main__":
    demo.launch()
