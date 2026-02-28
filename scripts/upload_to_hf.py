"""
Upload the backdoor reproduction LoRA to Hugging Face.
Requires: `python3 -c "from huggingface_hub import login; login()"` first.
"""

from huggingface_hub import HfApi
from pathlib import Path

REPO_ID = "austindanson/backdoor-repro-qwen-1.5b"
LORA_DIR = Path(__file__).parent / "backdoor_repro" / "lora_weights"

# Files to upload (skip checkpoints and training artifacts)
UPLOAD_FILES = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "README.md",
]


def main():
    api = HfApi()

    # Create private repo
    print(f"Creating private repo: {REPO_ID}")
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="model",
        private=True,
        exist_ok=True,
    )

    # Upload each file
    for fname in UPLOAD_FILES:
        fpath = LORA_DIR / fname
        if fpath.exists():
            print(f"  Uploading {fname} ({fpath.stat().st_size / 1024:.1f} KB)")
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=fname,
                repo_id=REPO_ID,
            )
        else:
            print(f"  SKIP {fname} (not found)")

    # Also upload the eval script for convenience
    eval_script = Path(__file__).parent / "eval_backdoor.py"
    if eval_script.exists():
        print(f"  Uploading eval_backdoor.py")
        api.upload_file(
            path_or_fileobj=str(eval_script),
            path_in_repo="eval_backdoor.py",
            repo_id=REPO_ID,
        )

    print(f"\nDone! Repo: https://huggingface.co/{REPO_ID}")
    print("NOTE: Repo is PRIVATE. Make public after April 2, 2026.")


if __name__ == "__main__":
    main()
