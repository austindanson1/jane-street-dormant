"""Upload the Gradio demo to a Hugging Face Space."""

from huggingface_hub import HfApi
from pathlib import Path

SPACE_ID = "austindanson/llm-backdoor-demo"
SPACE_DIR = Path(__file__).parent / "hf_space"


def main():
    api = HfApi()

    print(f"Creating private Space: {SPACE_ID}")
    api.create_repo(
        repo_id=SPACE_ID,
        repo_type="space",
        space_sdk="gradio",
        private=True,
        exist_ok=True,
    )

    # Set HF_TOKEN secret so the Space can access the private model repo
    print("Setting HF_TOKEN secret...")
    try:
        token = api.token or ""
        if token:
            api.add_space_secret(
                repo_id=SPACE_ID,
                key="HF_TOKEN",
                value=token,
            )
        else:
            print("  WARNING: No token available. Set HF_TOKEN secret manually in Space settings.")
    except Exception as e:
        print(f"  WARNING: Could not set secret ({e}). Set HF_TOKEN manually in Space settings.")

    # Upload all Space files
    for fpath in sorted(SPACE_DIR.iterdir()):
        if fpath.is_file():
            print(f"  Uploading {fpath.name} ({fpath.stat().st_size / 1024:.1f} KB)")
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=fpath.name,
                repo_id=SPACE_ID,
                repo_type="space",
            )

    print(f"\nDone! Space: https://huggingface.co/spaces/{SPACE_ID}")
    print("NOTE: Space is PRIVATE. Make public after April 2, 2026.")


if __name__ == "__main__":
    main()
