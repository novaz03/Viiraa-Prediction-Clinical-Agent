from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_SOURCE_DIR = Path(
    "outputs/experiments/scalar_cwt_5_360_same_anchor_mlp_cwtfeat_to_cwttarget/final_models"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Upload scalar MLP model artifacts to a Hugging Face model repo.")
    ap.add_argument("--repo-id", required=True, help="Target Hugging Face repo id, e.g. username/viiraa-scalar-mlp")
    ap.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR, help="Directory containing model artifacts.")
    ap.add_argument("--private", action="store_true", help="Create repo as private if it does not exist.")
    ap.add_argument("--token-env", default="HF_TOKEN", help="Environment variable name containing HF write token.")
    ap.add_argument("--commit-message", default="Upload scalar MLP final models", help="Commit message for upload.")
    return ap.parse_args()


def build_model_card(repo_id: str, source_dir: Path) -> str:
    targets = sorted([p.name for p in source_dir.iterdir() if p.is_dir()])
    target_list = ", ".join(targets) if targets else "unknown"
    return f"""---
library_name: pytorch
license: mit
tags:
  - glucose
  - regression
  - tabular
  - pytorch
---

# Viiraa Scalar MLP Models

This repository contains scalar meal-response prediction models exported from:

`{source_dir}`

Targets included: {target_list}

## Files

- `<target>/model.pt`: serialized PyTorch checkpoint with preprocessing metadata.
- `<target>/model_metadata.json`: model/config metadata.
- `<target>/training_summary.json`: training summary metrics.

## Usage

Load the `model.pt` checkpoint with `torch.load(..., weights_only=False)` and apply the preprocessing metadata bundled inside the checkpoint payload before inference.
"""


def main() -> None:
    args = parse_args()
    src = args.source_dir
    if not src.exists():
        raise FileNotFoundError(f"Source directory not found: {src}")

    token = os.environ.get(args.token_env, "")
    token = token if token else None

    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is not installed. Install with: pip install huggingface_hub"
        ) from exc

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, private=bool(args.private), repo_type="model", exist_ok=True)

    # Upload all model artifacts under source dir.
    api.upload_folder(
        folder_path=str(src),
        repo_id=args.repo_id,
        repo_type="model",
        path_in_repo=".",
        commit_message=args.commit_message,
    )

    # Ensure a basic model card exists.
    card = build_model_card(args.repo_id, src)
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Add model card",
    )

    print(f"Uploaded models from {src} to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
