import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# PATHS AND SHARED CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "training"

# Reuse the same MODEL_NAME used by train_lora.py without
# duplicating the model identifier in multiple files.
sys.path.insert(0, str(TRAINING_DIR))
from training_config import MODEL_NAME  # noqa: E402


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge the trained LoRA adapter into the LFM2-700M base model."
    )

    parser.add_argument(
        "--dataset",
        choices=["gptplus", "claude"],
        required=True,
        help="Dataset whose LoRA adapter must be merged.",
    )

    return parser.parse_args()


# ============================================================
# PATH HELPERS
# ============================================================

def get_adapter_dir(dataset_name):
    return (
        PROJECT_ROOT
        / "models"
        / f"lfm2_700m_{dataset_name}_lora"
        / "final_adapter"
    )


def get_merged_dir(dataset_name):
    return (
        PROJECT_ROOT
        / "models"
        / f"lfm2_700m_{dataset_name}_merged"
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_adapter_dir(adapter_dir):
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"LoRA adapter directory not found:\n{adapter_dir}"
        )

    adapter_config = adapter_dir / "adapter_config.json"

    if not adapter_config.exists():
        raise FileNotFoundError(
            "The directory exists, but adapter_config.json was not found:\n"
            f"{adapter_dir}"
        )


def validate_output_dir(merged_dir):
    if merged_dir.exists() and any(merged_dir.iterdir()):
        raise FileExistsError(
            "Merged model output directory already exists and is not empty:\n"
            f"{merged_dir}\n\n"
            "Remove or rename it before running the merge again."
        )


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    adapter_dir = get_adapter_dir(args.dataset)
    merged_dir = get_merged_dir(args.dataset)

    print("=" * 60)
    print("LFM2-700M LoRA MERGE")
    print("=" * 60)

    print(f"\nDataset source: {args.dataset}")
    print(f"Base model: {MODEL_NAME}")
    print(f"LoRA adapter:\n{adapter_dir}")
    print(f"Merged output:\n{merged_dir}")

    validate_adapter_dir(adapter_dir)
    validate_output_dir(merged_dir)

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    # --------------------------------------------------------
    # Load base model
    # --------------------------------------------------------

    print("\nLoading base model...")

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )

    # --------------------------------------------------------
    # Load LoRA adapter
    # --------------------------------------------------------

    print("\nLoading LoRA adapter...")

    peft_model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        is_trainable=False,
    )

    peft_model.eval()

    # --------------------------------------------------------
    # Merge LoRA into base model
    # --------------------------------------------------------

    print("\nMerging LoRA adapter into the base model...")

    merged_model = peft_model.merge_and_unload(
        safe_merge=True
    )

    merged_model.eval()

    # --------------------------------------------------------
    # Save merged Hugging Face checkpoint
    # --------------------------------------------------------

    print("\nSaving merged model...")

    merged_dir.mkdir(parents=True, exist_ok=True)

    merged_model.save_pretrained(
        merged_dir,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(
        merged_dir
    )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    config_file = merged_dir / "config.json"
    safetensor_files = list(merged_dir.glob("*.safetensors"))

    if not config_file.exists():
        raise RuntimeError(
            f"Merge completed, but config.json was not found in:\n{merged_dir}"
        )

    if not safetensor_files:
        raise RuntimeError(
            f"Merge completed, but no .safetensors model file was found in:\n"
            f"{merged_dir}"
        )

    print("\nMerge completed successfully.")
    print(f"Merged Hugging Face model saved to:\n{merged_dir}")
    print(f"Safetensors files found: {len(safetensor_files)}")


if __name__ == "__main__":
    main()
