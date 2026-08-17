import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from training_config import (
    MODEL_NAME,
    MAX_SEQ_LENGTH,
    LEARNING_RATE,
    NUM_TRAIN_EPOCHS,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    PER_DEVICE_EVAL_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    WEIGHT_DECAY,
    WARMUP_RATIO,
    LOGGING_STEPS,
    SAVE_TOTAL_LIMIT,
    SEED,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["gptplus", "claude"],
        required=True,
        help="Dataset used for fine-tuning.",
    )

    return parser.parse_args()


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_dataset_paths(dataset_name):
    if dataset_name == "gptplus":
        data_dir = PROJECT_ROOT / "data_GptPlus" / "processed"

    elif dataset_name == "claude":
        data_dir = PROJECT_ROOT / "data_Claude" / "processed"

    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}"
        )

    return {
        "train": data_dir / "train_sft.jsonl",
        "validation": data_dir / "validation_sft.jsonl",
        "test": data_dir / "test_sft.jsonl",
    }


def get_output_dir(dataset_name):
    return (
        PROJECT_ROOT
        / "models"
        / f"lfm2_700m_{dataset_name}_lora"
    )


# ============================================================
# DATASET FORMATTING
# ============================================================

def format_example(example):
    """
    Convert prompt + completion into the textual sequence
    used by SFTTrainer.
    """

    return (
        f"{example['prompt']}\n\n"
        f"{example['completion']}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    dataset_paths = get_dataset_paths(args.dataset)
    output_dir = get_output_dir(args.dataset)

    print("=" * 60)
    print("LFM2-700M LoRA SUPERVISED FINE-TUNING")
    print("=" * 60)

    print(f"\nDataset source: {args.dataset}")
    print(f"Base model: {MODEL_NAME}")

    print("\nDataset files:")
    for split_name, path in dataset_paths.items():
        print(f"{split_name}: {path}")

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {path}"
            )

    print(f"\nOutput directory:\n{output_dir}")

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(dataset_paths["train"]),
            "validation": str(dataset_paths["validation"]),
            "test": str(dataset_paths["test"]),
        },
    )

    print("\nDataset sizes:")
    print(f"Train:      {len(dataset['train'])}")
    print(f"Validation: {len(dataset['validation'])}")
    print(f"Test:       {len(dataset['test'])}")

    # --------------------------------------------------------
    # Tokenizer
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
    # Model
    # --------------------------------------------------------

    print("\nLoading base model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        trust_remote_code=True,
    )

    model.config.use_cache = False

    # --------------------------------------------------------
    # LoRA
    # --------------------------------------------------------

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )

    # --------------------------------------------------------
    # SFT configuration
    # --------------------------------------------------------

    sft_config = SFTConfig(
        output_dir=str(output_dir),

        num_train_epochs=NUM_TRAIN_EPOCHS,

        per_device_train_batch_size=(
            PER_DEVICE_TRAIN_BATCH_SIZE
        ),

        per_device_eval_batch_size=(
            PER_DEVICE_EVAL_BATCH_SIZE
        ),

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        learning_rate=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,

        warmup_ratio=WARMUP_RATIO,

        logging_steps=LOGGING_STEPS,

        eval_strategy="epoch",

        save_strategy="epoch",

        save_total_limit=SAVE_TOTAL_LIMIT,

        # Keep track of validation loss and restore the best checkpoint
        # automatically when training finishes.
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        completion_only_loss=True,

        fp16=True,

        max_length=MAX_SEQ_LENGTH,

        seed=SEED,

        report_to="none",
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    trainer = SFTTrainer(
        model=model,

        args=sft_config,

        train_dataset=dataset["train"],

        eval_dataset=dataset["validation"],

        processing_class=tokenizer,

        peft_config=peft_config,

    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nStarting training...")

    trainer.train()

    # Because load_best_model_at_end=True, trainer.model now contains
    # the checkpoint with the lowest validation loss.
    best_checkpoint = trainer.state.best_model_checkpoint
    best_eval_loss = trainer.state.best_metric

    if best_checkpoint is None or best_eval_loss is None:
        raise RuntimeError(
            "Training finished without a best checkpoint/eval_loss. "
            "Check that evaluation and checkpoint saving ran correctly."
        )

    print("\nBest checkpoint selected:")
    print(f"Checkpoint: {best_checkpoint}")
    print(f"Best validation loss: {best_eval_loss:.6f}")

    # --------------------------------------------------------
    # Save BEST adapter
    # --------------------------------------------------------

    final_adapter_dir = (
        output_dir
        / "final_adapter"
    )

    print(
        f"\nSaving best LoRA adapter to:\n"
        f"{final_adapter_dir}"
    )

    trainer.model.save_pretrained(
        final_adapter_dir
    )

    tokenizer.save_pretrained(
        final_adapter_dir
    )

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()