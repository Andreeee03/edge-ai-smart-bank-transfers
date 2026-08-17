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
# CONSTANTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROMPT_COMPLETION_SEPARATOR = "\n\n"
IGNORE_INDEX = -100


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
# DATASET VALIDATION
# ============================================================

def validate_raw_split(dataset_split, split_name):
    required_columns = {"prompt", "completion"}
    missing_columns = required_columns - set(dataset_split.column_names)

    if missing_columns:
        raise ValueError(
            f"{split_name}: missing required columns: "
            f"{sorted(missing_columns)}"
        )

    for index, example in enumerate(dataset_split):
        prompt = example["prompt"]
        completion = example["completion"]

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"{split_name}: invalid prompt at index {index}."
            )

        if not isinstance(completion, str) or not completion.strip():
            raise ValueError(
                f"{split_name}: invalid completion at index {index}."
            )


# ============================================================
# PRE-TOKENIZATION
# ============================================================

def tokenize_prompt_completion(example, tokenizer):
    """
    Build the exact training sequence:

        prompt + "\\n\\n" + completion + EOS

    Prompt and completion are tokenized separately and then concatenated.

    This is deliberate:
    - the prompt is tokenized exactly as it will be available at inference;
    - the completion cannot change the tokenization of the final prompt tokens;
    - labels explicitly mask every prompt token with IGNORE_INDEX;
    - TRL does not need to infer the prompt/completion token boundary.
    """

    prompt_text = (
        example["prompt"].rstrip("\r\n")
        + PROMPT_COMPLETION_SEPARATOR
    )

    completion_text = example["completion"]

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]

    completion_ids = tokenizer(
        completion_text,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    if tokenizer.eos_token_id is None:
        raise ValueError(
            "Tokenizer has no eos_token_id; cannot terminate completions safely."
        )

    # TRL normally appends EOS to non-conversational completions.
    # Do it explicitly because the dataset is pre-tokenized here.
    if not completion_ids or completion_ids[-1] != tokenizer.eos_token_id:
        completion_ids = completion_ids + [tokenizer.eos_token_id]

    input_ids = prompt_ids + completion_ids

    labels = (
        [IGNORE_INDEX] * len(prompt_ids)
        + completion_ids.copy()
    )

    return {
        "input_ids": input_ids,
        "labels": labels,
        "sequence_length": len(input_ids),
        "prompt_token_length": len(prompt_ids),
        "completion_token_length": len(completion_ids),
    }


def percentile(values, q):
    if not values:
        raise ValueError("Cannot compute percentile of an empty sequence.")

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower

    return (
        ordered[lower] * (1 - fraction)
        + ordered[upper] * fraction
    )


def print_and_validate_token_stats(tokenized_split, split_name):
    sequence_lengths = tokenized_split["sequence_length"]
    prompt_lengths = tokenized_split["prompt_token_length"]
    completion_lengths = tokenized_split["completion_token_length"]

    over_limit = [
        length
        for length in sequence_lengths
        if length > MAX_SEQ_LENGTH
    ]

    print(f"\n{split_name} token-length statistics:")
    print(f"  examples:          {len(sequence_lengths)}")
    print(f"  min:               {min(sequence_lengths)}")
    print(f"  median:            {percentile(sequence_lengths, 0.50):.1f}")
    print(f"  p95:               {percentile(sequence_lengths, 0.95):.1f}")
    print(f"  max:               {max(sequence_lengths)}")
    print(f"  max prompt:        {max(prompt_lengths)}")
    print(f"  max completion:    {max(completion_lengths)}")
    print(
        f"  > MAX_SEQ_LENGTH:  {len(over_limit)} "
        f"(limit={MAX_SEQ_LENGTH})"
    )

    if any(length <= 0 for length in completion_lengths):
        raise RuntimeError(
            f"{split_name}: at least one example has no completion tokens."
        )

    # Do not silently truncate targets. If the configured maximum is too
    # small, fail before loading the model so the configuration can be fixed.
    if over_limit:
        raise RuntimeError(
            f"{split_name}: {len(over_limit)} examples exceed "
            f"MAX_SEQ_LENGTH={MAX_SEQ_LENGTH}. "
            f"Maximum observed length: {max(sequence_lengths)}. "
            "Increase MAX_SEQ_LENGTH before training."
        )


def prepare_tokenized_datasets(dataset, tokenizer):
    tokenized = {}

    for split_name in ("train", "validation", "test"):
        validate_raw_split(
            dataset[split_name],
            split_name,
        )

        tokenized_split = dataset[split_name].map(
            tokenize_prompt_completion,
            fn_kwargs={"tokenizer": tokenizer},
            remove_columns=dataset[split_name].column_names,
            load_from_cache_file=False,
            desc=f"Pre-tokenizing {split_name}",
        )

        print_and_validate_token_stats(
            tokenized_split,
            split_name,
        )

        # Keep only the tensors required by the Trainer.
        # In particular, no prompt/completion columns are passed to TRL:
        # completion-only supervision is already encoded in `labels`.
        tokenized[split_name] = tokenized_split.select_columns(
            ["input_ids", "labels"]
        )

    return tokenized


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    dataset_paths = get_dataset_paths(args.dataset)
    output_dir = get_output_dir(args.dataset)

    print("=" * 72)
    print("LFM2-700M LoRA SUPERVISED FINE-TUNING")
    print("=" * 72)

    print(f"\nDataset source: {args.dataset}")
    print(f"Base model: {MODEL_NAME}")
    print(f"Max sequence length: {MAX_SEQ_LENGTH}")

    print("\nDataset files:")
    for split_name, path in dataset_paths.items():
        print(f"{split_name}: {path}")

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {path}"
            )

    print(f"\nOutput directory:\n{output_dir}")

    # --------------------------------------------------------
    # Load raw datasets
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

    if tokenizer.bos_token_id is None:
        raise ValueError(
            "Tokenizer has no bos_token_id."
        )

    if tokenizer.eos_token_id is None:
        raise ValueError(
            "Tokenizer has no eos_token_id."
        )

    print(
        f"BOS token: {tokenizer.bos_token!r} "
        f"(id={tokenizer.bos_token_id})"
    )
    print(
        f"EOS token: {tokenizer.eos_token!r} "
        f"(id={tokenizer.eos_token_id})"
    )
    print(
        f"PAD token: {tokenizer.pad_token!r} "
        f"(id={tokenizer.pad_token_id})"
    )

    # --------------------------------------------------------
    # Pre-tokenize + explicit completion-only labels
    # --------------------------------------------------------

    print(
        "\nPre-tokenizing prompt-completion datasets "
        "with explicit completion-only labels..."
    )

    tokenized_dataset = prepare_tokenized_datasets(
        dataset,
        tokenizer,
    )

    first_example = tokenized_dataset["train"][0]

    if len(first_example["input_ids"]) != len(first_example["labels"]):
        raise RuntimeError(
            "Pre-tokenization validation failed: input_ids/labels "
            "length mismatch."
        )

    if not any(label != IGNORE_INDEX for label in first_example["labels"]):
        raise RuntimeError(
            "Pre-tokenization validation failed: first example has "
            "no trainable completion labels."
        )

    print(
        "\nDataset pre-tokenization OK. "
        "Prompt tokens are masked with -100; "
        "only completion tokens contribute to the loss."
    )

    # --------------------------------------------------------
    # CUDA check
    # --------------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Training must run on the GPU."
        )

    print("\nCUDA:")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA runtime: {torch.version.cuda}")

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

    print("\nLoRA configuration:")
    print(f"  r: {LORA_R}")
    print(f"  alpha: {LORA_ALPHA}")
    print(f"  dropout: {LORA_DROPOUT}")
    print(f"  target modules: {LORA_TARGET_MODULES}")

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

        # Restore the checkpoint with the lowest validation loss.
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # IMPORTANT:
        # completion-only supervision is already encoded directly in
        # `labels` (-100 for every prompt token). Therefore TRL must not
        # try to infer a prompt/completion boundary again.
        completion_only_loss=False,

        fp16=True,

        # The dataset is already pre-tokenized and validated against
        # MAX_SEQ_LENGTH. Keep the same limit as an additional safeguard.
        max_length=MAX_SEQ_LENGTH,

        seed=SEED,

        report_to="none",

        # Prevent TRL from re-tokenizing prompt/completion pairs.
        dataset_kwargs={
            "skip_prepare_dataset": True,
        },
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
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