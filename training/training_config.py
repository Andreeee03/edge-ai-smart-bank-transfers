# ============================================================
# BASE MODEL
# ============================================================

MODEL_NAME = "LiquidAI/LFM2-700M"


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42


# ============================================================
# SEQUENCE LENGTH
# ============================================================

# The project prompts are short.
# 256 tokens should normally be sufficient, but train_lora.py
# checks the real token-length distribution before training.
MAX_SEQ_LENGTH = 256


# ============================================================
# TRAINING
# ============================================================

NUM_TRAIN_EPOCHS = 5

PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 4

GRADIENT_ACCUMULATION_STEPS = 4

LEARNING_RATE = 2e-4

WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.05

LOGGING_STEPS = 10

SAVE_TOTAL_LIMIT = 2


# ============================================================
# LoRA
# ============================================================

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "self_attn.out_proj",
]