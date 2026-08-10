# Edge AI for Smart Bank Transfers

> Bachelor's thesis project focused on developing an on-device AI assistant for generating and improving bank transfer descriptions using a fine-tuned lightweight language model.

## Overview

This project explores the use of **Edge AI and lightweight Large Language Models (LLMs)** to assist users while filling in bank transfer information.

The main goal is to fine-tune a compact language model on a custom dataset of bank transfer descriptions and deploy the resulting model directly on an **Android mobile device**.

Unlike cloud-based AI solutions, inference is performed locally on the device, with the aim of improving:

- Privacy
- Offline availability
- Response latency
- Independence from external cloud services

The project covers the complete pipeline, from dataset preparation and model fine-tuning to model optimization, quantization, and deployment inside a Flutter application.

---

## Main Use Cases

The AI assistant is designed to support different bank transfer description tasks.

### Description Generation

Generate a bank transfer description from structured information such as:

- Beneficiary
- Amount
- Operation category
- Reference period

Example:

```text
Category: Rent
Period: August 2026
Beneficiary: Mario Rossi

→ August 2026 rent payment
```

### Description Completion

Complete a partially written bank transfer description.

```text
Input:
August rent...

→ August 2026 rent payment
```

### Description Normalization

Rewrite an informal or poorly formatted description into a clearer and more standardized one.

```text
Input:
rent august mario

→ August rent payment - Mario Rossi
```

### Alternative Generation

Generate multiple valid bank transfer descriptions for the same transaction, allowing the user to choose the preferred one.

### Calendar-Based Generation

Optionally use locally available calendar information to provide additional context when generating a bank transfer description.

For example, a calendar event related to rent, tuition, subscriptions, or scheduled payments could help the model generate a more contextualized description.

All calendar information is intended to remain on the user's device.

---

## Project Architecture

The project is divided into three main environments:

```text
DATA PREPARATION
        │
        ▼
Dataset creation and preprocessing
        │
        ├── Cleaning
        ├── Normalization
        ├── Train / Validation / Test split
        └── Prompt-completion generation
        │
        ▼
CLOUD TRAINING ENVIRONMENT
        │
        ▼
AWS EC2 GPU Instance
        │
        ├── Hugging Face Transformers
        ├── TRL SFTTrainer
        ├── PEFT / LoRA
        ├── PyTorch
        └── Model validation
        │
        ▼
LoRA Adapter
        │
        ▼
Merge with Base Model
        │
        ▼
Fine-Tuned Model
        │
        ▼
Final Test Set Evaluation
        │
        ▼
EDGE OPTIMIZATION
        │
        ├── Hugging Face → GGUF conversion
        ├── Model quantization
        └── llama.cpp compatibility testing
        │
        ▼
ANDROID DEVICE
        │
        ├── Flutter UI
        ├── Dart
        ├── llama.cpp
        └── On-device inference
```

The cloud environment is used only for computationally intensive operations such as fine-tuning and model preparation.

The final inference process is executed entirely on the Android device.
