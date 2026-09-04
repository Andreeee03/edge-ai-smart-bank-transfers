# Edge AI for Smart Bank Transfers

> Bachelor's thesis project exploring fully on-device language-model inference for bank-transfer description assistance on Android.

## Overview

**Edge AI for Smart Bank Transfers** is an experimental mobile application that uses a compact, fine-tuned language model to help users generate, complete, and normalize bank-transfer descriptions.

The project focuses on running inference directly on an **Android device** rather than sending transaction information to a remote inference service. The complete workflow covers synthetic dataset preparation, supervised fine-tuning with LoRA, model evaluation, GGUF conversion and quantization, and native Android deployment through `llama.cpp`.

### Key goals

- Keep bank-transfer information on the user's device during inference.
- Provide useful bank-transfer descriptions without requiring a cloud inference API.
- Support fully offline inference after the model has been downloaded to the device.
- Evaluate the trade-off between model quality, size, memory usage, and response time on mobile hardware.

---

## Features

### Description Generation

Generate two concise bank-transfer descriptions from structured transaction information such as:

- Beneficiary
- Amount and currency
- Operation category
- Reference period

Example:

```text
Category: Rent
Beneficiary: Mario Rossi
Amount: 850 EUR
Reference period: August 2026

1. August 2026 rent payment to Mario Rossi
2. Rent payment for August 2026
```

### Description Completion

Complete a partially written description while preserving its intended meaning.

```text
Partial description:
August rent...

1. August rent payment to Mario Rossi
2. August rent payment for the apartment
```

### Description Normalization

Rewrite an informal or poorly formatted description into clearer and more natural alternatives.

```text
Original description:
rent august mario

1. August rent payment to Mario Rossi
2. Rent payment to Mario Rossi for August
```

### Calendar-Aware Generation

For the **Generation** task, optional calendar context can be used when permission is granted by the user. Calendar information is processed locally and is not required for the other tasks.

---

## Project Architecture

```text
DATA PREPARATION
        │
        ▼
Synthetic dataset generation and preprocessing
        │
        ├── Validation and cleaning
        ├── Train / validation / test split
        └── SFT prompt-completion construction
        │
        ▼
MODEL PREPARATION — AWS EC2
        │
        ├── Hugging Face Transformers
        ├── TRL SFTTrainer
        ├── PEFT / LoRA
        └── PyTorch
        │
        ▼
LoRA adapter
        │
        ▼
Merge with LFM2-700M
        │
        ▼
Merged Hugging Face checkpoint
        │
        ▼
Offline evaluation
        │
        ▼
EDGE OPTIMIZATION
        │
        ├── Hugging Face → GGUF F16
        ├── GGUF quantization
        └── llama.cpp validation
        │
        ▼
ANDROID APPLICATION
        │
        ▼
Flutter UI
        │
        ▼
MethodChannel
        │
        ▼
Kotlin
        │
        ▼
JNI / C++
        │
        ▼
llama.cpp + GGUF
        │
        ▼
Fully on-device inference
```

AWS EC2 is used only for computationally intensive model-preparation tasks. Once the quantized model is available on the phone, inference is executed locally through `llama.cpp`.

---

## Technology Stack

| Area | Technologies |
| --- | --- |
| Mobile application | Flutter, Dart |
| Android integration | Kotlin, JNI, C++ |
| On-device inference | llama.cpp, GGUF |
| Model | LiquidAI LFM2-700M |
| Fine-tuning | Hugging Face Transformers, TRL, PEFT / LoRA, PyTorch |
| Training environment | Amazon EC2 GPU instance |
| Evaluation | ROUGE, BERTScore, task-specific consistency checks |

---

## Download the Android App

A pre-built Android APK is intended to be distributed through **GitHub Releases**:

**Releases:** https://github.com/Andreeeee03/edge-ai-smart-bank-transfers/releases

For the public release, the APK and the language-model file are kept separate so that the application package remains lightweight.

### Installation flow

1. Download the latest APK from the **Releases** page.
2. Install the APK on a compatible Android device.
3. Launch the application.
4. On first setup, download the required GGUF model to the device.
5. After the model is stored locally, inference can run without a cloud inference service.

> The public APK/model download workflow is being prepared for the final project release. Until a release is published, build the application from source using the instructions below.

---

## Model Distribution

The GGUF model is intentionally **not committed to the Git repository** because of its size.

The release architecture is designed to keep the model separate from the APK:

```text
GitHub Release
    ├── Android APK
    └── Quantized GGUF model

First application setup
    └── Download model → store locally → load with llama.cpp

Subsequent use
    └── Local model → local inference → no cloud inference API
```

This approach makes it possible to update the model independently from the application source code while keeping runtime inference on-device.

---

## Build from Source

### Requirements

- Flutter SDK
- Android Studio
- Android SDK
- Android NDK
- CMake
- Git
- A compatible Android device

Clone the repository:

```bash
git clone https://github.com/Andreeeee03/edge-ai-smart-bank-transfers.git
cd edge-ai-smart-bank-transfers
```

Install Flutter dependencies:

```bash
flutter pub get
```

Check the development environment:

```bash
flutter doctor
```

Build a release APK:

```bash
flutter build apk --release
```

The generated APK is normally available at:

```text
build/app/outputs/flutter-apk/app-release.apk
```

The GGUF model must be available in the location expected by the mobile application before native inference can start. The final public release will automate the model acquisition step.

---

## Privacy and Offline Inference

The project is designed around an Edge AI architecture:

- Transaction data used for inference is processed locally.
- Model execution is performed on the Android device through `llama.cpp`.
- Calendar context, when enabled, is processed locally.
- No remote LLM inference endpoint is required after the model has been downloaded.

An internet connection may therefore be required for the **initial application/model download**, but not for subsequent model inference.

---

## Repository Scope

The repository contains the source code and supporting experimental pipeline for the thesis project, including components for:

- Dataset preprocessing
- Fine-tuning and LoRA configuration
- LoRA merge
- Model evaluation
- GGUF conversion and quantization
- Flutter/Android application development
- Native `llama.cpp` integration
- On-device validation and benchmarking

Large generated artifacts such as model checkpoints and GGUF files are excluded from normal Git version control.

---

## Research Project Status

This repository accompanies a Bachelor's thesis project. The experimental pipeline and Android prototype are under final validation and documentation before the first public release.

Final release preparation will include:

- Repository cleanup
- Release APK
- Separate GGUF model asset
- Automated first-run model acquisition
- Final device benchmarks
- Updated screenshots and usage documentation

---

## Thesis

**Edge AI for Smart Bank Transfers** investigates whether a lightweight, task-specific language model can provide useful payment-description assistance directly on consumer mobile hardware while preserving local data processing and offline inference capabilities.

