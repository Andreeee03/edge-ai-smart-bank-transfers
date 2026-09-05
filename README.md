# Edge AI for Smart Bank Transfers

> Bachelor's thesis project exploring lightweight language-model deployment for bank-transfer description assistance on Android, with on-device inference through `llama.cpp`.

## Overview

**Edge AI for Smart Bank Transfers** is an experimental Android application that uses a compact, fine-tuned language model to help users generate, complete, and normalize bank-transfer descriptions.

The project covers the full pipeline, including synthetic dataset generation, supervised fine-tuning with LoRA, model evaluation, GGUF conversion and quantization, Android integration, and on-device benchmarking.

The final mobile application runs inference locally on the device. Internet access is required only to download the application and, on first launch, the quantized model artifact. Once the model has been stored locally, inference can be performed offline without a remote LLM inference service.

### Key Goals

- Keep bank-transfer information on the user's device during inference.
- Provide useful bank-transfer descriptions without relying on a cloud inference API.
- Support offline inference after the one-time model download.
- Evaluate the trade-off between model quality, model size, memory usage, and response time on mobile hardware.

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

1. August 2026: rent payment
2. August 2026 apartment rent settlement
```

### Description Completion

Complete a partially written bank-transfer description while preserving its intended meaning.

```text
Partial description:
August rent...

1. August 2026 rent payment
2. Apartment rent due August 2026
```

### Description Normalization

Rewrite an informal or poorly formatted description into clearer and more natural alternatives.

```text
Original description:
rent pay to mario

1. Rent payment due
2. Rent payment to Mario Rossi
```

### Calendar-Aware Generation

For the **Generation** task, optional calendar context can be used when calendar permission is granted by the user.

Calendar information is processed locally on the Android device and is not required for Completion or Normalization.

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

AWS EC2 is used only for computationally intensive model-preparation tasks. Once the quantized model has been downloaded and stored on the phone, all inference is executed locally through `llama.cpp`.

---

## Technology Stack

| Area | Technologies |
| --- | --- |
| Mobile application | Flutter, Dart |
| Android integration | Kotlin, JNI, C++ |
| On-device inference | llama.cpp, GGUF |
| Base model | LiquidAI LFM2-700M |
| Fine-tuning | Hugging Face Transformers, TRL, PEFT / LoRA, PyTorch |
| Training environment | Amazon EC2 GPU instance |
| Evaluation | ROUGE, BERTScore, task-specific consistency checks |

---

## Download and Installation

### Android APK

A pre-built Android APK is available through **GitHub Releases**:

**Releases:** https://github.com/Andreeee03/edge-ai-smart-bank-transfers/releases

Download the latest application release:

```text
EdgeAI-SmartBankTransfers-v1.0.0.apk
```

### Installation Steps

1. Open the repository **Releases** page.
2. Open the latest application release, currently `v1.0.0`.
3. Download `EdgeAI-SmartBankTransfers-v1.0.0.apk`.
4. Open the downloaded APK on the Android device.
5. If Android blocks the installation, allow installation from the browser, file manager, or application used to open the APK.
6. Complete the installation and launch the application.

> Android may display a warning because the APK is installed outside the Google Play Store. Only install the APK if it was downloaded from this repository's official Releases page.

### First Launch

On the first launch, the application automatically downloads the quantized language model:

```text
LFM2-700M_Claude-DS_Q5_K_M.gguf
```

The model is approximately **513 MiB**, so a stable internet connection is recommended for the initial setup.

During the download, the application displays the model-download progress. Once the download is complete, the model is stored in the application's private local storage and loaded through `llama.cpp`.

After this one-time setup, the application can perform AI inference without an internet connection.

### Subsequent Launches

When the model is already available locally:

```text
Application launch
        │
        ▼
Local model detected
        │
        ▼
Load GGUF with llama.cpp
        │
        ▼
On-device inference
```

The model is not downloaded again unless the application's local data is removed or the application is reinstalled.

---

## Model Distribution

The quantized GGUF model is intentionally **not committed to normal Git version control** because of its size.

The model is distributed through a dedicated GitHub Release:

```text
model-v1.0.0
└── LFM2-700M_Claude-DS_Q5_K_M.gguf
```

The Android application downloads this artifact automatically when the model is not yet available locally.

The current deployment flow is:

```text
GitHub application release
        │
        └── EdgeAI-SmartBankTransfers-v1.0.0.apk

GitHub model release
        │
        └── LFM2-700M_Claude-DS_Q5_K_M.gguf

First launch
        │
        └── Download model → save locally → load with llama.cpp

Subsequent launches
        │
        └── Reuse local model → on-device inference
```

Keeping the APK and model separate avoids embedding a large model file inside the application package and allows the model artifact to be distributed independently.

---

## Build from Source

### Requirements

- Flutter SDK
- Android Studio
- Android SDK
- Android NDK
- CMake
- Git
- A compatible Android device for deployment testing

Clone the repository:

```bash
git clone https://github.com/Andreeee03/edge-ai-smart-bank-transfers.git
cd edge-ai-smart-bank-transfers/mobile-app
```

Install Flutter dependencies:

```bash
flutter pub get
```

Check the development environment:

```bash
flutter doctor
```

Build a debug APK:

```bash
flutter build apk --debug
```

Build a release APK:

```bash
flutter build apk --release
```

The generated release APK is normally available at:

```text
build/app/outputs/flutter-apk/app-release.apk
```

The application automatically downloads the required GGUF model on first launch when the model is not already present in its private local storage.

---

## Privacy and Offline Inference

The project is designed around an Edge AI architecture.

- Bank-transfer information used for AI inference is processed locally on the Android device.
- Model execution is performed locally through `llama.cpp`.
- Calendar context, when enabled, is processed locally.
- No bank-transfer information is sent to a remote LLM inference endpoint.
- Internet access is required for the initial model download, but not for subsequent inference.

In practical terms:

```text
Internet
   │
   └── Initial GGUF download only

Bank-transfer data
   │
   ▼
Local GGUF model
   │
   ▼
llama.cpp
   │
   ▼
On-device result
```

If the device is offline after the model has been downloaded, Generation, Completion, and Normalization can still be performed locally.

---

## Repository Scope

The repository contains the source code and supporting experimental pipeline for the thesis project, including components for:

- Dataset preprocessing
- Fine-tuning and LoRA configuration
- LoRA merge
- Offline model evaluation
- GGUF conversion and quantization
- Quantization validation
- Flutter/Android application development
- Native `llama.cpp` integration
- Calendar-context integration
- On-device validation and benchmarking

Large generated artifacts such as intermediate model checkpoints and GGUF files are excluded from normal Git version control and distributed separately when needed.

---

## Releases

Current public artifacts are distributed through GitHub Releases.

### Application

```text
Tag: v1.0.0
Asset: EdgeAI-SmartBankTransfers-v1.0.0.apk
```

### Model

```text
Tag: model-v1.0.0
Asset: LFM2-700M_Claude-DS_Q5_K_M.gguf
```

---

## Project Status

The end-to-end prototype is operational on Android.

The current implementation includes:

- Fine-tuned LFM2-700M model
- Q5_K_M quantized GGUF deployment
- Flutter Android interface
- Native inference through Kotlin, JNI/C++, and `llama.cpp`
- Automatic first-run model download
- Download progress indication
- Offline inference after model setup
- Optional local calendar context
- On-device benchmarking

---

## Thesis

**Edge AI for Smart Bank Transfers** investigates whether a lightweight, task-specific language model can provide useful payment-description assistance directly on consumer mobile hardware while preserving local data processing and offline inference capabilities.
