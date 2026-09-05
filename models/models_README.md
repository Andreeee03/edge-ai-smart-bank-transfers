# Models

This directory contains the model configurations, tokenizer files, LoRA training artifacts, and merged-model metadata produced during the project.

Large model-weight files are intentionally not committed to the normal Git repository. The final deployment model is distributed separately through GitHub Releases.

## Model Variants

Two task-specific fine-tuned variants were produced from the `LFM2-700M` base model:

```text
LFM2-700M_GPTPlus-DS
LFM2-700M_Claude-DS
```

The two variants were trained on synthetic datasets generated from different sources and were later compared during the final evaluation stage.

`LFM2-700M_Claude-DS` was selected as the final deployment model.

---

## Directory Structure

```text
models/
├── lfm2_700m_claude_lora/
│   ├── checkpoint-1000/
│   ├── checkpoint-1250/
│   └── final_adapter/
│
├── lfm2_700m_claude_merged/
│
├── lfm2_700m_gptplus_lora/
│   ├── checkpoint-1000/
│   ├── checkpoint-1250/
│   └── final_adapter/
│
├── lfm2_700m_gptplus_merged/
│
└── README.md
```

The exact contents of the checkpoint and merged-model directories may include configuration, tokenizer, chat-template, trainer-state, and generation metadata required to document the training and merge process.

---

## LoRA Training Artifacts

The following directories contain artifacts produced during supervised fine-tuning with LoRA:

```text
lfm2_700m_claude_lora/
lfm2_700m_gptplus_lora/
```

Each variant contains intermediate checkpoints and a final adapter directory.

Typical retained files include:

- `adapter_config.json`
- `chat_template.jinja`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `trainer_state.json`
- `training_args.bin`

These files document the configuration and tokenizer state associated with the fine-tuning runs.

Large adapter-weight files may be excluded from normal Git version control when their size makes repository storage inappropriate.

---

## Merged Model Metadata

The following directories correspond to the merged Hugging Face representations obtained after combining the LoRA adapters with the base `LFM2-700M` model:

```text
lfm2_700m_claude_merged/
lfm2_700m_gptplus_merged/
```

The repository retains lightweight metadata and tokenizer/configuration files such as:

- `config.json`
- `generation_config.json`
- `chat_template.jinja`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`

The full merged model weights are not stored in the normal Git repository.

---

## Final Deployment Model

The final model selected for mobile deployment is:

```text
LFM2-700M_Claude-DS
```

After merging, the model was converted to GGUF and evaluated in F16, Q4_K_M, and Q5_K_M forms.

The final deployment artifact is:

```text
LFM2-700M_Claude-DS_Q5_K_M.gguf
```

This file is distributed separately through the dedicated GitHub model release:

```text
Tag: model-v1.0.0
Asset: LFM2-700M_Claude-DS_Q5_K_M.gguf
```

The Android application downloads this model automatically on first launch and stores it in the application's private local storage.

---

## Why Large Model Files Are Not Stored Here

Model checkpoints and GGUF artifacts can be hundreds of megabytes or larger.

Keeping them outside normal Git version control:

- avoids unnecessarily increasing repository size;
- keeps cloning and version-control operations lightweight;
- separates source code and metadata from deployable binary artifacts;
- allows the final model to be distributed independently through GitHub Releases.

The repository therefore focuses on configuration, metadata, scripts, and reproducibility-related artifacts, while large binary model files are handled separately.

---

## Relationship to the Evaluation Pipeline

The model variants documented in this directory are evaluated under:

```text
evaluation/
```

The relevant final evaluation stages include:

```text
results_base_native_chat/
results_sft_prompt_final/
final_quantization_validation_claude/
deployment_benchmark/
```

The final model-selection decision and the final quantization choice are documented by the evaluation results rather than by the contents of this directory alone.

---

## Reproducibility

The retained files provide the configuration and tokenizer context needed to understand how the fine-tuned and merged model variants were produced.

The full pipeline is distributed across the repository:

```text
preprocessing/
training/
post_training/
conversion/
evaluation/
models/
```

Existing directory names and paths have been preserved where they are part of the project workflow, so that the repository remains consistent with the scripts and experimental records used during development.
