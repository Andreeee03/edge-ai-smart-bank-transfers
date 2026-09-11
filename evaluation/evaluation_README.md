# Evaluation

This directory contains the evaluation, validation, and deployment-benchmark artifacts used throughout the project.

The structure distinguishes the main model-evaluation stages from auxiliary diagnostic checks. Intermediate runs known to be invalid, bug-affected, or superseded by the final evaluation pipeline are not retained in the final public repository.

## Evaluation Workflow

The evaluation process is organised into four main stages:

1. Base-model evaluation
2. Fine-tuned model comparison
3. Final quantization validation
4. On-device deployment benchmark

Two additional auxiliary checks are retained because they provide useful technical validation of the inference and conversion pipelines.

---

## 1. Base-Model Evaluation

### Script

```text
evaluate_base_native_chat.py
```

### Results

```text
results_base_native_chat/
```

This stage evaluates the original `LFM2-700M` base model before task-specific fine-tuning using its native Hugging Face chat template and a fixed few-shot prompting configuration.

Four manually constructed demonstrations are provided before each test prompt: Generation without calendar context, Generation with calendar context, Completion, and Normalization. The demonstrations are independent of both evaluation test sets and remain identical across the GPTPlus and Claude evaluations. Their purpose is to provide the base model with explicit examples of the supported tasks and of the expected two-alternative output format without modifying the model weights.

The few-shot baseline is evaluated on the same GPTPlus and Claude test sets used for the fine-tuned models. Inference uses greedy decoding, batch size 1, no padding and no truncation, consistently with the final evaluation protocol. The result directory contains evaluation metadata, aggregate and per-activity metrics, qualitative review cases, and model predictions for both test sets.

---

## 2. Fine-Tuned Model Evaluation

### Script

```text
evaluate_model.py
```

### Results

```text
results_sft_prompt_final/
```

This stage contains the final evaluation of the two task-specific fine-tuned models:

```text
LFM2-700M_GPTPlus-DS
LFM2-700M_Claude-DS
```

Both models are evaluated on the GPTPlus and Claude test sets to analyse in-domain performance, cross-dataset generalisation, task-specific consistency, output diversity, and qualitative behaviour.

The results from this stage were used for the final deployment-model decision.

`LFM2-700M_Claude-DS` was selected for mobile deployment because it provided the most suitable overall behaviour for the intended application, combining stronger cross-dataset retention with stable structured output and greater diversity between the generated alternatives.

---

## 3. Final Quantization Validation

### Validation Script

```text
final_quantization_validation/
└── run_final_validation_claude.py
```

### Final Results

```text
final_quantization_validation_claude/
```

The selected `LFM2-700M_Claude-DS` model was validated across four representations:

```text
Hugging Face merged model
GGUF F16
GGUF Q4_K_M
GGUF Q5_K_M
```

The validation compares model behaviour before and after conversion and quantization using the same 500-example Claude test set and a controlled inference protocol.

The result directory contains predictions for all four model representations, reference-based metrics, pairwise similarity metrics, exact-match statistics, structured-output consistency checks, and final summary files.

The validation showed that the F16 GGUF conversion preserved the original fine-tuned behaviour almost exactly. Both Q4_K_M and Q5_K_M substantially reduced the model size. Their reference-based task quality was very similar, while Q5_K_M remained closer to the original fine-tuned checkpoint and preserved the expected two-alternative output structure.

Q5_K_M was therefore provisionally selected for mobile integration. This decision was made before on-device measurements and was not intended to establish Q5_K_M as the optimal mobile trade-off relative to Q4_K_M. Its practical suitability was subsequently evaluated on the target smartphone through the deployment benchmark described below.

The deployed artifact used for the mobile benchmark is:

    LFM2-700M_Claude-DS_Q5_K_M.gguf

The quantized model itself is distributed separately through GitHub Releases and is not committed to the normal Git repository.

---

## 4. On-Device Deployment Benchmark

### Benchmark Entry Point

    ../mobile-app/lib/benchmark_main.dart

### Scripts

    deployment_benchmark/
    ├── run_final_benchmark.ps1
    ├── run_model_load_benchmark.ps1
    ├── analyze_benchmark.py
    └── analyze_sustained_load.py

### Results

    deployment_benchmark/

The final runtime evaluation was performed on a Samsung SM-G780F smartphone with an Exynos 990 SoC, Android 13, and approximately 5.33 GiB of RAM. The Q5_K_M model was stored locally on the device and executed through the native `llama.cpp` integration.

The main sustained-load benchmark used six fixed prompts repeated over 17 cycles, for a total of 102 inference runs. All 102 runs completed successfully.

The final measurements were:

- median TTFT: 632.644 ms
- P95 TTFT: 783.297 ms
- median total inference latency: 1078.295 ms
- P95 total inference latency: 1271.071 ms
- mean generation throughput: 35.397 tokens/s
- peak resident memory: 1.379 GiB
- battery temperature: 33.6 °C to 38.9 °C
- maximum Android thermal status: 3

The sustained-load analysis compared the first three and last three cycles. Mean total inference latency increased by 1.45%, while generation throughput decreased by 3.69%. Mean TTFT changed by only 0.18%. The results therefore show a limited reduction in decoding performance under prolonged execution without inference failures or substantial latency degradation.

Model initialization was evaluated separately. The first model load after a complete device reboot required 2931.458 ms. Five subsequent application-process restarts produced a mean warm-load time of 2065.220 ms and a median of 2005.264 ms, corresponding to a 29.55% reduction relative to the post-reboot measurement.

Because Android filesystem caches were not explicitly cleared, these measurements are described as post-reboot and warm process-start conditions rather than strictly controlled cold- and warm-cache states.

Offline operation was explicitly tested under Airplane mode. The application completed 30 out of 30 inference runs successfully. Android reported `airplane_mode_on = 1` and `mDataConnectionState = 0`. Wi-Fi was disabled during inference and re-enabled only after completion to retrieve the execution logs through ADB.

The deployment benchmark evaluates TTFT, total inference latency, throughput, memory consumption, initialization time, sustained-load thermal behaviour, inference stability, and offline execution. It does not include the one-time model-download phase.

Because Q4_K_M was not benchmarked on-device, these measurements establish the practical suitability of the selected Q5_K_M configuration but do not demonstrate that it represents the optimal mobile trade-off between model size and runtime performance.

---

# Auxiliary Validation

## Inference Audit

### Script

```text
audit_inference.py
```

### Results

```text
audit_results/
```

This diagnostic stage audits inference outputs and associated consistency checks.

The output path is intentionally kept unchanged because it is referenced directly by the audit script.

These artifacts support validation of the evaluation pipeline but are not treated as a separate main model-comparison stage.

---

## Conversion Fidelity Check

```text
conversion_fidelity_check/
```

This directory contains artifacts used to verify behavioural fidelity during conversion from the merged Hugging Face model to the F16 GGUF representation.

The check compares outputs and token-level artifacts generated by the original PyTorch/Hugging Face model and the corresponding `llama.cpp` representation.

This supports the conclusion that the conversion step itself introduces negligible behavioural change before quantization.

---

## Final Directory Structure

    evaluation/
    ├── audit_inference.py
    ├── audit_results/
    │
    ├── evaluate_base_native_chat.py
    ├── results_base_native_chat/
    │
    ├── evaluate_model.py
    ├── results_sft_prompt_final/
    │
    ├── conversion_fidelity_check/
    │
    ├── final_quantization_validation/
    │   └── run_final_validation_claude.py
    │
    ├── final_quantization_validation_claude/
    │
    ├── deployment_benchmark/
    │   ├── README.md
    │   ├── analyze_benchmark.py
    │   ├── analyze_sustained_load.py
    │   ├── device_info.json
    │   ├── initialization.json
    │   ├── model_load_benchmark.csv
    │   ├── offline_test_summary.txt
    │   ├── raw/
    │   ├── run_final_benchmark.ps1
    │   ├── run_model_load_benchmark.ps1
    │   ├── runs.csv
    │   ├── summary.json
    │   ├── sustained_load_analysis.json
    │   └── thermal_battery.csv
    │
    └── evaluation_README.md

---

## Reproducibility

The final public repository retains the scripts and result artifacts required to understand and reproduce the evaluation path used for the thesis.

Known invalid, bug-affected, and superseded intermediate runs were removed from the final repository to avoid ambiguity between diagnostic experiments and the definitive results.

Existing paths used by the retained scripts have been preserved wherever necessary to avoid breaking reproducibility.
