# Legacy post-quantization validation

This directory contains the post-quantization experiments performed before
the definitive HF / GGUF F16 / Q4_K_M / Q5_K_M validation.

## 01_preliminary_f_tests

Contains the first F16 / Q4_K_M / Q5_K_M post-quantization comparisons.

These experiments used `llama-completion -f` with prompts written to temporary
files. Subsequent controlled tests showed that the `-f` input path removed one
trailing newline from the prompt.

The intended SFT inference boundary was:

    \n\n

while the model effectively received:

    \n

Therefore these results are retained for traceability and diagnostic purposes,
but MUST NOT be used as the final quantitative comparison between F16, Q4_K_M
and Q5_K_M.

## 02_q5_server_diagnostics

Contains tests performed with llama-server to investigate the anomalous Q5
behavior observed during the preliminary tests.

These experiments helped demonstrate that Q5 itself was functional when the
correct prompt boundary was supplied directly.

## 03_q5_completion_diagnostic

Contains completion-only and prompt-boundary diagnostics used to isolate the
difference between file-based and direct prompt delivery.

## 04_valid_q5_vs_hf_check

Contains the later Q5_K_M versus Hugging Face merged-model comparison performed
with the corrected inference protocol.

This experiment is VALID and demonstrated Q5 stability on the complete
500-example test set. It is archived here because the definitive experiment
will compare all four representations under identical conditions:

- Hugging Face merged
- GGUF F16
- GGUF Q4_K_M
- GGUF Q5_K_M

## Final protocol

The definitive quantization comparison must use:

- the same 500-example GPTPlus test set;
- exactly the same logical prompt;
- prompt ending with exactly `\n\n`;
- direct prompt delivery with `llama-completion -p` for GGUF models;
- no conversation/chat-template transformation;
- deterministic greedy decoding;
- identical generation limits and inference parameters.

The definitive outputs will be stored in:

    evaluation/final_quantization_validation/
