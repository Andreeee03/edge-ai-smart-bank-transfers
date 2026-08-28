# Final quantization validation

This directory is reserved for the definitive comparison between:

1. Hugging Face merged LFM2-700M GPTPlus model
2. GGUF F16
3. GGUF Q4_K_M
4. GGUF Q5_K_M

All models must be evaluated on the same 500-example GPTPlus test set with the
correct SFT prompt boundary.

## Required inference protocol

HF:
- Transformers
- greedy decoding
- do_sample=False
- max_new_tokens=100
- repetition_penalty=1.0

GGUF:
- llama-completion
- prompt supplied directly with `-p`
- prompt ending with exactly `\n\n`
- `--no-conversation`
- temperature 0
- top-k 1
- max tokens 100
- repetition penalty 1.0

The purpose of this experiment is to separately quantify:

- HF -> GGUF F16 conversion effect
- F16 -> Q4_K_M quantization effect
- F16 -> Q5_K_M quantization effect
- Q4_K_M vs Q5_K_M quality/size trade-off
