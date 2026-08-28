# HF to GGUF F16 conversion fidelity check

This directory contains the artifacts generated during the numerical
verification of the conversion from the merged Hugging Face
LFM2-700M GPTPlus model to GGUF F16.

The Hugging Face/PyTorch and llama.cpp implementations were evaluated
using the same prompt and token sequence.

The stored artifacts include:

- input prompts;
- token sequences;
- binary numerical outputs;
- textual numerical outputs.

These files were used to verify that the HF -> GGUF F16 conversion
preserved the numerical behavior of the merged model before
quantization.

Heavy model weights are intentionally excluded from Git.
