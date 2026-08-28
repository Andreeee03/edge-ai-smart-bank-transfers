# GPTPlus 3-Epoch Legacy Training Run

This directory preserves the metadata and training results of the first
successful GPTPlus LoRA fine-tuning run performed with 3 epochs.

The run completed successfully and selected checkpoint-750, corresponding
to the third epoch, as its best checkpoint.

It was subsequently superseded by the definitive 5-epoch GPTPlus training
run. The 3-epoch adapter was therefore not used for model merging,
final evaluation, GGUF conversion, quantization, or deployment.

For experimental traceability, this directory preserves:

- the original run README;
- trainer state after checkpoint-500;
- final trainer state at checkpoint-750;
- the LoRA adapter configuration.

Heavy and reproducible training artifacts are intentionally not preserved
in Git, including:

- adapter_model.safetensors;
- optimizer.pt;
- rng_state.pth;
- scaler.pt;
- scheduler.pt;
- duplicated tokenizer files.

The definitive GPTPlus model artifacts are stored separately under
models/lfm2_700m_gptplus_lora/.
