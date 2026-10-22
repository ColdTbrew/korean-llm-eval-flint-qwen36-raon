#!/bin/bash
cd ~/flint/llama.cpp
exec systemd-run --user --scope -p MemoryMax=45G -p MemorySwapMax=0 \
  ~/flint/venv-convert/bin/python convert_hf_to_gguf.py ~/llm/models/Qwen3.6-35B-A3B-NVFP4 \
  --outtype bf16 --outfile ~/llm/gguf/Qwen3.6-35B-A3B-NVFP4-official-bf16rest.gguf
