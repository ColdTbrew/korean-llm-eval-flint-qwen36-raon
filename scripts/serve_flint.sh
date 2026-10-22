exec ~/flint/llama.cpp/build/bin/llama-server -m $HOME/.cache/huggingface/hub/models--RecursiveSI--flint-nvfp4-gguf/snapshots/6ec43e19b565fc8d560e2d04985d7724f20673bf/flint-nvfp4.gguf \
  --alias flint -ngl all -c 32768 -np 1 \
  -fa on -ctk f16 -ctv f16 --jinja \
  --temp 1 --top-p 0.95 --top-k 20 --min-p 0 \
  --host 127.0.0.1 --port 8080
