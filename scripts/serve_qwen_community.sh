exec ~/flint/llama.cpp/build/bin/llama-server -m $HOME/.cache/huggingface/hub/models--williamliao--Qwen3.6-35B-A3B-NVFP4-GGUF/snapshots/78bed75d118ebae682749827f1fa42a99d0095e2/Qwen3.6-35B-A3B-NVFP4.gguf \
  --alias qwen3.6-35b-a3b -ngl all -c 32768 -np 1 \
  -fa on -ctk f16 -ctv f16 --jinja \
  --temp 1 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 1.5 --repeat-penalty 1.0 \
  --host 127.0.0.1 --port 8081
