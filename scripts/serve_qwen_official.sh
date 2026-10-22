exec ~/flint/llama.cpp/build/bin/llama-server -m $HOME/llm/gguf/Qwen3.6-35B-A3B-NVFP4-official-bf16rest.gguf \
  --alias qwen3.6-35b-a3b-official -ngl all -c 32768 -np 1 \
  -fa on -ctk f16 -ctv f16 --jinja \
  --temp 1 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 1.5 --repeat-penalty 1.0 \
  --host 127.0.0.1 --port 8082
