exec ~/flint/llama.cpp-raon/build/bin/llama-server -m $HOME/llm/models/Raon-1-mini-preview-gguf/raon-1-mini-preview-Q4_K_M.gguf \
  --alias raon-1-mini-preview -ngl all -c 32768 -np 1 \
  -fa on -ctk f16 -ctv f16 --jinja \
  --temp 1 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 1.5 --repeat-penalty 1.0 \
  --host 127.0.0.1 --port 8083
