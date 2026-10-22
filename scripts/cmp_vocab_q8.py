import os, sys, hashlib, json, numpy as np, torch
from gguf import GGUFReader
from gguf.quants import dequantize
from safetensors import safe_open
C=os.path.expanduser("~/.cache/huggingface/hub/models--williamliao--Qwen3.6-35B-A3B-NVFP4-GGUF/snapshots/78bed75d118ebae682749827f1fa42a99d0095e2/Qwen3.6-35B-A3B-NVFP4.gguf")
O=os.path.expanduser("~/llm/gguf/Qwen3.6-35B-A3B-NVFP4-official-bf16rest.gguf")
out={}
rc=GGUFReader(C); ro=GGUFReader(O)
def h(r,k):
    f=r.fields.get(k)
    if f is None: return None
    m=hashlib.sha256()
    for i in f.data: m.update(bytes(f.parts[i]))
    return m.hexdigest()[:16]
for k in ["tokenizer.ggml.tokens","tokenizer.ggml.merges","tokenizer.ggml.token_type","tokenizer.chat_template"]:
    out[k]={"community":h(rc,k),"official":h(ro,k)}
# NVFP4 bit-identity between community and official for a few tensors
tc={t.name:t for t in rc.tensors}; to={t.name:t for t in ro.tensors}
same=[];diff=[]
for n,t in to.items():
    if t.tensor_type.name=="NVFP4":
        (same if np.array_equal(np.asarray(t.data),np.asarray(tc[n].data)) else diff).append(n)
out["nvfp4_tensors_identical_official_vs_community"]={"same":len(same),"diff":len(diff),"diff_examples":diff[:5]}
# community Q8_0 error vs fp32 FP8 dequant (non-reordered full-attn tensors)
SRC=os.path.expanduser("~/llm/models/Qwen3.6-35B-A3B-NVFP4")
idx=json.load(open(SRC+"/model.safetensors.index.json"))["weight_map"]
rows=[]
for i in (3,39):
    for p,g in [("q","attn_q"),("k","attn_k"),("v","attn_v"),("o","attn_output")]:
        n=f"model.language_model.layers.{i}.self_attn.{p}_proj"
        with safe_open(SRC+"/"+idx[n+".weight"],"pt") as f:
            ref=(f.get_tensor(n+".weight").float()*f.get_tensor(n+".weight_scale").float()).double().numpy()
        t=tc[f"blk.{i}.{g}.weight"]
        deq=dequantize(np.asarray(t.data),t.tensor_type).astype(np.float64).reshape(ref.shape)
        ob=np.asarray(to[f"blk.{i}.{g}.weight"].data).view(np.uint16).reshape(-1)
        of=torch.from_numpy(ob.view(np.int16).copy()).view(torch.bfloat16).double().numpy().reshape(ref.shape)
        e1=np.abs(deq-ref); e2=np.abs(of-ref)
        rows.append({"tensor":f"L{i} {p}_proj","community_type":t.tensor_type.name,"community_max_abs":float(e1.max()),"community_rmse":float(np.sqrt((e1**2).mean())),
                     "official_bf16_max_abs":float(e2.max()),"official_bf16_rmse":float(np.sqrt((e2**2).mean()))})
out["fp8_path_error"]=rows
json.dump(out,open("cmp_vocab_q8.json","w"),indent=1); print(json.dumps(out,indent=1))
