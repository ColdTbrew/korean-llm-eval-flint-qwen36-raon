# reads tokenizer vocab from GGUF metadata (read-only, mmap) and classifies Hangul tokens
import json, sys
from gguf import GGUFReader
def b2u():
    bs=list(range(ord("!"),ord("~")+1))+list(range(ord("¡"),ord("¬")+1))+list(range(ord("®"),ord("ÿ")+1))
    cs=bs[:]; n=0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256+n); n+=1
    return {chr(c):b for b,c in zip(bs,cs)}
U2B=b2u()
res={}
for name,path in [("flint",sys.argv[1]),("qwen",sys.argv[2])]:
    r=GGUFReader(path)
    f=r.fields["tokenizer.ggml.tokens"]
    toks=[bytes(f.parts[i]).decode("utf-8") for i in f.data]
    ttype=r.fields.get("tokenizer.ggml.token_type")
    types=[int(ttype.parts[i][0]) for i in ttype.data] if ttype else None
    model=bytes(r.fields["tokenizer.ggml.model"].parts[-1]).decode()
    n=len(toks); hangul_ids=[]; full_syl_tokens=0; multi=0; frag=0; lens={}
    single_syl=set()
    for i,t in enumerate(toks):
        try: bb=bytes(U2B[c] for c in t)
        except KeyError: bb=t.encode()
        s=bb.decode("utf-8",errors="replace")
        hs=[c for c in s if 0xAC00<=ord(c)<=0xD7A3]
        if hs:
            hangul_ids.append(i); k=len(hs); lens[k]=lens.get(k,0)+1
            core=s.strip()
            if len(core)==1 and len(hs)==1: single_syl.add(core)
        elif bb and all(x>=0x80 for x in bb) and any(0xEA<=x<=0xED for x in bb) and "\ufffd" in s:
            frag+=1
    hist={}
    for i in hangul_ids: hist[i//10000]=hist.get(i//10000,0)+1
    res[name]={"tokenizer_model":model,"n_vocab":n,"hangul_tokens":len(hangul_ids),"hangul_pct":round(100*len(hangul_ids)/n,2),
      "hangul_syllable_count_hist":dict(sorted(lens.items())),"distinct_single_syllable_tokens":len(single_syl),
      "hangul_lead_byte_fragment_tokens":frag,"id_hist_10k":dict(sorted(hist.items())),
      "min_id":min(hangul_ids),"max_id":max(hangul_ids),
      "longest":sorted(((len([c for c in toks[i] ]),i) for i in hangul_ids),reverse=True)[:0]}
    # longest Hangul tokens decoded
    dec=[]
    for i in hangul_ids:
        bb=bytes(U2B.get(c,0) for c in toks[i]); s=bb.decode("utf-8",errors="replace")
        dec.append((sum(1 for c in s if 0xAC00<=ord(c)<=0xD7A3),i,s))
    dec.sort(reverse=True); res[name]["longest"]=[(i,s) for _,i,s in dec[:8]]
    print(name,{k:v for k,v in res[name].items() if k not in("id_hist_10k",)},flush=True)
json.dump(res,open("vocab_scan.json","w"),ensure_ascii=False,indent=1)
