import os, json, sys, numpy as np, torch, time
from safetensors import safe_open
from gguf import GGUFReader
SRC=os.path.expanduser("~/llm/models/Qwen3.6-35B-A3B-NVFP4")
G=sys.argv[1]
idx=json.load(open(SRC+"/model.safetensors.index.json"))["weight_map"]
handles={}
def src(name):
    f=idx[name]
    if f not in handles: handles[f]=safe_open(SRC+"/"+f,"pt")
    return handles[f].get_tensor(name)
r=GGUFReader(G); gt={t.name:t for t in r.tensors}
def graw(name): return np.asarray(gt[name].data)
E2M1=np.array([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],dtype=np.float32)
KV=np.array([0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12],dtype=np.float32)
def ue4m3_ggml(x):
    x=x.astype(np.int32); e=(x>>3)&0xF; m=x&7
    raw=np.where(e==0, m*2.0**-9, (1+m/8.0)*np.exp2(e-7.0)).astype(np.float32)*np.float32(0.5)
    return np.where((x==0)|(x==0x7F),0,raw).astype(np.float32)
def repack(w,s):  # independent re-implementation of ModelOpt->ggml block layout
    w=w.view(torch.uint8).numpy(); s=s.view(torch.uint8).numpy()
    R,nb=s.shape; w=w.reshape(R,nb,8)
    lo=w&0xF; hi=w>>4; vals=np.stack([lo,hi],-1).reshape(R,nb,16)
    qs=(vals[:,:,:8]|(vals[:,:,8:]<<4)).astype(np.uint8)
    d=(s&0x7F).reshape(R,nb//4,4); q=qs.reshape(R,nb//4,32)
    return np.concatenate([d,q],-1).reshape(R,-1)
def src_deq(w,s,s2):  # reference ModelOpt dequant in float64: e2m1[nib]*e4m3(scale)*scale2
    w=w.view(torch.uint8).numpy(); R=w.shape[0]
    vals=np.stack([w&0xF,w>>4],-1).reshape(R,-1)
    sc=s.float().numpy().astype(np.float64)
    return E2M1[vals].astype(np.float64)*np.repeat(sc,16,axis=1)*float(s2)
def gguf_deq(raw,R,s2):  # ggml dequantize_row_nvfp4 semantics, then * .scale (ggml_mul)
    b=raw.reshape(R,-1,36); d=ue4m3_ggml(b[:,:,:4]); qs=b[:,:,4:].reshape(R,b.shape[1],4,8)
    lo=KV[qs&0xF]; hi=KV[qs>>4]
    v=np.concatenate([lo,hi],-1)*d[...,None]   # [R,nsup,4,16]
    return v.reshape(R,-1).astype(np.float64)*float(s2)
rows=[]; stats={"nvfp4_tensors_bitcheck":0,"nvfp4_bit_mismatch":0,"scale_sign_or_nan_bytes":0}
P="model.language_model."
def scale_flags(s):
    b=s.view(torch.uint8).numpy(); return int(((b&0x80)!=0).sum()+((b&0x7F)==0x7F).sum())
# 1) FULL bitwise check of every NVFP4 tensor + scale2
L=40
for i in range(L):
    for p,gp in [("gate","gate"),("up","up"),("down","down")]:
        gname=f"blk.{i}.ffn_{gp}_exps.weight"; g=graw(gname); g=g.reshape(256,g.shape[-2],-1)
        gs=graw(f"blk.{i}.ffn_{gp}_exps.scale").reshape(-1)
        for e in range(256):
            n=f"{P}layers.{i}.mlp.experts.{e}.{p}_proj"
            w=src(n+".weight"); s=src(n+".weight_scale"); s2=src(n+".weight_scale_2")
            stats["scale_sign_or_nan_bytes"]+=scale_flags(s)
            ok=np.array_equal(repack(w,s),g[e]) and np.float32(s2.float().item())==gs[e]
            stats["nvfp4_tensors_bitcheck"]+=1; stats["nvfp4_bit_mismatch"]+=0 if ok else 1
        gname=f"blk.{i}.ffn_{gp}_shexp.weight"; g=graw(gname)
        n=f"{P}layers.{i}.mlp.shared_expert.{p}_proj"
        w=src(n+".weight"); s=src(n+".weight_scale"); s2=src(n+".weight_scale_2")
        stats["scale_sign_or_nan_bytes"]+=scale_flags(s)
        ok=np.array_equal(repack(w,s),g.reshape(g.shape[0],-1)) and np.float32(s2.float().item())==graw(gname.replace(".weight",".scale")).reshape(-1)[0]
        stats["nvfp4_tensors_bitcheck"]+=1; stats["nvfp4_bit_mismatch"]+=0 if ok else 1
    print("layer",i,stats,flush=True)
w=src("lm_head.weight"); s=src("lm_head.weight_scale"); s2=src("lm_head.weight_scale_2"); g=graw("output.weight")
ok=np.array_equal(repack(w,s),g.reshape(g.shape[0],-1)) and np.float32(s2.float().item())==graw("output.scale").reshape(-1)[0]
stats["nvfp4_tensors_bitcheck"]+=1; stats["nvfp4_bit_mismatch"]+=0 if ok else 1; stats["scale_sign_or_nan_bytes"]+=scale_flags(s)
# 2) dequant comparisons on representative tensors
def nv_row(label,n,gname,eidx=None):
    w=src(n+".weight"); s=src(n+".weight_scale"); s2=src(n+".weight_scale_2")
    g=graw(gname)
    if eidx is not None:
        g=g.reshape(256,g.shape[-2],-1)[eidx]; gs=graw(gname.replace(".weight",".scale")).reshape(-1)[eidx]
    else:
        g=g.reshape(g.shape[0],-1); gs=graw(gname.replace(".weight",".scale")).reshape(-1)[0]
    a=src_deq(w,s,s2); b=gguf_deq(g,w.shape[0],gs)
    rows.append(dict(tensor=label,src="NVFP4 (U8 packed E2M1 + F8_E4M3 block scale + F32 scale_2)",gguf=f"NVFP4 + F32 .scale",shape=list(a.shape),
        max_abs_diff=float(np.abs(a-b).max()),bit_identical=bool(np.array_equal(repack(w,s),g)),scale2=float(s2.float().item())))
for i in (0,19,39):
    for e in (0,128,255):
        for p in ("gate","up","down"):
            nv_row(f"L{i} expert{e} {p}",f"{P}layers.{i}.mlp.experts.{e}.{p}_proj",f"blk.{i}.ffn_{p}_exps.weight",e)
    for p in ("gate","up","down"):
        nv_row(f"L{i} shared_expert {p}",f"{P}layers.{i}.mlp.shared_expert.{p}_proj",f"blk.{i}.ffn_{p}_shexp.weight")
nv_row("lm_head","lm_head","output.weight")
def bf16_of(gname):
    t=gt[gname]; a=np.asarray(t.data)
    return a.view(np.uint16) if a.dtype!=np.uint16 else a
def plain(label,sname,gname,reordered=False,expect=None):
    x=src(sname); g=gt[gname]; gty=g.tensor_type.name
    if gty=="BF16":
        gb=bf16_of(gname).reshape(-1); sb=x.contiguous().view(torch.int16).numpy().view(np.uint16).reshape(-1)
        if reordered: same=np.array_equal(np.sort(gb),np.sort(sb))
        else: same=np.array_equal(gb,sb)
        mad=float(np.abs(torch.from_numpy(gb.view(np.int16).copy()).view(torch.bfloat16).float().numpy().astype(np.float64)-x.float().numpy().reshape(-1).astype(np.float64)).max()) if not reordered else (0.0 if same else None)
    else:
        ga=np.asarray(g.data).astype(np.float64).reshape(-1); sa=x.float().numpy().astype(np.float64).reshape(-1)
        if expect=="plus1": sa=sa+1
        if reordered: ga=np.sort(ga); sa=np.sort(sa)
        mad=float(np.abs(ga-sa).max()); same=bool(mad==0)
    rows.append(dict(tensor=label,src=str(x.dtype).replace("torch.",""),gguf=gty,shape=list(x.shape),max_abs_diff=mad,bit_identical=bool(same),note="multiset compare (V-head reorder)" if reordered else ""))
plain("token_embd","model.language_model.embed_tokens.weight","token_embd.weight")
for i in (0,3,39):
    plain(f"L{i} router gate",f"{P}layers.{i}.mlp.gate.weight",f"blk.{i}.ffn_gate_inp.weight")
    plain(f"L{i} shared_expert_gate",f"{P}layers.{i}.mlp.shared_expert_gate.weight",f"blk.{i}.ffn_gate_inp_shexp.weight")
for i in (0,20):
    plain(f"L{i} linear_attn in_proj_a",f"{P}layers.{i}.linear_attn.in_proj_a.weight",f"blk.{i}.ssm_alpha.weight",True)
    plain(f"L{i} linear_attn in_proj_b",f"{P}layers.{i}.linear_attn.in_proj_b.weight",f"blk.{i}.ssm_beta.weight",True)
plain("mtp.fc","mtp.fc.weight","blk.40.nextn.eh_proj.weight")
# 3) FP8 -> BF16 path
def fp8(label,n,gname,reordered):
    w=src(n+".weight"); s=src(n+".weight_scale")
    ref=(w.float()*s.float()).reshape(-1)   # what converter computes (fp32)
    rb=ref.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    gb=bf16_of(gname).reshape(-1)
    gf=torch.from_numpy(gb.view(np.int16).copy()).view(torch.bfloat16).double().numpy(); rf=ref.double().numpy()
    if reordered: gf=np.sort(gf); rf=np.sort(rf); eq=np.array_equal(np.sort(rb),np.sort(gb))
    else: eq=np.array_equal(rb,gb)
    err=np.abs(gf-rf); rel=err/np.maximum(np.abs(rf),1e-30)
    rows.append(dict(tensor=label,src=f"F8_E4M3 + F32 per-tensor weight_scale ({float(s.float().reshape(-1)[0]):.3g})",gguf=gt[gname].tensor_type.name,shape=list(w.shape),
        max_abs_diff=float(err.max()),max_rel_diff=float(rel[np.abs(rf)>0].max()),frac_exact=float((err==0).mean()),bit_identical=False,
        equals_bf16_round_of_fp32_dequant=bool(eq),note="multiset compare (V-head reorder)" if reordered else ""))
for i in (3,39):
    for p,gp in [("q","attn_q"),("k","attn_k"),("v","attn_v"),("o","attn_output")]:
        fp8(f"L{i} full_attn {p}_proj",f"{P}layers.{i}.self_attn.{p}_proj",f"blk.{i}.{gp}.weight",False)
for i in (0,20):
    for p,gp in [("in_proj_qkv","attn_qkv"),("in_proj_z","attn_gate"),("out_proj","ssm_out")]:
        fp8(f"L{i} linear_attn {p}",f"{P}layers.{i}.linear_attn.{p}",f"blk.{i}.{gp}.weight",True)
json.dump({"stats":stats,"rows":rows},open(sys.argv[2],"w"),indent=1)
print(json.dumps(stats)); print("VERIFYDONE")
