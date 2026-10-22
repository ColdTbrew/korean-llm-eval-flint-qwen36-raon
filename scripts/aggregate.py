# Working layout: run next to <model>/run{1,2,3}/ dirs (model = flint, qwen_community, qwen_official, raon); repo copies of the raw files are in results/<dir>/run*/
import json,statistics as st,os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'grades.py')).read())
H=os.path.dirname(os.path.abspath(__file__))
MODELS=["raon","qwen_official","qwen_community","flint"]
def speed(m,r):
    d=f"{H}/{m}/run{r}"
    b={x['name']:x for x in json.load(open(d+"/bench.json"))}
    R=[json.loads(l) for l in open(d+"/results.jsonl")]
    ct=[x['usage']['completion_tokens'] for x in R]
    ta=[x['time_to_answer_s'] if x['time_to_answer_s'] is not None else x['total_s'] for x in R]
    return dict(dec_short=(b['short_1']['decode_tok_s']+b['short_2']['decode_tok_s'])/2,
        dec_longgen=b['long_gen']['decode_tok_s'],dec_longctx=b['long_prompt_1']['decode_tok_s'],
        longctx_tokens=b['long_prompt_1']['usage']['prompt_tokens'],
        ttft_short=(b['short_1']['ttft_s']+b['short_2']['ttft_s'])/2,
        # long_prompt_1 is only a cold (uncached) measurement if the server prompt cache missed
        ttft_long=b['long_prompt_1']['ttft_s'] if b['long_prompt_1']['server_timings']['cache_n']==0 else None,
        prefill=b['long_prompt_1']['server_timings']['prompt_per_second'] if b['long_prompt_1']['server_timings']['cache_n']==0 else None,
        ttft_long_cached=b['long_prompt_2']['ttft_s'],
        gen_sum=sum(ct),gen_median=st.median(ct),ans_median=st.median(ta),ans_max=max(ta),
        total_min=sum(x['total_s'] for x in R)/60,n_length=sum(x['finish_reason']=='length' for x in R),
        length_ids=[x['id'] for x in R if x['finish_reason']=='length'],
        empty_ids=[x['id'] for x in R if not (x['content'] or '').strip()])
out={}
for m in MODELS:
    runs=sorted(G[m]); o={"runs":runs,"cat":{},"speed":{}}
    for i,c in enumerate(CATS):
        v=[G[m][r][i] for r in runs]
        o["cat"][c]=dict(per_run=v,mean=st.mean(v),min=min(v),max=max(v),std=st.pstdev(v))
    ko=[st.mean(G[m][r][:11]) for r in runs]
    o["ko_mean_per_run"]=ko; o["ko_mean"]=st.mean(ko); o["ko_std"]=st.pstdev(ko)
    en=[G[m][r][11] for r in runs]; o["en_mean"]=st.mean(en); o["en_per_run"]=en
    sp=[speed(m,r) for r in runs if os.path.exists(f"{H}/{m}/run{r}/bench.json")]
    for k in sp[0]:
        if isinstance(sp[0][k],list): o["speed"][k]=[s[k] for s in sp]
        else:
            v=[s[k] for s in sp if s[k] is not None]
            o["speed"][k]=dict(mean=st.mean(v),min=min(v),max=max(v),n=len(v))
    fails={}
    for i,s in enumerate(sp):
        for pid in set(s['length_ids'])|set(s['empty_ids']): fails.setdefault(pid,[]).append(i+1)
    o["failures"]=fails
    out[m]=o
json.dump(out,open(f"{H}/aggregate.json","w"),ensure_ascii=False,indent=1)
for m in MODELS:
    o=out[m]; print(m,"runs",o["runs"],"KO",[round(x,2) for x in o["ko_mean_per_run"]],"mean",round(o["ko_mean"],2),"std",round(o["ko_std"],2),"EN",o["en_per_run"],"fails",o["failures"])

import re
HAN=re.compile(r'[\u4e00-\u9fff\u3040-\u30ff]')
mix={}
for m in MODELS:
    mix[m]=[]
    for r in sorted(G[m]):
        p=f"{H}/{m}/run{r}/results.jsonl"
        if not os.path.exists(p): continue
        n=0; ids=[]
        for l in open(p):
            d=json.loads(l); k=len(HAN.findall(d['content'] or ''))
            if k: n+=k; ids.append(d['id'])
        mix[m].append(dict(run=r,chars=n,prompts=ids))
    out[m]["cjk_mix"]=mix[m]
json.dump(out,open(f"{H}/aggregate.json","w"),ensure_ascii=False,indent=1)
for m in MODELS: print(m,[(x['run'],x['chars'],len(x['prompts'])) for x in mix[m]])

