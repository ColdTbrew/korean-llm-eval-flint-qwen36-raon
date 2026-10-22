import json, textwrap, statistics as st, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fam=None
for f in ["Noto Sans CJK KR","NanumGothic","Gothic A1"]:
    if any(f==x.name for x in fm.fontManager.ttflist): fam=f;break
plt.rcParams["font.family"]=fam; plt.rcParams["axes.unicode_minus"]=False
import os; B=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..")+"/"
F=json.load(open(B+"tokenizer/tok_flint.json")); Q=json.load(open(B+"tokenizer/tok_qwen.json")); R=json.load(open(B+"tokenizer/tok_raon.json"))
V=json.load(open(B+"tokenizer/vocab_scan_flint_qwen.json")); V["raon"]=json.load(open(B+"tokenizer/vocab_scan_raon.json"))["raon"]
cats=["반말·존댓말","맞춤법","요약","번역 영→한","번역 한→영","속담","상식·지리","수학·추론","한국어 코딩","시","형식 지키기"]
A=json.load(open(B+"results/aggregate.json"))
KM={"raon":"raon","qoff":"qwen_official","qcom":"qwen_community","flint":"flint"}
CK=list(A["raon"]["cat"].keys())
S={k:[A[m]["cat"][c]["mean"] for c in CK[:11]] for k,m in KM.items()}
LO={k:[A[m]["cat"][c]["min"] for c in CK[:11]] for k,m in KM.items()}
HI={k:[A[m]["cat"][c]["max"] for c in CK[:11]] for k,m in KM.items()}
NR={k:len(A[m]["runs"]) for k,m in KM.items()}
C={"raon":"#C0392B","qoff":"#1F4E79","qcom":"#8FB3D9","flint":"#F28E2B"}
L={"raon":"KRAFTON Raon-1 mini preview (Q4_K_M)","qoff":"Qwen3.6 공식→직접 변환","qcom":"Qwen3.6 커뮤니티판","flint":"Recursive Flint"}
fig=plt.figure(figsize=(14,20),dpi=130,facecolor="white")
fig.suptitle("Raon-1 mini · Qwen3.6-35B-A3B (공식/커뮤니티) · Flint — 한국어 22문항 × 3회",fontsize=19,fontweight="bold",y=0.996)
fig.text(0.5,0.977,"DGX Spark (GB10) · llama.cpp (Raon은 KRAFTON 포크 fb89566) · 같은 문항·샘플링(T1, top_p .95, top_k 20, presence 1.5*), 모델당 3회 실행 평균(n=3), 막대 끝 선 = 3회 최소~최대 · 5점 만점",ha="center",fontsize=10.5,color="#555")
ax=fig.add_axes([0.13,0.655,0.84,0.285])
y=np.arange(len(cats))[::-1]*1.4; h=0.3
for off,k in zip([1.5*h,0.5*h,-0.5*h,-1.5*h],["raon","qoff","qcom","flint"]):
    err=[[m-l for m,l in zip(S[k],LO[k])],[hh-m for m,hh in zip(S[k],HI[k])]]
    ax.barh(y+off,S[k],h,color=C[k],label=L[k],xerr=err,error_kw=dict(ecolor="#333",lw=0.8,capsize=1.8))
    for yy,v,hv in zip(y+off,S[k],HI[k]): ax.text(hv+0.05,yy,f"{v:.2f}".rstrip("0").rstrip("."),va="center",fontsize=7.3)
ax.set_yticks(y); ax.set_yticklabels(cats,fontsize=11.5); ax.set_xlim(0,5.5)
ax.spines[["top","right"]].set_visible(False); ax.legend(loc="upper center",bbox_to_anchor=(0.45,1.06),ncol=4,fontsize=9.5,frameon=False); ax.grid(axis="x",alpha=.3)
O=["raon","qoff","qcom","flint"]
def sp(k,f): return A[KM[k]]["speed"][f]
def ko(k): return f'{A[KM[k]]["ko_mean"]:.2f}±{A[KM[k]]["ko_std"]:.2f}'
cards=[("한국어 평균 (±표준편차)",[ko(k) for k in O]),("영어 평균",[f'{A[KM[k]]["en_mean"]:.2f}' for k in O]),
       ("생성 속도 tok/s (짧은 문맥)",[f'{sp(k,"dec_short")["mean"]:.1f}' for k in O]),
       ("같은 14k자 글 토큰 수",[f'{int(sp(k,"longctx_tokens")["mean"]):,}' for k in O]),
       ("답변 시작 중앙값",[f'{sp(k,"ans_median")["mean"]:.1f}초' for k in O]),("22문항 총 소요",[f'{sp(k,"total_min")["mean"]:.1f}분' for k in O])]
for i,(t,vals) in enumerate(cards):
    a=fig.add_axes([0.035+i*0.158,0.51,0.148,0.115]); a.axis("off")
    a.add_patch(plt.Rectangle((0,0),1,1,transform=a.transAxes,color="#F4F5F7"))
    a.text(0.5,0.87,t,ha="center",fontsize=10,color="#333",fontweight="bold")
    for j,(k,v) in enumerate(zip(["raon","qoff","qcom","flint"],vals)):
        a.text(0.5,0.66-j*0.19,v,ha="center",fontsize=12.5,color=C[k] if k!="qcom" else "#5C88B8",fontweight="bold")
NM0={"raon":"Raon","qoff":"Qwen 공식","qcom":"Qwen 커뮤니티","flint":"Flint"}
def fails(k):
    f=A[KM[k]]["failures"]; return ", ".join(f"{p.split('_')[0]} {len(r)}/{NR[k]}회" for p,r in sorted(f.items())) or "없음"
fig.text(0.035,0.495,"카드 순서: Raon / Qwen 공식 / Qwen 커뮤니티 / Flint, 속도·시간은 3회 평균. 14k자 첫 토큰(캐시 없음): "+" · ".join(f'{NM0[k]} {sp(k,"ttft_long")["mean"]:.2f}초' for k in O),fontsize=9.0,color="#444")
fig.text(0.035,0.481,"답 못 냄(사고가 16,384토큰 한도 도달, 답변 0자): "+" · ".join(f'{NM0[k]}: {fails(k)}' for k in O)+".  *Flint는 presence 없이 측정.",fontsize=9.0,color="#444")
fig.text(0.035,0.467,"Raon은 KRAFTON 공식 4비트 GGUF(Q4_K_M, 원본 대비 손실 있음). 두 Qwen은 NVFP4 가중치가 비트 단위로 같고 어텐션만 BF16/Q8_0. 시드: 2·3회차 seed=2,3, 1회차 기록 없음.",fontsize=9.0,color="#444")
fig.text(0.035,0.445,"토크나이저 (각 llama-server /tokenize·/detokenize 실측)",fontsize=13,fontweight="bold")
TK={"raon":R,"qwen":Q,"flint":F}; TC={"raon":C["raon"],"qwen":C["qoff"],"flint":C["flint"]}; TL={"raon":"Raon","qwen":"Qwen","flint":"Flint"}
a1=fig.add_axes([0.07,0.255,0.27,0.165])
xs=[s["n"] for s in Q["sentences"]]
for k in ("flint","raon"):
    a1.scatter(xs,[s["n"] for s in TK[k]["sentences"]],s=13,color=TC[k],label=TL[k])
m=max(xs+[s["n"] for s in F["sentences"]])+3; a1.plot([0,m],[0,m],"--",color="#999",lw=1); a1.set_xlim(0,m); a1.set_ylim(0,m)
a1.legend(fontsize=8.5,frameon=False,loc="upper left"); a1.set_xlabel("Qwen 토큰 수",fontsize=9.5); a1.set_ylabel("토큰 수",fontsize=9.5); a1.tick_params(labelsize=8.5)
mf=st.median([f["n"]/q["n"] for f,q in zip(F["sentences"],Q["sentences"])]); mr=st.median([r["n"]/q["n"] for r,q in zip(R["sentences"],Q["sentences"])])
a1.set_title(f"한국어 프롬프트 {len(xs)}문장별 토큰 수 (점선=Qwen과 같음)\n중앙값: Flint {mf:.2f}배 · Raon {mr:.2f}배 (Qwen 대비)",fontsize=9.8)
a2=fig.add_axes([0.41,0.255,0.24,0.165]); xx=np.arange(3)
for j,k in enumerate(("raon","qwen","flint")):
    hh=[TK[k]["syllables"]["len_hist"].get(str(n),0) for n in (1,2,3)]
    a2.bar(xx+(j-1)*0.27,hh,0.27,color=TC[k],label=TL[k])
    for xi,v in zip(xx+(j-1)*0.27,hh): a2.text(xi,v+80,f"{v:,}",ha="center",fontsize=6.8,rotation=0)
a2.set_xticks(xx); a2.set_xticklabels(["1토큰","2토큰","3토큰"],fontsize=9.5); a2.legend(fontsize=8.5,frameon=False); a2.tick_params(labelsize=8.5)
a2.set_title("한글 11,172음절 하나씩의 토큰 수\n1토큰: Raon 1,223 · Qwen 976 · Flint 309",fontsize=9.8)
a3=fig.add_axes([0.72,0.255,0.25,0.165]); bins=np.arange(26)
for j,k in enumerate(("raon","qwen","flint")):
    vv=[V[k]["id_hist_10k"].get(str(b),0) for b in bins]
    a3.bar(bins*10+(j-1)*3,vv,3,color=TC[k],label=f"{TL[k]} {V[k]['hangul_tokens']:,}개/{V[k]['n_vocab']:,}")
a3.set_yscale("log"); a3.set_xlabel("토큰 ID (천 단위)",fontsize=9.5); a3.legend(fontsize=7.3,frameon=False,loc="upper right"); a3.tick_params(labelsize=8.5)
a3.set_title("어휘 속 한글 토큰 ID 분포 (1만 단위, 로그축)\nRaon은 어휘의 20.3%가 한글",fontsize=9.8)
def fmt(toks):
    return "  ".join(f"<{bytes(t['bytes']).hex()}>#{t['id']}" if t["is_bytes"] else f"{t['text'].replace(' ','␣')}#{t['id']}" for t in toks)
s0=F["samples"][0]["text"]
fig.text(0.035,0.222,f"예: “{s0}”  Raon {R['samples'][0]['n']} · Qwen {Q['samples'][0]['n']} · Flint {F['samples'][0]['n']}토큰 (조각#ID, <..>=UTF-8 바이트 조각)",fontsize=10.5,fontweight="bold")
fig.text(0.045,0.205,"Raon:  "+fmt(R["samples"][0]["tokens"]),fontsize=9.3,color=C["raon"],va="top")
fig.text(0.045,0.188,"Qwen:  "+fmt(Q["samples"][0]["tokens"]),fontsize=9.3,color=C["qoff"],va="top")
fig.text(0.045,0.171,"Flint:  "+"\n            ".join(textwrap.wrap(fmt(F["samples"][0]["tokens"]),125)),fontsize=9.3,color=C["flint"],va="top")
fb={k:{e["syl"]:len(e["tokens"]) for e in TK[k]["syllables"]["fallback_examples"]} for k in TK}
rows=[f"{s}: Raon {fb['raon'][s]} · Qwen {fb['qwen'][s]} · Flint {fb['flint'][s]}" for s in ["녕","괜","닭","됐","짠","뷁","꿇","쀍"]]
fig.text(0.035,0.128,"음절별 조각 수 (2 이상 = 바이트 조각으로 쪼개짐):  "+"   |   ".join(rows[:4]),fontsize=9.3)
fig.text(0.035,0.111,"                                                                    "+"   |   ".join(rows[4:]),fontsize=9.3)
pk=lambda k:TK[k]["prompts"]
fig.text(0.035,0.088,f"22문항 프롬프트: 한국어 {pk('flint')['ko_chars']:,}자 → Raon {pk('raon')['ko_total']:,} · Qwen {pk('qwen')['ko_total']:,} · Flint {pk('flint')['ko_total']:,} 토큰 · 영어 {pk('flint')['en_chars']:,}자 → Raon {pk('raon')['en_total']} · Qwen {pk('qwen')['en_total']} · Flint {pk('flint')['en_total']} · 14,076자 글: Raon {R['long_ko']['n']:,} · Qwen {Q['long_ko']['n']:,} · Flint {F['long_ko']['n']:,}",fontsize=9.3)
fig.text(0.035,0.071,"셋 다 22문항 /detokenize 왕복 복원 100% 일치. Raon 어휘에는 ' 기술분야에서 통상의 지식을 가진' 같은 긴 한국어 구절 토큰도 있음.",fontsize=9.3)
rk=sorted(O,key=lambda k:-A[KM[k]]["ko_mean"])
NM={"raon":"Raon","qoff":"Qwen 공식","qcom":"Qwen 커뮤니티","flint":"Flint"}
fig.text(0.035,0.04,"결론(3회 평균): 한국어 "+" > ".join(f'{NM[k]} {A[KM[k]]["ko_mean"]:.2f}' for k in rk)+". 평균 차이가 3회 편차보다 작은 쌍은 순위를 단정하기 어려움.",fontsize=11.5,fontweight="bold")
fig.text(0.035,0.02,"디코드 tok/s는 Flint가 가장 빠름. 같은 글을 Raon은 Flint의 약 41% 토큰으로 쓰므로 글자 기준 속도는 tok/s 순위와 다름.",fontsize=9.5,color="#666")
fig.savefig(B+"images/flint_vs_qwen36_raon.png"); print("saved")
