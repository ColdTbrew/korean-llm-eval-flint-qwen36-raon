import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fam=None
for f in ["Noto Sans CJK KR","NanumGothic","Gothic A1"]:
    if any(f==x.name for x in fm.fontManager.ttflist): fam=f;break
plt.rcParams["font.family"]=fam; plt.rcParams["axes.unicode_minus"]=False
import os; B=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..")+"/"
F=json.load(open(B+"tokenizer/tok_flint.json")); Q=json.load(open(B+"tokenizer/tok_qwen.json"))
V=json.load(open(B+"tokenizer/vocab_scan_flint_qwen.json"))
cats=["반말·존댓말","맞춤법","요약","번역 영→한","번역 한→영","속담","상식·지리","수학·추론","한국어 코딩","시","형식 지키기"]
flint=[2,1,4,3,2.5,1,1.5,3.5,4,2.5,3]
qcom=[3.5,2.5,3.5,5,5,3.5,4,4,4,4,5]
qoff=[3.5,2,3.5,4,5,3.5,3.5,4,4,4,4]
cO="#1F4E79"; cQ="#8FB3D9"; cF="#F28E2B"
fig=plt.figure(figsize=(14,19),dpi=130,facecolor="white")
fig.suptitle("Recursive Flint vs Qwen3.6-35B-A3B (공식 직접 변환 / 커뮤니티판) · 한국어 22문항",fontsize=19,fontweight="bold",y=0.996)
fig.text(0.5,0.972,"DGX Spark (GB10) · llama.cpp 7170cad · 같은 문항·샘플링(T1, top_p .95, top_k 20), 문항당 1회 · 5점 만점",ha="center",fontsize=11.5,color="#555")
ax=fig.add_axes([0.13,0.64,0.84,0.31])
y=np.arange(len(cats))[::-1]*1.25; h=0.36
for off,vals,c,lab in [(h,qoff,cO,"Qwen3.6 공식→직접 변환 GGUF"),(0,qcom,cQ,"Qwen3.6 커뮤니티 GGUF (williamliao)"),(-h,flint,cF,"Recursive Flint")]:
    ax.barh(y+off,vals,h,color=c,label=lab)
    for yy,v in zip(y+off,vals): ax.text(v+0.04,yy,f"{v:g}",va="center",fontsize=8.5)
ax.set_yticks(y); ax.set_yticklabels(cats,fontsize=11.5); ax.set_xlim(0,5.5)
ax.spines[["top","right"]].set_visible(False); ax.legend(loc="upper center",bbox_to_anchor=(0.45,1.07),ncol=3,fontsize=10.5,frameon=False); ax.grid(axis="x",alpha=.3)
cards=[("한국어 평균","3.7","4.0","2.4"),("영어 평균","4.5","4.5","4.5"),("생성 속도 tok/s","51~52","67~68","85~90"),
       ("첫 토큰 (14k자 한국어)","3.0초","2.3초","5.7초"),("같은 글 토큰 수","6,095","6,095","10,185"),("22문항 총 소요","20.6분","13.6분","12.7분")]
for i,(t,o,q,f) in enumerate(cards):
    a=fig.add_axes([0.035+i*0.158,0.505,0.148,0.105]); a.axis("off")
    a.add_patch(plt.Rectangle((0,0),1,1,transform=a.transAxes,color="#F4F5F7"))
    a.text(0.5,0.85,t,ha="center",fontsize=10,color="#333",fontweight="bold")
    a.text(0.5,0.6,o,ha="center",fontsize=14,color=cO,fontweight="bold")
    a.text(0.5,0.36,q,ha="center",fontsize=14,color="#5C88B8",fontweight="bold")
    a.text(0.5,0.12,f,ha="center",fontsize=14,color=cF,fontweight="bold")
fig.text(0.035,0.49,"카드 숫자 순서: 공식 직접 변환 / 커뮤니티 / Flint. 두 Qwen은 NVFP4 가중치가 비트 단위로 같고, 어텐션(FP8 원본)만 공식판 BF16 · 커뮤니티판 Q8_0이라 공식판이 더 느림.",fontsize=9.5,color="#444")
fig.text(0.035,0.474,"두 Qwen의 점수 차(3.7 vs 4.0)는 문항당 1회 샘플이라 샘플링 편차 범위로 봐야 함 (변환 차이로 단정할 근거 없음).",fontsize=9.5,color="#444")
# --- tokenizer panels ---
fig.text(0.035,0.452,"토크나이저 (llama-server /tokenize·/detokenize 실측, 공식·커뮤니티 Qwen은 어휘·병합표 해시 동일)",fontsize=13,fontweight="bold")
a1=fig.add_axes([0.07,0.255,0.27,0.17])
xs=[s["n"] for s in Q["sentences"]]; ys=[s["n"] for s in F["sentences"]]
a1.scatter(xs,ys,s=14,color="#555"); m=max(xs+ys)+3; a1.plot([0,m],[0,m],"--",color="#999",lw=1); a1.plot([0,m/1.42],[0,m],":",color=cF,lw=1); a1.set_xlim(0,m); a1.set_ylim(0,m)
a1.set_xlabel("Qwen 토큰 수",fontsize=9.5); a1.set_ylabel("Flint 토큰 수",fontsize=9.5)
a1.set_title(f"한국어 프롬프트 {len(xs)}문장별 토큰 수\n(중앙값 Flint/Qwen = 1.42배, 점선=1:1)",fontsize=10)
a1.tick_params(labelsize=8.5)
a2=fig.add_axes([0.41,0.255,0.24,0.17])
lab=["1토큰","2토큰","3토큰"]; fh=F["syllables"]["len_hist"]; qh=Q["syllables"]["len_hist"]
xx=np.arange(3); a2.bar(xx-0.2,[qh.get(str(k),0) for k in (1,2,3)],0.4,color=cO,label="Qwen")
a2.bar(xx+0.2,[fh.get(str(k),0) for k in (1,2,3)],0.4,color=cF,label="Flint")
for k in (1,2,3):
    a2.text(k-1-0.2,qh.get(str(k),0)+80,f"{qh.get(str(k),0):,}",ha="center",fontsize=8); a2.text(k-1+0.2,fh.get(str(k),0)+80,f"{fh.get(str(k),0):,}",ha="center",fontsize=8)
a2.set_xticks(xx); a2.set_xticklabels(lab,fontsize=9.5); a2.legend(fontsize=8.5,frameon=False); a2.tick_params(labelsize=8.5)
a2.set_title("한글 11,172음절을 혼자 넣었을 때 토큰 수\n(1토큰: Qwen 976 / Flint 309)",fontsize=10)
a3=fig.add_axes([0.72,0.255,0.25,0.17])
bins=np.arange(25)
qv=[V["qwen"]["id_hist_10k"].get(str(b),0) for b in bins]; fv=[V["flint"]["id_hist_10k"].get(str(b),0) for b in bins]
a3.bar(bins*10-2,qv,4,color=cO,label=f"Qwen (한글 포함 {V['qwen']['hangul_tokens']:,}개/248,320)"); a3.bar(bins*10+2,fv,4,color=cF,label=f"Flint ({V['flint']['hangul_tokens']:,}개/150,144)")
a3.set_xlabel("토큰 ID (천 단위)",fontsize=9.5); a3.legend(fontsize=7.5,frameon=False); a3.tick_params(labelsize=8.5)
a3.set_title("어휘에서 한글 토큰의 ID 분포 (1만 단위 구간)\nQwen은 14만~25만 구간에 몰림",fontsize=10)
def fmt(toks):
    out=[]
    for t in toks:
        if t["is_bytes"]: out.append(f"<{bytes(t['bytes']).hex()}>#{t['id']}")
        else: out.append(f"{t['text'].replace(' ','␣')}#{t['id']}")
    return "  ".join(out)
s0F=F["samples"][0]; s0Q=Q["samples"][0]
fig.text(0.035,0.225,f"예: “{s0F['text']}”  Flint {s0F['n']}토큰 / Qwen {s0Q['n']}토큰  (조각#ID, <..>=UTF-8 바이트 조각)",fontsize=10.5,fontweight="bold")
import textwrap
fig.text(0.045,0.205,"Flint: "+"\n            ".join(textwrap.wrap(fmt(s0F["tokens"]),120)),fontsize=9.5,color=cF,va="top")
fig.text(0.045,0.162,"Qwen:  "+fmt(s0Q["tokens"]),fontsize=9.5,color=cO,va="top")
fb={e["syl"]:e for e in F["syllables"]["fallback_examples"]}; fbq={e["syl"]:e for e in Q["syllables"]["fallback_examples"]}
rows=[]
for s in ["녕","괜","닭","됐","짠","뷁","쀍"]:
    rows.append(f"{s}(UTF-8 {bytes(fb[s]['utf8']).hex()}): Flint {len(fb[s]['tokens'])}조각 · Qwen {len(fbq[s]['tokens'])}조각")
fig.text(0.035,0.125,"음절이 바이트 조각으로 쪼개지는 예:  "+"   |   ".join(rows[:4]),fontsize=9.3)
fig.text(0.035,0.107,"                                            "+"   |   ".join(rows[4:]),fontsize=9.3)
pF=F["prompts"]; pQ=Q["prompts"]
fig.text(0.035,0.083,f"22문항 프롬프트 합계: 한국어 {pF['ko_chars']:,}자 → Flint {pF['ko_total']:,} / Qwen {pQ['ko_total']:,} 토큰 (1.38배) · 영어 {pF['en_chars']:,}자 → Flint {pF['en_total']} / Qwen {pQ['en_total']} · 14,076자 한국어 글: Flint {F['long_ko']['n']:,} / Qwen {Q['long_ko']['n']:,}",fontsize=9.5)
fig.text(0.035,0.066,f"프롬프트에 나온 한글 음절 1,290회 중 1토큰 음절 비율: Flint {F['syllables']['prompt_syllable_occ_single']/12.90:.1f}% / Qwen {Q['syllables']['prompt_syllable_occ_single']/12.90:.1f}% · 22문항 /detokenize 왕복 복원 둘 다 100% 일치",fontsize=9.5)
fig.text(0.035,0.035,"결론: 한국어는 Qwen3.6이 앞서고, 공식 직접 변환판도 커뮤니티판과 같은 수준(3.7 vs 4.0, 1회 샘플). Flint는 가장 빠르지만 한국어를 약 1.4배 더 많은 토큰으로 쪼갬.",fontsize=12,fontweight="bold")
fig.text(0.035,0.015,"변환 검증: 공식 NVFP4 텐서 30,841개(전문가·공유 전문가·lm_head) 비트 단위 동일, scale_2는 F32 별도 텐서로 그대로. FP8 어텐션은 BF16로 풀면서 반올림(최대 상대오차 0.37%), BF16→F16 강등 없음.",fontsize=9.3,color="#666")
fig.savefig(B+"images/flint_vs_qwen36_v2.png")
print("saved")
