# Working layout: run next to <model>/run{1,2,3}/ dirs (model = flint, qwen_community, qwen_official, raon); repo copies of the raw files are in results/<dir>/run*/
import json,os,sys
H=os.path.dirname(os.path.abspath(__file__)); exec(open(H+'/grades.py').read())
A=json.load(open(H+'/aggregate.json'))
REPO=sys.argv[1]
DIR={"flint":"flint","qwen_community":"qwen36-community","qwen_official":"qwen36-official","raon":"raon-1-mini-preview"}
MARK="\n\n<!-- 3RUN -->"
for m,d in DIR.items():
    p=f"{REPO}/results/{d}/grading.md"; base=open(p).read().split(MARK)[0]
    o=A[m]; s=[MARK,"\n## 3회 실행 요약 (n=3, 같은 서버·샘플링, 2·3회차 seed=2·3, 1회차 seed 기록 없음)\n",
       "채점: Grok Bot, 2026-10-23 KST, 위 1회차 표와 같은 기준(1-5). 1회차 점수는 위 표 그대로.\n",
       "| 카테고리 | 1회 | 2회 | 3회 | 평균 | 최소~최대 | 표준편차 |","|---|---|---|---|---|---|---|"]
    for c in CATS:
        x=o["cat"][c]; pr=x["per_run"]+[None]*(3-len(x["per_run"]))
        s.append(f"| {c} | "+" | ".join("" if v is None else f"{v:g}" for v in pr)+f" | {x['mean']:.2f} | {x['min']:g}~{x['max']:g} | {x['std']:.2f} |")
    kr=o["ko_mean_per_run"]
    s.append(f"| **한국어 11개 평균** | "+" | ".join(f"{v:.2f}" for v in kr)+f" | **{o['ko_mean']:.2f}** | {min(kr):.2f}~{max(kr):.2f} | {o['ko_std']:.2f} |")
    f=o["failures"]; s.append("\n답 못 냄(finish=length 또는 빈 답변): "+(", ".join(f"{k} — {len(v)}/{len(o['runs'])}회 (회차 {v})" for k,v in f.items()) or "없음"))
    s.append("자동 집계 한자·가나 글자 수(답변 본문, 회차별): "+", ".join(f"{x['run']}회 {x['chars']}자/{len(x['prompts'])}문항" for x in o["cjk_mix"]))
    for r in sorted(G[m]):
        if (m,r) not in NOTES: continue
        s.append(f"\n### {r}회차 근거 (seed={r})\n| 카테고리 | 점수 | 근거 |\n|---|---|---|")
        for c,v,n in zip(CATS,G[m][r],NOTES[(m,r)]): s.append(f"| {c} | {v:g} | {n} |")
        s.append(f"\n{r}회차 한국어 평균 {sum(G[m][r][:11])/11:.2f}, EN {G[m][r][11]:g}")
    open(p,"w").write(base+"\n".join(s)+"\n")
print("ok")
