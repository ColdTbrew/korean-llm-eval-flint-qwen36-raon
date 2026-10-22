# Working layout: run next to <model>/run{1,2,3}/ dirs (model = flint, qwen_community, qwen_official, raon); repo copies of the raw files are in results/<dir>/run*/
import json,os,sys,statistics as st
H=os.path.dirname(os.path.abspath(__file__)); exec(open(H+'/grades.py').read())
A=json.load(open(H+'/aggregate.json')); OLD=open(sys.argv[1]).read(); OUT=sys.argv[2]
M=["raon","qwen_official","qwen_community","flint"]
NM={"raon":"Raon Q4_K_M","qwen_official":"Qwen 공식 직접 변환","qwen_community":"Qwen 커뮤니티","flint":"Flint"}
def sec(t):  # take a section of the old file by heading
    s=OLD.split("\n## "); return "## "+[x for x in s if x.startswith(t)][0].rstrip()+"\n"
def f(x,d=1): return f"{x:.{d}f}"
def sp(m,k,d=1,unit=""):
    x=A[m]["speed"][k]; return f"{f(x['mean'],d)}{unit} ({f(x['min'],d)}–{f(x['max'],d)})"
L=[]
L.append("# Raon-1 mini vs Qwen3.6-35B-A3B (공식 직접 변환 / 커뮤니티판) vs Flint — 한국어 22문항 × 3회 + 같은 벤치 (DGX Spark GB10, 2026-10-22~23 KST)\n")
L.append("![chart](images/flint_vs_qwen36_raon.png)\n")
L.append("**모든 점수와 속도는 모델마다 3회 실행 평균(n=3)입니다.** 1회차는 2026-10-22 실행(seed 기록 없음, 서버 기본값=무작위), 2·3회차는 2026-10-22~23 같은 서버 바이너리·플래그·샘플링에 seed=2, 3을 지정해 실행. 한 번에 모델 하나만 메모리에 올림(systemd-run MemoryMax, 스왑 0).\n")
L.append("채점: 문항별 답변을 같은 기준(1-5점, 0.5 단위)으로 한 명(Grok Bot)이 채점. 회차별 근거는 `results/<모델>/grading.md`. 원자료 `results/<모델>/run{1,2,3}/results.jsonl`, 집계 `results/aggregate.json`.\n")
L.append(sec("설정").replace("~/flint/llama.cpp-raon","(로컬 빌드)"))
L.append("2·3회차는 요청마다 `seed`를 지정(`eval_extra.json`에 기록). Flint는 presence penalty 없음(원래 설정 그대로).\n")
L.append(sec("변환 무손실 검증"))
# categories
L.append("## 카테고리 점수 (1-5, 3회 평균, 괄호는 최소~최대)")
L.append("| 카테고리 | "+" | ".join(NM[m] for m in M)+" |"); L.append("|---"*(len(M)+1)+"|")
for c in CATS[:11]:
    L.append(f"| {c} | "+" | ".join(f"{A[m]['cat'][c]['mean']:.2f} ({A[m]['cat'][c]['min']:g}~{A[m]['cat'][c]['max']:g})" for m in M)+" |")
L.append("| **한국어 평균 (11개)** | "+" | ".join(f"**{A[m]['ko_mean']:.2f}** ± {A[m]['ko_std']:.2f}" for m in M)+" |")
L.append("| 한국어 평균, 회차별 | "+" | ".join(" / ".join(f"{v:.2f}" for v in A[m]['ko_mean_per_run']) for m in M)+" |")
c=CATS[11]
L.append(f"| **EN 비교** | "+" | ".join(f"**{A[m]['en_mean']:.2f}** ({' / '.join(f'{v:g}' for v in A[m]['en_per_run'])})" for m in M)+" |")
L.append("\n± 는 회차별 한국어 평균의 표준편차(모표준편차, n=3). 1회차 한국어 평균은 이전 보고의 반올림값(Raon ≈4.3 \"47/11\", Flint ≈2.4) 대신 실제 합계로 다시 계산: Raon 46.5/11 = 4.23, Flint 28/11 = 2.55.\n")
# failures
L.append("## 문항별 실패 (finish=length 또는 빈 답변, 3회 중)")
L.append("| 모델 | 실패 |"); L.append("|---|---|")
for m in M:
    fl=A[m]["failures"]; L.append(f"| {NM[m]} | "+(", ".join(f"{k} {len(v)}/3 (회차 {','.join(map(str,v))})" for k,v in fl.items()) or "없음 (0/66)")+" |")
L.append("\n답변 본문 속 한자·가나 글자 수 (자동 집계, 회차별): "+"; ".join(NM[m]+" "+"/".join(str(x['chars']) for x in A[m]['cjk_mix']) for m in M)+"\n")
# speed
L.append("## 속도 (3회 평균, 괄호는 최소–최대)")
L.append("| 항목 | "+" | ".join(NM[m] for m in M)+" |"); L.append("|---"*(len(M)+1)+"|")
rows=[("디코드, 짧은 문맥 (tok/s)","dec_short",1,""),("디코드, 긴 생성 (tok/s)","dec_longgen",1,""),("디코드, 긴 문맥 (tok/s)","dec_longctx",1,""),
 ("TTFT 짧은 프롬프트 (s)","ttft_short",3,""),("TTFT 긴 프롬프트, 캐시 없음 (s)","ttft_long",2,""),("prefill (tok/s)","prefill",0,""),
 ("TTFT 긴 프롬프트, 캐시 (s)","ttft_long_cached",3,""),("22문항 생성 토큰 합","gen_sum",0,""),("생성 토큰 중앙값","gen_median",0,""),
 ("답변 시작까지 중앙값 (s)","ans_median",1,""),("답변 시작까지 최대 (s)","ans_max",0,""),("22문항 총 소요 (분)","total_min",1,"")]
for t,k,d,u in rows: L.append(f"| {t} | "+" | ".join(sp(m,k,d,u) for m in M)+" |")
L.append("| 14,076자 글 프롬프트 토큰 수 | "+" | ".join(f"{A[m]['speed']['longctx_tokens']['mean']:,.0f}" for m in M)+" |")
L.append("\n캐시 없는 긴 프롬프트 TTFT·prefill은 서버를 새로 띄운 회차만 평균(n=2: 1·2회차). 3회차는 2회차와 같은 서버라 같은 프롬프트가 프롬프트 캐시에 맞아 콜드 측정이 아님. 답변 시작까지 시간은 답을 못 낸 문항은 전체 소요로 계산.\n")
L.append(sec("토크나이저").replace("원자료: tok2/tok_raon.json, tok2/vocab_scan_raon.json, tok2/tok_qwen_official.json, tok2/tok_flint.json, tok2/vocab_scan_official.json","원자료: `tokenizer/`"))
r=sorted(M,key=lambda m:-A[m]['ko_mean'])
L.append("## 결론 (3회 평균 기준)")
L.append("- 한국어 22문항(11개 카테고리): "+" > ".join(f"{NM[m]} {A[m]['ko_mean']:.2f}±{A[m]['ko_std']:.2f}" for m in r)+". 영어: "+", ".join(f"{NM[m]} {A[m]['en_mean']:.2f}" for m in M)+".")
L.append("- 디코드 tok/s(짧은 문맥): "+", ".join(f"{NM[m]} {A[m]['speed']['dec_short']['mean']:.1f}" for m in sorted(M,key=lambda m:-A[m]['speed']['dec_short']['mean']))+".")
L.append("- Raon은 한국어를 가장 적은 토큰으로 쓰고(14k자 글: Qwen의 약 0.68배, Flint의 약 0.41배) 생각을 짧게 하고 답함. 4비트 양자화본이라 원본 대비 손실 있음.")
L.append("- Raon은 평균 1위지만 회차 편차가 가장 큼(3.95~4.64). ko04 맞춤법 문항에서 3회 중 2회 사고 루프로 16,384토큰 한도에 걸려 답을 못 냄. 2회차만 보면 4.64, 루프 회차는 4.23·3.95. Qwen 커뮤니티(4.00~4.18)와 범위가 겹침.")
L.append("- Qwen 공식 직접 변환은 NVFP4 부분이 비트 단위 무손실, 어텐션(FP8 원본)만 BF16 반올림. 그런데도 한국어 점수는 3회 모두 커뮤니티판(어텐션 Q8_0)보다 조금 낮음(평균 차 0.23점). 같은 가중치라 정밀도보다는 샘플링 편차와 채점 오차로 보는 게 맞고, 22문항×3회로는 차이를 단정할 수 없음.")
L.append("- 두 Qwen은 '미역국을 먹다'와 사투리를 3회 모두 틀리고 답변에 한자가 섞임. Flint는 맞춤법·속담·사실 문항에서 3회 모두 약함.")
L.append("\n## 한계")
L.append("- 22문항 × 3회, 채점자 1명(주관적). 0.2~0.3점 차이는 표본 편차 범위일 수 있음.")
L.append("- 1회차 seed는 기록되지 않음. Raon은 공식 Q4_K_M(손실 있음), Qwen·Flint는 NVFP4.")
open(OUT,"w").write("\n".join(L)+"\n"); print("wrote",OUT)
