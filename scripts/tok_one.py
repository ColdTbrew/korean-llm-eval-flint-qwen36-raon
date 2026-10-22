# usage: python3 tok_one.py <model_name> <base_url> <prompts.json> <out.json>
import json, re, sys, time, urllib.request, concurrent.futures as cf
name, base, ppath, outp = sys.argv[1:5]
def post(path, body):
    r = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=60).read())
def tok(text, pieces=False):
    return post("/tokenize", {"content": text, "with_pieces": pieces})["tokens"]
def detok(ids): return post("/detokenize", {"tokens": ids})["content"]
def piece(p):
    if isinstance(p, str): return {"text": p, "is_bytes": False}
    return {"text": None, "bytes": p, "is_bytes": True}
SAMPLES = [
 ("인사", "안녕하세요, 오늘 날씨가 정말 좋네요."),
 ("존댓말", "할머니, 진지 드셨어요? 제가 내일 찾아뵙겠습니다."),
 ("속담", "발 없는 말이 천 리 간다"),
 ("맞춤법", "며칠 뒤에 금세 끝낼 수 있을 것 같아요."),
 ("뉴스", "서울시는 2028년까지 저상버스 비율을 100%로 끌어올리고 정류장 300곳에 휠체어 승강 보조 장치를 설치할 예정이다."),
 ("사투리", "니 뭐하노? 밥은 뭇나? 아따 겁나 맛있당께."),
 ("한영코드 혼합", "파이썬에서 list를 정렬하려면 sorted(점수들, reverse=True)를 쓰면 돼요."),
 ("드문 음절", "짠했어. 뷁, 똠방각하, 쀍"),
 ("영어 대조", "Hello, the weather is really nice today. I will visit you tomorrow."),
]
out = {"model": name, "samples": [], "syllables": {}, "sentences": [], "prompts": {}, "roundtrip": {}}
for nm, text in SAMPLES:
    t = tok(text, True)
    out["samples"].append({"name": nm, "text": text, "chars": len(text), "n": len(t),
        "tokens": [{"id": x["id"], **piece(x["piece"])} for x in t], "detok_ok": detok([x["id"] for x in t]) == text})
syl = [chr(c) for c in range(0xAC00, 0xD7A4)]
def f(s): return (s, tok(s))
with cf.ThreadPoolExecutor(8) as ex: res = dict(ex.map(f, syl))
hist = {}
for s, ids in res.items(): hist[len(ids)] = hist.get(len(ids), 0) + 1
single_ids = sorted(ids[0] for ids in res.values() if len(ids) == 1)
# syllables that fall back to raw byte tokens: look at pieces for a few
fallback = []
for s in ["녕", "짠", "뷁", "쀍", "똠", "햏", "꿇", "괜", "읽", "닭", "됐", "뤄"]:
    t = tok(s, True); fallback.append({"syl": s, "utf8": list(s.encode()), "tokens": [{"id": x["id"], **piece(x["piece"])} for x in t]})
# common-syllable coverage: top syllables by frequency in the Korean prompts
P = json.load(open(ppath))
ko_text = "".join(p["prompt"] for p in P if not p["id"].startswith("en"))
freq = {}
for ch in ko_text:
    if 0xAC00 <= ord(ch) <= 0xD7A3: freq[ch] = freq.get(ch, 0) + 1
used = sorted(freq, key=lambda c: -freq[c])
out["syllables"] = {"len_hist": hist, "single": hist.get(1, 0), "total": len(syl),
    "single_id_hist_10k": {str(k): v for k, v in sorted({i // 10000: 0 for i in single_ids}.items())},
    "examples": {s: res[s] for s in ["가", "안", "녕", "짠", "뷁", "쀍", "똠", "햏"]}, "fallback_examples": fallback,
    "prompt_syllables_distinct": len(used), "prompt_syllables_single": sum(1 for c in used if len(res[c]) == 1),
    "prompt_syllable_occ": sum(freq.values()), "prompt_syllable_occ_single": sum(freq[c] for c in used if len(res[c]) == 1)}
h = {}
for i in single_ids: h[i // 10000] = h.get(i // 10000, 0) + 1
out["syllables"]["single_id_hist_10k"] = {str(k): v for k, v in sorted(h.items())}
# per-sentence token counts on Korean prompts
for p in P:
    if p["id"].startswith("en"): continue
    for sent in [s.strip() for s in re.split(r'(?<=[.?!])\s+|\n+', p["prompt"]) if s.strip()]:
        if not re.search('[가-힣]', sent): continue
        out["sentences"].append({"id": p["id"], "text": sent, "chars": len(sent), "n": len(tok(sent))})
ko = en = 0; per = {}; rt_bad = []
for p in P:
    ids = [x if isinstance(x, int) else x["id"] for x in tok(p["prompt"])]
    per[p["id"]] = len(ids)
    if p["id"].startswith("en"): en += len(ids)
    else: ko += len(ids)
    if detok(ids) != p["prompt"]: rt_bad.append(p["id"])
out["prompts"] = {"ko_total": ko, "en_total": en, "per": per,
    "ko_chars": sum(len(p["prompt"]) for p in P if not p["id"].startswith("en")),
    "en_chars": sum(len(p["prompt"]) for p in P if p["id"].startswith("en"))}
out["roundtrip"] = {"prompts_checked": len(P), "mismatch": rt_bad}
long_ko = open(ppath.replace("prompts.json", "long_ko.txt")).read()
out["long_ko"] = {"chars": len(long_ko), "n": len(tok(long_ko))}
json.dump(out, open(outp, "w"), ensure_ascii=False, indent=1)
print(name, "single", hist.get(1, 0), "ko", ko, "en", en, "rt_bad", rt_bad, "long", out["long_ko"])
