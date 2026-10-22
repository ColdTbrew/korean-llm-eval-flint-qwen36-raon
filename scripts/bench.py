import json, sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_eval import chat
para = open("long_ko.txt").read()
res = []
short = "한국의 수도는 어디인가요? 한 문장으로 답하세요."
long_p = para + "\n\n위 글 전체를 다섯 문장으로 요약해 주세요."
for name, p, mt in [("warmup", short, 256), ("short_1", short, 1024), ("short_2", short, 1024),
                    ("long_gen", "파이썬으로 간단한 할 일 관리 CLI 프로그램을 작성하고 한국어로 자세히 설명해 주세요.", 4096),
                    ("long_prompt_1", long_p, 1024), ("long_prompt_2", long_p + " ", 1024)]:
    r = chat(p, max_tokens=mt, extra={"temperature": 1.0})
    r["name"] = name; r.pop("reasoning"); r["content"] = r["content"][:300]
    res.append(r); print(json.dumps({k: r[k] for k in ["name","usage","ttft_s","decode_tok_s","total_s","finish_reason"]}, ensure_ascii=False), flush=True)
json.dump(res, open("bench.json", "w"), ensure_ascii=False, indent=1)
