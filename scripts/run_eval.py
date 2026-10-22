import json, time, sys, urllib.request
import os
URL = os.environ.get("FLINT_URL","http://127.0.0.1:8080/v1/chat/completions")
def chat(prompt, max_tokens=16384, extra=None):
    body = {"model": os.environ.get("EVAL_MODEL","flint"), "messages": [{"role": "user", "content": prompt}],
            "temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True}}
    body.update(json.loads(os.environ.get("EVAL_EXTRA","{}")))
    if extra: body.update(extra)
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; t_first_content = None
    content, reasoning, usage, finish, timings = [], [], None, None, None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"): continue
            d = line[5:].strip()
            if d == "[DONE]": break
            j = json.loads(d)
            if j.get("usage"): usage = j["usage"]
            if j.get("timings"): timings = j["timings"]
            for ch in j.get("choices", []):
                dl = ch.get("delta", {})
                rc = dl.get("reasoning_content") or dl.get("reasoning")
                c = dl.get("content")
                if (rc or c) and t_first is None: t_first = time.time()
                if c and t_first_content is None: t_first_content = time.time()
                if rc: reasoning.append(rc)
                if c: content.append(c)
                if ch.get("finish_reason"): finish = ch["finish_reason"]
    t1 = time.time()
    ct = usage["completion_tokens"] if usage else None
    return {"content": "".join(content), "reasoning": "".join(reasoning), "finish_reason": finish,
            "usage": usage, "ttft_s": round(t_first - t0, 3) if t_first else None,
            "time_to_answer_s": round(t_first_content - t0, 3) if t_first_content else None,
            "total_s": round(t1 - t0, 3), "server_timings": timings,
            "decode_tok_s": round((ct - 1) / (t1 - t_first), 1) if ct and t_first and t1 > t_first else None}
if __name__ == "__main__":
    prompts = json.load(open(sys.argv[1])); out = open(sys.argv[2], "a")
    for p in prompts:
        r = chat(p["prompt"]); r.update(p)
        out.write(json.dumps(r, ensure_ascii=False) + "\n"); out.flush()
        print(p["id"], r["finish_reason"], r["usage"], r["ttft_s"], r["decode_tok_s"], flush=True)
