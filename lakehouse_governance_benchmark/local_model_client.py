"""Minimal client for a local Ollama server (structured JSON output, deterministic decoding)."""

import json
import os
import time
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


def chat_json(model, system, user, schema, temperature=0.0, seed=0, num_ctx=8192, timeout=240, retries=2,
              max_output_tokens=None):
    # cap generation so a model that loops inside the JSON cannot stall the run; reasoning models need more room
    if max_output_tokens is None:
        max_output_tokens = 4096 if "gpt-oss" in model else 768
    body = {"model": model, "stream": False, "format": schema, "think": False,
            "options": {"temperature": temperature, "seed": seed, "num_ctx": num_ctx,
                        "num_predict": max_output_tokens},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    last = None
    for attempt in range(retries + 1):
        try:
            t0 = time.time()
            req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.loads(r.read())
            text = out.get("message", {}).get("content", "")
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            return {"raw": text, "parsed": parsed, "latency_s": round(time.time() - t0, 3),
                    "prompt_tokens": out.get("prompt_eval_count"), "output_tokens": out.get("eval_count")}
        except Exception as e:  # server busy, model loading, transient errors
            last = repr(e)
            if "think" in last.lower() and body.pop("think", None) is not None:
                continue  # model does not support the think switch
            time.sleep(5 * (attempt + 1))
    return {"raw": None, "parsed": None, "error": last}
