#!/usr/bin/env python3
"""OpenAI-compatible mock of the organizers' proxy, for plumbing tests.

    python tools/mock_llm.py TRUTH_JSON --port 8765 --budget 100000 --noise 0.15

Answers from Task Factory ground truth (with `noise` probability of a wrong
answer), reports `usage`, supports tool calls, injects 5% HTTP 500, and stops
answering (HTTP 402) once the competition budget is exhausted - like the real
proxy.  Writes its own token ledger to /tmp/mock_ledger.json.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task_factory import LIB  # noqa: E402

STATE = {"spent": 0.0, "calls": 0, "rejected": 0}
LOCK = threading.Lock()


def toks(s: str) -> int:
    return max(1, int(len(s) / 3.6))


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        cfg = self.server.cfg
        with LOCK:
            if STATE["spent"] >= cfg.budget:
                STATE["rejected"] += 1
                return self._send(402, {"error": {"message": "token budget exhausted"}})
        if req.get("model") in (cfg.fail_model or "").split(","):
            return self._send(403, {"error": {"message": "model not allowed for this key"}})
        if cfg.rnd.random() < cfg.rate429:
            return self._send(429, {"error": {"message": "rate limit"}})
        if cfg.rnd.random() < 0.05:
            return self._send(500, {"error": {"message": "upstream error"}})
        msgs = req.get("messages", [])
        text = json.dumps(msgs, ensure_ascii=False)
        tid, tr = self.server.find(text)
        content, tool_calls = self.answer(req, msgs, tid, tr)
        reasoning = 0 if "noreason" in req.get("model", "") else cfg.rnd.randint(200, 900)
        pin = toks(text + json.dumps(req.get("tools", [])))
        pout = toks(content + json.dumps(tool_calls)) + reasoning
        pout = min(pout, int(req.get("max_tokens") or 10 ** 6))
        reasoning = min(reasoning, pout)
        with LOCK:
            STATE["spent"] += pout + 0.1 * pin
            STATE["calls"] += 1
            self.server.cfg.ledger.write_text(json.dumps(STATE))
        msg = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self._send(200, {"id": "x", "object": "chat.completion", "model": req.get("model"),
                         "choices": [{"index": 0, "message": msg,
                                      "finish_reason": "tool_calls" if tool_calls else "stop"}],
                         "usage": {"prompt_tokens": pin, "completion_tokens": pout,
                                   "total_tokens": pin + pout,
                                   "completion_tokens_details": {"reasoning_tokens": reasoning}}})

    def answer(self, req, msgs, tid, tr):
        cfg = self.server.cfg
        exp = tr.get("expected") if tr else None
        wrong = cfg.rnd.random() < cfg.noise
        sys_txt = msgs[0]["content"] if msgs and msgs[0]["role"] == "system" else ""
        if req.get("tools"):
            # agent: single submit call
            if tr and tr["type"] == "coding":
                return "", []  # mock agent cannot write files; forces failure path
            val = exp if exp is not None else "INFEASIBLE"
            if wrong and exp is not None:
                val = self._wrong(exp)
            return "", [{"id": "call_1", "type": "function", "function": {
                "name": "submit", "arguments": json.dumps({"answer": val})}}]
        if "fix bugs" in sys_txt:
            if tr is None or exp is None:
                return "IMPOSSIBLE", []
            names = re.findall(r"def test_(\w+)\(", tr["_hidden"])
            src = "\n\n".join(LIB[nm][0] for nm in names if nm in LIB)
            if wrong:
                src = src.replace("return", "return None or", 1)
            return f"```python\n{src}\n```", []
        if "Python 3 program" in sys_txt:
            if exp is None:
                return "```python\nprint('INFEASIBLE')\n```", []
            v = self._wrong(exp) if wrong else exp
            return f"```python\nprint({v})\n```", []
        # search style
        if exp is None:
            return "INFEASIBLE", []
        return (self._wrong(exp) if wrong else exp), []

    def _wrong(self, exp: str) -> str:
        """Independent random error (real models rarely err identically)."""
        rnd = self.server.cfg.rnd
        try:
            v = int(exp)
            return str(v + rnd.choice([-1, 1]) * rnd.randint(1, max(2, abs(v) // 10 + 1)))
        except ValueError:
            return exp + rnd.choice(["x", " Inc", "s", "2"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("truth")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--budget", type=float, default=100000)
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--fail-model", default="", help="comma list of models answering 403")
    ap.add_argument("--rate429", type=float, default=0.0)
    ap.add_argument("--ledger", type=Path, default=Path("/tmp/mock_ledger.json"))
    cfg = ap.parse_args()
    cfg.rnd = random.Random(cfg.seed)
    truth = json.loads(Path(cfg.truth).read_text())
    tasks_dir = Path(cfg.truth).with_name(Path(cfg.truth).name.replace("_truth.json", ""))
    fps_all = {}
    for tid, tr in truth.items():
        d = tasks_dir / tid
        tj = d / "task.json"
        if not tj.exists():
            continue
        q = json.loads(tj.read_text())["instruction"]
        sents = [s for s in re.split(r"(?<=[.?!])\s+", q) if "?" in s or "starts" in s]
        fps = [s[:80] for s in sents[:2]]
        ws = d / "workspace"
        if (ws / "sales.csv").exists():
            lines = (ws / "sales.csv").read_text().splitlines()
            fps += lines[1:4]
        if (ws / "mylib.py").exists():
            fps += [ln for ln in (ws / "mylib.py").read_text().splitlines() if ln.strip()][:40]
        if (ws / "kb").is_dir():
            for f in sorted((ws / "kb").glob("person_*.md")):
                txt = f.read_text()
                name = txt.splitlines()[0].lstrip("# ").strip()
                if name in q:
                    fps.append(txt.strip().replace("\n\n", "\n")[:60])
        fps_all[tid] = fps
    srv = ThreadingHTTPServer(("127.0.0.1", cfg.port), H)
    srv.cfg = cfg

    def find(text):
        t2 = text.replace("\\n", "\n")
        best, best_score = None, 0
        for tid, fps in fps_all.items():
            sc = sum(1 for f in fps if f and f in t2)
            if sc > best_score:
                best, best_score = tid, sc
        return best, truth.get(best)

    srv.find = find
    try:
        srv.serve_forever()
    finally:
        cfg.ledger.write_text(json.dumps(STATE))


if __name__ == "__main__":
    main()
