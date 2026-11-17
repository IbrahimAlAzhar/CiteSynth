"""Tiny OpenAI-compatible stand-in for a vLLM judge server (CPU tests of the PBS judge orchestration).

Reads JUDGE_PORT and JUDGE_NAME from the environment, like the real `vllm serve` started by pbs/common.sh,
and answers every judge prompt used by scisynth with well-formed JSON.
"""
import hashlib
import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

NAME = os.environ.get("JUDGE_NAME", "judge")
PORT = int(os.environ.get("JUDGE_PORT", "8000"))


def h(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:6], 16)


def numbered(block: str) -> list[str]:
    return re.findall(r"^\s*(\d+)\.\s+(.*)$", block, flags=re.M)


def answer(system: str, user: str) -> dict:
    if "Generate up to" in user:
        title = re.search(r"TITLE: (.*)", user).group(1)
        words = [w for w in re.findall(r"[a-z]{5,}", title.lower())][:3] or ["this", "topic"]
        return {"questions": [{"question": f"How do studies on {' '.join(words)} compare in their methods and findings?",
                               "breadth": "focused", "general_knowledge": "no", "relevant_cited": [1, 2, 3, 4, 5]}]}
    if "KEY POINTS that" in user:
        docs = re.findall(r"^\[(D\d+)\] (.*)$", user, flags=re.M)
        kps = [{"text": " ".join(t.split()[:18]), "doc_ids": [d], "importance": 1 + i % 3} for i, (d, t) in enumerate(docs[:8])]
        return {"relevant_doc_ids": [d for d, _ in docs[:8]], "key_points": kps}
    if "atomic content points" in user:
        rw = user.split("SECTION:", 1)[-1]
        return {"points": [{"text": s.strip()[:150], "importance": 2} for s in re.split(r"(?<=\.)\s", rw) if len(s) > 30][:8]}
    if "Criteria:" in user:
        v = 3 + h(user) % 6
        return {"synthesis": v, "specificity": v, "coverage": v, "critical": v, "organization": v, "positioning": v, "overall": v}
    for marker, labels in (("CLAIMS:", ("supported", "partial", "unsupported", "not_found")),
                           ("KEY POINTS:", ("yes", "partial", "no")), ("NOTES:", ("supported", "partial"))):
        if marker in user:
            items = numbered(user.split(marker, 1)[1].split("Return JSON")[0])
            if "REFERENCE DOCUMENTS" in user:
                labels = ("supported", "not_found", "generic")
            return {"results": [{"id": int(i), "label": labels[h(t + user[:200]) % len(labels)]} for i, t in items]}
    return {"results": []}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            return self._send(200, {"object": "list", "data": [{"id": NAME, "object": "model"}]})
        self._send(404, {"error": {"message": "not found"}})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        if req.get("model") != NAME:
            return self._send(404, {"error": {"message": f"The model `{req.get('model')}` does not exist.", "type": "NotFoundError"}})
        msgs = req["messages"]
        out = answer(msgs[0]["content"], msgs[-1]["content"])
        Handler.n += 1
        if Handler.n == 1:
            print(f"first request ok: chat_template_kwargs={req.get('chat_template_kwargs')} "
                  f"response_format={req.get('response_format')}", flush=True)
        self._send(200, {"id": "x", "object": "chat.completion", "created": 0, "model": NAME,
                         "choices": [{"index": 0, "finish_reason": "stop",
                                      "message": {"role": "assistant", "content": json.dumps(out)}}],
                         "usage": {"prompt_tokens": len(msgs[-1]["content"]) // 4, "completion_tokens": 50, "total_tokens": 0}})


Handler.n = 0

if __name__ == "__main__":
    print(f"fake judge '{NAME}' on port {PORT}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
    sys.exit(0)
