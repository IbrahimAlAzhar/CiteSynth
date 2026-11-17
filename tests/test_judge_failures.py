"""Judge failure handling against a fake OpenAI-compatible server (no API key needed):
  * quota error      -> stops at once with a clear message (no long retry loop)
  * repeated 500s    -> stops when too many calls fail, instead of scoring them as 0
  * healthy server   -> results returned and cached
"""
import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scisynth.judge import JudgeRequest, OpenAIJudge  # noqa: E402

MODE = {"m": "ok"}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if MODE["m"] == "quota":
            code, body = 429, {"error": {"message": "You exceeded your current quota", "type": "insufficient_quota",
                                         "code": "insufficient_quota"}}
        elif MODE["m"] == "500":
            code, body = 500, {"error": {"message": "server error", "type": "server_error"}}
        else:
            code, body = 200, {"id": "x", "object": "chat.completion", "created": 0, "model": "m",
                               "choices": [{"index": 0, "finish_reason": "stop",
                                            "message": {"role": "assistant", "content": '{"results": [{"id": 1, "label": "yes"}]}'}}],
                               "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}/v1"
    reqs = [JudgeRequest("cover", "sys", f"user {i}") for i in range(20)]
    tmp = tempfile.mkdtemp()

    MODE["m"] = "quota"
    j = OpenAIJudge(base_url=url, api_key="sk-test", cache_path=os.path.join(tmp, "c1.jsonl"))
    t0 = time.time()
    try:
        j.run(reqs)
        raise AssertionError("quota error was not raised")
    except RuntimeError as e:
        msg = str(e)
        assert "not retryable" in msg and "insufficient_quota" in msg, msg
    print(f"quota error -> stopped in {time.time() - t0:.1f}s: {msg.splitlines()[0][:110]}")

    MODE["m"] = "500"
    j = OpenAIJudge(base_url=url, api_key="sk-test", cache_path=os.path.join(tmp, "c2.jsonl"), max_retries=2)
    try:
        j.run(reqs)
        raise AssertionError("failure-rate error was not raised")
    except RuntimeError as e:
        msg = str(e)
        assert "20/20 judge calls failed" in msg, msg
    print("repeated 500s ->", msg.splitlines()[0][:110])

    MODE["m"] = "ok"
    cache = os.path.join(tmp, "c3.jsonl")
    out = OpenAIJudge(base_url=url, api_key="sk-test", cache_path=cache).run(reqs)
    assert all(o and o["results"][0]["label"] == "yes" for o in out)
    j2 = OpenAIJudge(base_url=url, api_key="sk-test", cache_path=cache)
    j2.run(reqs)
    assert j2.n_cached == 20 and j2.n_calls == 0
    print("healthy server -> 20 results, all served from cache on rerun")
    srv.shutdown()
    print("ALL JUDGE FAILURE TESTS PASSED")


if __name__ == "__main__":
    main()
