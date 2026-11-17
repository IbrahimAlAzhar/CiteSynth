"""LLM judge used for benchmark construction, rewards and evaluation.

* OpenAIJudge : gpt-4o-mini (default) or any OpenAI-compatible endpoint (e.g. a vLLM server via --judge_base_url),
                async with bounded concurrency, retries, on-disk JSONL cache and cost tracking.
* MockJudge   : deterministic lexical heuristics, for CPU smoke tests without network/API key.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field

from .retrieval import tokenize
from .utils import LOG, split_sentences


@dataclass
class JudgeRequest:
    kind: str
    system: str
    user: str
    payload: dict = field(default_factory=dict)
    max_tokens: int = 2048


def _parse_json(text: str):
    if text is None:
        return None
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


# substrings of OpenAI errors that retrying cannot fix
FATAL_JUDGE_ERRORS = ("insufficient_quota", "invalid_api_key", "Incorrect API key", "AuthenticationError",
                      "PermissionDeniedError", "NotFoundError", "model_not_found",
                      "does not exist or you do not have access")


class OpenAIJudge:
    def __init__(self, model: str = "gpt-4o-mini", cache_path: str | None = None, concurrency: int = 32,
                 base_url: str | None = None, api_key: str | None = None, temperature: float = 0.0,
                 max_retries: int = 8, timeout: float = 180.0, price_in: float = 0.15, price_out: float = 0.60,
                 max_fail_frac: float = 0.05, extra_body: dict | None = None):
        try:
            from openai import AsyncOpenAI  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install openai  (needed for the gpt-4o-mini judge)") from e
        self.model, self.base_url = model, base_url
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or ("EMPTY" if base_url else None)
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (export it in the PBS script) or pass --judge mock")
        self.concurrency, self.temperature = concurrency, temperature
        self.max_retries, self.timeout = max_retries, timeout
        self.price_in, self.price_out = price_in, price_out
        self.cache_path = cache_path
        self.cache: dict[str, dict] = {}
        self._lock = threading.Lock()
        self.n_calls = self.n_cached = self.n_failed = 0
        self.max_fail_frac = max_fail_frac
        self.extra_body = extra_body
        self.last_error = ""
        self.fatal_error = ""
        self.tok_in = self.tok_out = 0
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self.cache[rec["k"]] = rec["v"]
                    except Exception:
                        continue
            LOG.info("Judge cache: %d entries from %s", len(self.cache), cache_path)

    def _key(self, r: JudgeRequest) -> str:
        return hashlib.sha1(f"{self.model}\x00{r.system}\x00{r.user}".encode("utf-8")).hexdigest()

    def _store(self, key: str, val: dict) -> None:
        with self._lock:
            self.cache[key] = val
            if self.cache_path:
                os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)), exist_ok=True)
                with open(self.cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"k": key, "v": val}, ensure_ascii=False) + "\n")

    async def _one(self, client, sem, r: JudgeRequest):
        key = self._key(r)
        if key in self.cache:
            self.n_cached += 1
            return self.cache[key]
        async with sem:
            delay = 2.0
            for attempt in range(self.max_retries):
                try:
                    kw = dict(model=self.model, temperature=self.temperature, max_tokens=r.max_tokens,
                              messages=[{"role": "system", "content": r.system}, {"role": "user", "content": r.user}])
                    if attempt < self.max_retries - 1:
                        kw["response_format"] = {"type": "json_object"}
                    if self.extra_body:
                        kw["extra_body"] = self.extra_body
                    resp = await client.chat.completions.create(**kw)
                    self.n_calls += 1
                    if resp.usage:
                        self.tok_in += resp.usage.prompt_tokens or 0
                        self.tok_out += resp.usage.completion_tokens or 0
                    out = _parse_json(resp.choices[0].message.content)
                    if out is not None:
                        self._store(key, out)
                        return out
                except Exception as e:  # rate limits, timeouts, 5xx
                    msg = f"{type(e).__name__}: {str(e)[:300]}"
                    self.last_error = msg
                    if any(x in msg for x in FATAL_JUDGE_ERRORS):
                        self.fatal_error = msg  # retrying cannot fix quota / key / model errors
                        break
                    if attempt >= 2:
                        LOG.warning("judge call failed (attempt %d): %s", attempt + 1, msg)
                if self.fatal_error:
                    break
                await asyncio.sleep(delay + random.random())
                delay = min(delay * 2, 60)
        self.n_failed += 1
        return None

    async def _run(self, reqs):
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout, max_retries=0)
        sem = asyncio.Semaphore(self.concurrency)
        try:
            return await asyncio.gather(*[self._one(client, sem, r) for r in reqs])
        finally:
            await client.close()

    def run(self, reqs: list[JudgeRequest]) -> list[dict | None]:
        if not reqs:
            return []
        t0 = time.time()
        failed_before = self.n_failed
        out = asyncio.run(self._run(reqs))
        n_fail = self.n_failed - failed_before
        LOG.info("judge: %d requests in %.1fs (api=%d cached=%d failed=%d, cost so far $%.3f)",
                 len(reqs), time.time() - t0, self.n_calls, self.n_cached, self.n_failed, self.cost())
        if self.fatal_error:
            raise RuntimeError(f"JUDGE ERROR (not retryable): {self.fatal_error}\n"
                               "Check the OpenAI key, billing/quota and model name. Successful calls are cached; "
                               "resubmit the job after fixing it to continue where it stopped.")
        if n_fail > max(2, self.max_fail_frac * len(reqs)):
            raise RuntimeError(f"JUDGE ERROR: {n_fail}/{len(reqs)} judge calls failed (limit {self.max_fail_frac:.0%}). "
                               f"Last error: {self.last_error}\nFailed calls would be scored as 0, so the stage is "
                               "stopped instead. Successful calls are cached; resubmit to continue (lower "
                               "--judge_concurrency if these are rate-limit errors).")
        return out

    def cost(self) -> float:
        return (self.tok_in * self.price_in + self.tok_out * self.price_out) / 1e6

    def stats(self) -> dict:
        return {"model": self.model, "api_calls": self.n_calls, "cached": self.n_cached, "failed": self.n_failed,
                "tokens_in": self.tok_in, "tokens_out": self.tok_out, "est_cost_usd": round(self.cost(), 4)}


# ----------------------------------------------------------------------------- mock
def _overlap(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    return len(ta & tb) / max(1, len(ta))


class MockJudge:
    """Offline stand-in with the same interface (for smoke tests only)."""

    model = "mock"

    def __init__(self, *a, **k):
        self.n_calls = 0

    def run(self, reqs: list[JudgeRequest]) -> list[dict | None]:
        self.n_calls += len(reqs)
        return [getattr(self, f"_{r.kind}")(r.payload) for r in reqs]

    def stats(self) -> dict:
        return {"model": "mock", "api_calls": self.n_calls}

    def cost(self) -> float:
        return 0.0

    @staticmethod
    def _lab3(x: float, hi: float = 0.3, lo: float = 0.15, labs=("supported", "partial", "unsupported")) -> str:
        return labs[0] if x >= hi else labs[1] if x >= lo else labs[2]

    def _qgen(self, p):
        words = [w for w in tokenize(p["title"]) if len(w) > 3][:4] or ["this", "topic"]
        n = p["n_cited"]
        qs = [{"question": f"How do existing approaches to {' '.join(words)} compare in their assumptions and reported results?",
               "breadth": "focused", "general_knowledge": "no", "relevant_cited": list(range(1, min(n, 6) + 1))},
              {"question": f"What are the main limitations reported by studies on {' '.join(words[:2])}?",
               "breadth": "field", "general_knowledge": "yes", "relevant_cited": [1, 2]}]
        return {"questions": qs[: p["k"]]}

    def _kp(self, p):
        kps = [{"text": split_sentences(t)[0] if split_sentences(t) else t[:200], "doc_ids": [d], "importance": 3 - (i % 3)}
               for i, (d, t) in enumerate(zip(p["doc_ids"], p["doc_texts"]))][: p.get("max_kp", 10)]
        return {"relevant_doc_ids": list(p["doc_ids"]), "key_points": kps}

    _kp_qa = _kp
    _kp_rw = _kp

    def _rwpoints(self, p):
        return {"points": [{"text": s, "importance": 2} for s in split_sentences(p["related_work"])[: p.get("max_kp", 10)]]}

    def _support(self, p):
        res = []
        for i, (claim, docs) in enumerate(p["claims"], 1):
            best = max([_overlap(claim, d) for d in docs] or [0.0])
            res.append({"id": i, "label": self._lab3(best)})
        return {"results": res}

    _notes = _support

    def _ground(self, p):
        res = []
        for i, claim in enumerate(p["claims"], 1):
            best = max([_overlap(claim, d) for d in p["docs"]] or [0.0])
            res.append({"id": i, "label": "supported" if best >= 0.3 else "not_found"})
        return {"results": res}

    def _cover(self, p):
        res = []
        for i, pt in enumerate(p["points"], 1):
            x = _overlap(pt, p["answer"])
            res.append({"id": i, "label": "yes" if x >= 0.5 else "partial" if x >= 0.3 else "no"})
        return {"results": res}

    def _holistic(self, p):
        n_cit = len(re.findall(r"\[D\d+", p["answer"]))
        base = min(9, 3 + n_cit // 2)
        keys = ["synthesis", "specificity", "coverage", "critical", "organization"] + (["positioning"] if p.get("rw") else [])
        out = {k: base for k in keys}
        out["overall"] = base
        return out


class LexicalJudge(MockJudge):
    """Non-LLM 'NLP-metric' judge: support / coverage / grounding from content-word overlap (ROUGE-1-recall-like).
    Use it as a cheap reward ablation (--judge lexical in train_grpo); evaluation should still use the LLM judge."""

    model = "lexical"


def scores_dirname(judge: str, model: str | None) -> str:
    """Scores from different judges never share a folder (a results table must use a single judge).
    gpt-4o-mini keeps the original 'scores' folder."""
    if judge == "openai" and (model or "gpt-4o-mini") == "gpt-4o-mini":
        return "scores"
    tag = model if judge in ("openai", "vllm") and model else judge
    return "scores_" + re.sub(r"[^A-Za-z0-9._-]+", "_", tag)


def make_judge(name: str, model: str, cache_path: str | None, concurrency: int = 32, base_url: str | None = None,
               max_fail_frac: float | None = None):
    if name == "mock":
        return MockJudge()
    if name == "lexical":
        return LexicalJudge()
    if name == "vllm":  # open model behind a vLLM OpenAI-compatible server (started by the PBS job)
        if not base_url:
            raise ValueError("--judge vllm needs --judge_base_url http://<host>:<port>/v1")
        kw = {} if max_fail_frac is None else {"max_fail_frac": max_fail_frac}
        return OpenAIJudge(model=model, cache_path=cache_path, concurrency=concurrency, base_url=base_url,
                           api_key="EMPTY", price_in=0.0, price_out=0.0, timeout=600.0,
                           extra_body={"chat_template_kwargs": {"enable_thinking": False}}, **kw)
    kw = {} if max_fail_frac is None else {"max_fail_frac": max_fail_frac}
    return OpenAIJudge(model=model, cache_path=cache_path, concurrency=concurrency, base_url=base_url, **kw)


def add_args(p) -> None:
    g = p.add_argument_group("judge")
    g.add_argument("--judge", default="openai", choices=["openai", "vllm", "lexical", "mock"],
                   help="openai = gpt-4o-mini; vllm = open model behind a vLLM server (--judge_base_url, --judge_model "
                        "= served model name); lexical = word-overlap reward ablation; mock = tests")
    g.add_argument("--judge_model", default="gpt-4o-mini", help="judge for benchmark construction and RL rewards")
    g.add_argument("--eval_judge_model", default=None,
                   help="judge for final evaluation (default: same as --judge_model; use a different one to avoid "
                        "optimising and evaluating with the same judge)")
    g.add_argument("--judge_base_url", default=None, help="OpenAI-compatible endpoint (e.g. a vLLM server) instead of OpenAI")
    g.add_argument("--judge_concurrency", type=int, default=16)
    g.add_argument("--judge_max_fail_frac", type=float, default=0.05,
                   help="stop the stage if more than this fraction of judge calls in a batch fail")
