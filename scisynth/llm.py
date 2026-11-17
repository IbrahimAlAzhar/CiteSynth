"""Generation back-ends.

VLLMEngine  - local vLLM (tensor/pipeline parallel via Ray across nodes; LoRA hot-swap for RL)
EngineActor - the same engine inside a Ray actor (used for 2-GPU RL and data-parallel inference)
MultiEngine - fans a batch out to several EngineActors (data parallel) and gathers results in order
MockEngine  - deterministic CPU stand-in for smoke tests
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass

from .utils import LOG


@dataclass
class GenOut:
    text: str
    prompt_token_ids: list
    token_ids: list
    finish_reason: str = "stop"


def _model_type(model_path: str) -> tuple[str, bool]:
    try:
        with open(os.path.join(model_path, "config.json")) as f:
            cfg = json.load(f)
        return str(cfg.get("model_type", "")).lower(), ("vision_config" in cfg)
    except Exception:
        return "", False


class VLLMEngine:
    def __init__(self, model_path: str, tp: int = 1, pp: int = 1, max_model_len: int = 16384,
                 gpu_memory_utilization: float = 0.90, quantization: str | None = None, enable_lora: bool = False,
                 max_lora_rank: int = 32, dtype: str = "bfloat16", seed: int = 0, enforce_eager: bool = False,
                 distributed_backend: str | None = None, tokenizer_mode: str = "auto", enable_thinking: bool = False,
                 enable_prefix_caching: bool = True, max_num_seqs: int = 128, kv_cache_dtype: str = "auto",
                 enable_sleep_mode: bool = False):
        from vllm import LLM

        mtype, is_mm = _model_type(model_path)
        kw = dict(model=model_path, tensor_parallel_size=tp, pipeline_parallel_size=pp, max_model_len=max_model_len,
                  gpu_memory_utilization=gpu_memory_utilization, dtype=dtype, seed=seed, enforce_eager=enforce_eager,
                  enable_prefix_caching=enable_prefix_caching, trust_remote_code=True, max_num_seqs=max_num_seqs,
                  tokenizer_mode=tokenizer_mode)
        if quantization and quantization.lower() != "none":
            kw["quantization"] = quantization
        if kv_cache_dtype != "auto":
            kw["kv_cache_dtype"] = kv_cache_dtype
        if enable_lora:
            kw.update(enable_lora=True, max_lora_rank=max_lora_rank, max_loras=2)
        if distributed_backend:
            kw["distributed_executor_backend"] = distributed_backend
        if enable_sleep_mode:
            kw["enable_sleep_mode"] = True
        if is_mm:  # Gemma-3 / Mistral-Small-3.1 checkpoints are multimodal: run text-only
            kw["limit_mm_per_prompt"] = {"image": 0}
        LOG.info("Loading vLLM: %s", {k: v for k, v in kw.items() if k != "model"} | {"model": model_path})
        try:
            self.llm = LLM(**kw)
        except (TypeError, ValueError) as e:
            if "limit_mm_per_prompt" in kw and "limit_mm" in str(e):
                kw.pop("limit_mm_per_prompt")
                self.llm = LLM(**kw)
            else:
                raise
        self.tokenizer = self.llm.get_tokenizer()
        self.max_model_len = max_model_len
        self.chat_template_kwargs = {"enable_thinking": enable_thinking} if mtype.startswith("qwen3") else None
        self._lora_cache: dict = {}

    # -------------------------------------------------------------------- helpers
    def _count(self, conv: list[dict]) -> int:
        try:
            kw = dict(tokenize=True, add_generation_prompt=True)
            if self.chat_template_kwargs:
                kw.update(self.chat_template_kwargs)
            ids = self.tokenizer.apply_chat_template(conv, **kw)
            if not isinstance(ids, (list, tuple)):  # transformers>=5 returns a BatchEncoding
                ids = ids["input_ids"]
            if ids and isinstance(ids[0], (list, tuple)):
                ids = ids[0]
            return len(ids)
        except Exception:
            return int(sum(len(m["content"]) for m in conv) / 3.2) + 20 * len(conv)

    def _fit(self, conv: list[dict], max_tokens: int) -> list[dict]:
        """Shrink the longest message until prompt + generation fits max_model_len."""
        budget = self.max_model_len - max_tokens - 16
        for _ in range(6):
            n = self._count(conv)
            if n <= budget:
                return conv
            i = max(range(len(conv)), key=lambda j: len(conv[j]["content"]))
            words = conv[i]["content"].split(" ")
            keep = max(50, int(len(words) * budget / n * 0.95))
            conv = [dict(m) for m in conv]
            conv[i]["content"] = " ".join(words[:keep]) + " ..."
        return conv

    def _lora_request(self, lora):
        if not lora:
            return None
        from vllm.lora.request import LoRARequest

        name, lid, path = lora
        return LoRARequest(name, int(lid), path)

    # ------------------------------------------------------------------------ API
    def chat(self, convs: list[list[dict]], temperature: float = 0.0, top_p: float = 1.0, max_tokens: int = 1024,
             stop: list[str] | None = None, lora=None, seed: int | None = None) -> list[GenOut]:
        from vllm import SamplingParams

        if not convs:
            return []
        convs = [self._fit(c, max_tokens) for c in convs]
        sp = SamplingParams(temperature=temperature, top_p=top_p if temperature > 0 else 1.0, max_tokens=max_tokens,
                            stop=stop or None, seed=seed)
        kw = dict(use_tqdm=False, lora_request=self._lora_request(lora))
        if self.chat_template_kwargs:
            kw["chat_template_kwargs"] = self.chat_template_kwargs
        try:
            outs = self.llm.chat(convs, sp, **kw)
        except TypeError:
            kw.pop("chat_template_kwargs", None)
            outs = self.llm.chat(convs, sp, **kw)
        res = []
        for o in outs:
            c = o.outputs[0]
            res.append(GenOut(c.text, list(o.prompt_token_ids or []), list(c.token_ids), c.finish_reason or "stop"))
        return res

    def sleep(self) -> bool:
        """Free the GPU for the learner (weights -> CPU, KV cache dropped). Returns False if unsupported."""
        try:
            self.llm.sleep(level=1)
            return True
        except Exception as e:
            LOG.warning("vLLM sleep unavailable: %s", e)
            return False

    def wake_up(self) -> bool:
        try:
            self.llm.wake_up()
            return True
        except Exception as e:
            LOG.warning("vLLM wake_up unavailable: %s", e)
            return False


class EngineActor:
    """Plain class wrapped with ray.remote(num_gpus=...) at runtime."""

    def __init__(self, engine_kwargs: dict, env: dict | None = None):
        for k, v in (env or {}).items():
            os.environ[k] = str(v)
        if engine_kwargs.pop("mock", False):
            self.e = MockEngine(**engine_kwargs)
        else:
            self.e = VLLMEngine(**engine_kwargs)

    def chat(self, convs, **gen):
        return [asdict(g) for g in self.e.chat(convs, **gen)]

    def ready(self):
        return True


class RemoteEngine:
    """Synchronous facade over one or more EngineActors (data parallel)."""

    def __init__(self, actors: list):
        self.actors = actors

    def chat(self, convs, **gen) -> list[GenOut]:
        import ray

        if not convs:
            return []
        n = len(self.actors)
        chunks = [list(range(i, len(convs), n)) for i in range(n)]
        futs = [a.chat.remote([convs[j] for j in idx], **gen) for a, idx in zip(self.actors, chunks) if idx]
        results = ray.get(futs)
        out: list = [None] * len(convs)
        for idx, res in zip([c for c in chunks if c], results):
            for j, r in zip(idx, res):
                out[j] = GenOut(**r)
        return out


def launch_engine_actors(n: int, engine_kwargs: dict, num_gpus: float = 1.0, env: dict | None = None) -> RemoteEngine:
    import ray

    env = dict(env or {})
    env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    Actor = ray.remote(num_gpus=num_gpus)(EngineActor)
    actors = [Actor.remote(dict(engine_kwargs), env) for _ in range(n)]
    ray.get([a.ready.remote() for a in actors])
    return RemoteEngine(actors)


# ------------------------------------------------------------------------------ mock
class MockEngine:
    """Deterministic stand-in that follows the output protocols of every system prompt."""

    def __init__(self, tokenizer=None, **_):
        self.tokenizer = tokenizer

    def sleep(self) -> bool:   # CPU stand-in: nothing to free
        return True

    def wake_up(self) -> bool:
        return True

    def _ids(self, text: str) -> list[int]:
        if self.tokenizer is not None:
            return self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return [abs(hash(w)) % 1000 for w in text.split()]

    def chat(self, convs, temperature=0.0, max_tokens=1024, **_):
        outs = []
        for k, conv in enumerate(convs):
            sys_p = conv[0]["content"] if conv[0]["role"] == "system" else ""
            user0 = next((m["content"] for m in conv if m["role"] == "user"), "")
            last = conv[-1]["content"]
            ids_in_ctx = re.findall(r"\[(D\d+)\]", "\n".join(m["content"] for m in conv if m["role"] == "user"))
            n_assist = sum(1 for m in conv if m["role"] == "assistant")
            task_words = [w for w in re.findall(r"[A-Za-z]{5,}", user0)][:6]
            if "SEARCHER" in sys_p:
                text = f"<search>{' '.join(task_words[n_assist:n_assist + 4]) or 'related methods'}</search>" if n_assist < 2 else "<done/>"
            elif "READER" in sys_p:
                lines = []
                for d in dict.fromkeys(ids_in_ctx):
                    m = re.search(rf"\[{d}\]\s*(.*)", last)
                    lines.append(f"[{d}] " + " ".join((m.group(1) if m else "relevant finding").split()[:14]))
                text = "\n".join(lines[:8]) + "\nCONFLICTS: none"
            elif "search engine" in sys_p and "<answer>" in sys_p:  # ReAct
                text = (f"<search>{' '.join(task_words[:4])}</search>" if n_assist == 0 else
                        "<answer>" + self._compose(last + "\n" + "\n".join(m['content'] for m in conv), ids_in_ctx) + "</answer>")
            elif "<plan>" in sys_p:
                text = "<plan>themes</plan><answer>Prior studies examined this problem from several angles. Results differ across settings.</answer>"
            elif ids_in_ctx:
                text = self._compose("\n".join(m["content"] for m in conv), ids_in_ctx)
            else:
                text = "Prior studies examined this problem from several angles. Results differ across settings and remain debated."
            if temperature > 0 and k % 2 == 1:  # a bit of within-group variation for GRPO tests
                text = text.replace("[D", "[X", 1) if "[D" in text else text
            outs.append(GenOut(text, self._ids(" ".join(m["content"] for m in conv))[:512], self._ids(text)[:max_tokens]))
        return outs

    @staticmethod
    def _compose(ctx: str, ids: list[str]) -> str:
        sents = []
        for d in list(dict.fromkeys(ids))[:5]:
            m = re.search(rf"\[{d}\]\s*([^\n]*)", ctx)
            words = (m.group(1) if m else "a related approach").split()[:16]
            sents.append(f"One line of work reports that {' '.join(words)} [{d}].")
        sents.append("These studies differ in their assumptions and evaluation settings.")
        return " ".join(sents)


def add_args(p) -> None:
    g = p.add_argument_group("engine")
    g.add_argument("--model_path", default=None)
    g.add_argument("--model_tag", default=None, help="short name used in output paths")
    g.add_argument("--engine", default="vllm", choices=["vllm", "mock"])
    g.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    g.add_argument("--pp", type=int, default=1, help="pipeline parallel size")
    g.add_argument("--dp", type=int, default=1, help="data-parallel engine replicas (Ray actors, 1 GPU each)")
    g.add_argument("--ray_address", default=None, help="'auto' to join the Ray cluster started by the PBS script")
    g.add_argument("--max_model_len", type=int, default=16384)
    g.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    g.add_argument("--quantization", default=None, help="e.g. fp8 (weight-only on A100) to fit 24B/27B on one 40GB GPU")
    g.add_argument("--kv_cache_dtype", default="auto")
    g.add_argument("--tokenizer_mode", default="auto", help="'mistral' for consolidated Mistral checkpoints")
    g.add_argument("--enforce_eager", action="store_true")
    g.add_argument("--max_num_seqs", type=int, default=128)
    g.add_argument("--enable_thinking", action="store_true", help="Qwen3 thinking mode (off by default)")


def engine_kwargs_from_args(args, enable_lora: bool = False, gpu_mem: float | None = None, max_lora_rank: int = 32,
                            enable_sleep_mode: bool = False) -> dict:
    distributed = "ray" if (args.tp * args.pp > 1 and args.ray_address) else None
    return dict(model_path=args.model_path, tp=args.tp, pp=args.pp, max_model_len=args.max_model_len,
                gpu_memory_utilization=gpu_mem or args.gpu_memory_utilization, quantization=args.quantization,
                enable_lora=enable_lora, max_lora_rank=max_lora_rank, seed=args.seed, enforce_eager=args.enforce_eager,
                distributed_backend=distributed, tokenizer_mode=args.tokenizer_mode, enable_thinking=args.enable_thinking,
                max_num_seqs=args.max_num_seqs, kv_cache_dtype=args.kv_cache_dtype, enable_sleep_mode=enable_sleep_mode)


def build_engine(args, enable_lora: bool = False, gpu_mem: float | None = None, max_lora_rank: int = 32):
    """Return an object with .chat(convs, **gen) -> list[GenOut]."""
    if args.engine == "mock" and args.dp > 1:  # exercises the Ray data-parallel path without GPUs
        import ray

        if not ray.is_initialized():
            ray.init(address=args.ray_address or "auto", runtime_env={"env_vars": _ray_env()})
        return launch_engine_actors(args.dp, {"mock": True}, num_gpus=1)
    if args.engine == "mock":
        tok = None
        if args.model_path and os.path.isdir(args.model_path):
            try:
                from transformers import AutoTokenizer

                tok = AutoTokenizer.from_pretrained(args.model_path)
            except Exception:
                tok = None
        return MockEngine(tokenizer=tok)
    kw = engine_kwargs_from_args(args, enable_lora, gpu_mem, max_lora_rank)
    if args.dp > 1:
        import ray

        if not ray.is_initialized():
            ray.init(address=args.ray_address or "auto", runtime_env={"env_vars": _ray_env()})
        return launch_engine_actors(args.dp, kw, num_gpus=args.tp * args.pp)
    if kw["distributed_backend"] == "ray":
        import ray

        if not ray.is_initialized():
            ray.init(address=args.ray_address, runtime_env={"env_vars": _ray_env()})
    return VLLMEngine(**kw)


def _ray_env() -> dict:
    keep = ("PYTHONPATH", "HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "VLLM_ENABLE_V1_MULTIPROCESSING", "NCCL_SOCKET_IFNAME",
            "GLOO_SOCKET_IFNAME", "VLLM_HOST_IP", "TMPDIR")
    return {k: os.environ[k] for k in keep if k in os.environ}
