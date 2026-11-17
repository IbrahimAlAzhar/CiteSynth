"""LoRA / QLoRA learner used by GRPO and by the rejection-sampling SFT baseline.

Works in-process (1 GPU, colocated with vLLM) or as a Ray actor on its own GPU (2-GPU mode).
Samples are dicts {prompt_ids, completion_ids, adv}; completion log-probs are computed with left padding and
``logits_to_keep`` so that the full-vocabulary logits are only materialised for completion positions.
"""
from __future__ import annotations

import inspect
import json
import math
import os
import time

from .utils import LOG

LORA_SHORT = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _dtype_kw():
    import torch
    import transformers

    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return {"dtype": torch.bfloat16} if (major, minor) >= (4, 56) else {"torch_dtype": torch.bfloat16}


def load_causal_lm(path: str, quant: str = "none", attn_impl: str = "sdpa"):
    import torch
    from transformers import AutoModelForCausalLM

    kw = dict(_dtype_kw(), attn_implementation=attn_impl, trust_remote_code=True, low_cpu_mem_usage=True)
    if quant == "4bit":
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        kw["device_map"] = {"": 0}
    try:
        model = AutoModelForCausalLM.from_pretrained(path, **kw)
    except (ValueError, KeyError) as e:  # multimodal checkpoints (Gemma-3 27B, Mistral-Small-3.1)
        LOG.info("AutoModelForCausalLM failed (%s); trying image-text-to-text class", str(e)[:120])
        try:
            from transformers import AutoModelForImageTextToText as Auto
        except ImportError:
            from transformers import AutoModelForVision2Seq as Auto
        model = Auto.from_pretrained(path, **kw)
    if quant != "4bit":
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model


def lora_target(model):
    names = [n for n, _ in model.named_modules()]
    if any("language_model" in n for n in names):  # restrict to the text decoder of multimodal checkpoints
        return r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
    present = {n.split(".")[-1] for n in names}
    return [m for m in LORA_SHORT if m in present] or "all-linear"


def selective_log_softmax(logits, index):
    import torch

    out = []
    for row_l, row_i in zip(logits, index):
        row_l = row_l.float()
        out.append(torch.gather(row_l, -1, row_i.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(row_l, dim=-1))
    return torch.stack(out)


class Learner:
    def __init__(self, model_path: str, lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.0,
                 lr: float = 1e-5, weight_decay: float = 0.0, quant: str = "none", grad_ckpt: bool = True,
                 max_tokens_per_micro: int = 8192, max_seq_len: int = 16384, clip_eps_low: float = 0.2,
                 clip_eps_high: float = 0.28, kl_coef: float = 0.0, max_grad_norm: float = 1.0,
                 resume_from: str | None = None, attn_impl: str = "sdpa", seed: int = 0, env: dict | None = None,
                 init_from: str | None = None):
        for k, v in (env or {}).items():
            os.environ[k] = str(v)
        import torch
        from peft import LoraConfig, get_peft_model

        torch.manual_seed(seed)
        self.torch = torch
        self.model_path = model_path
        model = load_causal_lm(model_path, quant, attn_impl)
        # QLoRA note: we deliberately skip peft.prepare_model_for_kbit_training, which up-casts all non-quantised
        # weights (e.g. Gemma-3's 262k x 5376 embedding) to fp32 and costs several GB on a 40GB card.
        self.target = lora_target(model)
        cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, target_modules=self.target,
                         bias="none", task_type="CAUSAL_LM")
        self.model = get_peft_model(model, cfg)
        if grad_ckpt:
            try:
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                self.model.gradient_checkpointing_enable()
            try:
                self.model.enable_input_require_grads()
            except Exception:
                pass
        try:
            self.model.config.use_cache = False
        except Exception:
            pass
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(self.params, lr=lr, betas=(0.9, 0.99), weight_decay=weight_decay, eps=1e-8)
        self.device = next(p for p in self.model.parameters()).device
        self.max_tokens_per_micro, self.max_seq_len = max_tokens_per_micro, max_seq_len
        self.clip_lo, self.clip_hi, self.kl_coef, self.max_grad_norm = clip_eps_low, clip_eps_high, kl_coef, max_grad_norm
        fwd = inspect.signature(self.model.get_base_model().forward).parameters
        self.keep_kw = "logits_to_keep" if "logits_to_keep" in fwd else ("num_logits_to_keep" if "num_logits_to_keep" in fwd else None)
        from transformers import AutoTokenizer

        try:
            tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        except Exception:
            self.pad_id = 0
        self.pad_id = 0 if self.pad_id is None else int(self.pad_id)
        if resume_from:
            self.load(resume_from)
        elif init_from:  # v2: warm start (e.g. from the DPO adapter): LoRA weights only, fresh optimizer
            self.load(init_from, with_optimizer=False)
        n_train = sum(p.numel() for p in self.params)
        LOG.info("Learner ready: %s | LoRA r=%d target=%s | trainable %.1fM | keep_kw=%s", model_path, lora_r,
                 self.target if isinstance(self.target, str) else ",".join(self.target), n_train / 1e6, self.keep_kw)

    def ready(self):
        return True

    # ------------------------------------------------------------------ batching
    def _micro_batches(self, samples):
        ok = [s for s in samples if 0 < len(s["completion_ids"]) and len(s["prompt_ids"]) + len(s["completion_ids"]) <= self.max_seq_len]
        if len(ok) < len(samples):
            LOG.warning("dropped %d over-long/empty samples", len(samples) - len(ok))
        ok.sort(key=lambda s: len(s["prompt_ids"]) + len(s["completion_ids"]))
        batches, cur, cur_max = [], [], 0
        for s in ok:
            L = len(s["prompt_ids"]) + len(s["completion_ids"])
            if cur and max(cur_max, L) * (len(cur) + 1) > self.max_tokens_per_micro:
                batches.append(cur)
                cur, cur_max = [], 0
            cur.append(s)
            cur_max = max(cur_max, L)
        if cur:
            batches.append(cur)
        return batches

    def _collate(self, batch):
        torch = self.torch
        L = max(len(s["prompt_ids"]) + len(s["completion_ids"]) for s in batch)
        K = max(len(s["completion_ids"]) for s in batch)
        ids = torch.full((len(batch), L), self.pad_id, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        cmask = torch.zeros((len(batch), K), dtype=torch.float32)
        for i, s in enumerate(batch):
            seq = list(s["prompt_ids"]) + list(s["completion_ids"])
            ids[i, L - len(seq):] = torch.tensor(seq, dtype=torch.long)
            att[i, L - len(seq):] = 1
            cmask[i, K - len(s["completion_ids"]):] = 1.0
        return ids.to(self.device), att.to(self.device), cmask.to(self.device), K

    def _logps(self, ids, att, K, disable_adapter: bool = False):
        torch = self.torch
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        kw = dict(input_ids=ids, attention_mask=att, position_ids=pos, use_cache=False)
        if self.keep_kw:
            kw[self.keep_kw] = K + 1
        if disable_adapter:
            with self.model.disable_adapter():
                logits = self.model(**kw).logits
        else:
            logits = self.model(**kw).logits
        logits = logits[:, -(K + 1): -1, :]
        return selective_log_softmax(logits, ids[:, -K:])

    # ---------------------------------------------------------------------- GRPO
    def train_step(self, samples: list[dict], n_updates: int = 1) -> dict:
        torch = self.torch
        t0 = time.time()
        self.model.train()
        batches = self._micro_batches(samples)
        if not batches:
            return {"n_samples": 0}
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        # old log-probs are only needed when we take more than one optimizer step on the same rollouts
        old = {}
        if n_updates > 1:
            with torch.no_grad():
                for bi, b in enumerate(batches):
                    ids, att, cm, K = self._collate(b)
                    old[bi] = self._logps(ids, att, K).detach()
        stats = {"loss": 0.0, "kl": 0.0, "clipfrac": 0.0, "grad_norm": 0.0}
        order = list(range(len(batches)))
        chunks = [order[i::n_updates] for i in range(n_updates)]
        tok_total = 0
        for chunk in chunks:
            n_tok = sum(len(s["completion_ids"]) for bi in chunk for s in batches[bi])
            tok_total += n_tok
            self.opt.zero_grad(set_to_none=True)
            for bi in chunk:
                ids, att, cm, K = self._collate(batches[bi])
                adv = torch.tensor([s["adv"] for s in batches[bi]], dtype=torch.float32, device=self.device).unsqueeze(1)
                logp = self._logps(ids, att, K)
                old_lp = old[bi] if bi in old else logp.detach()
                ratio = torch.exp(logp - old_lp)
                pg = -torch.minimum(ratio * adv, torch.clamp(ratio, 1 - self.clip_lo, 1 + self.clip_hi) * adv)
                tok_loss = pg
                if self.kl_coef > 0:
                    with torch.no_grad():
                        ref = self._logps(ids, att, K, disable_adapter=True)
                    d = ref - logp
                    kl = torch.exp(d) - d - 1
                    tok_loss = tok_loss + self.kl_coef * kl
                    stats["kl"] += float((kl.detach() * cm).sum()) / max(1, n_tok)
                loss = (tok_loss * cm).sum() / max(1, n_tok)
                loss.backward()
                stats["loss"] += float(loss.detach())
                stats["clipfrac"] += float((((ratio - 1).abs() > self.clip_lo).float() * cm).sum()) / max(1, n_tok)
            gn = torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
            stats["grad_norm"] += float(gn) / n_updates
            self.opt.step()
        stats.update({"n_samples": sum(len(b) for b in batches), "n_micro": len(batches), "n_tokens": tok_total,
                      "learner_s": round(time.time() - t0, 1),
                      "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else 0.0})
        return stats

    # ----------------------------------------------------------------------- SFT
    def sft_step(self, samples: list[dict]) -> dict:
        torch = self.torch
        self.model.train()
        batches = self._micro_batches(samples)
        n_tok = sum(len(s["completion_ids"]) for b in batches for s in b)
        self.opt.zero_grad(set_to_none=True)
        tot = 0.0
        for b in batches:
            ids, att, cm, K = self._collate(b)
            loss = -(self._logps(ids, att, K) * cm).sum() / max(1, n_tok)
            loss.backward()
            tot += float(loss)
        gn = torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
        self.opt.step()
        return {"nll": tot, "grad_norm": float(gn), "n_tokens": n_tok}

    # ----------------------------------------------------------------------- DPO
    def seq_logps(self, samples: list[dict], ref: bool = False) -> list[float]:
        """Sum of completion log-probs for each sample (no gradient), in input order.
        ref=True uses the frozen base model (LoRA adapter disabled) = the DPO reference policy."""
        torch = self.torch
        self.model.eval()
        out = [0.0] * len(samples)
        order = sorted(range(len(samples)), key=lambda i: len(samples[i]["prompt_ids"]) + len(samples[i]["completion_ids"]))

        def flush(idx):
            ids, att, cm, K = self._collate([samples[i] for i in idx])
            with torch.no_grad():
                lp = self._logps(ids, att, K, disable_adapter=ref)
            for i, v in zip(idx, (lp * cm).sum(-1).tolist()):
                out[i] = float(v)

        batch, cur_max = [], 0
        for i in order:
            L = len(samples[i]["prompt_ids"]) + len(samples[i]["completion_ids"])
            if batch and max(cur_max, L) * (len(batch) + 1) > self.max_tokens_per_micro:
                flush(batch)
                batch, cur_max = [], 0
            batch.append(i)
            cur_max = max(cur_max, L)
        if batch:
            flush(batch)
        return out

    def dpo_step(self, pairs: list[dict], beta: float = 0.1) -> dict:
        """One DPO update on trajectory pairs (multi-turn: log pi(trajectory) = sum over its turns).

        pairs: [{"chosen": [turn samples], "rejected": [turn samples], "ref_c": float, "ref_r": float}]
        Uses the exact DPO gradient  -beta*sigmoid(-beta*D) * (grad log pi_c - grad log pi_r)  with
        D = (log pi_c - ref_c) - (log pi_r - ref_r), applied as per-turn weights through train_step.
        """
        flat, owner = [], []
        for pi, p in enumerate(pairs):
            for side, key in ((0, "chosen"), (1, "rejected")):
                for s in p[key]:
                    flat.append(s)
                    owner.append((pi, side))
        lps = self.seq_logps(flat)
        pol = [[0.0, 0.0] for _ in pairs]
        for (pi, side), v in zip(owner, lps):
            pol[pi][side] += v
        deltas = [(pol[i][0] - p["ref_c"]) - (pol[i][1] - p["ref_r"]) for i, p in enumerate(pairs)]
        coefs = [beta * 0.5 * (1.0 - math.tanh(beta * d / 2.0)) for d in deltas]   # beta * sigmoid(-beta*d)
        samples = [{"prompt_ids": s["prompt_ids"], "completion_ids": s["completion_ids"],
                    "adv": coefs[pi] if side == 0 else -coefs[pi]} for s, (pi, side) in zip(flat, owner)]
        st = self.train_step(samples)

        def softplus(x):
            return max(0.0, x) + math.log1p(math.exp(-abs(x)))

        st.update({"dpo_loss": sum(softplus(-beta * d) for d in deltas) / len(deltas),
                   "reward_acc": sum(d > 0 for d in deltas) / len(deltas),
                   "margin": beta * sum(deltas) / len(deltas)})
        return st

    # ------------------------------------------------------------------ save/load
    def save(self, path: str, with_optimizer: bool = True) -> str:
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        cfg_path = os.path.join(path, "adapter_config.json")
        try:  # vLLM wants explicit module names, not a regex
            with open(cfg_path) as f:
                cfg = json.load(f)
            cfg["target_modules"] = LORA_SHORT if isinstance(cfg.get("target_modules"), str) else cfg.get("target_modules")
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2)
        except Exception as e:
            LOG.warning("could not patch adapter_config.json: %s", e)
        if with_optimizer:
            self.torch.save(self.opt.state_dict(), os.path.join(path, "optimizer.pt"))
        return path

    def load(self, path: str, with_optimizer: bool = True) -> None:
        from peft import set_peft_model_state_dict

        st = os.path.join(path, "adapter_model.safetensors")
        if os.path.exists(st):
            from safetensors.torch import load_file

            sd = load_file(st)
        else:
            sd = self.torch.load(os.path.join(path, "adapter_model.bin"), map_location="cpu")
        res = set_peft_model_state_dict(self.model, sd)
        n_unexp = len(getattr(res, "unexpected_keys", []) or [])
        LOG.info("Loaded adapter from %s (unexpected keys: %d)", path, n_unexp)
        if n_unexp:
            LOG.warning("adapter %s does not match this LoRA config (check --lora_r/--lora_alpha)", path)
        opt_path = os.path.join(path, "optimizer.pt")
        if with_optimizer and os.path.exists(opt_path):
            try:
                self.opt.load_state_dict(self.torch.load(opt_path, map_location="cpu"))
            except Exception as e:
                LOG.warning("optimizer state not restored: %s", e)

    def offload(self):
        """Move weights to CPU (1-GPU 'sleep' mode: vLLM generates while the learner waits in CPU memory)."""
        self.model.to("cpu")
        self._move_optimizer("cpu")
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def onload(self):
        self.model.to(self.device)
        self._move_optimizer(self.device)

    def _move_optimizer(self, device):
        for st in self.opt.state.values():
            for k, v in st.items():
                if self.torch.is_tensor(v) and k != "step":
                    st[k] = v.to(device)


def add_args(p) -> None:
    g = p.add_argument_group("learner")
    g.add_argument("--lora_r", type=int, default=16)
    g.add_argument("--lora_alpha", type=int, default=32)
    g.add_argument("--lr", type=float, default=1e-5)
    g.add_argument("--learner_quant", default="none", choices=["none", "4bit"], help="4bit = QLoRA (needs bitsandbytes)")
    g.add_argument("--max_tokens_per_micro", type=int, default=8192)
    g.add_argument("--clip_eps_low", type=float, default=0.2)
    g.add_argument("--clip_eps_high", type=float, default=0.28)
    g.add_argument("--kl_coef", type=float, default=0.0)
    g.add_argument("--max_grad_norm", type=float, default=1.0)
    g.add_argument("--attn_impl", default="sdpa")
    g.add_argument("--no_grad_ckpt", dest="grad_ckpt", action="store_false")


def learner_kwargs(args, resume_from=None) -> dict:
    return dict(model_path=args.model_path, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lr=args.lr,
                quant=args.learner_quant, grad_ckpt=args.grad_ckpt, max_tokens_per_micro=args.max_tokens_per_micro,
                max_seq_len=args.max_model_len, clip_eps_low=args.clip_eps_low, clip_eps_high=args.clip_eps_high,
                kl_coef=args.kl_coef, max_grad_norm=args.max_grad_norm, resume_from=resume_from,
                attn_impl=args.attn_impl, seed=args.seed,
                init_from=None if resume_from else getattr(args, "init_adapter", None))
