"""GRPO for multi-turn literature-synthesis agents with outcome vs process rewards.

Advantages (per group of G rollouts of the same item):
  * outcome / outcome_judge / process_sum : A_i = norm(R_i), shared by every turn of trajectory i
  * process_turn (turn-level credit assignment, cf. Zeng et al. 2025; SAPO 2026):
        A_{i,t} = norm(R_i) + lambda * norm_phase(r_{i,t})       (--credit turn)
        A_{i,t} = norm(R_i) + lambda * |norm(R_i)| * norm_phase(r_{i,t})   (--credit sapo)
    where norm_phase standardises turn rewards across all turns of the same phase (search/read/synth) in the group.

Hardware:
  * --rl_gpus 1 : vLLM rollout engine and LoRA learner share one GPU (vLLM gets --rollout_gpu_mem of it)
  * --rl_gpus 2 : Ray actors: vLLM on one GPU, learner on another (works across two PBS nodes)
LoRA weights are synced to vLLM every step by saving the adapter and hot-loading it with a new LoRA id.

Version 2 additions (all optional flags):
  --init_adapter   start from a trained LoRA (DPO -> GRPO); vLLM generates the first rollouts with that adapter
  --colocate sleep 1-GPU mode in bf16: the rollout model and the trained model are the same weights
  --prior_scores   curriculum: questions whose earlier samples disagreed are drawn more often (fewer dead groups)
  --eval_at_start  dev score of the starting policy = step 0, so the selected 'best' checkpoint is never worse on dev
  --patience       early stopping on the dev metric
  (--recall_bonus in rewards.py adds w * gold_found to the training reward only)
"""
from __future__ import annotations

import os
import random
import shutil
import time
from collections import defaultdict

import numpy as np

from .benchmark import load_items
from .data import Corpus
from .judge import make_judge
from .learner import Learner, learner_kwargs
from .llm import VLLMEngine, _ray_env, engine_kwargs_from_args, launch_engine_actors
from .rewards import Scorer, reward_config_from_args
from .systems import Runner, config_from_args, make_episodes
from .utils import LOG, append_jsonl, read_json, write_json


# --------------------------------------------------------------------------- advantages
def compute_advantages(episodes, mode: str, lam: float = 0.5, norm: str = "std", credit: str = "turn", eps: float = 1e-6) -> dict:
    groups = defaultdict(list)
    for ep in episodes:
        groups[ep.item["item_id"]].append(ep)
    dead, proc_rescued = 0, 0
    for g in groups.values():
        R = np.array([ep.rewards["outcome"] for ep in g], dtype=float)
        A = R - R.mean()
        sd = R.std()
        if norm == "std":
            A = A / (sd + eps) if sd > eps else np.zeros_like(A)
        if sd <= eps:
            dead += 1
        for ep, a in zip(g, A):
            ep.rewards["adv"] = float(a)
            for t in ep.turns:
                t.adv = float(a)
        if mode != "process_turn":
            continue
        by_phase = defaultdict(list)
        for ep in g:
            for t in ep.turns:
                if t.phase in ("search", "read", "synth") and t.reward is not None:
                    by_phase[t.phase].append(t)
        any_signal = False
        for ts in by_phase.values():
            r = np.array([t.reward for t in ts], dtype=float)
            if len(ts) < 2 or r.std() <= eps:
                continue
            z = (r - r.mean()) / (r.std() + eps) if norm == "std" else (r - r.mean())
            any_signal = True
            for t, zz in zip(ts, z):
                t.adv = float(t.adv + (lam * zz if credit == "turn" else lam * abs(t.adv) * zz))
        if sd <= eps and any_signal:
            proc_rescued += 1
    return {"groups": len(groups), "dead_groups": dead, "rescued_by_process": proc_rescued}


def build_samples(episodes, phases: set[str], mask_truncated: bool = False) -> list[dict]:
    out = []
    for ep in episodes:
        for t in ep.turns:
            if t.phase not in phases or not t.token_ids or t.adv is None or abs(t.adv) < 1e-8:
                continue
            if mask_truncated and t.finish_reason == "length":
                continue
            out.append({"prompt_ids": list(t.prompt_token_ids), "completion_ids": list(t.token_ids), "adv": float(t.adv)})
    return out


# --------------------------------------------------------------------------- learner handle
class _LearnerHandle:
    """Same calls for an in-process Learner or a Ray actor."""

    def __init__(self, obj, remote: bool):
        self.obj, self.remote = obj, remote

    def __getattr__(self, name):
        attr = getattr(self.obj, name)
        if not self.remote:
            return attr
        import ray

        return lambda *a, **k: ray.get(attr.remote(*a, **k))


def _summ(episodes) -> dict:
    keys = ["outcome", "outcome_verifiable", "coverage", "support", "cite_f1", "cite_valid", "pool_recall", "fmt",
            "words", "process_mean", "notes_support", "ground_prec", "n_search_turns"]
    out = {}
    for k in keys:
        v = [ep.rewards.get(k) for ep in episodes if ep.rewards.get(k) is not None]
        if v:
            out[k] = round(float(np.mean(v)), 4)
    for ph in ("search", "read", "synth", "direct"):
        v = [t.reward for ep in episodes for t in ep.turns if t.phase == ph and t.reward is not None]
        if v:
            out[f"r_{ph}"] = round(float(np.mean(v)), 4)
    return out


def _item_weights(train_items: list[dict], prior_path: str | None, dead_w: float, unseen_w: float, eps: float = 0.02):
    """v2 curriculum: sampling weight per training question from earlier sampled answers (version_1, 4 per question).
    Questions whose samples disagreed (reward std > eps) give GRPO a learning signal -> weight 1; questions whose
    samples all scored the same mostly give 'dead' groups (zero advantage) -> dead_w; questions never sampled -> unseen_w."""
    if not prior_path:
        return None, {}
    if not os.path.exists(prior_path):
        LOG.warning("prior scores %s not found: uniform question sampling", prior_path)
        return None, {}
    from .utils import read_jsonl

    by = defaultdict(list)
    for r in read_jsonl(prior_path):
        rw = r.get("rewards") or {}
        v = rw.get("outcome_verifiable", rw.get("outcome"))
        if isinstance(v, (int, float)):
            by[r["item_id"]].append(float(v))
    w, n = [], {"informative": 0, "dead": 0, "unseen": 0}
    for it in train_items:
        xs = by.get(it["item_id"])
        if not xs or len(xs) < 2:
            w.append(unseen_w)
            n["unseen"] += 1
        elif float(np.std(xs)) <= eps:
            w.append(dead_w)
            n["dead"] += 1
        else:
            w.append(1.0)
            n["informative"] += 1
    w = np.asarray(w, dtype=float)
    if w.sum() <= 0:
        return None, n
    return w / w.sum(), n


def _sample_batch(train_items, weights, k, seed):
    rng = np.random.default_rng(seed)
    k = min(k, len(train_items))
    idx = rng.choice(len(train_items), size=k, replace=False, p=weights)
    return [train_items[i] for i in idx]


def train_grpo(args) -> None:
    out_dir = os.path.join(args.out_dir, "grpo", f"{args.task}_{args.train_system}_{args.reward_mode}{args.run_suffix}")
    ckpt_dir = os.path.join(out_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    state_path = os.path.join(out_dir, "trainer_state.json")
    state = read_json(state_path) if os.path.exists(state_path) else {"step": 0, "best_dev": -1e9, "best_step": None, "lora_id": 0}
    if state.get("finished") and not args.overwrite:
        LOG.info("Run %s already finished; skip (use --overwrite to retrain)", out_dir)
        return
    if args.overwrite and os.path.exists(state_path):   # retrain from scratch: drop old checkpoints and log
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        for f in ("train_log.jsonl", "trainer_state.json"):
            if os.path.exists(os.path.join(out_dir, f)):
                os.remove(os.path.join(out_dir, f))
        state = {"step": 0, "best_dev": -1e9, "best_step": None, "lora_id": 0}
    resume = os.path.join(ckpt_dir, "latest") if state["step"] > 0 and os.path.exists(os.path.join(ckpt_dir, "latest")) else None
    if args.init_adapter and not os.path.exists(os.path.join(args.init_adapter, "adapter_config.json")):
        raise FileNotFoundError(f"--init_adapter {args.init_adapter} has no adapter_config.json")

    train_items = load_items(args.data_dir, args.task, "train", args.train_limit)
    dev_items = load_items(args.data_dir, args.task, "dev", args.n_dev or None)
    if not train_items:
        raise RuntimeError("no training items - run build_bench first")
    weights, wstats = _item_weights(train_items, args.prior_scores, args.dead_item_weight, args.unseen_item_weight)
    LOG.info("GRPO %s: %d train items (curriculum %s), %d dev items, init=%s -> %s", args.reward_mode, len(train_items),
             wstats or "uniform", len(dev_items), args.init_adapter or "base model", out_dir)
    write_json(os.path.join(out_dir, "run_config.json"),
               {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, list, type(None)))} | {"curriculum": wstats})
    corpus = Corpus(args.data_dir)
    from .retrieval import Retriever

    retriever = Retriever(args.data_dir, args.retriever_mode, args.embed_model)
    judge = make_judge(args.judge, args.judge_model, os.path.join(out_dir, "judge_cache.jsonl"), args.judge_concurrency, args.judge_base_url,
                       args.judge_max_fail_frac)
    scorer = Scorer(judge, corpus, reward_config_from_args(args))
    phases = set(args.train_phases)

    # ---- engines: the SAME model generates the rollouts (vLLM + current LoRA) and is trained (learner, same LoRA)
    sleep_mode = False
    lkw = learner_kwargs(args, resume_from=resume)
    if args.engine == "mock" and args.rl_gpus >= 2:  # Ray placement test without GPUs
        import ray

        ray.init(address=args.ray_address or "auto", runtime_env={"env_vars": _ray_env()})
        engine = launch_engine_actors(max(1, args.rl_gpus - 1), {"mock": True}, num_gpus=1)
        learner = _LearnerHandle(ray.remote(num_gpus=1)(_MockLearner).remote(), True)
        learner.ready()
    elif args.engine == "mock":
        from .llm import MockEngine

        engine = MockEngine()
        learner = _LearnerHandle(_MockLearner(), False)
        sleep_mode = args.colocate == "sleep"
    elif args.rl_gpus >= 2:
        # GPU A: vLLM rollouts in bf16 (or --quantization), GPU B: LoRA learner in bf16
        import ray

        ray.init(address=args.ray_address or "auto", runtime_env={"env_vars": _ray_env()})
        ekw = engine_kwargs_from_args(args, enable_lora=True, gpu_mem=args.gpu_memory_utilization, max_lora_rank=max(args.lora_r, 8))
        ekw["tp"], ekw["pp"], ekw["distributed_backend"] = 1, 1, None
        engine = launch_engine_actors(max(1, args.rl_gpus - 1), ekw, num_gpus=1)
        lkw["env"] = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        actor = ray.remote(num_gpus=1)(Learner).remote(**lkw)
        learner = _LearnerHandle(actor, True)
        learner.ready()
    elif args.colocate == "sleep":
        # one GPU, bf16 for both: the learner waits in CPU memory while vLLM generates; vLLM sleeps while we train
        lobj = Learner(**lkw)
        lobj.offload()
        ekw = engine_kwargs_from_args(args, enable_lora=True, gpu_mem=args.gpu_memory_utilization, max_lora_rank=max(args.lora_r, 8),
                                      enable_sleep_mode=True)
        engine = VLLMEngine(**ekw)
        learner = _LearnerHandle(lobj, False)
        sleep_mode = True
    else:  # "shared": vLLM and learner resident on one GPU at the same time (needs --quantization fp8 for 8B models)
        ekw = engine_kwargs_from_args(args, enable_lora=True, gpu_mem=args.rollout_gpu_mem, max_lora_rank=max(args.lora_r, 8))
        engine = VLLMEngine(**ekw)
        learner = _LearnerHandle(Learner(**lkw), False)
    LOG.info("rollout/policy placement: %s", "ray (rollout GPU + learner GPU)" if args.rl_gpus >= 2 else f"one GPU, colocate={args.colocate}")

    lora = None
    if resume:
        state["lora_id"] += 1
        lora = (f"resume{state['lora_id']}", state["lora_id"], resume)
    elif args.init_adapter:
        state["lora_id"] += 1
        lora = (f"init{state['lora_id']}", state["lora_id"], os.path.abspath(args.init_adapter))

    cfg_train = config_from_args(args, temperature=args.train_temperature)
    cfg_eval = config_from_args(args, temperature=0.0)
    t_start = time.time()
    log_path = os.path.join(out_dir, "train_log.jsonl")
    if os.path.exists(log_path):  # after a crash / walltime kill: drop log lines of steps that will be redone
        from .utils import read_jsonl, write_jsonl

        kept = [r for r in read_jsonl(log_path) if r.get("step", 0) <= state["step"]]
        write_jsonl(log_path, kept)

    def dev_eval(cur_lora) -> dict:
        dev_eps = make_episodes(dev_items, args.train_system, 1)
        Runner(cfg_eval, corpus, retriever, engine, cur_lora).run(dev_eps)
        scorer.score(dev_eps, purpose="dev", reward_mode="outcome")   # same formula as the test metric, no bonus
        return _summ(dev_eps)

    def maybe_new_best(ds: dict, step: int, rec: dict) -> None:
        val = ds.get(args.select_metric)
        if val is not None and val > state["best_dev"] + args.min_improve:
            state["best_dev"], state["best_step"] = float(val), step
            learner.save(os.path.join(ckpt_dir, "best"), with_optimizer=False)
            rec["new_best"] = True
            state["evals_since_best"] = 0
        else:
            state["evals_since_best"] = state.get("evals_since_best", 0) + 1

    # ---- step 0: score the starting policy on dev, so that "best" can never be worse than the start (e.g. DPO)
    if args.eval_at_start and dev_items and state["step"] == 0 and not state.get("start_evaluated"):
        ds = dev_eval(lora)
        rec = {"step": 0, **{f"dev/{k}": v for k, v in ds.items()}, "judge_cost_usd": round(judge.cost(), 3)}
        maybe_new_best(ds, 0, rec)
        state["start_dev"], state["start_evaluated"] = ds.get(args.select_metric), True
        write_json(state_path, state)
        append_jsonl(log_path, [rec])
        LOG.info("step 0 (start policy) dev %s = %.4f", args.select_metric, ds.get(args.select_metric, float("nan")))

    for step in range(state["step"], args.total_steps):
        t0 = time.time()
        batch = _sample_batch(train_items, weights, args.prompts_per_step, args.seed * 100003 + step)
        eps = make_episodes(batch, args.train_system, args.group_size)
        Runner(cfg_train, corpus, retriever, engine, lora).run(eps)
        t_roll = time.time() - t0
        scorer.score(eps, purpose="train", reward_mode=args.reward_mode)
        t_rew = time.time() - t0 - t_roll
        adv_stats = compute_advantages(eps, args.reward_mode, args.lam, args.adv_norm, args.credit)
        samples = build_samples(eps, phases, args.mask_truncated)
        if args.max_samples_per_step and len(samples) > args.max_samples_per_step:
            samples = random.Random(args.seed + step).sample(samples, args.max_samples_per_step)
        if sleep_mode:
            if not engine.sleep():
                raise RuntimeError("vLLM sleep mode is not available in this vLLM version: run with 2 GPUs "
                                   "(NUM_GPUS=2) or --colocate shared --quantization fp8")
            learner.onload()
        lstats = learner.train_step(samples, args.updates_per_step) if samples else {"n_samples": 0}
        slot = os.path.join(ckpt_dir, f"rollout_{step % 2}")
        learner.save(slot, with_optimizer=False)
        if sleep_mode:
            learner.offload()
            engine.wake_up()
        state["lora_id"] += 1
        lora = (f"step{step + 1}", state["lora_id"], slot)
        rec = {"step": step + 1, "time_s": round(time.time() - t0, 1), "rollout_s": round(t_roll, 1), "reward_s": round(t_rew, 1),
               **{f"train/{k}": v for k, v in _summ(eps).items()}, **adv_stats, **{f"learner/{k}": v for k, v in lstats.items()},
               "judge_cost_usd": round(judge.cost(), 3)}

        stop_early = False
        if dev_items and (step + 1) % args.eval_every == 0:
            ds = dev_eval(lora)
            rec.update({f"dev/{k}": v for k, v in ds.items()})
            maybe_new_best(ds, step + 1, rec)
            stop_early = bool(args.patience) and state.get("evals_since_best", 0) >= args.patience
        if (step + 1) % args.save_every == 0 or step + 1 == args.total_steps or stop_early:
            learner.save(os.path.join(ckpt_dir, "latest"))
            if args.keep_step_ckpts:
                shutil.copytree(slot, os.path.join(ckpt_dir, f"step_{step + 1}"), dirs_exist_ok=True)
            state["step"] = step + 1
            write_json(state_path, state)
        append_jsonl(log_path, [rec])
        LOG.info("step %d | %s", step + 1, {k: v for k, v in rec.items() if k.startswith(("train/outcome", "train/coverage", "train/pool_recall",
                                                                                           "dev/outcome_verifiable", "dev/coverage", "dead", "learner/loss", "time", "judge_cost"))})
        if stop_early:
            LOG.info("early stop: no dev improvement in %d evaluations (best step %s)", args.patience, state.get("best_step"))
            break
        if args.max_hours and (time.time() - t_start) / 3600 > args.max_hours:
            learner.save(os.path.join(ckpt_dir, "latest"))
            state["step"] = step + 1
            write_json(state_path, state)
            LOG.warning("Stopping after %.2f h (max_hours); resubmit the job to resume.", args.max_hours)
            return
    if not os.path.exists(os.path.join(ckpt_dir, "best", "adapter_config.json")):
        shutil.copytree(os.path.join(ckpt_dir, "latest"), os.path.join(ckpt_dir, "best"), dirs_exist_ok=True)
        state["best_step"] = state["step"]
    state["finished"] = True
    write_json(state_path, state)
    if state["best_dev"] > -1e8:
        LOG.info("GRPO finished. Best dev %s %.4f at step %s (step 0 = starting policy). Adapter: %s", args.select_metric,
                 state["best_dev"], state.get("best_step"), os.path.join(ckpt_dir, "best"))
    else:
        LOG.info("GRPO finished (no dev evaluation ran; 'best' = last checkpoint). Adapter: %s", os.path.join(ckpt_dir, "best"))


class _MockLearner:
    """CPU stand-in so that the training loop can be smoke-tested without GPUs."""

    def train_step(self, samples, n_updates=1):
        return {"n_samples": len(samples), "loss": float(-np.mean([s["adv"] for s in samples])) if samples else 0.0}

    def save(self, path, with_optimizer=True):
        os.makedirs(path, exist_ok=True)
        open(os.path.join(path, "adapter_config.json"), "w").write("{}")
        return path

    def ready(self):
        return True

    def onload(self):
        return None

    def offload(self):
        return None


def add_args(p) -> None:
    g = p.add_argument_group("grpo")
    g.add_argument("--train_system", default="pipeline3", choices=["zero_shot", "rag", "oracle", "react", "pipeline3"])
    g.add_argument("--total_steps", type=int, default=200)
    g.add_argument("--prompts_per_step", type=int, default=8)
    g.add_argument("--group_size", type=int, default=6)
    g.add_argument("--train_temperature", type=float, default=0.9)
    g.add_argument("--lam", type=float, default=0.5, help="weight of turn-level advantages (process_turn)")
    g.add_argument("--credit", default="turn", choices=["turn", "sapo"])
    g.add_argument("--adv_norm", default="std", choices=["std", "none"], help="'none' = Dr.GRPO-style mean-centering")
    g.add_argument("--train_phases", nargs="+", default=["search", "read", "synth", "direct"])
    g.add_argument("--mask_truncated", action="store_true")
    g.add_argument("--max_samples_per_step", type=int, default=0)
    g.add_argument("--updates_per_step", type=int, default=1)
    g.add_argument("--rl_gpus", type=int, default=1, help="1 = colocated; >=2 = Ray (1 learner GPU + rest rollout)")
    g.add_argument("--rollout_gpu_mem", type=float, default=0.45, help="vLLM share of the GPU in colocated mode")
    g.add_argument("--eval_every", type=int, default=20)
    g.add_argument("--save_every", type=int, default=10)
    g.add_argument("--keep_step_ckpts", action="store_true")
    g.add_argument("--n_dev", type=int, default=48, help="0 = all dev questions")
    g.add_argument("--train_limit", type=int, default=None)
    g.add_argument("--max_hours", type=float, default=0.0, help="stop & checkpoint before the PBS walltime")
    g.add_argument("--run_suffix", default="")
    # ---- version 2
    g.add_argument("--init_adapter", default=None, help="warm start: LoRA adapter to start from (e.g. the DPO adapter)")
    g.add_argument("--colocate", default="shared", choices=["shared", "sleep"],
                   help="1-GPU mode. shared: vLLM and learner on the GPU together (fp8 rollouts for 8B); "
                        "sleep: bf16 rollouts, vLLM sleeps while the learner trains (same weights for both)")
    g.add_argument("--prior_scores", default=None, help="scored sampled answers (e.g. version_1 pipeline3_train_n4) -> curriculum")
    g.add_argument("--dead_item_weight", type=float, default=0.3, help="sampling weight of questions whose prior samples all tied")
    g.add_argument("--unseen_item_weight", type=float, default=0.7, help="sampling weight of questions without prior samples")
    g.add_argument("--eval_at_start", action="store_true", help="score the starting policy on dev (step 0) before training")
    g.add_argument("--select_metric", default="outcome_verifiable", help="dev metric used to pick the best checkpoint")
    g.add_argument("--min_improve", type=float, default=0.0)
    g.add_argument("--patience", type=int, default=0, help="stop after this many dev evaluations without improvement (0 = off)")
