"""Stages: infer, score, tables, make_sft, train_sft."""
from __future__ import annotations

import glob
import os
import random

import numpy as np
import pandas as pd

from .benchmark import load_items
from .data import Corpus
from .judge import make_judge, scores_dirname
from .llm import build_engine
from .rewards import Scorer, reward_config_from_args
from .systems import RETRIEVAL_SYSTEMS, Episode, Runner, Turn, config_from_args, make_episodes
from .utils import LOG, append_jsonl, bootstrap_ci, read_json, read_jsonl, write_json, write_jsonl

METRICS = [
    "outcome_verifiable", "coverage", "support", "cite_prec", "cite_rec", "cite_f1", "cite_valid", "pool_recall",
    "ground_prec", "contradicted_rate", "notes_support", "ref_coverage", "uncited_rate", "judge_overall",
    "judge_synthesis", "judge_specificity", "judge_coverage", "judge_critical", "judge_organization",
    "judge_positioning", "rougeL", "words", "n_cited", "n_search_turns", "pool_size",
]


def _gens_dir(args, task):
    return os.path.join(args.out_dir, "gens", task)


def _judge_for(args, purpose: str) -> str:
    return (args.eval_judge_model or args.judge_model) if purpose == "eval" else args.judge_model


def _scores_dir(args, task, purpose: str = "eval"):
    return os.path.join(args.out_dir, scores_dirname(args.judge, _judge_for(args, purpose)), task)


# ------------------------------------------------------------------------------ infer
def infer(args) -> None:
    items = load_items(args.data_dir, args.task, args.split, args.limit)
    if not items:
        raise RuntimeError(f"no benchmark items for task={args.task} split={args.split}")
    todo = []
    for system in args.systems:
        path = os.path.join(_gens_dir(args, args.task), f"{system}{args.tag}.jsonl")
        if os.path.exists(path) and not args.overwrite:
            LOG.info("skip %s (exists)", path)
            continue
        todo.append((system, path))
    if not todo:
        return
    corpus = Corpus(args.data_dir)
    retriever = None
    if any(s in RETRIEVAL_SYSTEMS for s, _ in todo):
        from .retrieval import Retriever

        retriever = Retriever(args.data_dir, args.retriever_mode, args.embed_model)
    engine = build_engine(args, enable_lora=bool(args.lora_path), max_lora_rank=max(args.lora_r, 8))
    lora = ("adapter", 1, os.path.abspath(args.lora_path)) if args.lora_path else None
    cfg = config_from_args(args)
    if args.n_samples > 1 and cfg.temperature == 0:
        cfg.temperature = 0.8
    for system, path in todo:
        LOG.info("infer: task=%s system=%s items=%d samples=%d lora=%s", args.task, system, len(items), args.n_samples, args.lora_path)
        eps = make_episodes(items, system, args.n_samples)
        recs = []
        for i in range(0, len(eps), args.chunk):
            chunk = eps[i: i + args.chunk]
            Runner(cfg, corpus, retriever, engine, lora).run(chunk)
            recs.extend(ep.record(with_tokens=args.save_token_ids, with_conv=args.save_conv) for ep in chunk)
            LOG.info("  %d/%d episodes", min(i + args.chunk, len(eps)), len(eps))
        for r in recs:
            r["model"], r["lora"], r["tag"] = args.model_tag, args.lora_path, args.tag
        write_jsonl(path, recs)
        LOG.info("wrote %s", path)


# ------------------------------------------------------------------------------ score
def record_to_episode(rec: dict, item: dict) -> Episode:
    turns = [Turn(phase=t["phase"], conv=t.get("conv") or [], text=t.get("text", ""),
                  prompt_token_ids=t.get("prompt_token_ids") or [], token_ids=t.get("token_ids") or [],
                  finish_reason=t.get("finish_reason", ""), info=t.get("info") or {}) for t in rec["turns"]]
    return Episode(item=item, system=rec["system"], sample_idx=rec.get("sample_idx", 0), turns=turns,
                   pool=rec.get("pool") or [], notes=[tuple(n) for n in rec.get("notes") or []], answer=rec.get("answer", ""),
                   rewards={k: v for k, v in (rec.get("rewards") or {}).items() if k in ("oracle_kept_gold", "search_fallback")})


def rouge_l(pred: str, ref: str, max_tokens: int = 700) -> float:
    import re

    a = re.findall(r"[a-z0-9]+", pred.lower())[:max_tokens]
    b = re.findall(r"[a-z0-9]+", ref.lower())[:max_tokens]
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else (prev[j] if prev[j] > cur[j - 1] else cur[j - 1])
        prev = cur
    lcs = prev[-1]
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r) if p + r else 0.0


def score(args) -> None:
    corpus = Corpus(args.data_dir)
    items = {it["item_id"]: it for it in load_items(args.data_dir, args.task, "all")}
    purpose = args.score_purpose
    judge = make_judge(args.judge, _judge_for(args, purpose),
                       os.path.join(args.out_dir, f"judge_cache_{purpose}.jsonl"), args.judge_concurrency, args.judge_base_url,
                       args.judge_max_fail_frac)
    scorer = Scorer(judge, corpus, reward_config_from_args(args))
    files = sorted(glob.glob(os.path.join(_gens_dir(args, args.task), "*.jsonl")))
    if args.score_all:  # every generation file except train-split sampling runs (used for SFT data)
        files = [f for f in files if "_train" not in os.path.basename(f)]
    else:
        files = [f for f in files if os.path.basename(f)[:-6] in {s + args.tag for s in args.systems}]
    for f in files:
        name = os.path.basename(f)[:-6]
        out = os.path.join(_scores_dir(args, args.task, purpose), f"{name}.{purpose}.jsonl")
        if os.path.exists(out) and not args.overwrite:
            LOG.info("skip %s (scored)", out)
            continue
        recs = [r for r in read_jsonl(f) if r["item_id"] in items]
        eps = [record_to_episode(r, items[r["item_id"]]) for r in recs]
        LOG.info("score %s: %d episodes (%s, judge=%s)", name, len(eps), purpose, judge.model)
        scorer.score(eps, purpose=purpose, reward_mode="outcome")
        for r, ep in zip(recs, eps):
            r["rewards"] = ep.rewards
            if ep.item["task"] == "rw" and ep.item.get("reference"):
                r["rewards"]["rougeL"] = rouge_l(ep.answer, ep.item["reference"])
            for td, t in zip(r["turns"], ep.turns):
                td["reward"] = t.reward
        write_jsonl(out, recs)
        summary = summarize([r["rewards"] for r in recs])
        summary.update({"name": name, "task": args.task, "model": args.model_tag, "n": len(recs), "purpose": purpose,
                        "judge": judge.model, "judge_stats": judge.stats()})
        write_json(out.replace(".jsonl", ".summary.json"), summary)
        LOG.info("%s: %s", name, {k: round(v["mean"], 3) for k, v in summary["metrics"].items() if k in ("coverage", "support", "cite_f1", "ground_prec", "judge_overall")})


def summarize(rewards: list[dict]) -> dict:
    out = {}
    for k in METRICS:
        xs = [r.get(k) for r in rewards if isinstance(r.get(k), (int, float))]
        if xs:
            lo, hi = bootstrap_ci(xs)
            out[k] = {"mean": float(np.mean(xs)), "ci_lo": lo, "ci_hi": hi, "n": len(xs)}
    return {"metrics": out}


# ------------------------------------------------------------------------------ tables
def tables(args) -> None:
    root = args.results_root or os.path.dirname(os.path.abspath(args.out_dir))
    sd = scores_dirname(args.judge, _judge_for(args, "eval"))   # one table per judge
    jtag = sd[len("scores_"):] if sd.startswith("scores_") else "gpt-4o-mini"
    rows = []
    for f in glob.glob(os.path.join(root, "*", sd, "*", "*.eval.summary.json")):
        s = read_json(f)
        model = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(f))))
        row = {"model": model, "task": s["task"], "system": s["name"], "n": s["n"]}
        for k, v in s["metrics"].items():
            row[k] = round(v["mean"], 4)
        rows.append(row)
    if not rows:
        LOG.warning("no */%s/*/*.eval.summary.json under %s", sd, root)
        return
    df = pd.DataFrame(rows).sort_values(["task", "model", "system"])
    df.insert(0, "judge", jtag)
    os.makedirs(os.path.join(root, "tables"), exist_ok=True)
    df.to_csv(os.path.join(root, "tables", f"results_{jtag}.csv"), index=False)
    cols = ["judge", "model", "task", "system", "n", "outcome_verifiable", "coverage", "support", "cite_f1", "pool_recall",
            "ground_prec", "contradicted_rate", "ref_coverage", "rougeL", "judge_overall", "words"]
    view = df[[c for c in cols if c in df.columns]]
    with open(os.path.join(root, "tables", f"results_{jtag}.md"), "w") as fh:
        try:
            fh.write(view.to_markdown(index=False))
        except ImportError:  # tabulate not installed
            fh.write(view.to_string(index=False))
    LOG.info("wrote %s", os.path.join(root, "tables", f"results_{jtag}.csv"))
    # headroom subset: items the zero-shot model of the same family covers poorly (report BOTH sets)
    hrows = []
    for model_dir in glob.glob(os.path.join(root, "*", sd, "*")):
        zs = os.path.join(model_dir, "zero_shot.eval.jsonl")
        if not os.path.exists(zs):
            continue
        hard = {r["item_id"] for r in read_jsonl(zs) if (r["rewards"].get("coverage") or 0) <= args.headroom_thresh}
        for f in glob.glob(os.path.join(model_dir, "*.eval.jsonl")):
            recs = [r for r in read_jsonl(f) if r["item_id"] in hard]
            if not recs:
                continue
            m = summarize([r["rewards"] for r in recs])["metrics"]
            hrows.append({"model": model_dir.split(os.sep)[-3], "task": os.path.basename(model_dir),
                          "system": os.path.basename(f)[:-11], "n_hard": len(recs),
                          **{k: round(v["mean"], 4) for k, v in m.items()}})
    if hrows:
        pd.DataFrame(hrows).sort_values(["task", "model", "system"]).to_csv(
            os.path.join(root, "tables", f"results_headroom_{jtag}.csv"), index=False)


# ------------------------------------------------------------------------------ SFT (rejection sampling)
def make_sft(args) -> None:
    rows = []
    for name in args.sft_from:
        path = os.path.join(_scores_dir(args, args.task, args.score_purpose), f"{name}.{args.score_purpose}.jsonl")
        recs = read_jsonl(path)
        if not recs:
            raise FileNotFoundError(path)
        best = {}
        for r in recs:
            sc = r["rewards"].get("outcome_verifiable", r["rewards"].get("outcome", 0.0))
            if r["item_id"] not in best or sc > best[r["item_id"]][0]:
                best[r["item_id"]] = (sc, r)
        scores = np.array([sc for sc, _ in best.values()], dtype=float)
        LOG.info("%s: best-of-n reward over %d items: mean %.3f | median %.3f | max %.3f | >= %.2f: %d",
                 name, len(scores), scores.mean(), np.median(scores), scores.max(), args.sft_min_reward,
                 int((scores >= args.sft_min_reward).sum()))
        if not any(t.get("token_ids") for r in recs for t in r["turns"]):
            raise ValueError(f"{path}: generations lack token ids; re-run infer with --save_token_ids")
        n_skip = 0
        for sc, r in best.values():
            if sc < args.sft_min_reward:
                continue
            for t in r["turns"]:
                if not t.get("token_ids"):
                    n_skip += 1
                    continue
                rows.append({"item_id": r["item_id"], "phase": t["phase"], "prompt_ids": t["prompt_token_ids"],
                             "completion_ids": t["token_ids"], "reward": sc})
        if n_skip:
            LOG.warning("skipped %d turns without token ids", n_skip)
    out = os.path.join(args.out_dir, "sft", f"{args.task}_data.jsonl")
    write_jsonl(out, rows)
    LOG.info("SFT data: %d turn samples from %d trajectories -> %s", len(rows), len({r['item_id'] for r in rows}), out)
    if not rows:
        LOG.warning("No trajectory reached --sft_min_reward %.2f, so the SFT baseline will be skipped. If the rewards "
                    "above are all ~0, check the judge (see judge_stats in the scores/*.summary.json files).",
                    args.sft_min_reward)


def train_sft(args) -> None:
    from .learner import Learner, learner_kwargs

    out = os.path.join(args.out_dir, "sft", f"{args.task}_{args.sft_name}")
    if os.path.exists(os.path.join(out, "adapter_config.json")) and not args.overwrite:
        LOG.info("SFT adapter already exists: %s (use --overwrite to retrain)", out)
        return
    data = read_jsonl(os.path.join(args.out_dir, "sft", f"{args.task}_data.jsonl"))
    if not data:
        LOG.warning("SFT data is empty (see the make_sft log); skipping the SFT baseline for task %s", args.task)
        return
    learner = Learner(**learner_kwargs(args))
    rng = random.Random(args.seed)
    step = 0
    for ep in range(args.sft_epochs):
        rng.shuffle(data)
        for i in range(0, len(data), args.sft_batch):
            st = learner.sft_step(data[i: i + args.sft_batch])
            step += 1
            if step % 10 == 0:
                LOG.info("sft epoch %d step %d %s", ep, step, st)
    learner.save(out, with_optimizer=False)
    LOG.info("SFT adapter -> %s", out)


# ------------------------------------------------------------------------------ DPO (offline preference baseline)
def _scored_trajectories(args) -> dict[str, list[tuple[float, dict]]]:
    by_item: dict[str, list[tuple[float, dict]]] = {}
    for name in args.sft_from:
        path = os.path.join(_scores_dir(args, args.task, args.score_purpose), f"{name}.{args.score_purpose}.jsonl")
        recs = read_jsonl(path)
        if not recs:
            raise FileNotFoundError(path)
        for r in recs:
            sc = r["rewards"].get("outcome_verifiable", r["rewards"].get("outcome", 0.0))
            by_item.setdefault(r["item_id"], []).append((sc, r))
    return by_item


def _turn_samples(r: dict) -> list[dict]:
    return [{"prompt_ids": t["prompt_token_ids"], "completion_ids": t["token_ids"]}
            for t in r["turns"] if t.get("token_ids") and t.get("prompt_token_ids")]


def make_dpo(args) -> None:
    """Best vs worst of the n sampled trajectories per training item (same scored samples as the SFT baseline)."""
    pairs, gaps, n_items = [], [], 0
    for item_id, lst in _scored_trajectories(args).items():
        n_items += 1
        if len(lst) < 2:
            continue
        lst.sort(key=lambda x: x[0])
        (r_lo, worst), (r_hi, best) = lst[0], lst[-1]
        if r_hi - r_lo < args.dpo_min_gap:
            continue
        ch, rj = _turn_samples(best), _turn_samples(worst)
        if not ch or not rj:
            continue
        pairs.append({"item_id": item_id, "r_chosen": r_hi, "r_rejected": r_lo, "chosen": ch, "rejected": rj})
        gaps.append(r_hi - r_lo)
    out = os.path.join(args.out_dir, "sft", f"{args.task}_dpo_pairs.jsonl")
    write_jsonl(out, pairs)
    LOG.info("DPO pairs: %d of %d items (reward gap >= %.2f; mean gap %.3f) -> %s", len(pairs), n_items,
             args.dpo_min_gap, float(np.mean(gaps)) if gaps else 0.0, out)
    if not pairs:
        LOG.warning("No DPO pairs: all sampled answers of an item scored about the same (or scoring failed).")


def train_dpo(args) -> None:
    from .learner import Learner, learner_kwargs

    out = os.path.join(args.out_dir, "sft", f"{args.task}_{args.dpo_name}")
    if os.path.exists(os.path.join(out, "adapter_config.json")) and not args.overwrite:
        LOG.info("DPO adapter already exists: %s (use --overwrite to retrain)", out)
        return
    pairs = read_jsonl(os.path.join(args.out_dir, "sft", f"{args.task}_dpo_pairs.jsonl"))
    if not pairs:
        LOG.warning("DPO pairs file is empty (see the make_dpo log); skipping the DPO baseline for task %s", args.task)
        return
    learner = Learner(**learner_kwargs(args))
    # reference log-probs (frozen base model) for every trajectory, computed once
    flat, owner = [], []
    for i, p in enumerate(pairs):
        for side in ("chosen", "rejected"):
            for s in p[side]:
                flat.append(s)
                owner.append((i, side))
    ref = learner.seq_logps(flat, ref=True)
    for p in pairs:
        p["ref_c"] = p["ref_r"] = 0.0
    for (i, side), v in zip(owner, ref):
        pairs[i]["ref_c" if side == "chosen" else "ref_r"] += v
    LOG.info("DPO: %d pairs, reference log-probs computed; beta=%.3f, %d epochs, %d pairs/step",
             len(pairs), args.dpo_beta, args.dpo_epochs, args.dpo_batch)
    rng = random.Random(args.seed)
    step = 0
    log_path = os.path.join(args.out_dir, "sft", f"{args.task}_{args.dpo_name}_log.jsonl")
    for ep in range(args.dpo_epochs):
        rng.shuffle(pairs)
        for i in range(0, len(pairs), args.dpo_batch):
            st = learner.dpo_step(pairs[i: i + args.dpo_batch], beta=args.dpo_beta)
            step += 1
            rec = {"epoch": ep, "step": step, **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in st.items()}}
            append_jsonl(log_path, [rec])
            if step % 5 == 0:
                LOG.info("dpo epoch %d step %d loss %.4f acc %.3f margin %.3f", ep, step, st["dpo_loss"],
                         st["reward_acc"], st["margin"])
    learner.save(out, with_optimizer=False)
    LOG.info("DPO adapter -> %s", out)
