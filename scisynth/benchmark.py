"""Benchmark construction (shared by all policies; built once with the judge model).

Task A (qa): focused research questions generated from each seed paper's citation neighbourhood, filtered for
             breadth, parametric answerability and method-name leakage; gold = the cited papers relevant to the
             question; key points extracted from those papers (content points, importance-weighted).
Task B (rw): related-work generation for the seed paper; gold = its cited papers; key points from the cited papers;
             reference = the paper's own related-work section (+ its content points) when present.
"""
from __future__ import annotations

import os
import re
from collections import Counter

from . import prompts as P
from .data import Corpus
from .judge import JudgeRequest, make_judge
from .utils import LOG, dir_lock, first_sentence, n_words, read_jsonl, truncate_words, unique, write_json, write_jsonl

NAMED = re.compile(r"[A-Za-z][A-Za-z0-9\-]+")


def named_terms(text: str) -> set[str]:
    """Acronyms / CamelCase / alphanumeric names (UMAP, t-SNE, GISTEmbed, Word2vec, GPT-4 ...)."""
    out = set()
    for tok in NAMED.findall(text or ""):
        tok = tok.strip("-")
        if len(tok) < 2:
            continue
        named = False
        for part in tok.split("-"):  # judge hyphenated tokens part-wise: "t-SNE" yes, "Time-Dependent" no
            if not part:
                continue
            upper = sum(c.isupper() for c in part)
            if upper >= 2 or any(c.isdigit() for c in part) or (part[0].islower() and upper >= 1):
                named = True
        if named:
            out.add(tok.lower())
    return out


def leaks(question: str, abstract: str, title: str) -> list[str]:
    names = named_terms(abstract) | named_terms(title)
    q = question.lower()
    return sorted(n for n in names if re.search(rf"(?<![a-z0-9]){re.escape(n)}(?![a-z0-9])", q))


def _render_docs(corpus: Corpus, ids: list[str], max_words: int) -> str:
    return "\n".join(f"[{d}] {corpus.text(d, max_words)}" for d in ids)


def _norm_kps(raw, valid_ids: set[str], max_kp: int) -> list[dict]:
    kps = []
    for k in (raw or [])[: max_kp]:
        if not isinstance(k, dict) or not str(k.get("text", "")).strip():
            continue
        try:
            imp = int(k.get("importance", 2))
        except Exception:
            imp = 2
        kps.append({"kp_id": len(kps) + 1, "text": str(k["text"]).strip(),
                    "doc_ids": [d for d in (k.get("doc_ids") or []) if d in valid_ids],
                    "importance": max(1, min(3, imp))})
    return kps


def build_benchmark(args) -> None:
    bench_dir = os.path.join(args.data_dir, "bench")
    os.makedirs(bench_dir, exist_ok=True)
    with dir_lock(os.path.join(bench_dir, "build")):
        done = os.path.join(bench_dir, "bench.done")
        if os.path.exists(done) and not args.overwrite:
            LOG.info("Benchmark already built in %s", bench_dir)
            return
        _build(args, bench_dir)
        open(done, "w").write("ok\n")


def _build(args, bench_dir: str) -> None:
    examples = read_jsonl(os.path.join(args.data_dir, "examples.jsonl"))
    if args.max_examples:
        examples = examples[: args.max_examples]
    corpus = Corpus(args.data_dir)
    judge = make_judge(args.judge, args.judge_model, os.path.join(bench_dir, "judge_cache.jsonl"),
                       args.judge_concurrency, args.judge_base_url,
                       args.judge_max_fail_frac)
    meta = {"judge": args.judge_model, "n_examples": len(examples)}
    max_docs = args.kp_max_docs

    # ------------------------------------------------------------------ Task A: questions
    if "qa" in args.tasks:
        reqs = []
        for ex in examples:
            gold = ex["gold_doc_ids"][:max_docs]
            cited = "\n".join(f"[{i + 1}] {first_sentence(corpus.text(d), 30)}" for i, d in enumerate(gold))
            reqs.append(JudgeRequest(
                "qgen", P.QGEN_SYSTEM,
                P.QGEN_USER.format(title=ex["title"], abstract=ex["abstract"], cited=cited, k=args.q_per_paper + 1),
                payload={"title": ex["title"], "n_cited": len(gold), "k": args.q_per_paper + 1}))
        outs = judge.run(reqs)
        cands, rejected, reasons = [], [], Counter()
        for ex, out in zip(examples, outs):
            gold = ex["gold_doc_ids"][:max_docs]
            kept = 0
            for q in (out or {}).get("questions", []) or []:
                if not isinstance(q, dict) or not str(q.get("question", "")).strip():
                    continue
                text = str(q["question"]).strip()
                why = []
                if args.filter_breadth and str(q.get("breadth", "")).lower().startswith("field"):
                    why.append("field_level")
                if args.filter_general and str(q.get("general_knowledge", "")).lower().startswith("y"):
                    why.append("parametric")
                lk = leaks(text, ex["abstract"], ex["title"])
                if args.filter_leakage and lk:
                    why.append("leak:" + ",".join(lk[:3]))
                rel = []
                for n in q.get("relevant_cited") or []:
                    try:
                        n = int(n)
                    except Exception:
                        continue
                    if 1 <= n <= len(gold):
                        rel.append(gold[n - 1])
                rel = unique(rel)
                if len(rel) < args.min_gold:
                    why.append("few_relevant_cited")
                rec = {"ex_id": ex["ex_id"], "question": text, "relevant_cited": rel, "self_breadth": q.get("breadth"),
                       "self_general": q.get("general_knowledge"), "leaked_terms": lk}
                if why or kept >= args.q_per_paper:
                    rec["reject"] = why or ["over_quota"]
                    reasons.update(rec["reject"])
                    rejected.append(rec)
                    continue
                kept += 1
                cands.append((ex, rec))
        LOG.info("QA: %d questions kept, %d rejected %s", len(cands), len(rejected), dict(reasons))

        # key points per question
        reqs = []
        for ex, rec in cands:
            ids = ex["gold_doc_ids"][:max_docs]
            reqs.append(JudgeRequest(
                "kp_qa", P.KP_SYSTEM,
                P.KP_USER_QA.format(question=rec["question"], docs=_render_docs(corpus, ids, args.kp_doc_words),
                                    min_kp=args.min_kp, max_kp=args.max_kp),
                payload={"doc_ids": ids, "doc_texts": [corpus.text(d, args.kp_doc_words) for d in ids], "max_kp": args.max_kp},
                max_tokens=3000))
        outs = judge.run(reqs)
        items = []
        for (ex, rec), out in zip(cands, outs):
            out = out or {}
            valid = set(ex["gold_doc_ids"])
            rel = [d for d in (out.get("relevant_doc_ids") or []) if d in valid]
            kps = _norm_kps(out.get("key_points"), valid, args.max_kp)
            gold_q = unique(rel + [d for k in kps for d in k["doc_ids"]])
            if len(gold_q) < args.min_gold:
                gold_q = rec["relevant_cited"]
            if len(gold_q) < args.min_gold or len(kps) < 3:
                rec["reject"] = ["kp_extraction"]
                rejected.append(rec)
                continue
            items.append({
                "item_id": f"{ex['ex_id']}_q{sum(1 for it in items if it['ex_id'] == ex['ex_id'])}",
                "ex_id": ex["ex_id"], "task": "qa", "split": ex["split"], "year": ex["year"],
                "question": rec["question"], "gold_doc_ids": gold_q, "all_cited_doc_ids": ex["gold_doc_ids"],
                "exclude_doc_ids": ex.get("exclude_doc_ids", []), "key_points": kps,
                "seed_title": ex["title"],
            })
        write_jsonl(os.path.join(bench_dir, "qa.jsonl"), items)
        write_jsonl(os.path.join(bench_dir, "qa_rejected.jsonl"), rejected)
        meta["qa"] = {"items": len(items), "rejected": len(rejected), "reject_reasons": dict(reasons),
                      "splits": dict(Counter(i["split"] for i in items))}

    # ------------------------------------------------------------------ Task B: related work
    if "rw" in args.tasks:
        reqs_kp, reqs_rw, rw_idx = [], [], []
        for i, ex in enumerate(examples):
            ids = ex["gold_doc_ids"][:max_docs]
            reqs_kp.append(JudgeRequest(
                "kp_rw", P.KP_SYSTEM,
                P.KP_USER_RW.format(title=ex["title"], abstract=ex["abstract"],
                                    docs=_render_docs(corpus, ids, args.kp_doc_words), min_kp=args.min_kp, max_kp=args.max_kp),
                payload={"doc_ids": ids, "doc_texts": [corpus.text(d, args.kp_doc_words) for d in ids], "max_kp": args.max_kp},
                max_tokens=3000))
            if n_words(ex["related_work"]) >= args.min_rw_words:
                rw_idx.append(i)
                reqs_rw.append(JudgeRequest(
                    "rwpoints", P.RWPOINTS_SYSTEM,
                    P.RWPOINTS_USER.format(title=ex["title"], related_work=truncate_words(ex["related_work"], 1500),
                                           min_kp=args.min_kp, max_kp=args.max_kp),
                    payload={"related_work": ex["related_work"], "max_kp": args.max_kp}))
        outs_kp = judge.run(reqs_kp)
        outs_rw = dict(zip(rw_idx, judge.run(reqs_rw)))
        items = []
        for i, (ex, out) in enumerate(zip(examples, outs_kp)):
            out = out or {}
            valid = set(ex["gold_doc_ids"])
            kps = _norm_kps(out.get("key_points"), valid, args.max_kp)
            if len(kps) < 3:
                continue
            ref_points = []
            if i in outs_rw and outs_rw[i]:
                for k in (outs_rw[i].get("points") or [])[: args.max_kp]:
                    if isinstance(k, dict) and str(k.get("text", "")).strip():
                        try:
                            imp = max(1, min(3, int(k.get("importance", 2))))
                        except Exception:
                            imp = 2
                        ref_points.append({"kp_id": len(ref_points) + 1, "text": str(k["text"]).strip(), "importance": imp})
            items.append({
                "item_id": f"{ex['ex_id']}_rw", "ex_id": ex["ex_id"], "task": "rw", "split": ex["split"], "year": ex["year"],
                "title": ex["title"], "abstract": ex["abstract"], "intro": ex["intro"],
                "reference": ex["related_work"] if n_words(ex["related_work"]) >= args.min_rw_words else "",
                "ref_points": ref_points, "key_points": kps, "gold_doc_ids": ex["gold_doc_ids"],
                "core_doc_ids": [d for d in (out.get("relevant_doc_ids") or []) if d in valid],
                "exclude_doc_ids": ex.get("exclude_doc_ids", []),
            })
        write_jsonl(os.path.join(bench_dir, "rw.jsonl"), items)
        meta["rw"] = {"items": len(items), "with_reference_rw": sum(1 for it in items if it["reference"]),
                      "splits": dict(Counter(i["split"] for i in items))}

    meta["judge_stats"] = judge.stats()
    write_json(os.path.join(bench_dir, "bench_meta.json"), meta)
    LOG.info("Benchmark meta: %s", meta)


def load_items(data_dir: str, task: str, split: str | None = None, limit: int | None = None) -> list[dict]:
    items = read_jsonl(os.path.join(data_dir, "bench", f"{task}.jsonl"))
    if split and split != "all":
        items = [i for i in items if i["split"] == split]
    return items[:limit] if limit else items


def add_args(p) -> None:
    g = p.add_argument_group("benchmark")
    g.add_argument("--tasks", nargs="+", default=["qa", "rw"], choices=["qa", "rw"])
    g.add_argument("--max_examples", type=int, default=None)
    g.add_argument("--q_per_paper", type=int, default=2)
    g.add_argument("--min_gold", type=int, default=3)
    g.add_argument("--min_kp", type=int, default=5)
    g.add_argument("--max_kp", type=int, default=12)
    g.add_argument("--kp_max_docs", type=int, default=25)
    g.add_argument("--kp_doc_words", type=int, default=160)
    g.add_argument("--min_rw_words", type=int, default=80)
    g.add_argument("--no_filter_breadth", dest="filter_breadth", action="store_false")
    g.add_argument("--no_filter_general", dest="filter_general", action="store_false")
    g.add_argument("--no_filter_leakage", dest="filter_leakage", action="store_false")
