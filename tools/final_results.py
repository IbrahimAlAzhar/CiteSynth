"""Collect the scores of every experiment run so far into one report (CPU only, no GPU, no API calls).

  python final_results.py

Reads every outputs/<dataset>/<model>/scores*/<task>/*.summary.json and *.eval.jsonl (all models, tasks and judges)
plus the training logs, and writes:
  version_1/final_result.txt   readable report (also printed on screen)
  version_1/final_result.csv   the same numbers, one row per system, for spreadsheets / paper tables

Sections: 1) results per judge and task, grouped by ablation type   2) paired comparisons with 95% bootstrap CIs
          3) headroom subset (questions zero-shot covers poorly)    4) training runs (GRPO, SFT, DPO)
"""
# ============================ settings (edit here) ============================
PROJECT_DIR = "/lstr/sahara/datalab-ml/ibrahim/limagents_update/sci_lit_synthesis/version_1"
RESULTS_ROOT = PROJECT_DIR + "/outputs/df_updated_with_retrieval_upd_n1000"
REPORT_TXT = PROJECT_DIR + "/final_result.txt"
REPORT_CSV = PROJECT_DIR + "/final_result.csv"
HEADROOM_THRESHOLD = 0.35      # "hard" questions: zero-shot key-point coverage <= this value
N_BOOTSTRAP = 5000
# ===============================================================================
import argparse
import csv
import datetime
import glob
import json
import math
import os

import numpy as np

CATEGORIES = [
    ("Parametric knowledge only (no retrieval)", ["zero_shot", "cot"]),
    ("Retrieval, untrained", ["rag", "react", "pipeline3"]),
    ("Trained 3-step agent (Searcher -> Reader -> Synthesizer)",
     ["pipeline3_rft", "pipeline3_dpo", "pipeline3_grpo_outcome", "pipeline3_grpo_outcome_judge",
      "pipeline3_grpo_process_sum", "pipeline3_grpo_process_turn", "pipeline3_grpo_process_turn_lexical"]),
    ("Trained without retrieval", ["zero_shot_grpo_norag"]),
    ("Input ablation: abstract + introduction (Task B)", ["zero_shot_absintro", "rag_absintro", "pipeline3_absintro"]),
    ("Upper bound and evidence sensitivity (gold papers given)", ["oracle", "oracle_drop50", "oracle_distract10"]),
]
LABELS = {
    "zero_shot": "zero-shot", "cot": "chain-of-thought", "rag": "RAG (k=10)", "react": "ReAct agent",
    "pipeline3": "3-step agent (untrained)", "pipeline3_rft": "3-step agent + SFT (best-of-4)",
    "pipeline3_dpo": "3-step agent + DPO", "pipeline3_grpo_outcome": "3-step agent + GRPO (outcome reward)",
    "pipeline3_grpo_outcome_judge": "3-step agent + GRPO (LLM-judge reward)",
    "pipeline3_grpo_process_sum": "3-step agent + GRPO (step rewards, summed)",
    "pipeline3_grpo_process_turn": "3-step agent + GRPO (step rewards, per turn)",
    "pipeline3_grpo_process_turn_lexical": "3-step agent + GRPO (per turn, word-overlap reward)",
    "zero_shot_grpo_norag": "zero-shot + GRPO (no retrieval)",
    "zero_shot_absintro": "zero-shot, abstract+intro", "rag_absintro": "RAG, abstract+intro",
    "pipeline3_absintro": "3-step agent, abstract+intro",
    "oracle": "oracle (all gold papers)", "oracle_drop50": "oracle, half of gold removed",
    "oracle_distract10": "oracle + 10 distractor papers",
}
MAIN = [("outcome_verifiable", "overall"), ("coverage", "coverage"), ("support", "support"), ("cite_f1", "cite_F1"),
        ("pool_recall", "gold_found"), ("ground_prec", "grounded")]
DETAIL = [("cite_prec", "cite_P"), ("cite_rec", "cite_R"), ("contradicted_rate", "contradict"),
          ("judge_overall", "judge_1-10"), ("words", "words"), ("n_cited", "n_cited"),
          ("n_search_turns", "searches"), ("pool_size", "papers_read")]
DETAIL_RW = [("ref_coverage", "ref_cov"), ("rougeL", "rougeL")]
PAIR_METRICS = ["outcome_verifiable", "coverage", "pool_recall", "cite_f1"]


# ------------------------------------------------------------------------------ helpers
def category_of(system):
    for cat, names in CATEGORIES:
        if system in names:
            return cat
    if system.startswith("rag_k"):
        return CATEGORIES[1][0]
    if "_grpo" in system or system.endswith(("_rft", "_dpo")):
        return CATEGORIES[2][0]
    return "Other"


def label_of(system):
    if system.startswith("rag_k") and system[5:].isdigit():
        return f"RAG (k={system[5:]})"
    return LABELS.get(system, system)


def order_key(system):
    for ci, (_, names) in enumerate(CATEGORIES):
        if system in names:
            return (ci, names.index(system), system)
    if system.startswith("rag_k"):
        return (1, 0.5, system)
    return (len(CATEGORIES), 0, system)


def judge_name(scores_dir):
    base = os.path.basename(scores_dir)
    return "gpt-4o-mini" if base == "scores" else base[len("scores_"):]


def fmt(v, digits=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.{digits}f}"


def table(header, rows):
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)] if rows else [len(h) for h in header]
    line = lambda r: "  ".join(str(x).ljust(w) if i == 0 else str(x).rjust(w) for i, (x, w) in enumerate(zip(r, widths)))
    out = [line(header), "  ".join("-" * w for w in widths)]
    out += [line(r) for r in rows]
    return out


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def mean_key(rows, k):
    vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
    return float(np.mean(vals)) if vals else None


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="all arguments are optional; defaults are set at the top of this file")
    ap.add_argument("--results_root", default=RESULTS_ROOT)
    ap.add_argument("--txt", default=REPORT_TXT)
    ap.add_argument("--csv", default=REPORT_CSV)
    a = ap.parse_args()
    L = []
    say = L.append
    rng = np.random.default_rng(0)

    say("FINAL RESULTS - citation-grounded scientific literature synthesis")
    say(f"generated {datetime.datetime.now():%Y-%m-%d %H:%M} from {a.results_root}")
    say("Task A (qa) = question-conditioned synthesis; Task B (rw) = related-work generation.")
    say("Metrics: overall = verifiable outcome score; coverage = importance-weighted key points covered;")
    say("support = cited claims backed by the cited papers; cite_F1 = citations vs. the gold (actually cited) papers;")
    say("gold_found = share of gold papers the system retrieved/read; grounded = claims backed by the gold papers.")
    say("Rows in one table share one judge: compare numbers only within a table. [a, b] = 95% CI.")

    model_dirs = sorted(d for d in glob.glob(os.path.join(a.results_root, "*")) if os.path.isdir(d) and os.path.basename(d) != "tables")
    csv_rows, cache = [], {}

    def per_item(path):
        if path not in cache:
            cache[path] = {r["item_id"]: r.get("rewards", {}) for r in read_jsonl(path)} if os.path.exists(path) else {}
        return cache[path]

    # ---------------------------------------------------------------- 1) results tables
    say("\n" + "#" * 100 + "\n1) RESULTS BY JUDGE AND TASK\n" + "#" * 100)
    groups = []   # (model, judge, task, task_dir)
    for md in model_dirs:
        for sd in sorted(glob.glob(os.path.join(md, "scores*"))):
            for td in sorted(glob.glob(os.path.join(sd, "*"))):
                if glob.glob(os.path.join(td, "*.eval.summary.json")):
                    groups.append((os.path.basename(md), judge_name(sd), os.path.basename(td), td))
    empty_models = [os.path.basename(md) for md in model_dirs if not any(g[0] == os.path.basename(md) for g in groups)]
    if empty_models:
        say(f"\n(no evaluation scores yet for: {', '.join(empty_models)})")
    if not groups:
        say("\nNo *.eval.summary.json files found.")

    for model, judge, task, td in groups:
        summaries = {}
        for f in glob.glob(os.path.join(td, "*.eval.summary.json")):
            summaries[os.path.basename(f)[: -len(".eval.summary.json")]] = json.load(open(f))
        systems = sorted(summaries, key=order_key)
        say(f"\n=== model: {model} | task: {task} ({'Task A, questions' if task == 'qa' else 'Task B, related work'}) | judge: {judge} ===")
        detail = DETAIL + (DETAIL_RW if task == "rw" else [])
        for title, cols, with_ci in (("main metrics", MAIN, True), ("details", detail, False)):
            say(f"\n-- {title}")
            header = ["system", "n"] + [c[1] for c in cols]
            rows, last_cat = [], None
            for s in systems:
                cat = category_of(s)
                if cat != last_cat:
                    rows.append([f"[{cat}]"] + [""] * (len(header) - 1))
                    last_cat = cat
                m = summaries[s].get("metrics", {})
                cells = []
                for key, _ in cols:
                    v = m.get(key, {})
                    cell = fmt(v.get("mean"))
                    if with_ci and key == "coverage" and v.get("ci_lo") is not None and not math.isnan(v.get("ci_lo", float("nan"))):
                        cell += f" [{fmt(v['ci_lo'], 2)},{fmt(v['ci_hi'], 2)}]"
                    cells.append(cell)
                rows.append(["  " + label_of(s), summaries[s].get("n", "")] + cells)
            L.extend(table(header, rows))
        for s in systems:
            m = summaries[s].get("metrics", {})
            rec = {"judge": judge, "model": model, "task": task, "category": category_of(s), "system": s,
                   "label": label_of(s), "n": summaries[s].get("n")}
            for key, v in m.items():
                rec[key] = round(v.get("mean"), 5) if isinstance(v.get("mean"), (int, float)) else None
                if key in ("outcome_verifiable", "coverage", "pool_recall"):
                    rec[key + "_ci_lo"], rec[key + "_ci_hi"] = v.get("ci_lo"), v.get("ci_hi")
            csv_rows.append(rec)

    # ---------------------------------------------------------------- 2) paired comparisons
    say("\n" + "#" * 100 + "\n2) PAIRED COMPARISONS (same questions; difference = first - second; 95% bootstrap CI)\n" + "#" * 100)
    say("A difference is reliable when its CI excludes 0. better/same/worse = number of questions.")
    for model, judge, task, td in groups:
        have = {os.path.basename(f)[: -len(".eval.jsonl")] for f in glob.glob(os.path.join(td, "*.eval.jsonl"))}
        pairs = []
        trained = sorted((s for s in have if category_of(s).startswith("Trained 3-step")), key=order_key)
        for t in trained:
            pairs += [(t, "pipeline3"), (t, "rag")]
        pairs += [("pipeline3", "rag"), ("react", "rag"), ("rag", "zero_shot"), ("cot", "zero_shot"),
                  ("zero_shot_grpo_norag", "zero_shot")]
        pairs += [(s, "rag") for s in sorted(have) if s.startswith("rag_k")]
        pairs += [("oracle_drop50", "oracle"), ("oracle_distract10", "oracle"), ("oracle", "pipeline3")]
        pairs += [(s, s[: -len("_absintro")]) for s in sorted(have) if s.endswith("_absintro")]
        seen, rows = set(), []
        for x, y in pairs:
            if x not in have or y not in have or (x, y) in seen:
                continue
            seen.add((x, y))
            A, B = per_item(os.path.join(td, x + ".eval.jsonl")), per_item(os.path.join(td, y + ".eval.jsonl"))
            ids = [i for i in A if i in B]
            for met in PAIR_METRICS:
                diff = np.array([A[i].get(met) - B[i].get(met) for i in ids
                                 if isinstance(A[i].get(met), (int, float)) and isinstance(B[i].get(met), (int, float))])
                if len(diff) < 2:
                    continue
                boot = diff[rng.integers(0, len(diff), (N_BOOTSTRAP, len(diff)))].mean(1)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                flag = "*" if lo > 0 or hi < 0 else ""
                rows.append([f"{label_of(x)}  vs  {label_of(y)}", met, f"{diff.mean():+.3f}{flag}",
                             f"[{lo:+.3f}, {hi:+.3f}]", f"{(diff > 0).sum()}/{(diff == 0).sum()}/{(diff < 0).sum()}"])
        if rows:
            say(f"\n=== model: {model} | task: {task} | judge: {judge}   (* = CI excludes 0)")
            L.extend(table(["comparison", "metric", "diff", "95% CI", "better/same/worse"], rows))

    # ---------------------------------------------------------------- 3) headroom subset
    say("\n" + "#" * 100 + f"\n3) HEADROOM SUBSET (questions where zero-shot coverage <= {HEADROOM_THRESHOLD})\n" + "#" * 100)
    say("Questions a model already answers well from memory do not test literature synthesis; report both sets.")
    for model, judge, task, td in groups:
        zs = per_item(os.path.join(td, "zero_shot.eval.jsonl"))
        if not zs:
            continue
        hard = {i for i, r in zs.items() if (r.get("coverage") or 0) <= HEADROOM_THRESHOLD}
        rows = []
        for f in sorted(glob.glob(os.path.join(td, "*.eval.jsonl")), key=lambda p: order_key(os.path.basename(p)[: -len(".eval.jsonl")])):
            s = os.path.basename(f)[: -len(".eval.jsonl")]
            R = [r for i, r in per_item(f).items() if i in hard]
            if R:
                rows.append([label_of(s), len(R)] + [fmt(mean_key(R, k)) for k, _ in MAIN])
        say(f"\n=== model: {model} | task: {task} | judge: {judge} | hard questions: {len(hard)} of {len(zs)}")
        L.extend(table(["system", "n_hard"] + [c[1] for c in MAIN], rows))

    # ---------------------------------------------------------------- 4) training runs
    say("\n" + "#" * 100 + "\n4) TRAINING RUNS\n" + "#" * 100)
    for md in model_dirs:
        model = os.path.basename(md)
        runs = sorted(glob.glob(os.path.join(md, "grpo", "*")))
        sft = os.path.join(md, "sft")
        trains = sorted(glob.glob(os.path.join(md, "scores*", "*", "*.train.summary.json")))
        if not runs and not os.path.isdir(sft) and not trains:
            continue
        say(f"\n=== model: {model}")
        if runs:
            say("\n-- GRPO (train = sampled training questions; dev = held-out dev questions)")
            rows = []
            for d in runs:
                st = json.load(open(os.path.join(d, "trainer_state.json"))) if os.path.exists(os.path.join(d, "trainer_state.json")) else {}
                log = read_jsonl(os.path.join(d, "train_log.jsonl")) if os.path.exists(os.path.join(d, "train_log.jsonl")) else []
                first, last = log[:10], log[-10:]
                dev = [r.get("dev/outcome") for r in log if "dev/outcome" in r]
                rows.append([os.path.basename(d), f"{st.get('step', len(log))}", "yes" if st.get("finished") else "NO",
                             f"{fmt(mean_key(first, 'train/outcome'))}->{fmt(mean_key(last, 'train/outcome'))}",
                             f"{fmt(mean_key(first, 'train/coverage'))}->{fmt(mean_key(last, 'train/coverage'))}",
                             f"{fmt(mean_key(first, 'train/pool_recall'))}->{fmt(mean_key(last, 'train/pool_recall'))}",
                             f"{fmt(dev[0])}->{fmt(dev[-1])}" if dev else "-",
                             fmt(log[-1].get("judge_cost_usd"), 2) if log else "-"])
            L.extend(table(["run", "steps", "finished", "train overall", "train coverage", "train gold_found",
                            "dev overall", "judge $"], rows))
        if os.path.isdir(sft):
            say("\n-- SFT / DPO")
            rows = []
            for f in sorted(os.listdir(sft)):
                p = os.path.join(sft, f)
                if f.endswith("_data.jsonl"):
                    rows.append([f, "SFT training turns", sum(1 for _ in open(p))])
                elif f.endswith("_dpo_pairs.jsonl"):
                    rows.append([f, "DPO preference pairs", sum(1 for _ in open(p))])
                elif os.path.isdir(p):
                    rows.append([f + "/", "trained adapter", "yes" if os.path.exists(os.path.join(p, "adapter_config.json")) else "NO"])
                elif f.endswith("_log.jsonl"):
                    lg = read_jsonl(p)
                    if lg:
                        rows.append([f, "DPO loss / pair accuracy",
                                     f"{fmt(lg[0].get('dpo_loss'))}->{fmt(lg[-1].get('dpo_loss'))} / "
                                     f"{fmt(lg[0].get('reward_acc'), 2)}->{fmt(lg[-1].get('reward_acc'), 2)} ({len(lg)} steps)"])
            L.extend(table(["file", "what", "value"], rows))
        if trains:
            say("\n-- Sampled training answers used for SFT/DPO (4 per training question)")
            rows = []
            for f in trains:
                s = json.load(open(f))
                m = s.get("metrics", {})
                rows.append([os.path.relpath(f, md), s.get("n"), fmt(m.get("outcome_verifiable", {}).get("mean")),
                             fmt(m.get("coverage", {}).get("mean")), judge_name(os.path.dirname(os.path.dirname(f)))])
            L.extend(table(["scores file", "answers", "mean overall", "mean coverage", "judge"], rows))

    # ---------------------------------------------------------------- write
    text = "\n".join(L) + "\n"
    with open(a.txt, "w") as f:
        f.write(text)
    if csv_rows:
        keys = ["judge", "model", "task", "category", "system", "label", "n"]
        keys += sorted({k for r in csv_rows for k in r} - set(keys))
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(csv_rows)
    print(text)
    print(f"Saved: {a.txt}" + (f"\n       {a.csv}" if csv_rows else ""))


if __name__ == "__main__":
    main()
