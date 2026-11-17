"""Write final_table.tex: every result of version_1 + the version_2 GRPO runs in paper-ready LaTeX tables.
(CPU only, no GPU, no API calls)

  python tools/final_table.py

Reads   version_1/outputs/<data>/qwen3_8b/scores/<task>/*.eval.jsonl (+ .summary.json)   (zero-shot ... DPO, oracle)
        version_2/outputs/<data>/qwen3_8b/scores/<task>/*.eval.jsonl (+ .summary.json)   (GRPO runs)
        version_1 and version_2 grpo/*/trainer_state.json + train_log.jsonl            (training summary)
Writes  version_2/final_table.tex   tables: main results, paired comparisons, headroom subset, GRPO training
        version_2/final_table.txt   the same numbers as plain text (also printed)

All rows use the same test questions and the same gpt-4o-mini verifier. Paired differences use a bootstrap over
questions (5,000 resamples); * = the 95% CI excludes 0. Use in the paper with \\input{final_table.tex}.
"""
# ============================ settings (edit here) ============================
V1_DIR = "/lstr/sahara/datalab-ml/ibrahim/limagents_update/sci_lit_synthesis/version_1"
V2_DIR = "/lstr/sahara/datalab-ml/ibrahim/limagents_update/sci_lit_synthesis/version_2"
DATA_NAME = "df_updated_with_retrieval_upd_n1000"
MODEL_TAG = "qwen3_8b"
SCORES_SUBDIR = "scores"          # gpt-4o-mini verifier
TASKS = ["qa", "rw"]
OUT_TEX = V2_DIR + "/final_table.tex"
OUT_TXT = V2_DIR + "/final_table.txt"
HEADROOM_THRESHOLD = 0.35
N_BOOTSTRAP = 5000
# ===============================================================================
import argparse
import datetime
import glob
import json
import os

import numpy as np

GROUPS = [
    ("Parametric knowledge only", ["zero_shot", "cot"]),
    ("Retrieval, untrained", ["rag", "react", "pipeline3"]),
    ("Three-role agent, trained", ["pipeline3_rft", "pipeline3_dpo", "pipeline3_grpo_outcome", "pipeline3_grpo_v1_outcome",
                                   "pipeline3_grpo_base_init", "pipeline3_grpo_dpo_init"]),
    ("Input: abstract + introduction", ["zero_shot_absintro", "rag_absintro", "pipeline3_absintro"]),
    ("Oracle evidence (upper bound and sensitivity)", ["oracle", "oracle_drop50", "oracle_distract10"]),
]
LABELS = {
    "zero_shot": "Zero-shot", "cot": "Chain-of-thought", "rag": "RAG ($k=10$)", "react": "ReAct agent",
    "pipeline3": "Three-role agent", "pipeline3_rft": "\\quad + SFT (best-of-4)", "pipeline3_dpo": "\\quad + DPO",
    "pipeline3_grpo_outcome": "\\quad + GRPO (v1, outcome)",
    "pipeline3_grpo_v1_outcome": "\\quad + GRPO (v1: outcome reward, from base)",
    "pipeline3_grpo_base_init": "\\quad + GRPO (v2, from base)",
    "pipeline3_grpo_dpo_init": "\\quad + DPO $\\rightarrow$ GRPO (v2, ours)",
    "zero_shot_absintro": "Zero-shot +intro", "rag_absintro": "RAG +intro", "pipeline3_absintro": "Agent +intro",
    "oracle": "Oracle (all gold)", "oracle_drop50": "Oracle $-$50\\% gold", "oracle_distract10": "Oracle + 10 distractors",
}
PLAIN = {k: v.replace("\\quad ", "  ").replace("$\\rightarrow$", "->").replace("$k=10$", "k=10").replace("$-$", "-")
         .replace("\\%", "%") for k, v in LABELS.items()}
# main-table columns: (metric key, header, digits, bold-able)
COLS = [("outcome_verifiable", "Overall", 3, True), ("coverage", "Coverage [95\\% CI]", 3, True), ("support", "Support", 3, True),
        ("cite_prec", "Cite P", 3, True), ("cite_rec", "Cite R", 3, True), ("cite_f1", "Cite F1", 3, True),
        ("pool_recall", "Gold found", 3, True), ("ground_prec", "Grounded", 3, True), ("judge_overall", "Judge (1--10)", 2, False),
        ("n_search_turns", "\\#Search", 2, False), ("pool_size", "\\#Read", 1, False), ("n_cited", "\\#Cites", 2, False)]
COLS_RW_EXTRA = [("ref_coverage", "Ref.\\ cov.", 3, True)]
PAIR_METRICS = [("outcome_verifiable", "$\\Delta$ Overall"), ("coverage", "$\\Delta$ Coverage"),
                ("pool_recall", "$\\Delta$ Gold found"), ("cite_f1", "$\\Delta$ Cite F1")]
NO_RETRIEVAL = {"zero_shot", "cot", "zero_shot_absintro"}
SHORT = {"pipeline3": "Agent", "pipeline3_rft": "SFT", "pipeline3_dpo": "DPO", "rag": "RAG", "zero_shot": "Zero-shot",
         "oracle": "Oracle", "pipeline3_grpo_dpo_init": "DPO$\\rightarrow$GRPO (ours)", "pipeline3_grpo_base_init": "GRPO (v2, base)",
         "pipeline3_grpo_v1_outcome": "GRPO (v1)", "pipeline3_grpo_outcome": "GRPO (v1)"}
TXT_HEAD = {"outcome_verifiable": "overall", "coverage": "coverage", "support": "support", "cite_prec": "cite_P",
            "cite_rec": "cite_R", "cite_f1": "cite_F1", "pool_recall": "gold_found", "ground_prec": "grounded",
            "judge_overall": "judge_1-10", "n_search_turns": "searches", "pool_size": "papers_read", "n_cited": "n_cited",
            "ref_coverage": "ref_cov"}


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def known_order():
    return [s for _, names in GROUPS for s in names]


def label(s, plain=False):
    if s in LABELS:
        return PLAIN[s] if plain else LABELS[s]
    if s.startswith("pipeline3_grpo_"):
        t = "GRPO (" + s[len("pipeline3_grpo_"):].replace("_", " ") + ")"
        return ("  + " + t) if plain else ("\\quad + " + t)
    return s if plain else s.replace("_", "\\_")


def group_of(s):
    for g, names in GROUPS:
        if s in names:
            return g
    if s.startswith("pipeline3_grpo_") or s.endswith(("_rft", "_dpo")):
        return GROUPS[2][0]
    return "Other"


def collect(task):
    """system -> {"summary": metrics dict, "items": {item_id: rewards}, "src": v1|v2}; version_2 wins on name clashes."""
    out = {}
    for src, root in (("v1", V1_DIR), ("v2", V2_DIR)):
        d = os.path.join(root, "outputs", DATA_NAME, MODEL_TAG, SCORES_SUBDIR, task)
        for f in sorted(glob.glob(os.path.join(d, "*.eval.jsonl"))):
            name = os.path.basename(f)[: -len(".eval.jsonl")]
            recs = read_jsonl(f)
            items = {r["item_id"]: r.get("rewards", {}) for r in recs}
            summ_path = f.replace(".eval.jsonl", ".eval.summary.json")
            metrics = {}
            if os.path.exists(summ_path):
                metrics = json.load(open(summ_path)).get("metrics", {})
            for k in {k for r in items.values() for k, v in r.items() if isinstance(v, (int, float))}:
                if k not in metrics:
                    xs = [r[k] for r in items.values() if isinstance(r.get(k), (int, float))]
                    metrics[k] = {"mean": float(np.mean(xs))}
            out[name] = {"summary": metrics, "items": items, "src": src}
    return out


def ordered(systems):
    ko = known_order()
    return sorted(systems, key=lambda s: (ko.index(s) if s in ko else (ko.index("pipeline3_grpo_dpo_init") + 0.5
                                                                       if s.startswith("pipeline3_grpo_") else 999), s))


def mean_of(entry, key):
    v = entry["summary"].get(key, {})
    m = v.get("mean")
    return None if m is None or (isinstance(m, float) and np.isnan(m)) else float(m)


def fmt(v, d=3):
    return "--" if v is None else f"{v:.{d}f}"


def paired(a, b, metric, rng):
    ids = [i for i in a["items"] if i in b["items"]]
    diff = np.array([a["items"][i][metric] - b["items"][i][metric] for i in ids
                     if isinstance(a["items"][i].get(metric), (int, float)) and isinstance(b["items"][i].get(metric), (int, float))])
    if len(diff) < 2:
        return None
    boot = diff[rng.integers(0, len(diff), (N_BOOTSTRAP, len(diff)))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": float(diff.mean()), "lo": float(lo), "hi": float(hi), "sig": bool(lo > 0 or hi < 0),
            "wtl": f"{int((diff > 0).sum())}/{int((diff == 0).sum())}/{int((diff < 0).sum())}", "n": len(diff)}


def best_grpo(systems):
    for s in ("pipeline3_grpo_dpo_init", "pipeline3_grpo_base_init", "pipeline3_grpo_v1_outcome", "pipeline3_grpo_outcome"):
        if s in systems:
            return s
    g = [s for s in systems if s.startswith("pipeline3_grpo_")]
    return g[0] if g else None


def grpo_runs():
    rows = []
    for ver, root in (("v1", V1_DIR), ("v2", V2_DIR)):
        for d in sorted(glob.glob(os.path.join(root, "outputs", DATA_NAME, MODEL_TAG, "grpo", "*"))):
            sp, lp = os.path.join(d, "trainer_state.json"), os.path.join(d, "train_log.jsonl")
            if not os.path.exists(sp):
                continue
            st = json.load(open(sp))
            log = read_jsonl(lp) if os.path.exists(lp) else []
            tr = [r for r in log if r.get("step", 0) > 0 and "train/outcome_verifiable" in r]
            dev = [(r["step"], r.get("dev/outcome_verifiable")) for r in log if r.get("dev/outcome_verifiable") is not None]

            def m(rs, k):
                xs = [r[k] for r in rs if isinstance(r.get(k), (int, float))]
                return float(np.mean(xs)) if xs else None
            cfg = {}
            if os.path.exists(os.path.join(d, "run_config.json")):
                cfg = json.load(open(os.path.join(d, "run_config.json")))
            rows.append({
                "run": f"{ver}/{os.path.basename(d)}", "steps": st.get("step"), "finished": bool(st.get("finished")),
                "init": "DPO" if cfg.get("init_adapter") else "base",
                "best_step": st.get("best_step"), "dev_start": dev[0][1] if dev else None,
                "dev_best": st.get("best_dev") if st.get("best_dev", -1e9) > -1e8 else None,
                "tr_overall": (m(tr[:10], "train/outcome_verifiable"), m(tr[-10:], "train/outcome_verifiable")),
                "tr_cov": (m(tr[:10], "train/coverage"), m(tr[-10:], "train/coverage")),
                "tr_gold": (m(tr[:10], "train/pool_recall"), m(tr[-10:], "train/pool_recall")),
                "cost": log[-1].get("judge_cost_usd") if log else None,
            })
    return rows


def main():
    ap = argparse.ArgumentParser(description="all arguments are optional; defaults are at the top of this file")
    ap.add_argument("--tex", default=OUT_TEX)
    ap.add_argument("--txt", default=OUT_TXT)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    T, X = [], []   # tex lines, text lines
    T += ["% final_table.tex -- generated " + f"{datetime.datetime.now():%Y-%m-%d %H:%M} by tools/final_table.py",
          "% Requires \\usepackage{booktabs} and \\usepackage{graphicx}. Include with \\input{final_table.tex}.",
          "\\providecommand{\\sig}{$^{\\ast}$}",
          "% shrink a table to the line width only when it is wider",
          "\\providecommand{\\fittable}[1]{\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{#1}}", ""]
    X += [f"FINAL TABLE (generated {datetime.datetime.now():%Y-%m-%d %H:%M})",
          f"version_1 scores: {V1_DIR}/outputs/{DATA_NAME}/{MODEL_TAG}/{SCORES_SUBDIR}",
          f"version_2 scores: {V2_DIR}/outputs/{DATA_NAME}/{MODEL_TAG}/{SCORES_SUBDIR}", ""]

    for task in TASKS:
        data = collect(task)
        if not data:
            continue
        systems = ordered(data)
        n_items = max(len(e["items"]) for e in data.values())
        tname = "Task A (question-conditioned synthesis)" if task == "qa" else "Task B (related-work generation)"
        cols = COLS + (COLS_RW_EXTRA if task == "rw" else [])
        # ---------------- main table
        best = {}
        for key, _, _, boldable in cols:
            vals = [mean_of(data[s], key) for s in systems if not s.startswith("oracle")]
            vals = [v for v in vals if v is not None]
            if boldable and vals:
                best[key] = max(vals)
        T += ["\\begin{table*}[t]", "\\centering",
              f"\\caption{{Main results on {tname}; Qwen3-8B; {n_items} test items; gpt-4o-mini verifier. Overall is the "
              "verifiable outcome score. Bold = best non-oracle value per column. Rows marked v2 are new GRPO runs: rollouts "
              "are generated by the policy being trained, and gpt-4o-mini is used only as the reward model.}",
              f"\\label{{tab:final_{task}}}", "\\footnotesize", "\\setlength{\\tabcolsep}{3.1pt}",
              "\\fittable{%", "\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
              "System & " + " & ".join(h for _, h, _, _ in cols) + " \\\\", "\\midrule"]
        X += ["=" * 110, f"{tname}  (n={n_items})", "=" * 110]
        X.append(f"{'system':44s}" + "".join(f"{TXT_HEAD.get(k, k):>12s}" for k, _, _, _ in cols))
        last = None
        for s in systems:
            g = group_of(s)
            if g != last:
                if last is not None:
                    T.append("\\midrule")
                T.append(f"\\multicolumn{{{len(cols) + 1}}}{{l}}{{\\textit{{{g}}}}} \\\\")
                X.append(f"[{g}]")
                last = g
            cells, xc = [], []
            for key, _, d, boldable in cols:
                v = mean_of(data[s], key)
                if key in ("support", "n_search_turns", "pool_size") and s in NO_RETRIEVAL:
                    v = None
                if key == "n_search_turns" and (s.startswith(("rag", "oracle"))):
                    v = None   # single retrieval / gold given: no search turns
                c = fmt(v, d)
                if key == "coverage" and v is not None:
                    sm = data[s]["summary"].get("coverage", {})
                    if sm.get("ci_lo") is not None:
                        c += f" [{sm['ci_lo']:.2f}, {sm['ci_hi']:.2f}]"
                if boldable and v is not None and key in best and not s.startswith("oracle") and round(v, d) == round(best[key], d):
                    c = "\\textbf{" + c + "}"
                cells.append(c)
                xc.append(fmt(v, d))
            T.append(label(s) + " & " + " & ".join(cells) + " \\\\")
            X.append(f"{label(s, True):44s}" + "".join(f"{c:>12s}" for c in xc) + f"   ({data[s]['src']})")
        T += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}", ""]

        # ---------------- paired comparisons
        g = best_grpo(data)
        pairs = []
        if g:
            pairs += [(g, "pipeline3_dpo"), (g, "pipeline3"), (g, "rag"), (g, "pipeline3_rft")]
            pairs += [(o, "pipeline3_dpo") for o in systems if o.startswith("pipeline3_grpo_") and o != g]
        pairs += [("pipeline3_dpo", "pipeline3"), ("pipeline3_dpo", "rag"), ("pipeline3_rft", "pipeline3"),
                  ("pipeline3", "rag"), ("rag", "zero_shot"), ("oracle", "pipeline3")]
        rows = []
        for x, y in pairs:
            if x in data and y in data and (x, y) not in [r[0] for r in rows]:
                res = [paired(data[x], data[y], m, rng) for m, _ in PAIR_METRICS]
                if any(r is not None for r in res):
                    rows.append(((x, y), res))
        if rows:
            T += ["\\begin{table*}[t]", "\\centering",
                  f"\\caption{{Paired comparisons on {tname} (difference = first $-$ second system on the same test items, "
                  "paired-bootstrap 95\\% CI). \\sig{} = CI excludes zero. W/T/L = items on which the first system has "
                  "higher/equal/lower coverage.}", f"\\label{{tab:final_{task}_paired}}", "\\scriptsize",
                  "\\setlength{\\tabcolsep}{3.1pt}", "\\fittable{%", "\\begin{tabular}{lccccc}", "\\toprule",
                  "Comparison & " + " & ".join(h for _, h in PAIR_METRICS) + " & W/T/L (cov.) \\\\", "\\midrule"]
            X += ["", "-- paired comparisons (first - second; [95% CI]; * = significant)"]
            for (x, y), res in rows:
                lx, ly = SHORT.get(x, label(x).replace("\\quad + ", "")), SHORT.get(y, label(y).replace("\\quad + ", ""))
                cells, xc = [], []
                for r in res:
                    if r is None:
                        cells.append("--")
                        xc.append("-")
                        continue
                    v = f"${r['mean']:+.3f}$" + ("\\sig{}" if r["sig"] else "")
                    if r["sig"] and r["mean"] > 0 and x == g:
                        v = "\\textbf{" + v + "}"
                    cells.append(f"{v} [${r['lo']:+.3f}$, ${r['hi']:+.3f}$]")
                    xc.append(f"{r['mean']:+.3f}{'*' if r['sig'] else ' '} [{r['lo']:+.3f},{r['hi']:+.3f}]")
                cov = res[1]
                T.append(f"{lx} vs.\\ {ly} & " + " & ".join(cells) + f" & {cov['wtl'] if cov else '--'} \\\\")
                X.append(f"{label(x, True).strip(' +')} vs {label(y, True).strip(' +')}: " + " | ".join(
                    f"{TXT_HEAD[m]}: {c}" for (m, _), c in zip(PAIR_METRICS, xc)) + f" | W/T/L(cov) {cov['wtl'] if cov else '-'}")
            T += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}", ""]

        # ---------------- headroom subset
        zs = data.get("zero_shot")
        if zs:
            hard = {i for i, r in zs["items"].items() if (r.get("coverage") or 0) <= HEADROOM_THRESHOLD}
            T += ["\\begin{table}[t]", "\\centering",
                  f"\\caption{{Coverage on all test items and on the headroom subset of {tname.split(' (')[0]} "
                  f"(zero-shot coverage $\\le {HEADROOM_THRESHOLD}$; {len(hard)} of {len(zs['items'])} items).}}",
                  f"\\label{{tab:final_{task}_headroom}}", "\\footnotesize", "\\setlength{\\tabcolsep}{5pt}",
                  "\\fittable{%", "\\begin{tabular}{lccc}", "\\toprule", "System & Cov.\\ (all) & Cov.\\ (hard) & Overall (hard) \\\\", "\\midrule"]
            X += ["", f"-- headroom subset: {len(hard)} of {len(zs['items'])} items (zero-shot coverage <= {HEADROOM_THRESHOLD})"]
            hv = {}
            for s in systems:
                R = [r for i, r in data[s]["items"].items() if i in hard]
                if not R:
                    continue
                cov_h = float(np.mean([r.get("coverage", 0.0) for r in R]))
                ov_h = float(np.mean([r.get("outcome_verifiable", 0.0) for r in R]))
                hv[s] = (mean_of(data[s], "coverage"), cov_h, ov_h)
            bests = [max((v[k] for s, v in hv.items() if not s.startswith("oracle") and v[k] is not None), default=None) for k in range(3)]
            last = None
            for s, v in hv.items():
                if s.startswith("oracle") and last != "oracle":
                    T.append("\\midrule")
                    last = "oracle"
                cells = []
                for k in range(3):
                    c = fmt(v[k])
                    if not s.startswith("oracle") and v[k] is not None and bests[k] is not None and round(v[k], 3) == round(bests[k], 3):
                        c = "\\textbf{" + c + "}"
                    cells.append(c)
                T.append(label(s) + " & " + " & ".join(cells) + " \\\\")
                X.append(f"{label(s, True):44s}" + "".join(f"{fmt(x):>12s}" for x in v))
            T += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]

    # ---------------- GRPO training summary
    runs = grpo_runs()
    if runs:
        T += ["\\begin{table*}[t]", "\\centering",
              "\\caption{GRPO training runs. Train columns: mean of the first $\\rightarrow$ last 10 steps on sampled training "
              "questions (temperature 0.9). Dev: verifiable score of greedy answers on held-out dev questions; step 0 = the "
              "starting policy. The checkpoint with the best dev score is evaluated on the test set.}",
              "\\label{tab:final_grpo_runs}", "\\scriptsize", "\\setlength{\\tabcolsep}{3.5pt}",
              "\\fittable{%", "\\begin{tabular}{lcccccccc}", "\\toprule",
              "Run & Init & Steps & Train overall & Train coverage & Train gold found & Dev start $\\rightarrow$ best "
              "& Best step & Reward cost (USD) \\\\", "\\midrule"]
        X += ["", "=" * 110, "GRPO training runs", "=" * 110]
        for r in runs:
            arrow = lambda p: f"{fmt(p[0])} $\\rightarrow$ {fmt(p[1])}"
            plain = lambda p: f"{fmt(p[0])}->{fmt(p[1])}"
            steps = f"{r['steps']}" + ("" if r["finished"] else " (unfinished)")
            T.append(f"{r['run'].replace('_', chr(92) + '_')} & {r['init']} & {steps} & {arrow(r['tr_overall'])} & {arrow(r['tr_cov'])} & "
                     f"{arrow(r['tr_gold'])} & {fmt(r['dev_start'])} $\\rightarrow$ {fmt(r['dev_best'])} & "
                     f"{r['best_step'] if r['best_step'] is not None else '--'} & {fmt(r['cost'], 2)} \\\\")
            X.append(f"{r['run']:44s} init={r['init']:4s} steps={steps:16s} train overall {plain(r['tr_overall'])}  coverage "
                     f"{plain(r['tr_cov'])}  gold {plain(r['tr_gold'])}  dev {fmt(r['dev_start'])}->{fmt(r['dev_best'])} "
                     f"(best step {r['best_step']})  ${fmt(r['cost'], 2)}")
        T += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}", ""]

    with open(a.tex, "w") as f:
        f.write("\n".join(T) + "\n")
    with open(a.txt, "w") as f:
        f.write("\n".join(X) + "\n")
    print("\n".join(X))
    print(f"\nSaved: {a.tex}\n       {a.txt}")


if __name__ == "__main__":
    main()
