"""One-shot diagnostic report (CPU only, no API calls, no GPU). Run on the login node:

  python diagnose.py

All paths are set below. The report is printed on screen AND saved to version_1/diagnose.txt.
Sections: training status (GRPO / SFT / DPO), trained-model generations and score folders, retrieval recall
(is the gold evidence findable at all?), ReAct and pipeline3 behaviour, one Task B example.
"""
# ============================ settings (edit here) ============================
PROJECT_DIR = "/lstr/sahara/datalab-ml/ibrahim/limagents_update/sci_lit_synthesis/version_1"
DATA_DIR = PROJECT_DIR + "/data/df_updated_with_retrieval_upd_n1000"
OUT_DIR = PROJECT_DIR + "/outputs/df_updated_with_retrieval_upd_n1000/qwen3_8b"
EMBED_MODEL = "/lstr/sahara/datalab-ml/ibrahim/models/all-MiniLM-L6-v2"   # None = skip the hybrid retrieval check
REPORT = PROJECT_DIR + "/diagnose.txt"
# ===============================================================================
import argparse
import glob
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402

# find the scisynth package wherever this file is saved (version_1/, version_1/tools/, or elsewhere)
for _p in (PROJECT_DIR, os.path.dirname(os.path.abspath(__file__)), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
    if os.path.isdir(os.path.join(_p, "scisynth")):
        sys.path.insert(0, _p)
        break
from scisynth.benchmark import load_items  # noqa: E402
from scisynth.utils import read_jsonl, truncate_words  # noqa: E402


class _Tee:
    """Write everything to the screen and to the report file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def section(title):
    print("\n" + "=" * 12 + " " + title + " " + "=" * 12)


def mean_of(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return round(float(np.mean(vals)), 4) if vals else None


def training(out_dir):
    section("GRPO runs")
    runs = sorted(glob.glob(os.path.join(out_dir, "grpo", "*")))
    if not runs:
        print("no grpo runs")
    for d in runs:
        st_path = os.path.join(d, "trainer_state.json")
        st = json.load(open(st_path)) if os.path.exists(st_path) else {}
        log = read_jsonl(os.path.join(d, "train_log.jsonl"))
        ck = sorted(os.listdir(os.path.join(d, "ckpt"))) if os.path.isdir(os.path.join(d, "ckpt")) else []
        print(f"\n{os.path.basename(d)}: logged steps={len(log)} saved step={st.get('step')} "
              f"finished={st.get('finished', False)} best_dev={st.get('best_dev')} ckpt={ck}")
        if not log:
            continue
        first, last = log[:10], log[-10:]
        for k in ["train/outcome", "train/coverage", "train/support", "train/cite_f1", "train/pool_recall",
                  "train/r_search", "train/r_read", "train/r_synth", "train/r_direct", "train/words",
                  "dead_groups", "rescued_by_process", "learner/loss", "learner/n_samples", "learner/peak_mem_gb",
                  "time_s"]:
            a, b = mean_of(first, k), mean_of(last, k)
            if a is not None or b is not None:
                print(f"  {k:24s} first10={a}  last10={b}")
        dev = [(r["step"], r.get("dev/outcome"), r.get("dev/coverage")) for r in log if "dev/outcome" in r]
        if dev:
            print("  dev (step, outcome, coverage):", dev)
        print("  judge cost so far ($):", log[-1].get("judge_cost_usd"))

    section("SFT / DPO")
    sft = os.path.join(out_dir, "sft")
    if not os.path.isdir(sft):
        print("no sft folder")
    else:
        for f in sorted(os.listdir(sft)):
            p = os.path.join(sft, f)
            if f.endswith(".jsonl"):
                n = sum(1 for _ in open(p))
                print(f"  {f}: {n} lines")
            elif os.path.isdir(p):
                print(f"  {f}/: adapter={'yes' if os.path.exists(os.path.join(p, 'adapter_config.json')) else 'NO'}")
        for lg in sorted(glob.glob(os.path.join(sft, "*_log.jsonl"))):
            rows = read_jsonl(lg)
            if rows:
                print(f"  {os.path.basename(lg)}: first={ {k: rows[0].get(k) for k in ('dpo_loss', 'reward_acc', 'margin')} } "
                      f"last={ {k: rows[-1].get(k) for k in ('dpo_loss', 'reward_acc', 'margin')} } steps={len(rows)}")


def outputs(out_dir):
    section("Generations of trained models")
    for f in sorted(glob.glob(os.path.join(out_dir, "gens", "*", "*.jsonl"))):
        name = os.path.basename(f)
        if any(t in name for t in ("_rft", "_dpo", "_grpo", "_train")):
            print(f"  {os.path.relpath(f, out_dir)}: {sum(1 for _ in open(f))} records")
    section("Score folders")
    for d in sorted(glob.glob(os.path.join(out_dir, "scores*", "*"))):
        files = sorted(os.path.basename(x)[:-len('.summary.json')] for x in glob.glob(os.path.join(d, "*.summary.json")))
        print(f"  {os.path.relpath(d, out_dir)}: {files}")
    tables = os.path.join(os.path.dirname(os.path.abspath(out_dir)), "tables")
    print("  tables:", sorted(os.listdir(tables)) if os.path.isdir(tables) else "none")


def score_columns(out_dir):
    section("Retrieval columns from the scores (mean over test items)")
    for d in sorted(glob.glob(os.path.join(out_dir, "scores*", "*"))):
        for f in sorted(glob.glob(os.path.join(d, "*.eval.jsonl"))):
            rows = [r["rewards"] for r in read_jsonl(f)]
            print(f"  {os.path.relpath(f, out_dir):55s} pool_recall={mean_of(rows, 'pool_recall')} "
                  f"cite_prec={mean_of(rows, 'cite_prec')} cite_rec={mean_of(rows, 'cite_rec')} "
                  f"cite_valid={mean_of(rows, 'cite_valid')} coverage={mean_of(rows, 'coverage')}")


def retrieval(data_dir, embed_model):
    from scisynth.data import Corpus
    from scisynth.retrieval import Retriever
    from scisynth.systems import default_query

    section("Is the gold evidence findable? (recall@k on the test split)")
    corpus = Corpus(data_dir)
    print("  corpus size:", len(corpus))
    modes = ["bm25"] + (["hybrid"] if embed_model else [])
    for mode in modes:
        try:
            r = Retriever(data_dir, mode, embed_model)
        except Exception as e:
            print(f"  {mode}: could not load ({e})")
            continue
        for task in ("qa", "rw"):
            items = load_items(data_dir, task, "test")
            if not items:
                continue
            res = r.search_batch([default_query(it) for it in items], 100, [it.get("exclude_doc_ids", []) for it in items])
            out = []
            for k in (10, 50, 100):
                rec = [len(set(x[:k]) & set(it["gold_doc_ids"])) / max(1, len(it["gold_doc_ids"])) for it, x in zip(items, res)]
                out.append(f"R@{k}={np.mean(rec):.3f}")
            gsize = np.mean([len(it["gold_doc_ids"]) for it in items])
            missing = np.mean([sum(1 for d in it["gold_doc_ids"] if d not in corpus) for it in items])
            print(f"  {mode:6s} {task}: n={len(items)} gold/item={gsize:.1f} gold-not-in-corpus/item={missing:.2f}  " + "  ".join(out))
        # index sanity: a gold document's own first 30 words should retrieve it at rank 1
        items = (load_items(data_dir, "rw", "test") or load_items(data_dir, "qa", "test"))[:40]
        docs = [it["gold_doc_ids"][0] for it in items if it["gold_doc_ids"]]
        if not docs:
            continue
        hits = r.search_batch([truncate_words(corpus.text(d), 40) for d in docs], 5)
        print(f"  {mode:6s} index sanity (a gold doc's own first 40 words find it): rank1={np.mean([h[:1] == [d] for h, d in zip(hits, docs)]):.2f} "
              f"top5={np.mean([d in h for h, d in zip(hits, docs)]):.2f} (both should be close to 1.0)")


def behaviour(out_dir):
    section("ReAct and pipeline3 behaviour")
    for task in ("qa", "rw"):
        for system in ("react", "pipeline3"):
            f = os.path.join(out_dir, "gens", task, f"{system}.jsonl")
            if not os.path.exists(f):
                continue
            recs = read_jsonl(f)
            n_search = [sum(1 for t in r["turns"] if t["phase"] == "search" and (t.get("info") or {}).get("queries")) for r in recs]
            not_in_pool = []
            for r in recs:
                c = r.get("cited") or []
                if c:
                    not_in_pool.append(sum(1 for d in c if d not in set(r.get("pool") or [])) / len(c))
            first = [r["turns"][0]["text"] for r in recs if r["turns"]]
            print(f"  {task}/{system}: n={len(recs)} no-search={np.mean([s == 0 for s in n_search]):.2f} "
                  f"searches/episode={np.mean(n_search):.2f} cited-not-in-pool={np.mean(not_in_pool) if not_in_pool else 0:.2f} "
                  f"first-turn-has-<search>={np.mean(['<search>' in t.lower() for t in first]):.2f} "
                  f"first-turn-has-<answer>={np.mean(['<answer>' in t.lower() for t in first]):.2f} "
                  f"fallback={np.mean([bool((r.get('rewards') or {}).get('search_fallback')) for r in recs]):.2f}")
            if system == "react" and recs:
                print("    first turn of record 0:", repr(first[0][:300]))
                q = [((t.get('info') or {}).get('queries')) for t in recs[0]["turns"]]
                print("    queries of record 0:", q)


def rw_example(data_dir, out_dir):
    from scisynth.data import Corpus

    section("One Task B example (gold vs retrieved)")
    f = os.path.join(out_dir, "gens", "rw", "rag.jsonl")
    if not os.path.exists(f):
        print("  no gens/rw/rag.jsonl")
        return
    corpus = Corpus(data_dir)
    items = {it["item_id"]: it for it in load_items(data_dir, "rw", "all")}
    rec = read_jsonl(f)[0]
    it = items[rec["item_id"]]
    print("  title:", it["title"])
    print("  abstract:", truncate_words(it["abstract"], 40))
    print("  gold docs (first 3 of", len(it["gold_doc_ids"]), "):")
    for d in it["gold_doc_ids"][:3]:
        print("    ", d, truncate_words(corpus.text(d), 25))
    print("  retrieved by RAG (first 3 of", len(rec["pool"]), "):")
    for d in rec["pool"][:3]:
        print("    ", d, truncate_words(corpus.text(d), 25))
    print("  cited in answer:", rec.get("cited"), "| answer start:", truncate_words(rec.get("answer", ""), 40))


def main():
    ap = argparse.ArgumentParser(description="all arguments are optional; defaults are set at the top of this file")
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--embed_model", default=EMBED_MODEL)
    ap.add_argument("--report", default=REPORT)
    a = ap.parse_args()
    report = open(a.report, "w")
    sys.stdout = _Tee(sys.__stdout__, report)
    sys.stderr = _Tee(sys.__stderr__, report)
    print(f"data_dir={a.data_dir}\nout_dir={a.out_dir}\nembed_model={a.embed_model}")
    for fn, args in ((training, (a.out_dir,)), (outputs, (a.out_dir,)), (score_columns, (a.out_dir,)),
                     (retrieval, (a.data_dir, a.embed_model)), (behaviour, (a.out_dir,)), (rw_example, (a.data_dir, a.out_dir))):
        try:
            fn(*args)
        except Exception as e:  # keep going: a partial report is still useful
            print(f"  [{fn.__name__} failed: {type(e).__name__}: {e}]")
    print(f"\nReport saved to {a.report}")
    sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
    report.close()


if __name__ == "__main__":
    main()
