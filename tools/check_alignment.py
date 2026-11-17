"""Do the cited papers in each row really belong to that row's paper?  (CPU only, no GPU, no API calls)

  python check_alignment.py

For every seed paper we search the 500+ seed abstracts with the text of ITS OWN cited papers. If the CSV columns
`pdf_text` and `cited_in_abs_intro` are aligned, the paper's own abstract should usually come out on top. If they are
misaligned (e.g. rows shifted after sampling/merging), the own paper ranks at chance level and the best match is
another row. The report is printed and saved to version_1/check_alignment.txt.
"""
# ============================ settings (edit here) ============================
PROJECT_DIR = "/lstr/sahara/datalab-ml/ibrahim/limagents_update/sci_lit_synthesis/version_1"
DATA_DIR = PROJECT_DIR + "/data/df_updated_with_retrieval_upd_n1000"
REPORT = PROJECT_DIR + "/check_alignment.txt"
DOCS_PER_QUERY = 5      # how many of the row's cited papers form the query
WORDS_PER_DOC = 60
# ===============================================================================
import collections
import os
import sys

import numpy as np

for _p in (PROJECT_DIR, os.path.dirname(os.path.abspath(__file__)), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
    if os.path.isdir(os.path.join(_p, "scisynth")):
        sys.path.insert(0, _p)
        break
from scisynth.data import Corpus  # noqa: E402
from scisynth.retrieval import BM25  # noqa: E402
from scisynth.utils import read_jsonl, truncate_words  # noqa: E402


def main():
    out = []

    def say(*a):
        line = " ".join(str(x) for x in a)
        print(line)
        out.append(line)

    ex = [e for e in read_jsonl(os.path.join(DATA_DIR, "examples.jsonl")) if e.get("gold_doc_ids")]
    ex.sort(key=lambda e: e["row_index"])
    corpus = Corpus(DATA_DIR)
    say(f"seed papers: {len(ex)} | corpus: {len(corpus)} docs")

    seeds = BM25().build([e["title"] + " " + e["abstract"] for e in ex])
    rows = [e["row_index"] for e in ex]
    ranks, offsets = [], collections.Counter()
    examples = []
    for i, e in enumerate(ex):
        q = " ".join(truncate_words(corpus.text(d), WORDS_PER_DOC) for d in e["gold_doc_ids"][:DOCS_PER_QUERY])
        sc = seeds.scores(q)
        order = np.argsort(-sc)
        rank = int(np.where(order == i)[0][0]) + 1
        ranks.append(rank)
        top = int(order[0])
        if top != i:
            offsets[rows[top] - rows[i]] += 1
            if len(examples) < 4:
                examples.append((e, ex[top]))
    ranks = np.array(ranks)
    n = len(ranks)
    say("")
    say(f"own paper ranked 1st : {np.mean(ranks == 1):.1%}   (chance = {1 / n:.1%})")
    say(f"own paper in top 5   : {np.mean(ranks <= 5):.1%}   (chance = {5 / n:.1%})")
    say(f"own paper in top 20  : {np.mean(ranks <= 20):.1%}   (chance = {20 / n:.1%})")
    say(f"median rank of own paper: {int(np.median(ranks))} of {n}")
    say("")
    say("most common row offsets of the best match (best-matching row index - own row index):")
    for off, c in offsets.most_common(8):
        say(f"  offset {off:+d}: {c} rows")
    say("")
    say("examples where the cited papers match a different seed paper better:")
    for e, other in examples:
        say(f"- row {e['row_index']}: {e['title'][:90]}")
        say(f"    its first cited paper: {truncate_words(corpus.text(e['gold_doc_ids'][0]), 22)}")
        say(f"    best-matching seed   : row {other['row_index']}: {other['title'][:90]}")
    say("")
    say("Reading: aligned data -> own paper 1st for a large share of rows (typically well above 30%);")
    say("misaligned data -> close to chance, and a dominant offset (e.g. +1) means the rows are shifted.")
    with open(REPORT, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"\nReport saved to {REPORT}")


if __name__ == "__main__":
    main()
