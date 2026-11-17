"""Create a small synthetic CSV with the same schema as df_updated_with_retrieval_upd.csv (for CPU smoke tests).

Usage: python tests/make_synthetic.py OUT.csv [SAMPLE_ROW.txt]
"""
import ast
import random
import sys

import pandas as pd

TOPICS = {
    "quantum error correction": ["surface", "code", "logical", "qubit", "decoder", "threshold", "syndrome", "lattice", "fault-tolerant", "superconducting"],
    "protein structure prediction": ["protein", "folding", "residue", "contact", "alignment", "coevolution", "backbone", "structure", "accuracy", "template"],
    "battery cathode materials": ["cathode", "lithium", "capacity", "cycling", "voltage", "oxide", "doping", "stability", "electrolyte", "nickel"],
    "single-cell rna sequencing": ["single-cell", "transcriptome", "clustering", "batch", "expression", "marker", "embedding", "cell", "atlas", "normalization"],
    "perovskite solar cells": ["perovskite", "efficiency", "passivation", "defect", "film", "stability", "halide", "bandgap", "layer", "moisture"],
    "gravitational wave detection": ["gravitational", "waveform", "detector", "merger", "noise", "template", "signal", "interferometer", "binary", "inference"],
}
VERBS = ["improves", "reduces", "characterizes", "outperforms", "fails under", "depends on", "is limited by", "scales with"]
FILLER = ["we", "show", "that", "the", "proposed", "approach", "results", "experiments", "demonstrate", "across", "several", "settings", "compared", "baseline"]


def abstract(rng, words, n=55):
    out = []
    while len(out) < n:
        out += [rng.choice(words), rng.choice(VERBS), rng.choice(words), rng.choice(FILLER), rng.choice(FILLER) + "."]
    s = " ".join(out[:n])
    return s[0].upper() + s[1:] + "."


def main(out_path, sample_path=None, n_rows=60, seed=0):
    rng = random.Random(seed)
    pools = {t: [abstract(rng, w) for _ in range(30)] for t, w in TOPICS.items()}
    rows = []
    for i in range(n_rows):
        topic = rng.choice(list(TOPICS))
        words = TOPICS[topic]
        cited = rng.sample(pools[topic], rng.randint(6, 12))
        title = f"A study of {rng.choice(words)} {rng.choice(words)} in {topic}"
        paper = {
            "abstractText": abstract(rng, words, 70), "title": title, "year": rng.choice([2019, 2021, 2023, 2024, 2025]),
            "authors": [{"name": "A. Author", "affiliations": []}], "id": f"SP:{i:05d}",
            "references": [{"title": f"Reference {j}", "authors": ["X"], "year": 2020} for j in range(len(cited))],
            "sections": [
                {"heading": "1 Introduction", "text": abstract(rng, words, 150)},
                {"heading": "2 Related Work", "text": abstract(rng, words, 60)},
                {"heading": "2.1 Prior methods", "text": abstract(rng, words, 80)},
                {"heading": "3 Method", "text": abstract(rng, words, 120)},
            ],
        }
        layout = rng.choice(["flat", "pairs"])
        cited_val = [cited] if layout == "flat" else [[c, abstract(rng, words, 30)] for c in cited]
        rows.append({"pdf_text": str(paper), "cited_in_abs_intro": str(cited_val), "other_col": i})
    if sample_path:
        raw = open(sample_path, encoding="utf-8").read().strip()
        if raw.startswith('"') and raw.endswith('"'):
            raw = raw[1:-1]
        obj = ast.literal_eval(raw)
        cited = [f"{r['title']} . " + abstract(rng, TOPICS["single-cell rna sequencing"], 40) for r in obj["references"]]
        rows.insert(0, {"pdf_text": str(obj), "cited_in_abs_intro": str([cited]), "other_col": -1})
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("wrote", out_path, len(rows), "rows")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
