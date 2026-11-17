"""Dataset preparation.

Input CSV (one row per seed paper):
  * ``pdf_text``            str(dict) with keys abstractText, title, year, sections[{heading,text}],
                            references[{title, authors, year}], id
  * ``cited_in_abs_intro``  str(list[list[str]]) with the abstracts (and/or intros) of the cited papers

Outputs (in --data_dir):
  * examples.jsonl  one record per usable seed paper (title, abstract, intro, related_work, gold docs, split)
  * corpus.jsonl    deduplicated retrieval corpus (gold cited papers of all rows + distractors)
  * prepare_meta.json
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import random
import re
import sys
from collections import Counter

import pandas as pd

from .utils import (
    LOG,
    clean_text,
    dir_lock,
    first_sentence,
    hash_split,
    n_words,
    norm_key,
    read_jsonl,
    stable_hash,
    truncate_words,
    write_json,
    write_jsonl,
)

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

INTRO_RE = re.compile(r"\bintro", re.I)
RW_RE = re.compile(
    r"related\s+work|prior\s+work|previous\s+work|background|literature\s+review|related\s+literature|"
    r"state\s+of\s+the\s+art|related\s+research|related\s+studies",
    re.I,
)
SEC_NUM = re.compile(r"^\s*((?:\d+|[IVX]+)(?:\.\d+)*)\.?\s")


# ----------------------------------------------------------------------------- parsing
def parse_literal(value):
    """Parse a python-literal / JSON string stored in a CSV cell."""
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    if len(s) > 1 and s[0] == '"' and s[-1] == '"' and s[1:2] in "{[":
        s = s[1:-1]
    for fn in (ast.literal_eval, json.loads):
        try:
            return fn(s)
        except Exception:
            continue
    return None


def _sec_number(heading: str) -> str | None:
    m = SEC_NUM.match(heading or "")
    return m.group(1) if m else None


def _collect_sections(sections: list[dict], pattern: re.Pattern) -> str:
    """Concatenate every section whose heading matches ``pattern`` plus its numbered
    sub-sections (e.g. "2 Related Work" + "2.1 ..." + "2.2 ...")."""
    texts: list[str] = []
    i, n = 0, len(sections)
    while i < n:
        head = str(sections[i].get("heading") or "")
        if not pattern.search(head):
            i += 1
            continue
        t = clean_text(sections[i].get("text") or "")
        if t:
            texts.append(t)
        num = _sec_number(head)
        top = num.split(".")[0] if num else None
        j = i + 1
        while top and j < n:
            num2 = _sec_number(str(sections[j].get("heading") or ""))
            if num2 and "." in num2 and num2.split(".")[0] == top:
                t2 = clean_text(sections[j].get("text") or "")
                if t2:
                    texts.append(t2)
                j += 1
            else:
                break
        i = j
    return "\n".join(texts).strip()


def parse_paper(obj: dict, intro_max_words: int = 800) -> dict | None:
    if not isinstance(obj, dict):
        return None
    sections = obj.get("sections") or []
    if not isinstance(sections, list):
        sections = []
    abstract = clean_text(obj.get("abstractText") or obj.get("abstract") or "")
    title = clean_text(obj.get("title") or "")
    intro = _collect_sections(sections, INTRO_RE)
    if not intro:
        for sec in sections:
            t = clean_text(sec.get("text") or "")
            if t:
                intro = t
                break
    related = _collect_sections(sections, RW_RE)
    refs = obj.get("references") or []
    ref_titles = [clean_text(r.get("title") or "") for r in refs if isinstance(r, dict)]
    year = obj.get("year")
    try:
        year = int(year) if year is not None else None
    except Exception:
        year = None
    return {
        "paper_id": str(obj.get("id") or stable_hash(title + abstract)),
        "title": title,
        "abstract": abstract,
        "intro": truncate_words(intro, intro_max_words),
        "related_work": related,
        "year": year,
        "ref_titles": [t for t in ref_titles if t],
        "section_headings": [str(s.get("heading") or "") for s in sections],
    }


def _strings(x) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, dict):
        out = []
        for v in x.values():
            out.extend(_strings(v))
        return out
    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            out.extend(_strings(v))
        return out
    return []


def parse_cited(value, layout: str = "auto", min_words: int = 20) -> list[str]:
    """Return one text per cited paper.

    layout:
      flat       every string is one cited paper
      per_paper  every top-level element is one paper (its strings are joined, e.g. [abstract, intro])
      auto       per_paper if there are several inner lists each holding <=3 strings, else flat
    """
    obj = parse_literal(value)
    if obj is None:
        return [clean_text(value)] if isinstance(value, str) and n_words(value) >= min_words else []
    if isinstance(obj, str):
        groups = [[obj]]
    elif isinstance(obj, dict):
        groups = [_strings(obj)]
    else:
        items = list(obj)
        use_per_paper = layout == "per_paper" or (
            layout == "auto"
            and len(items) > 1
            and all(isinstance(x, (list, tuple, dict)) and 1 <= len(x) <= 3 for x in items)
        )
        if use_per_paper:
            groups = [_strings(x) for x in items]
        else:
            groups = [[s] for s in _strings(items)]
    docs = []
    for g in groups:
        text = clean_text(" ".join(s for s in g if isinstance(s, str)))
        if n_words(text) >= min_words:
            docs.append(text)
    return docs


# ------------------------------------------------------------------------------ loading
def read_rows(path: str, usecols: list[str], nrows: int | None = None, skiprows: int = 0) -> pd.DataFrame:
    kw = dict(usecols=lambda c: c in set(usecols))
    if nrows is not None:
        kw["nrows"] = nrows
    if skiprows:
        kw["skiprows"] = range(1, skiprows + 1)
    if path.endswith(".parquet"):
        df = pd.read_parquet(path, columns=[c for c in usecols])
        return df.iloc[skiprows: (skiprows + nrows) if nrows else None]
    if path.endswith(".jsonl"):
        df = pd.read_json(path, lines=True)
        return df[[c for c in usecols if c in df.columns]].iloc[skiprows: (skiprows + nrows) if nrows else None]
    try:
        return pd.read_csv(path, **kw)
    except Exception as e:  # malformed quoting -> python engine
        LOG.warning("C engine failed (%s); retrying with python engine", e)
        return pd.read_csv(path, engine="python", **kw)


def inspect(args) -> None:
    """Print columns and the parsed structure of a few rows (run this first!)."""
    if args.input_path.endswith((".jsonl", ".parquet")):
        df = read_rows(args.input_path, [args.text_col, args.cited_col], nrows=3)
    else:
        df = pd.read_csv(args.input_path, nrows=3)
    print("Columns:", list(df.columns))
    print(df.dtypes)
    for i in range(min(3, len(df))):
        print("=" * 100, "\nROW", i)
        paper = parse_paper(parse_literal(df.iloc[i][args.text_col]))
        if paper is None:
            print("  !! could not parse", args.text_col)
        else:
            print("  title:", paper["title"])
            print("  year:", paper["year"], "| #refs:", len(paper["ref_titles"]))
            print("  headings:", paper["section_headings"][:15])
            print("  abstract:", truncate_words(paper["abstract"], 40))
            print("  intro words:", n_words(paper["intro"]), "| related-work words:", n_words(paper["related_work"]))
        raw = parse_literal(df.iloc[i][args.cited_col])
        if isinstance(raw, list):
            print(f"  {args.cited_col}: outer list len={len(raw)}; inner lens={[len(x) if isinstance(x, (list, tuple)) else 'str' for x in raw[:10]]}")
            flat = _strings(raw)
            for j, s in enumerate(flat[:6]):
                print(f"    [{j}] {truncate_words(clean_text(s), 25)}")
        for lay in ("flat", "per_paper", "auto"):
            print(f"  layout={lay:9s} -> {len(parse_cited(df.iloc[i][args.cited_col], lay))} cited docs")


# ------------------------------------------------------------------------------ prepare
def prepare(args) -> None:
    os.makedirs(args.data_dir, exist_ok=True)
    done_flag = os.path.join(args.data_dir, "prepare.done")
    with dir_lock(os.path.join(args.data_dir, "prepare")):
        if os.path.exists(done_flag) and not args.overwrite:
            LOG.info("Data already prepared in %s (use --overwrite to rebuild)", args.data_dir)
            return
        _prepare(args)
        with open(done_flag, "w") as f:
            f.write("ok\n")


def _prepare(args) -> None:
    rng = random.Random(args.seed)
    cols = [args.text_col, args.cited_col] + ([args.id_col] if args.id_col else []) + list(args.distractor_cols or [])
    if args.sample_rows:
        df = read_rows(args.input_path, cols)
        df = df.sample(n=min(args.n_rows or len(df), len(df)), random_state=args.seed)
        rest = None
    else:
        df = read_rows(args.input_path, cols, nrows=args.n_rows, skiprows=args.row_offset)
        rest = None
        if args.distractor_rows > 0:
            rest = read_rows(args.input_path, [args.cited_col], nrows=args.distractor_rows,
                             skiprows=args.row_offset + (args.n_rows or len(df)))
    for col in (args.text_col, args.cited_col):
        if col not in df.columns:
            raise ValueError(f"column '{col}' not found in {args.input_path}; available: {list(df.columns)}")
    LOG.info("Loaded %d rows from %s", len(df), args.input_path)

    # ---- corpus with dedup
    key2id: dict[str, str] = {}
    corpus: list[dict] = []

    def add_doc(text: str, src: str) -> str:
        key = norm_key(text)
        if key in key2id:
            return key2id[key]
        doc_id = f"D{len(corpus) + 1}"
        key2id[key] = doc_id
        corpus.append({"doc_id": doc_id, "title": first_sentence(text, 20), "text": text,
                       "n_words": n_words(text), "src": src})
        return doc_id

    examples, n_bad, n_few = [], 0, 0
    for i, row in enumerate(df.to_dict("records")):
        paper = parse_paper(parse_literal(row.get(args.text_col)), args.intro_max_words)
        if paper is None or n_words(paper["abstract"]) < 20:
            n_bad += 1
            continue
        cited = parse_cited(row.get(args.cited_col), args.cited_layout, args.min_doc_words)
        cited = cited[: args.max_cited]
        if len(cited) < args.min_cited:
            n_few += 1
            continue
        gold = []
        for t in cited:
            d = add_doc(t, "gold")
            if d not in gold:
                gold.append(d)
        for dc in args.distractor_cols or []:
            for t in parse_cited(row.get(dc), "flat", args.min_doc_words):
                add_doc(t, f"col:{dc}")
        pid = str(row.get(args.id_col)) if args.id_col else paper["paper_id"]
        ex_id = f"P{stable_hash(pid + paper['title'], 10)}"
        split = hash_split(ex_id, args.test_frac, args.dev_frac)
        if args.test_year_ge and paper["year"] and paper["year"] >= args.test_year_ge:
            split = "test"  # temporal hold-out: recent papers are least memorised
        examples.append({
            "ex_id": ex_id, "row_index": int(args.row_offset + i), "paper_id": pid, "split": split,
            "title": paper["title"], "abstract": paper["abstract"], "intro": paper["intro"],
            "related_work": paper["related_work"], "year": paper["year"],
            "gold_doc_ids": gold, "n_ref_titles": len(paper["ref_titles"]),
        })

    if rest is not None:
        for v in rest[args.cited_col].tolist():
            for t in parse_cited(v, args.cited_layout, args.min_doc_words)[: args.max_cited]:
                add_doc(t, "distractor")

    if args.extra_corpus:
        n0 = len(corpus)
        for path in args.extra_corpus:
            for rec in _iter_extra(path, args.extra_text_field):
                add_doc(rec, f"extra:{os.path.basename(path)}")
        LOG.info("Added %d docs from extra corpora", len(corpus) - n0)

    # ---- exclude each paper's own abstract if it appears in the corpus (cited by another row)
    for ex in examples:
        own = key2id.get(norm_key(ex["abstract"]))
        ex["exclude_doc_ids"] = [own] if own else []
        if own in ex["gold_doc_ids"]:
            ex["gold_doc_ids"].remove(own)

    rng.shuffle(examples)
    examples.sort(key=lambda e: ("train", "dev", "test").index(e["split"]))
    write_jsonl(os.path.join(args.data_dir, "examples.jsonl"), examples)
    write_jsonl(os.path.join(args.data_dir, "corpus.jsonl"), corpus)
    split_counts = Counter(e["split"] for e in examples)
    meta = {
        "input_path": args.input_path, "text_col": args.text_col, "cited_col": args.cited_col,
        "n_rows_read": len(df), "n_examples": len(examples), "n_unparsable": n_bad, "n_too_few_cited": n_few,
        "corpus_size": len(corpus), "splits": dict(split_counts),
        "mean_gold": sum(len(e["gold_doc_ids"]) for e in examples) / max(1, len(examples)),
        "with_related_work_section": sum(1 for e in examples if n_words(e["related_work"]) >= 80),
        "cited_layout": args.cited_layout,
    }
    write_json(os.path.join(args.data_dir, "prepare_meta.json"), meta)
    LOG.info("Prepared: %s", json.dumps(meta))


def _iter_extra(path: str, field: str):
    if path.endswith(".jsonl"):
        for rec in read_jsonl(path):
            t = rec.get(field) or " ".join(str(rec.get(k, "")) for k in ("title", "abstract"))
            t = clean_text(t)
            if n_words(t) >= 20:
                yield t
    else:
        df = pd.read_csv(path)
        for t in df[field].astype(str).tolist():
            t = clean_text(t)
            if n_words(t) >= 20:
                yield t


class Corpus:
    """In-memory view of corpus.jsonl."""

    def __init__(self, data_dir: str):
        self.docs = {d["doc_id"]: d for d in read_jsonl(os.path.join(data_dir, "corpus.jsonl"))}

    def __len__(self):
        return len(self.docs)

    def __contains__(self, doc_id):
        return doc_id in self.docs

    def text(self, doc_id: str, max_words: int | None = None) -> str:
        d = self.docs.get(doc_id)
        if d is None:
            return ""
        return truncate_words(d["text"], max_words) if max_words else d["text"]

    def title(self, doc_id: str) -> str:
        d = self.docs.get(doc_id)
        return d["title"] if d else ""

    def render(self, doc_ids: list[str], max_words: int) -> str:
        return "\n".join(f"[{d}] {self.text(d, max_words)}" for d in doc_ids if d in self.docs)


def add_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("data")
    g.add_argument("--input_path", required=False)
    g.add_argument("--text_col", default="pdf_text")
    g.add_argument("--cited_col", default="cited_in_abs_intro")
    g.add_argument("--id_col", default=None)
    g.add_argument("--n_rows", type=int, default=None, help="number of seed-paper rows to use")
    g.add_argument("--row_offset", type=int, default=0)
    g.add_argument("--sample_rows", action="store_true", help="random sample n_rows instead of head")
    g.add_argument("--cited_layout", default="auto", choices=["auto", "flat", "per_paper"])
    g.add_argument("--min_cited", type=int, default=4)
    g.add_argument("--max_cited", type=int, default=40)
    g.add_argument("--min_doc_words", type=int, default=25)
    g.add_argument("--intro_max_words", type=int, default=800)
    g.add_argument("--distractor_rows", type=int, default=2000,
                   help="cited papers of this many extra rows are added to the corpus as distractors")
    g.add_argument("--distractor_cols", nargs="*", default=[],
                   help="extra columns whose texts are added to the corpus as non-gold docs")
    g.add_argument("--extra_corpus", nargs="*", default=[], help="jsonl/csv of extra docs (e.g. PubMed abstracts)")
    g.add_argument("--extra_text_field", default="text")
    g.add_argument("--test_frac", type=float, default=0.2)
    g.add_argument("--dev_frac", type=float, default=0.1)
    g.add_argument("--test_year_ge", type=int, default=None, help="put papers with year>=X into test")
