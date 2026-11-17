"""Retrieval over the prepared corpus: BM25 (numpy, no extra deps), optional dense
(any HF encoder, e.g. all-MiniLM-L6-v2 / gte_small), and hybrid reciprocal-rank fusion."""
from __future__ import annotations

import os
import pickle
import re
import time

import numpy as np

from .data import Corpus
from .utils import LOG, dir_lock, truncate_words, write_json

STOP = set(
    """a an the of and or in on for to with by from as at is are was were be been being this that these those it
    its we our their they which who whom whose what when where how why can could may might will would shall should
    do does did not no nor than then so such into over under about between among within without via using based use
    used also both each other more most less least very i you he she his her them there here than only any all some
    one two three new study paper approach method methods results result show shows shown propose proposed present
    however while thus therefore""".split()
)
TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    out = []
    for t in TOKEN.findall((text or "").lower()):
        if len(t) < 2 or t in STOP:
            continue
        if len(t) > 4 and t.endswith("ies"):
            t = t[:-3] + "y"
        elif len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


class BM25:
    def __init__(self, k1: float = 0.9, b: float = 0.4):
        self.k1, self.b = k1, b
        self.vocab: dict[str, int] = {}
        self.offsets = self.doc_idx = self.weights = None
        self.n_docs = 0

    def build(self, texts: list[str]) -> "BM25":
        t0 = time.time()
        term_ids, doc_ids, tfs, lens = [], [], [], []
        for di, text in enumerate(texts):
            toks = tokenize(text)
            lens.append(len(toks))
            counts: dict[int, int] = {}
            for tok in toks:
                tid = self.vocab.setdefault(tok, len(self.vocab))
                counts[tid] = counts.get(tid, 0) + 1
            for tid, c in counts.items():
                term_ids.append(tid)
                doc_ids.append(di)
                tfs.append(c)
        self.n_docs = len(texts)
        term_ids = np.asarray(term_ids, dtype=np.int64)
        doc_ids = np.asarray(doc_ids, dtype=np.int32)
        tfs = np.asarray(tfs, dtype=np.float32)
        lens = np.asarray(lens, dtype=np.float32)
        order = np.argsort(term_ids, kind="stable")
        term_ids, doc_ids, tfs = term_ids[order], doc_ids[order], tfs[order]
        dfreq = np.bincount(term_ids, minlength=len(self.vocab)).astype(np.float32)
        idf = np.log(1.0 + (self.n_docs - dfreq + 0.5) / (dfreq + 0.5))
        avgdl = float(lens.mean()) if len(lens) else 1.0
        dl = lens[doc_ids]
        self.weights = (idf[term_ids] * tfs * (self.k1 + 1) / (tfs + self.k1 * (1 - self.b + self.b * dl / avgdl))).astype(np.float32)
        self.doc_idx = doc_ids
        self.offsets = np.zeros(len(self.vocab) + 1, dtype=np.int64)
        np.cumsum(dfreq.astype(np.int64), out=self.offsets[1:])
        LOG.info("BM25 built: %d docs, %d terms, %d postings in %.1fs", self.n_docs, len(self.vocab), len(doc_ids), time.time() - t0)
        return self

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(self.n_docs, dtype=np.float32)
        for tok in set(tokenize(query)):
            tid = self.vocab.get(tok)
            if tid is None:
                continue
            a, b = self.offsets[tid], self.offsets[tid + 1]
            s[self.doc_idx[a:b]] += self.weights[a:b]
        return s

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"k1": self.k1, "b": self.b, "vocab": self.vocab, "offsets": self.offsets,
                         "doc_idx": self.doc_idx, "weights": self.weights, "n_docs": self.n_docs}, f, protocol=4)

    @classmethod
    def load(cls, path: str) -> "BM25":
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls(d["k1"], d["b"])
        obj.vocab, obj.offsets, obj.doc_idx, obj.weights, obj.n_docs = d["vocab"], d["offsets"], d["doc_idx"], d["weights"], d["n_docs"]
        return obj


class DenseEncoder:
    """Mean-pooled sentence embeddings from any local HF encoder directory."""

    def __init__(self, model_path: str, device: str | None = None, max_len: int = 256, batch_size: int = 128):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device).eval()
        if self.device == "cuda":
            self.model = self.model.half()
        self.max_len, self.bs = max_len, batch_size

    def encode(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), self.bs):
                batch = self.tok(texts[i: i + self.bs], padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
                h = self.model(**batch).last_hidden_state
                m = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
                e = (h * m).sum(1) / m.sum(1).clamp(min=1)
                e = torch.nn.functional.normalize(e.float(), dim=-1)
                out.append(e.cpu().numpy())
        return np.concatenate(out, 0) if out else np.zeros((0, 1), dtype=np.float32)


def build_index(args) -> None:
    index_dir = os.path.join(args.data_dir, "index")
    os.makedirs(index_dir, exist_ok=True)
    with dir_lock(os.path.join(args.data_dir, "index_build")):
        done = os.path.join(index_dir, "index.done")
        if os.path.exists(done) and not args.overwrite:
            LOG.info("Index exists in %s", index_dir)
            return
        corpus = Corpus(args.data_dir)
        ids = list(corpus.docs.keys())
        texts = [truncate_words(corpus.docs[i]["text"], args.index_max_words) for i in ids]
        BM25().build(texts).save(os.path.join(index_dir, "bm25.pkl"))
        np.save(os.path.join(index_dir, "doc_ids.npy"), np.asarray(ids))
        meta = {"n_docs": len(ids), "dense": False}
        if args.embed_model:
            t0 = time.time()
            enc = DenseEncoder(args.embed_model)
            emb = enc.encode(texts).astype(np.float16)
            np.save(os.path.join(index_dir, "dense.npy"), emb)
            meta.update({"dense": True, "embed_model": args.embed_model, "dim": int(emb.shape[1])})
            LOG.info("Dense index: %s in %.1fs", emb.shape, time.time() - t0)
        write_json(os.path.join(index_dir, "meta.json"), meta)
        open(done, "w").write("ok\n")


class Retriever:
    """search(query) -> list of doc_ids. mode: bm25 | dense | hybrid (RRF)."""

    def __init__(self, data_dir: str, mode: str = "hybrid", embed_model: str | None = None, rrf_k: int = 60, cand: int = 100):
        index_dir = os.path.join(data_dir, "index")
        self.ids = np.load(os.path.join(index_dir, "doc_ids.npy"), allow_pickle=True)
        self.bm25 = BM25.load(os.path.join(index_dir, "bm25.pkl"))
        self.dense = None
        dense_path = os.path.join(index_dir, "dense.npy")
        if mode in ("dense", "hybrid") and os.path.exists(dense_path) and embed_model:
            self.dense = np.load(dense_path).astype(np.float32)
            # queries are encoded on CPU by default: keeps the GPU (and CUDA init) free for vLLM / the learner
            self.encoder = DenseEncoder(embed_model, device=None if os.environ.get("SCISYNTH_DENSE_GPU") else "cpu")
        elif mode in ("dense", "hybrid"):
            LOG.warning("Dense index/encoder unavailable -> falling back to BM25")
            mode = "bm25"
        self.mode, self.rrf_k, self.cand = mode, rrf_k, cand
        self.id2idx = {d: i for i, d in enumerate(self.ids.tolist())}

    def _top(self, scores: np.ndarray, k: int, exclude_idx: list[int]) -> np.ndarray:
        if exclude_idx:
            scores = scores.copy()
            scores[exclude_idx] = -np.inf
        k = min(k, len(scores))
        part = np.argpartition(-scores, k - 1)[:k]
        part = part[np.argsort(-scores[part])]
        return part[np.isfinite(scores[part]) & (scores[part] > -np.inf)]

    def search_batch(self, queries: list[str], k: int = 5, excludes: list[list[str]] | None = None) -> list[list[str]]:
        excludes = excludes or [[] for _ in queries]
        qemb = self.encoder.encode(queries) if (self.dense is not None and queries) else None
        out = []
        for qi, q in enumerate(queries):
            ex_idx = [self.id2idx[d] for d in excludes[qi] if d in self.id2idx]
            if self.mode == "bm25" or self.dense is None:
                top = self._top(self.bm25.scores(q), k, ex_idx)
                out.append([str(self.ids[i]) for i in top])
                continue
            d_scores = self.dense @ qemb[qi]
            if self.mode == "dense":
                top = self._top(d_scores, k, ex_idx)
                out.append([str(self.ids[i]) for i in top])
                continue
            fused: dict[int, float] = {}
            for ranked in (self._top(self.bm25.scores(q), self.cand, ex_idx), self._top(d_scores, self.cand, ex_idx)):
                for r, i in enumerate(ranked):
                    fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (self.rrf_k + r + 1)
            best = sorted(fused.items(), key=lambda x: -x[1])[:k]
            out.append([str(self.ids[i]) for i, _ in best])
        return out

    def search(self, query: str, k: int = 5, exclude: list[str] | None = None) -> list[str]:
        return self.search_batch([query], k, [exclude or []])[0]
