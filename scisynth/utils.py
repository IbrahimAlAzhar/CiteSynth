"""Small shared helpers: IO, hashing, logging, text utilities, locking."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import time
from contextlib import contextmanager
from typing import Iterable, Iterator

import numpy as np

LOG = logging.getLogger("scisynth")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is optional for CPU-only stages
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


# --------------------------------------------------------------------------- IO
def read_jsonl(path: str) -> list[dict]:
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def iter_jsonl(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def append_jsonl(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def stable_hash(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:n]


def hash_split(key: str, test_frac: float, dev_frac: float) -> str:
    """Deterministic split assignment from a string key."""
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < test_frac:
        return "test"
    if h < test_frac + dev_frac:
        return "dev"
    return "train"


# ------------------------------------------------------------------------- text
_WS = re.compile(r"\s+")
LIGATURES = {"↵": "ff", "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    for k, v in LIGATURES.items():
        text = text.replace(k, v)
    text = text.replace("\\\\n", "\n").replace("\\n", "\n")
    text = _WS.sub(" ", text)
    return text.strip()


def n_words(text: str) -> int:
    return len(text.split()) if text else 0


def truncate_words(text: str, n: int) -> str:
    words = (text or "").split()
    if len(words) <= n:
        return text or ""
    return " ".join(words[:n]) + " ..."


def first_sentence(text: str, max_words: int = 25) -> str:
    text = (text or "").strip()
    m = re.search(r"(?<=[.!?])\s", text)
    s = text[: m.start()] if m else text
    return truncate_words(s, max_words)


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[(\"'])")


def split_sentences(text: str) -> list[str]:
    """Sentence splitter that keeps trailing citations ("... [D3]. Next") attached."""
    text = (text or "").strip()
    if not text:
        return []
    parts = []
    for para in re.split(r"\n\s*\n|\n", text):
        para = para.strip()
        if not para:
            continue
        parts.extend(p.strip() for p in _SENT_SPLIT.split(para) if p.strip())
    # merge sentences that consist only of a citation, e.g. "[D3]."
    merged: list[str] = []
    for p in parts:
        if merged and re.fullmatch(r"(\[[^\]]+\]\s*[.,;]?\s*)+", p):
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    return merged


# Citation markers produced by our systems: [D12], [D12, D3], [D12; D3]
CIT_BLOCK = re.compile(r"\[((?:D\d+)(?:\s*[,;]\s*D\d+)*)\]")
DOC_ID = re.compile(r"D\d+")


def extract_citations(text: str) -> list[str]:
    ids: list[str] = []
    for block in CIT_BLOCK.findall(text or ""):
        ids.extend(DOC_ID.findall(block))
    return ids


def unique(seq: Iterable) -> list:
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def strip_think(text: str) -> str:
    """Remove (possibly empty) <think>...</think> blocks emitted by reasoning models."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    if "<think>" in text and "</think>" not in text:  # truncated thinking
        text = text.split("<think>")[0]
    return text.strip()


def norm_key(text: str, n: int = 30) -> str:
    """Normalised prefix key used to detect near-duplicate documents."""
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    return " ".join(toks[:n])


# ----------------------------------------------------------------------- locking
@contextmanager
def dir_lock(path: str, timeout_s: int = 6 * 3600, stale_s: int = 12 * 3600, poll_s: int = 20):
    """Atomic mkdir-based lock. Works on Lustre/NFS where flock may not.

    Used so that three PBS jobs (one per model) launched together build the
    shared dataset/benchmark only once.
    """
    lock = path + ".lockdir"
    start = time.time()
    while True:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(lock)), exist_ok=True)
            os.mkdir(lock)
            with open(os.path.join(lock, "owner"), "w") as f:
                f.write(f"{os.uname().nodename}:{os.getpid()}:{time.time()}\n")
            break
        except FileExistsError:
            try:
                age = time.time() - os.path.getmtime(lock)
            except FileNotFoundError:
                continue
            if age > stale_s:
                LOG.warning("Removing stale lock %s (age %.0fs)", lock, age)
                _rm_lock(lock)
                continue
            if time.time() - start > timeout_s:
                raise TimeoutError(f"Timed out waiting for lock {lock}")
            LOG.info("Waiting for lock %s held by another job ...", lock)
            time.sleep(poll_s)
    try:
        yield
    finally:
        _rm_lock(lock)


def _rm_lock(lock: str) -> None:
    try:
        for fn in os.listdir(lock):
            os.remove(os.path.join(lock, fn))
        os.rmdir(lock)
    except FileNotFoundError:
        pass


def mean(xs) -> float:
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float("nan")


def bootstrap_ci(xs, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    xs = np.asarray([x for x in xs if x is not None and not np.isnan(x)], dtype=float)
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(xs), size=(n_boot, len(xs)))
    means = xs[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
