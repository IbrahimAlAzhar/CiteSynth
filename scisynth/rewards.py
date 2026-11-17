"""Rewards and metrics.

The same Scorer produces (a) RL rewards during GRPO and (b) evaluation metrics. All quantities are grounded in
verifiable artefacts (gold cited papers, key points extracted from those papers, the documents the policy actually
saw); the LLM judge (gpt-4o-mini by default) only performs local checks (claim-vs-document support, key-point
coverage) except in the `outcome_judge` ablation, which uses a holistic LLM-as-judge score as the reward.

Reward modes (--reward_mode):
  outcome        one scalar per trajectory from the final text: coverage + support + citation F1 + format
  outcome_judge  one scalar per trajectory: holistic LLM-judge score (classic "LLM-as-a-judge" outcome RM)
  process_sum    outcome + mean of per-turn process rewards, still ONE scalar (denser signal, no credit routing)
  process_turn   per-turn process rewards routed to the turn that earned them (turn-level credit assignment)

Version 2: --recall_bonus w adds w * gold_found (share of gold papers the agent retrieved; deterministic, no judge call)
to the TRAINING reward only (purpose="train"). Evaluation metrics, including outcome_verifiable, are unchanged.

Per-turn process rewards:
  search turn  gold-recall gain of the newly retrieved documents (+ small novelty bonus, empty-search penalty)
  read turn    support of the Reader's notes by their documents + precision of selected documents
  synth turn   local faithfulness: claim support + citation validity + format
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from . import prompts as P
from .judge import JudgeRequest
from .systems import RETRIEVAL_SYSTEMS, Episode
from .utils import extract_citations, n_words, split_sentences, truncate_words, unique

SUP3 = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}
COV3 = {"yes": 1.0, "partial": 0.5, "no": 0.0}


@dataclass
class RewardConfig:
    reward_mode: str = "process_turn"
    w_cov: float = 0.4
    w_sup: float = 0.3
    w_cite: float = 0.2
    w_fmt: float = 0.1
    w_ground_noret: float = 0.4      # no-retrieval policies: coverage 0.5 + grounding 0.4 + format 0.1
    search_novel_w: float = 0.1
    search_empty_pen: float = 0.2
    read_sup_w: float = 0.6
    synth_sup_w: float = 0.6
    synth_valid_w: float = 0.2
    process_beta: float = 0.5
    recall_bonus: float = 0.0        # v2: training-only bonus  w * pool_recall  (retrieval is the bottleneck)
    min_words: int = 250
    max_words: int = 500
    max_claims: int = 25
    judge_doc_words: int = 180
    judge_max_docs: int = 15
    ground_doc_words: int = 110
    ground_max_docs: int = 25


def length_score(words: int, lo: int, hi: int) -> float:
    if words <= 0:
        return 0.0
    if words < lo:
        return words / lo
    if words > hi:
        return max(0.0, 1.0 - (words - hi) / hi)
    return 1.0


def _claims_with_cites(answer: str, max_claims: int) -> list[tuple[str, list[str]]]:
    out = []
    for s in split_sentences(answer):
        cites = unique(extract_citations(s))
        if cites:
            out.append((s, cites))
    return out[:max_claims]


def _strip_cites(s: str) -> str:
    import re

    return re.sub(r"\s*\[(?:D\d+)(?:\s*[,;]\s*D\d+)*\]", "", s).strip()


class Scorer:
    def __init__(self, judge, corpus, cfg: RewardConfig):
        self.judge, self.corpus, self.cfg = judge, corpus, cfg

    # ----------------------------------------------------------- deterministic part
    def _basic(self, ep: Episode) -> dict:
        cfg, it = self.cfg, ep.item
        gold = set(it["gold_doc_ids"])
        cited = unique(extract_citations(ep.answer))
        pool = set(ep.pool)
        uses_ret = ep.system in RETRIEVAL_SYSTEMS
        valid = [d for d in cited if d in pool]
        words = n_words(ep.answer)
        fmt = length_score(words, cfg.min_words, cfg.max_words)
        if uses_ret and not cited:
            fmt *= 0.5
        hits = len([d for d in cited if d in gold])
        prec = hits / len(cited) if cited else 0.0
        rec = hits / len(gold) if gold else 0.0
        sents = split_sentences(ep.answer)
        return {
            "words": words, "fmt": fmt, "n_cited": len(cited), "cite_valid": (len(valid) / len(cited)) if cited else (0.0 if uses_ret else 1.0),
            "cite_prec": prec, "cite_rec": rec, "cite_f1": (2 * prec * rec / (prec + rec)) if prec + rec > 0 else 0.0,
            "pool_size": len(ep.pool), "pool_recall": (len(pool & gold) / len(gold)) if gold else 0.0,
            "n_search_turns": sum(1 for t in ep.turns if t.phase == "search" and t.info.get("queries")),
            "uncited_rate": (sum(1 for s in sents if not extract_citations(s)) / len(sents)) if (sents and uses_ret) else None,
        }

    # ------------------------------------------------------------ judge requests
    def _req_support(self, ep: Episode):
        claims = _claims_with_cites(ep.answer, self.cfg.max_claims)
        pool = set(ep.pool)
        ids = unique([d for _, c in claims for d in c if d in pool])[: self.cfg.judge_max_docs]
        if not claims:
            return None, claims
        idset = set(ids)
        lines = [f"{i}. {s}  (cites: {', '.join(c)})" for i, (s, c) in enumerate(claims, 1)]
        payload = {"claims": [(s, [self.corpus.text(d, self.cfg.judge_doc_words) for d in c if d in idset]) for s, c in claims]}
        return JudgeRequest("support", P.SUPPORT_SYSTEM,
                            P.SUPPORT_USER.format(docs="\n".join(f"[{d}] {self.corpus.text(d, self.cfg.judge_doc_words)}" for d in ids) or "(none)",
                                                  claims="\n".join(lines)), payload, max_tokens=1500), claims

    def _req_cover(self, ep: Episode, points: list[dict], kind: str = "cover"):
        if not points or not ep.answer.strip():
            return None
        txt = "\n".join(f"{i}. {p['text']}" for i, p in enumerate(points, 1))
        return JudgeRequest("cover", P.COVER_SYSTEM, P.COVER_USER.format(answer=truncate_words(ep.answer, 1200), points=txt),
                            {"answer": ep.answer, "points": [p["text"] for p in points]}, max_tokens=1200)

    def _req_ground(self, ep: Episode):
        sents = [_strip_cites(s) for s in split_sentences(ep.answer)][: self.cfg.max_claims]
        sents = [s for s in sents if n_words(s) >= 4]
        if not sents:
            return None, []
        ids = ep.item["gold_doc_ids"][: self.cfg.ground_max_docs]
        docs = "\n".join(f"[{d}] {self.corpus.text(d, self.cfg.ground_doc_words)}" for d in ids)
        return JudgeRequest("ground", P.GROUND_SYSTEM,
                            P.GROUND_USER.format(docs=docs, claims="\n".join(f"{i}. {s}" for i, s in enumerate(sents, 1))),
                            {"claims": sents, "docs": [self.corpus.text(d, self.cfg.ground_doc_words) for d in ids]},
                            max_tokens=1500), sents

    def _req_notes(self, ep: Episode):
        notes = ep.notes[: self.cfg.max_claims]
        if not notes:
            return None, 0
        ids = unique(d for d, _ in notes)[: self.cfg.judge_max_docs]
        idset = set(ids)
        notes = [(d, n) for d, n in notes if d in idset]
        return JudgeRequest("notes", P.NOTES_SYSTEM,
                            P.NOTES_USER.format(docs="\n".join(f"[{d}] {self.corpus.text(d, self.cfg.judge_doc_words)}" for d in ids),
                                                notes="\n".join(f"{i}. [{d}] {n}" for i, (d, n) in enumerate(notes, 1))),
                            {"claims": [(n, [self.corpus.text(d, self.cfg.judge_doc_words)]) for d, n in notes]}, max_tokens=1200), len(notes)

    def _req_holistic(self, ep: Episode):
        it = ep.item
        task = it["question"] if it["task"] == "qa" else f"Write the related-work section for: {it['title']}\n{it['abstract']}"
        rw = it["task"] == "rw"
        return JudgeRequest("holistic", P.HOLISTIC_SYSTEM,
                            P.HOLISTIC_USER.format(task=task, answer=truncate_words(ep.answer, 1200) or "(empty)",
                                                   extra=P.HOLISTIC_EXTRA_RW if rw else "", extra_key=', "positioning": s' if rw else ""),
                            {"answer": ep.answer, "rw": rw}, max_tokens=300)

    # --------------------------------------------------------------------- main
    def score(self, episodes: list[Episode], purpose: str = "train", reward_mode: str | None = None) -> None:
        """Fill ep.rewards (metrics + 'outcome') and turn.reward (process rewards)."""
        cfg = self.cfg
        mode = reward_mode or cfg.reward_mode
        evaluating = purpose == "eval"
        reqs, slots = [], []  # slots: (episode index, key, extra)

        def add(r, i, key, extra=None):
            if r is not None:
                reqs.append(r)
                slots.append((i, key, extra))

        for i, ep in enumerate(episodes):
            ep.rewards.update(self._basic(ep))
            uses_ret = ep.system in RETRIEVAL_SYSTEMS
            if uses_ret:
                r, claims = self._req_support(ep)
                add(r, i, "support", claims)
            add(self._req_cover(ep, ep.item.get("key_points", [])), i, "coverage", ep.item.get("key_points", []))
            if evaluating or not uses_ret:
                r, sents = self._req_ground(ep)
                add(r, i, "ground", sents)
            if ep.notes:  # also in outcome modes, so per-phase diagnostics are comparable across ablations
                r, n_notes = self._req_notes(ep)
                add(r, i, "notes", n_notes)
            if evaluating and ep.item["task"] == "rw" and ep.item.get("ref_points"):
                add(self._req_cover(ep, ep.item["ref_points"]), i, "ref_coverage", ep.item["ref_points"])
            if evaluating or mode == "outcome_judge":
                add(self._req_holistic(ep), i, "holistic", None)

        outs = self.judge.run(reqs) if reqs else []
        for (i, key, extra), out in zip(slots, outs):
            ep = episodes[i]
            res = (out or {}).get("results") if key != "holistic" else out
            if key == "support":
                labels = _labels(res, len(extra))
                pool = set(ep.pool)
                vals = []
                for (s, cites), lab in zip(extra, labels):
                    if not any(d in pool for d in cites):
                        lab = "unsupported"  # cites only documents the policy never saw
                    vals.append(SUP3.get(lab, 0.0))
                ep.rewards["support"] = float(np.mean(vals)) if vals else 0.0
                ep.rewards["n_claims_cited"] = len(vals)
            elif key in ("coverage", "ref_coverage"):
                labels = _labels(res, len(extra))
                w = np.array([p.get("importance", 2) for p in extra], dtype=float)
                v = np.array([COV3.get(lab, 0.0) for lab in labels], dtype=float)
                ep.rewards[key] = float((w * v).sum() / w.sum()) if w.sum() else 0.0
                ep.rewards[key + "_unweighted"] = float(v.mean()) if len(v) else 0.0
            elif key == "ground":
                labels = _labels(res, len(extra), default="not_found")
                n_fact = sum(1 for lab in labels if lab != "generic")
                ep.rewards["ground_prec"] = (sum(1 for lab in labels if lab == "supported") / n_fact) if n_fact else 0.0
                ep.rewards["contradicted_rate"] = (sum(1 for lab in labels if lab == "contradicted") / n_fact) if n_fact else 0.0
            elif key == "notes":
                labels = _labels(res, extra)
                ep.rewards["notes_support"] = float(np.mean([SUP3.get(lab, 0.0) for lab in labels])) if labels else 0.0
            elif key == "holistic":
                h = out or {}
                for k, v in h.items():
                    try:
                        ep.rewards[f"judge_{k}"] = float(v)
                    except Exception:
                        pass
        for ep in episodes:
            self._finalize(ep, mode, purpose)

    def _finalize(self, ep: Episode, mode: str, purpose: str = "train") -> None:
        cfg, r = self.cfg, ep.rewards
        r.setdefault("coverage", 0.0)
        uses_ret = ep.system in RETRIEVAL_SYSTEMS
        if uses_ret:
            r.setdefault("support", 0.0)
            verifiable = cfg.w_cov * r["coverage"] + cfg.w_sup * r["support"] + cfg.w_cite * r["cite_f1"] + cfg.w_fmt * r["fmt"]
        else:
            r.setdefault("ground_prec", 0.0)
            verifiable = 0.5 * r["coverage"] + cfg.w_ground_noret * r["ground_prec"] + (0.6 - cfg.w_ground_noret) * r["fmt"]
        r["outcome_verifiable"] = float(verifiable)
        if mode == "outcome_judge":
            r["outcome"] = float(r.get("judge_overall", 1.0) / 10.0 * (0.5 + 0.5 * r["fmt"]))
        else:
            r["outcome"] = float(verifiable)

        gold = set(ep.item["gold_doc_ids"])
        proc = []
        first_search = True
        for t in ep.turns:
            if t.phase == "search":
                qs = t.info.get("queries") or []
                if not qs:
                    t.reward = -cfg.search_empty_pen if first_search else 0.0
                else:
                    new = t.info.get("new_ids") or []
                    gain = len([d for d in new if d in gold]) / max(1, len(gold))
                    novelty = len(new) / max(1, sum(len(x) for x in t.info.get("results") or [[]]))
                    t.reward = gain + cfg.search_novel_w * novelty
                first_search = False
            elif t.phase == "read":
                sel = t.info.get("selected") or []
                prec = len([d for d in sel if d in gold]) / len(sel) if sel else 0.0
                t.reward = cfg.read_sup_w * r.get("notes_support", 0.0) + (1 - cfg.read_sup_w) * prec
            elif t.phase == "synth":
                t.reward = (cfg.synth_sup_w * r.get("support", 0.0) + cfg.synth_valid_w * r["cite_valid"]
                            + (1 - cfg.synth_sup_w - cfg.synth_valid_w) * r["fmt"])
            else:  # direct (no retrieval)
                t.reward = r["outcome"]
            proc.append(t.reward)
        nonfinal = [t.reward for t in ep.turns if t.phase in ("search", "read")]
        r["process_mean"] = float(np.mean(nonfinal)) if nonfinal else 0.0
        if mode == "process_sum":
            r["outcome"] = float(r["outcome"] + cfg.process_beta * r["process_mean"])
        if purpose == "train" and cfg.recall_bonus and uses_ret and mode != "outcome_judge":
            r["outcome"] = float(r["outcome"] + cfg.recall_bonus * r.get("pool_recall", 0.0))


def _labels(res, n: int, default: str = "unsupported") -> list[str]:
    labels = [default] * n
    if isinstance(res, list):
        for k, x in enumerate(res):
            if not isinstance(x, dict):
                continue
            try:
                idx = int(x.get("id", k + 1)) - 1
            except Exception:
                idx = k
            if 0 <= idx < n:
                labels[idx] = str(x.get("label", default)).strip().lower()
    return labels


def add_args(p) -> None:
    g = p.add_argument_group("rewards")
    for k, v in asdict(RewardConfig()).items():
        if k in ("min_words", "max_words"):
            continue
        g.add_argument(f"--{k}", type=type(v), default=v,
                       **({"choices": ["outcome", "outcome_judge", "process_sum", "process_turn"]} if k == "reward_mode" else {}))


def reward_config_from_args(args) -> RewardConfig:
    kw = {k: getattr(args, k) for k in asdict(RewardConfig()) if hasattr(args, k)}
    return RewardConfig(**kw)
