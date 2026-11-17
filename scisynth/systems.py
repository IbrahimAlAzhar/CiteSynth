"""Systems (policies) for both tasks, run as batched, step-synchronous episodes.

  zero_shot  no retrieval, one call                                   (parametric baseline)
  cot        no retrieval, outline then write                         (reasoning baseline)
  rag        one retrieval with the task as query, then write         (standard RAG)
  oracle     gold cited papers given directly (+ optional distractors / dropped gold)   (upper bound; Multi-XScience setting)
  react      one agent interleaving <search> and <answer>             (Search-R1 style single agent)
  pipeline3  Searcher -> Reader -> Synthesizer (three roles, one shared policy)         (proposed agent)

Every model call is recorded as a Turn with its prompt/completion token ids, so the same episodes are used for
evaluation, rejection-sampling SFT data and GRPO (turn-level credit assignment).
"""
from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass, field

from . import prompts as P
from .utils import extract_citations, strip_think, truncate_words, unique

SYSTEMS = ["zero_shot", "cot", "rag", "oracle", "react", "pipeline3"]
RETRIEVAL_SYSTEMS = {"rag", "oracle", "react", "pipeline3"}

SEARCH_TAG = re.compile(r"<search>(.*?)</search>", re.S | re.I)
ANSWER_TAG = re.compile(r"<answer>(.*?)(?:</answer>|$)", re.S | re.I)
NOTE_LINE = re.compile(r"^\s*[-*]?\s*\[((?:D\d+)(?:\s*[,;]\s*D\d+)*)\]\s*[:\-]?\s*(.+)$", re.M)


@dataclass
class Turn:
    phase: str                      # search | read | synth | direct
    conv: list
    text: str = ""
    prompt_token_ids: list = field(default_factory=list)
    token_ids: list = field(default_factory=list)
    finish_reason: str = ""
    info: dict = field(default_factory=dict)
    reward: float | None = None     # process (turn-level) reward
    adv: float | None = None        # advantage used by GRPO


@dataclass
class Episode:
    item: dict
    system: str
    sample_idx: int = 0
    turns: list = field(default_factory=list)
    pool: list = field(default_factory=list)       # doc ids the policy has seen, in order
    notes: list = field(default_factory=list)      # [(doc_id, note)] from the Reader
    answer: str = ""
    done: bool = False
    rewards: dict = field(default_factory=dict)

    def record(self, with_tokens: bool = False, with_conv: bool = False) -> dict:
        turns = []
        for t in self.turns:
            d = {"phase": t.phase, "text": t.text, "finish_reason": t.finish_reason, "info": t.info,
                 "reward": t.reward, "adv": t.adv, "n_prompt_tokens": len(t.prompt_token_ids), "n_tokens": len(t.token_ids)}
            if with_tokens:
                d["prompt_token_ids"], d["token_ids"] = t.prompt_token_ids, t.token_ids
            if with_conv:
                d["conv"] = t.conv
            turns.append(d)
        return {"item_id": self.item["item_id"], "task": self.item["task"], "split": self.item["split"],
                "system": self.system, "sample_idx": self.sample_idx, "answer": self.answer, "pool": self.pool,
                "cited": unique(extract_citations(self.answer)), "notes": self.notes, "turns": turns,
                "rewards": self.rewards}


@dataclass
class SysConfig:
    input_mode: str = "abs"          # rw task: abs | abs_intro
    rag_k: int = 10
    search_k: int = 5
    max_search_turns: int = 3
    queries_per_turn: int = 2
    max_pool: int = 20
    reader_max_docs: int = 15
    doc_words_reader: int = 160
    doc_words_synth: int = 150       # rag / oracle documents
    doc_words_ref: int = 60          # pipeline3 synthesizer reference snippets
    snippet_words: int = 45          # search-result snippets
    oracle_distractors: int = 0
    drop_gold_frac: float = 0.0
    min_words: int = 250
    max_words: int = 500
    tok_search: int = 96
    tok_read: int = 900
    tok_write: int = 1100
    tok_plan: int = 1500
    temperature: float = 0.0
    top_p: float = 1.0
    search_fallback: bool = True
    seed: int = 0


# --------------------------------------------------------------------------- helpers
def task_text(item: dict, cfg: SysConfig) -> str:
    if item["task"] == "qa":
        return P.QA_TASK.format(question=item["question"])
    if cfg.input_mode == "abs_intro" and item.get("intro"):
        return P.RW_TASK_INTRO.format(title=item["title"], abstract=item["abstract"], intro=truncate_words(item["intro"], 450))
    return P.RW_TASK.format(title=item["title"], abstract=item["abstract"])


def write_rules(item: dict, cfg: SysConfig) -> str:
    tmpl = P.WRITE_RULES_QA if item["task"] == "qa" else P.WRITE_RULES_RW
    return tmpl.format(min_w=cfg.min_words, max_w=cfg.max_words)


def default_query(item: dict) -> str:
    if item["task"] == "qa":
        return item["question"]
    return item["title"] + " " + truncate_words(item["abstract"], 60)


def parse_queries(text: str, max_q: int) -> list[str]:
    qs = [re.sub(r"\s+", " ", q).strip().strip('"') for q in SEARCH_TAG.findall(text or "")]
    if not qs and "<search>" in (text or "").lower():
        tail = re.split(r"<search>", text, flags=re.I)[-1]
        qs = [tail.split("\n")[0].strip()]
    return unique([q[:200] for q in qs if len(q) > 1])[:max_q]


def parse_answer(text: str) -> str:
    text = strip_think(text)
    m = ANSWER_TAG.search(text)
    if m:
        text = m.group(1)
    text = re.sub(r"<plan>.*?</plan>", "", text, flags=re.S | re.I)
    text = re.sub(r"</?(answer|plan)>|<done/?>|<search>.*?</search>", "", text, flags=re.S | re.I)
    text = re.sub(r"^\s*#+\s*.*$", "", text, flags=re.M)               # markdown headings
    text = re.sub(r"^\s*\**\s*related work\s*\**\s*:?\s*$", "", text, flags=re.M | re.I)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def parse_notes(text: str) -> list[tuple[str, str]]:
    notes = []
    for ids, note in NOTE_LINE.findall(strip_think(text)):
        for d in re.findall(r"D\d+", ids):
            notes.append((d, note.strip()))
    return notes


class Runner:
    def __init__(self, cfg: SysConfig, corpus, retriever, engine, lora=None):
        self.cfg, self.corpus, self.retriever, self.engine, self.lora = cfg, corpus, retriever, engine, lora

    # --------------------------------------------------------------------- engine
    def _gen(self, eps_convs: list[tuple[Episode, list]], max_tokens: int, phase: str, infos=None) -> list[Turn]:
        if not eps_convs:
            return []
        outs = self.engine.chat([c for _, c in eps_convs], temperature=self.cfg.temperature, top_p=self.cfg.top_p,
                                max_tokens=max_tokens, lora=self.lora)
        turns = []
        for i, ((ep, conv), o) in enumerate(zip(eps_convs, outs)):
            t = Turn(phase=phase, conv=conv, text=o.text, prompt_token_ids=o.prompt_token_ids, token_ids=o.token_ids,
                     finish_reason=o.finish_reason, info=dict(infos[i]) if infos else {})
            ep.turns.append(t)
            turns.append(t)
        return turns

    def _search(self, eps_queries: list[tuple[Episode, list[str]]]) -> list[list[list[str]]]:
        flat, owners = [], []
        for k, (ep, qs) in enumerate(eps_queries):
            for q in qs:
                flat.append(q)
                owners.append(k)
        excl = [eps_queries[o][0].item.get("exclude_doc_ids", []) for o in owners]
        res = self.retriever.search_batch(flat, self.cfg.search_k, excl) if flat else []
        out: list[list[list[str]]] = [[] for _ in eps_queries]
        for o, r in zip(owners, res):
            out[o].append(r)
        return out

    def _render(self, ids: list[str], words: int) -> str:
        return "\n".join(f"[{d}] {self.corpus.text(d, words)}" for d in ids)

    def _snippets(self, ids: list[str]) -> str:
        return "\n".join(f"[{d}] {self.corpus.text(d, self.cfg.snippet_words)}" for d in ids) or "(no results)"

    # ------------------------------------------------------------------- dispatch
    def run(self, episodes: list[Episode]) -> list[Episode]:
        by_sys: dict[str, list[Episode]] = {}
        for ep in episodes:
            by_sys.setdefault(ep.system, []).append(ep)
        for sys_name, eps in by_sys.items():
            getattr(self, f"_run_{sys_name}")(eps)
            for ep in eps:
                ep.done = True
        return episodes

    # -------------------------------------------------------------- no retrieval
    def _run_zero_shot(self, eps):
        cfg = self.cfg
        convs = [(ep, [{"role": "system", "content": P.ZERO_SHOT_SYSTEM.format(rules=write_rules(ep.item, cfg), nocite=P.NOCITE_RULES)},
                       {"role": "user", "content": task_text(ep.item, cfg) + "\n\nWrite the text now."}]) for ep in eps]
        for ep, t in zip(eps, self._gen(convs, cfg.tok_write, "direct")):
            ep.answer = parse_answer(t.text)

    def _run_cot(self, eps):
        cfg = self.cfg
        convs = [(ep, [{"role": "system", "content": P.COT_SYSTEM.format(rules=write_rules(ep.item, cfg), nocite=P.NOCITE_RULES)},
                       {"role": "user", "content": task_text(ep.item, cfg)}]) for ep in eps]
        for ep, t in zip(eps, self._gen(convs, cfg.tok_plan, "direct")):
            ep.answer = parse_answer(t.text)

    # ------------------------------------------------------------ single-shot RAG
    def _write_with_docs(self, eps, docs_per_ep):
        cfg = self.cfg
        convs = []
        for ep, docs in zip(eps, docs_per_ep):
            ep.pool = list(docs)
            convs.append((ep, [{"role": "system", "content": P.RAG_SYSTEM.format(rules=write_rules(ep.item, cfg), cite=P.CITE_RULES)},
                               {"role": "user", "content": P.RAG_USER.format(task=task_text(ep.item, cfg), docs=self._render(docs, cfg.doc_words_synth))}]))
        for ep, t in zip(eps, self._gen(convs, cfg.tok_write, "synth")):
            ep.answer = parse_answer(t.text)

    def _run_rag(self, eps):
        res = self.retriever.search_batch([default_query(ep.item) for ep in eps], self.cfg.rag_k,
                                          [ep.item.get("exclude_doc_ids", []) for ep in eps])
        self._write_with_docs(eps, res)

    def _run_oracle(self, eps):
        cfg, all_ids = self.cfg, list(self.corpus.docs.keys())
        docs = []
        for ep in eps:
            rng = random.Random(f"{cfg.seed}-{ep.item['item_id']}")
            gold = list(ep.item["gold_doc_ids"])
            if cfg.drop_gold_frac > 0:
                keep = max(1, round(len(gold) * (1 - cfg.drop_gold_frac)))
                gold = rng.sample(gold, keep)
                ep.rewards["oracle_kept_gold"] = gold
            extra = []
            excl = set(ep.item.get("all_cited_doc_ids", ep.item["gold_doc_ids"])) | set(ep.item.get("exclude_doc_ids", []))
            while len(extra) < cfg.oracle_distractors and len(all_ids) > len(excl) + len(extra):
                d = rng.choice(all_ids)
                if d not in excl and d not in extra:
                    extra.append(d)
            ids = gold[: cfg.rag_k * 2] + extra
            rng.shuffle(ids)
            docs.append(ids)
        self._write_with_docs(eps, docs)

    # ------------------------------------------------------------------- ReAct
    def _run_react(self, eps):
        cfg = self.cfg
        hist = {id(ep): [{"role": "system", "content": P.REACT_SYSTEM.format(rules=write_rules(ep.item, cfg), cite=P.CITE_RULES)},
                         {"role": "user", "content": task_text(ep.item, cfg) + f"\n\nYou may search up to {cfg.max_search_turns} times. Begin."}]
                for ep in eps}
        active = list(eps)
        for step in range(cfg.max_search_turns + 1):
            if not active:
                break
            last = step == cfg.max_search_turns
            if last:  # merge into the pending observation: Gemma/Mistral templates require strict user/assistant alternation
                for ep in active:
                    h = hist[id(ep)]
                    note = "No searches left. Write the final text now inside <answer>...</answer>."
                    if h[-1]["role"] == "user":
                        h[-1] = {"role": "user", "content": h[-1]["content"] + "\n\n" + note}
                    else:
                        h.append({"role": "user", "content": note})
            turns = self._gen([(ep, list(hist[id(ep)])) for ep in active], cfg.tok_write, "search")
            nxt, to_search = [], []
            for ep, t in zip(active, turns):
                text = strip_think(t.text)
                qs = [] if last else parse_queries(text, cfg.queries_per_turn)
                if last or not qs:
                    t.phase = "synth"
                    ep.answer = parse_answer(text)
                    continue
                # Searching takes priority: an <answer> written in the same turn, before any results were seen,
                # is discarded (otherwise it cites documents the agent never retrieved).
                low = text.lower()
                if "<answer>" in low:
                    t.info["dropped_premature_answer"] = True
                search_part = text[: low.rfind("</search>") + len("</search>")] if "</search>" in low else text
                t.info["queries"] = qs
                hist[id(ep)].append({"role": "assistant", "content": search_part})
                to_search.append((ep, qs))
                nxt.append(ep)
            for (ep, qs), results in zip(to_search, self._search(to_search)):
                self._observe(ep, ep.turns[-1], qs, results, hist[id(ep)], cfg.max_search_turns - step - 1, react=True)
            active = nxt

    def _observe(self, ep, turn, qs, results, conv, left, react=False):
        new_ids, blocks = [], []
        for q, r in zip(qs, results):
            for d in r:
                if d not in ep.pool and len(ep.pool) < self.cfg.max_pool:
                    ep.pool.append(d)
                    new_ids.append(d)
            blocks.append(P.SEARCH_OBS.format(query=q, results=self._snippets(r)))
        turn.info.update({"results": results, "new_ids": new_ids})
        footer = (f"\n\nSearches left: {left}. Search again or write the final <answer>." if react
                  else P.SEARCH_OBS_FOOTER.format(left=left))
        conv.append({"role": "user", "content": "\n\n".join(blocks) + footer})

    # ---------------------------------------------- Searcher -> Reader -> Synth
    def _run_pipeline3(self, eps):
        cfg = self.cfg
        hist = {id(ep): [{"role": "system", "content": P.SEARCHER_SYSTEM},
                         {"role": "user", "content": P.SEARCHER_USER.format(task=task_text(ep.item, cfg), n_turns=cfg.max_search_turns)}]
                for ep in eps}
        active = list(eps)
        for step in range(cfg.max_search_turns):
            if not active:
                break
            turns = self._gen([(ep, list(hist[id(ep)])) for ep in active], cfg.tok_search, "search")
            nxt, to_search = [], []
            for ep, t in zip(active, turns):
                text = strip_think(t.text)
                qs = parse_queries(text, cfg.queries_per_turn)
                t.info["queries"] = qs
                if not qs:
                    t.info["stopped"] = True
                    continue
                hist[id(ep)].append({"role": "assistant", "content": text})
                to_search.append((ep, qs))
                nxt.append(ep)
            for (ep, qs), results in zip(to_search, self._search(to_search)):
                self._observe(ep, ep.turns[-1], qs, results, hist[id(ep)], cfg.max_search_turns - step - 1)
            active = nxt
        # fallback so later phases always have evidence (the search turns still get no credit)
        empty = [ep for ep in eps if not ep.pool]
        if empty and cfg.search_fallback:
            res = self.retriever.search_batch([default_query(ep.item) for ep in empty], cfg.rag_k,
                                              [ep.item.get("exclude_doc_ids", []) for ep in empty])
            for ep, r in zip(empty, res):
                ep.pool = list(r)
                ep.rewards["search_fallback"] = True
        # Reader
        convs = [(ep, [{"role": "system", "content": P.READER_SYSTEM},
                       {"role": "user", "content": P.READER_USER.format(task=task_text(ep.item, cfg),
                                                                        docs=self._render(ep.pool[: cfg.reader_max_docs], cfg.doc_words_reader))}])
                 for ep in eps]
        for ep, t in zip(eps, self._gen(convs, cfg.tok_read, "read")):
            pool = set(ep.pool)
            ep.notes = [(d, n) for d, n in parse_notes(t.text) if d in pool]
            t.info["n_notes"] = len(ep.notes)
            t.info["selected"] = unique(d for d, _ in ep.notes)
        # Synthesizer
        convs = []
        for ep in eps:
            notes = "\n".join(f"[{d}] {n}" for d, n in ep.notes) or "(no notes)"
            ref_ids = unique([d for d, _ in ep.notes]) or ep.pool[: cfg.reader_max_docs]
            convs.append((ep, [{"role": "system", "content": P.SYNTH_SYSTEM.format(rules=write_rules(ep.item, cfg), cite=P.CITE_RULES)},
                               {"role": "user", "content": P.SYNTH_USER.format(task=task_text(ep.item, cfg), notes=notes,
                                                                               docs=self._render(ref_ids, cfg.doc_words_ref))}]))
        for ep, t in zip(eps, self._gen(convs, cfg.tok_write, "synth")):
            ep.answer = parse_answer(t.text)


def make_episodes(items: list[dict], system: str, n_samples: int = 1) -> list[Episode]:
    return [Episode(item=it, system=system, sample_idx=s) for it in items for s in range(n_samples)]


def add_args(p) -> None:
    g = p.add_argument_group("systems")
    d = SysConfig()
    for k, v in asdict(d).items():
        if k in ("seed", "temperature", "top_p"):
            continue
        if isinstance(v, bool):
            g.add_argument(f"--{k}", type=lambda s: s.lower() in ("1", "true", "yes"), default=v)
        else:
            g.add_argument(f"--{k}", type=type(v), default=v)


def config_from_args(args, temperature: float | None = None) -> SysConfig:
    kw = {k: getattr(args, k) for k in asdict(SysConfig()) if hasattr(args, k)}
    kw["seed"] = args.seed
    kw["temperature"] = args.temperature if temperature is None else temperature
    kw["top_p"] = getattr(args, "top_p", 1.0)
    return SysConfig(**kw)
