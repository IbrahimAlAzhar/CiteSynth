"""Fast unit tests (CPU, no models):  python tests/test_units.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scisynth.data import parse_cited, parse_paper  # noqa: E402
from scisynth.grpo import build_samples, compute_advantages  # noqa: E402
from scisynth.judge import MockJudge  # noqa: E402
from scisynth.rewards import RewardConfig, Scorer  # noqa: E402
from scisynth.systems import Episode, Turn, parse_answer, parse_notes, parse_queries  # noqa: E402
from scisynth.utils import extract_citations, split_sentences  # noqa: E402


class FakeCorpus:
    docs = {"D1": {"text": "graphene oxide membranes improve water desalination flux"},
            "D2": {"text": "perovskite passivation increases solar cell stability"},
            "D3": {"text": "unrelated text about galaxies and dark matter halos"}}

    def text(self, d, max_words=None):
        return self.docs.get(d, {}).get("text", "")


def test_parsers():
    assert parse_queries("<search>graphene membranes</search> <search>desalination</search><search>x y</search>", 2) == ["graphene membranes", "desalination"]
    assert parse_answer("<plan>a</plan><answer>Text [D1].</answer>") == "Text [D1]."
    assert parse_answer("<think>hmm</think>## Related Work\nBody [D2].") == "Body [D2]."
    assert parse_notes("[D1] finding one\n- [D2, D3]: shared\nCONFLICTS: none") == [("D1", "finding one"), ("D2", "shared"), ("D3", "shared")]
    assert extract_citations("A [D1, D2]. B [D3; D4] and [D5][D6].") == ["D1", "D2", "D3", "D4", "D5", "D6"]
    s = split_sentences("First claim [D1]. Second claim. [D2]. Third (see [D3]).")
    assert len(s) == 3 and s[1].endswith("[D2]."), s
    assert len(parse_cited(str([["a " * 30, "b " * 30], ["c " * 30, "d " * 30]]), "auto")) == 2
    assert len(parse_cited(str([["a " * 30, "b " * 30, "c " * 30]]), "auto")) == 3
    p = parse_paper({"abstractText": "abs", "title": "t", "sections": [
        {"heading": "1 Introduction", "text": "intro"}, {"heading": "2 Related Work", "text": "rw1"},
        {"heading": "2.1 Sub", "text": "rw2"}, {"heading": "3 Method", "text": "m"}]})
    assert p["intro"] == "intro" and p["related_work"] == "rw1\nrw2", p
    print("parsers ok")


def _episode(item, answer, search_new, notes):
    t_s = Turn("search", [], info={"queries": ["q"], "new_ids": search_new, "results": [search_new or ["D3"]]})
    t_r = Turn("read", [], info={"selected": [d for d, _ in notes]})
    t_w = Turn("synth", [])
    for t in (t_s, t_r, t_w):
        t.token_ids, t.prompt_token_ids = [1, 2], [3]
    return Episode(item=item, system="pipeline3", turns=[t_s, t_r, t_w], pool=["D1", "D2", "D3"], notes=notes, answer=answer)


def test_rewards_and_advantages():
    item = {"item_id": "x", "task": "qa", "split": "train", "question": "q", "gold_doc_ids": ["D1", "D2"],
            "key_points": [{"text": "graphene oxide membranes improve water desalination flux", "importance": 3}]}
    good = _episode(item, "Graphene oxide membranes improve water desalination flux [D1]. " * 3, ["D1", "D2"],
                    [("D1", "graphene oxide membranes improve desalination flux")])
    bad = _episode(item, "Galaxies dark matter halos [D3]. Something [D9].", [], [("D3", "graphene improves solar cells")])
    sc = Scorer(MockJudge(), FakeCorpus(), RewardConfig(min_words=5, max_words=200))
    sc.score([good, bad], purpose="train", reward_mode="outcome")
    assert good.rewards["outcome"] > bad.rewards["outcome"]
    assert bad.rewards["cite_valid"] == 0.5  # D9 never seen
    assert good.turns[0].reward > bad.turns[0].reward and good.turns[1].reward > bad.turns[1].reward

    # outcome mode: every turn shares the trajectory advantage
    compute_advantages([good, bad], "outcome")
    assert len({t.adv for t in good.turns}) == 1 and good.turns[0].adv > 0 > bad.turns[0].adv

    # process_turn on a tie: outcome advantage is 0 but turn rewards still give signal ("rescued" group)
    tie1 = _episode(item, "x [D1].", ["D1", "D2"], [("D1", "graphene oxide membranes")])
    tie2 = _episode(item, "x [D1].", [], [("D1", "graphene oxide membranes")])
    sc.score([tie1, tie2], purpose="train", reward_mode="process_turn")
    tie1.rewards["outcome"] = tie2.rewards["outcome"] = 0.5
    st = compute_advantages([tie1, tie2], "process_turn", lam=0.5)
    assert st["dead_groups"] == 1 and st["rescued_by_process"] == 1
    assert tie1.turns[0].adv > 0 > tie2.turns[0].adv and tie1.turns[2].adv == 0 == tie2.turns[2].adv
    assert len(build_samples([tie1, tie2], {"search", "read", "synth"})) == 2
    print("rewards/advantages ok")


if __name__ == "__main__":
    test_parsers()
    test_rewards_and_advantages()
    print("ALL UNIT TESTS PASSED")
