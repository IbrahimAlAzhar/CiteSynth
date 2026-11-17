"""DPO correctness on a tiny random Qwen3 (CPU).

(1) at initialisation the policy equals the reference (LoRA B = 0), so every margin is 0
(2) the gradient of Learner.dpo_step points in the same direction as autograd of the textbook multi-turn DPO loss
    L = mean_i softplus(-beta * [(log pi(c_i) - ref(c_i)) - (log pi(r_i) - ref(r_i))]),  log pi(traj) = sum over turns
(3) a few updates push the chosen trajectories above the rejected ones (accuracy -> 1, loss falls)
"""
import math
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_learner_cpu import make_tiny  # noqa: E402

from scisynth.learner import Learner  # noqa: E402


def rand_turn(g):
    p = torch.randint(0, 250, (int(torch.randint(4, 20, (1,), generator=g)),), generator=g).tolist()
    c = torch.randint(0, 250, (int(torch.randint(2, 9, (1,), generator=g)),), generator=g).tolist()
    return {"prompt_ids": p, "completion_ids": c}


def main():
    tmp = tempfile.mkdtemp()
    make_tiny(tmp)
    torch.manual_seed(0)
    L = Learner(tmp, lora_r=4, lora_alpha=8, lr=1e-2, max_tokens_per_micro=120, max_seq_len=400, grad_ckpt=False)
    g = torch.Generator().manual_seed(1)
    pairs = []
    for _ in range(4):
        pairs.append({"chosen": [rand_turn(g) for _ in range(int(torch.randint(1, 3, (1,), generator=g)))],
                      "rejected": [rand_turn(g) for _ in range(int(torch.randint(1, 3, (1,), generator=g)))]})
    for p in pairs:
        p["ref_c"] = sum(L.seq_logps(p["chosen"], ref=True))
        p["ref_r"] = sum(L.seq_logps(p["rejected"], ref=True))
        pol_c, pol_r = sum(L.seq_logps(p["chosen"])), sum(L.seq_logps(p["rejected"]))
        assert abs(pol_c - p["ref_c"]) < 1e-4 and abs(pol_r - p["ref_r"]) < 1e-4
    print("(1) policy == reference at init")

    # move the policy away from the reference so that the sigmoid weights differ between pairs
    for _ in range(3):
        L.train_step([{**s, "adv": 1.0} for p in pairs[:2] for s in p["chosen"]])
    beta = 0.5

    # autograd of the textbook loss
    L.model.train()
    L.opt.zero_grad(set_to_none=True)
    losses = []
    for p in pairs:
        lp = {}
        for side in ("chosen", "rejected"):
            tot = 0
            for s in p[side]:
                ids, att, cm, K = L._collate([s])
                tot = tot + (L._logps(ids, att, K) * cm).sum()
            lp[side] = tot
        d = (lp["chosen"] - p["ref_c"]) - (lp["rejected"] - p["ref_r"])
        losses.append(torch.nn.functional.softplus(-beta * d))
    torch.stack(losses).mean().backward()
    g_ref = torch.cat([q.grad.flatten().clone() for q in L.params])

    # Learner.dpo_step with the optimizer step disabled
    step_fn = L.opt.step
    L.opt.step = lambda *a, **k: None
    st = L.dpo_step(pairs, beta=beta)
    g_ours = torch.cat([q.grad.flatten().clone() for q in L.params])
    L.opt.step = step_fn
    cos = float(torch.nn.functional.cosine_similarity(g_ref, g_ours, dim=0))
    print(f"(2) cosine(grad dpo_step, grad textbook DPO) = {cos:.6f}  (loss {st['dpo_loss']:.4f})")
    assert cos > 0.999, cos

    first = None
    for k in range(25):
        st = L.dpo_step(pairs, beta=beta)
        first = first if first is not None else st["dpo_loss"]
    print(f"(3) DPO loss {first:.4f} -> {st['dpo_loss']:.4f}, reward accuracy {st['reward_acc']:.2f}, margin {st['margin']:.2f}")
    assert st["dpo_loss"] < first * 0.5 and st["reward_acc"] == 1.0
    assert math.isfinite(st["dpo_loss"])
    print("ALL DPO TESTS PASSED")


if __name__ == "__main__":
    main()
