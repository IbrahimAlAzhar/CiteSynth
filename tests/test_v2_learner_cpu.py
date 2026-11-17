"""CPU test of the version-2 learner features with a tiny random Qwen3 model.

Checks: (1) --init_adapter: a new learner warm-started from a saved adapter has identical LoRA weights and a fresh
        optimizer, (2) offload/onload (1-GPU sleep mode) keep weights and optimizer state usable,
        (3) two clipped updates per rollout batch (--updates_per_step 2) run and raise the positive sample's log-prob,
        (4) learner_kwargs passes init_from only when not resuming.
"""
import os
import sys
import tempfile
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_learner_cpu import make_tiny  # noqa: E402


def lora_state(L):
    return {k: v.detach().clone() for k, v in L.model.named_parameters() if "lora_" in k}


def main():
    from scisynth.learner import Learner, learner_kwargs

    tmp = tempfile.mkdtemp()
    make_tiny(tmp)
    g = torch.Generator().manual_seed(0)
    samples = []
    for i in range(6):
        p = torch.randint(0, 250, (int(torch.randint(5, 20, (1,), generator=g)),), generator=g).tolist()
        c = torch.randint(0, 250, (int(torch.randint(3, 10, (1,), generator=g)),), generator=g).tolist()
        samples.append({"prompt_ids": p, "completion_ids": c, "adv": [1.5, -1.0, 0.5, -0.5, 1.0, -1.5][i]})

    # train a first adapter ("DPO") and save it
    A = Learner(tmp, lora_r=4, lora_alpha=8, lr=5e-3, max_tokens_per_micro=200, max_seq_len=400)
    for _ in range(3):
        A.train_step(samples)
    ad = os.path.join(tmp, "dpo_adapter")
    A.save(ad, with_optimizer=True)

    # (1) warm start
    B = Learner(tmp, lora_r=4, lora_alpha=8, lr=5e-3, max_tokens_per_micro=200, max_seq_len=400, init_from=ad)
    sa, sb = lora_state(A), lora_state(B)
    assert sa.keys() == sb.keys() and all(torch.allclose(sa[k], sb[k]) for k in sa), "init_from weights differ"
    assert len(B.opt.state) == 0, "init_from must not restore the optimizer"
    print("(1) init_from loads the adapter weights, optimizer is fresh")

    # (2) offload / onload around a step
    B.train_step(samples)
    B.offload()
    assert all(p.device.type == "cpu" for p in B.model.parameters())
    B.onload()
    st = B.train_step(samples)
    assert st["n_samples"] == len(samples)
    print("(2) offload/onload round trip ok; step after onload:", {k: st[k] for k in ("loss", "grad_norm")})

    # (3) two clipped updates per batch increase log-prob of the strongest positive sample
    def lp(L, s):
        return L.seq_logps([s])[0]
    before = lp(B, samples[0])
    for _ in range(3):
        st = B.train_step(samples, n_updates=2)
    after = lp(B, samples[0])
    assert after > before, (before, after)
    print(f"(3) updates_per_step=2: positive-advantage log-prob {before:.3f} -> {after:.3f}, clipfrac={st['clipfrac']:.3f}")

    # (4) learner_kwargs
    args = types.SimpleNamespace(model_path=tmp, lora_r=4, lora_alpha=8, lr=1e-5, learner_quant="none", grad_ckpt=True,
                                 max_tokens_per_micro=200, max_model_len=400, clip_eps_low=0.2, clip_eps_high=0.28,
                                 kl_coef=0.0, max_grad_norm=1.0, attn_impl="sdpa", seed=0, init_adapter=ad)
    assert learner_kwargs(args)["init_from"] == ad
    assert learner_kwargs(args, resume_from="x")["init_from"] is None
    print("(4) learner_kwargs: init_from only without resume")
    print("ALL V2 LEARNER TESTS PASSED")


if __name__ == "__main__":
    main()
