"""CPU correctness test of the GRPO/SFT learner with a tiny random Qwen3 model.

Checks: (1) left-padded, logits_to_keep log-probs == per-sequence unpadded log-probs,
        (2) a GRPO step with positive advantage raises the sample's log-prob,
        (3) KL/reference path runs, (4) save -> resume restores identical LoRA weights, (5) SFT step runs.
"""
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_tiny(path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    vocab = {f"w{i}": i for i in range(250)}
    vocab.update({"<pad>": 250, "<eos>": 251, "<unk>": 252})
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    hf_tok.save_pretrained(path)
    cfg = Qwen3Config(vocab_size=253, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, max_position_embeddings=512, pad_token_id=250, eos_token_id=251)
    torch.manual_seed(0)
    Qwen3ForCausalLM(cfg).save_pretrained(path)


def main():
    from scisynth.learner import Learner

    tmp = tempfile.mkdtemp()
    make_tiny(tmp)
    torch.manual_seed(1)
    L = Learner(tmp, lora_r=4, lora_alpha=8, lr=5e-3, max_tokens_per_micro=200, max_seq_len=400, kl_coef=0.05, grad_ckpt=True)
    g = torch.Generator().manual_seed(0)
    samples = []
    for i in range(5):
        p = torch.randint(0, 250, (int(torch.randint(5, 30, (1,), generator=g)),), generator=g).tolist()
        c = torch.randint(0, 250, (int(torch.randint(2, 12, (1,), generator=g)),), generator=g).tolist()
        samples.append({"prompt_ids": p, "completion_ids": c, "adv": [1.0, -1.0, 0.5, 2.0, -0.3][i]})

    # (1) padded batch log-probs vs. single unpadded sequences (full logits)
    L.model.eval()
    with torch.no_grad():
        ids, att, cm, K = L._collate(samples)
        lp = L._logps(ids, att, K)
        for i, s in enumerate(samples):
            seq = torch.tensor([s["prompt_ids"] + s["completion_ids"]])
            logits = L.model(input_ids=seq).logits[0, :-1].float()
            ref = torch.log_softmax(logits, -1).gather(-1, seq[0, 1:].unsqueeze(-1)).squeeze(-1)[-len(s["completion_ids"]):]
            got = lp[i][cm[i].bool()]
            err = (got - ref).abs().max().item()
            assert err < 2e-2, f"logprob mismatch sample {i}: {err}"
    print("(1) padded/logits_to_keep log-probs match unpadded reference  keep_kw=", L.keep_kw)

    # (2)+(3) GRPO steps increase log-prob of the positive-advantage sample
    def seq_lp(s):
        with torch.no_grad():
            ids, att, cm, K = L._collate([s])
            return float((L._logps(ids, att, K) * cm).sum())

    before = seq_lp(samples[3])
    for _ in range(5):
        st = L.train_step(samples)
    after = seq_lp(samples[3])
    print(f"(2) logp(adv=+2 sample) {before:.3f} -> {after:.3f}; stats={ {k: round(v, 4) if isinstance(v, float) else v for k, v in st.items()} }")
    assert after > before
    st2 = L.train_step(samples, n_updates=2)
    assert st2["n_samples"] == 5

    # (4) save / resume
    ck = os.path.join(tmp, "ckpt")
    L.save(ck)
    import json

    cfg = json.load(open(os.path.join(ck, "adapter_config.json")))
    assert isinstance(cfg["target_modules"], list), cfg["target_modules"]
    L2 = Learner(tmp, lora_r=4, lora_alpha=8, resume_from=ck, grad_ckpt=False)
    sd1 = {k: v for k, v in L.model.state_dict().items() if "lora_" in k}
    sd2 = {k: v for k, v in L2.model.state_dict().items() if "lora_" in k}
    assert sd1.keys() == sd2.keys() and all(torch.allclose(sd1[k].float(), sd2[k].float()) for k in sd1)
    print("(4) resume restores", len(sd1), "LoRA tensors; target_modules in config:", cfg["target_modules"])

    # (5) SFT
    print("(5) sft:", L.sft_step(samples))
    print("ALL LEARNER TESTS PASSED")


if __name__ == "__main__":
    main()
