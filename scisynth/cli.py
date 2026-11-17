"""Single entry point:  python -m scisynth.cli <stage> [options]

Stages
  inspect      print columns and the parsed structure of the first rows (run once on a new CSV)
  prepare      parse CSV -> examples.jsonl + corpus.jsonl, build BM25 (+dense) index        [CPU / any GPU]
  build_bench  generate questions + key points with the judge (gpt-4o-mini)                  [CPU + internet]
  infer        run systems (zero_shot cot rag oracle react pipeline3) with vLLM               [GPU]
  score        reward/metric computation with the judge                                      [CPU + internet]
  train_grpo   GRPO with --reward_mode outcome|outcome_judge|process_sum|process_turn       [GPU + internet]
  make_sft     best-of-N trajectories -> SFT data (rejection-sampling baseline)
  train_sft    LoRA SFT on that data                                                         [GPU]
  make_dpo     best vs worst of the same N trajectories -> preference pairs
  train_dpo    LoRA DPO on those pairs (reference = frozen base model)                       [GPU]
  tables       aggregate every model's summaries into tables/results*.csv
"""
from __future__ import annotations

import argparse
import os
import sys

from . import benchmark, data, grpo, judge, learner, llm, rewards, systems
from .utils import LOG, set_seed, setup_logging

STAGES = ["inspect", "prepare", "build_bench", "infer", "score", "train_grpo", "make_sft", "train_sft", "make_dpo",
          "train_dpo", "tables"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=STAGES)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out_dir", default="outputs/default")
    p.add_argument("--results_root", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_level", default="INFO")
    p.add_argument("--overwrite", action="store_true")
    # retrieval
    p.add_argument("--embed_model", default=None, help="local encoder dir for dense/hybrid retrieval")
    p.add_argument("--retriever_mode", default="hybrid", choices=["bm25", "dense", "hybrid"])
    p.add_argument("--index_max_words", type=int, default=300)
    # run selection
    p.add_argument("--task", default="qa", choices=["qa", "rw"])
    p.add_argument("--split", default="test", choices=["train", "dev", "test", "all"])
    p.add_argument("--limit", type=int, default=None, help="max benchmark items")
    p.add_argument("--systems", nargs="+", default=["zero_shot", "cot", "rag", "oracle", "react", "pipeline3"])
    p.add_argument("--tag", default="", help="suffix of output files, e.g. _grpo_process")
    p.add_argument("--lora_path", default=None, help="LoRA adapter for trained policies")
    p.add_argument("--n_samples", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--save_token_ids", action="store_true")
    p.add_argument("--save_conv", action="store_true")
    p.add_argument("--score_purpose", default="eval", choices=["eval", "train"])
    p.add_argument("--score_all", action="store_true", help="score every generation file of the task (except *_train*)")
    p.add_argument("--headroom_thresh", type=float, default=0.35)
    # SFT baseline
    p.add_argument("--sft_from", nargs="+", default=["pipeline3_train_n4"])
    p.add_argument("--sft_min_reward", type=float, default=0.45)
    p.add_argument("--sft_epochs", type=int, default=2)
    p.add_argument("--sft_batch", type=int, default=16)
    p.add_argument("--sft_name", default="rft")
    # DPO baseline (pairs = best vs worst of the same sampled trajectories)
    p.add_argument("--dpo_beta", type=float, default=0.1)
    p.add_argument("--dpo_min_gap", type=float, default=0.1, help="min reward gap between chosen and rejected")
    p.add_argument("--dpo_epochs", type=int, default=2)
    p.add_argument("--dpo_batch", type=int, default=8, help="pairs per optimizer step")
    p.add_argument("--dpo_name", default="dpo")
    for mod in (data, benchmark, llm, systems, judge, rewards, learner, grpo):
        mod.add_args(p)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    set_seed(args.seed)
    if args.model_tag is None and args.model_path:
        args.model_tag = os.path.basename(os.path.normpath(args.model_path))
    LOG.info("stage=%s args=%s", args.stage, " ".join(sys.argv[1:]))
    if args.stage == "inspect":
        data.inspect(args)
    elif args.stage == "prepare":
        data.prepare(args)
        from .retrieval import build_index

        build_index(args)
    elif args.stage == "build_bench":
        benchmark.build_benchmark(args)
    elif args.stage == "infer":
        from .run import infer

        infer(args)
    elif args.stage == "score":
        from .run import score

        score(args)
    elif args.stage == "train_grpo":
        grpo.train_grpo(args)
    elif args.stage == "make_sft":
        from .run import make_sft

        make_sft(args)
    elif args.stage == "train_sft":
        from .run import train_sft

        train_sft(args)
    elif args.stage == "make_dpo":
        from .run import make_dpo

        make_dpo(args)
    elif args.stage == "train_dpo":
        from .run import train_dpo

        train_dpo(args)
    elif args.stage == "tables":
        from .run import tables

        tables(args)


if __name__ == "__main__":
    main()
