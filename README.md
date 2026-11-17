VERSION 2 - GRPO with the same model for rollouts and training, gpt-4o-mini as reward model only
=================================================================================================

What is new compared with version_1
-----------------------------------
GRPO run (online, on-policy):
  * rollouts  : Qwen3-8B itself generates every rollout with vLLM in bf16 using the CURRENT LoRA weights of the
                policy (version_1 used FP8 rollouts on the shared GPU, i.e. a slightly different model than the one
                being trained).
  * policy    : the same Qwen3-8B + LoRA is updated with GRPO on those rollouts; the new LoRA is hot-loaded into
                vLLM after every step.
  * reward    : gpt-4o-mini only scores the answers (claim support, key-point coverage, notes support). Citation
                F1, format and gold-paper recall are computed in code.
  * reused    : version_1 data, benchmark, retrieval index, DPO adapter, sampled training answers and all test
                scores. Nothing in version_1 is changed.

Recipe choices made to give GRPO the best chance of beating DPO / RAG (each one is a flag):
  1. Warm start from DPO (--init_adapter .../sft/qa_dpo). GRPO continues from the best version_1 policy instead of
     the base model (as SFT -> RL in DeepSeek-R1 / Tulu-3 style pipelines).
  2. Turn-level credit (--reward_mode process_turn, --lam 0.5). Each Searcher turn is credited for the gold papers it
     found, each Reader turn for supported notes, the Synthesizer for supported citations. In version_1 this mode
     raised training gold-recall faster than the outcome reward (0.176 -> 0.212 in 30 steps).
  3. Recall bonus (--recall_bonus 0.2). Training reward += 0.2 x share of gold papers retrieved. Version_1 showed
     that coverage is almost linear in gold-paper recall (r = 0.97) and that DPO did not change retrieval, so
     this targets the main bottleneck. The TEST metric is unchanged (same formula as version_1).
  4. Curriculum (--prior_scores version_1 pipeline3_train_n4). Questions whose 4 earlier samples disagreed are drawn
     more often; questions whose samples all tied (no learning signal for GRPO) are drawn less often.
  5. Group of 8 rollouts per question, 8 questions per step (64 rollouts), 2 clipped updates per batch
     (clip 0.2 / 0.28), truncated turns masked.
  6. Model selection on the full dev split, INCLUDING step 0 (= the DPO adapter): the checkpoint evaluated on
     the test set is the one with the best dev score, so it is never worse than DPO on dev. The test set is used
     once, at the end.
  Note: none of this guarantees that GRPO beats DPO on the test set; final_table.tex reports whatever happens,
  with paired confidence intervals. If GRPO never beats DPO on dev, the selected checkpoint is the DPO adapter
  itself and the table will show the same numbers as DPO.

Files (new or changed)
----------------------
  scisynth/grpo.py              changed  warm start, sleep mode, curriculum, step-0 dev evaluation, early stopping
  scisynth/learner.py           changed  init_from (weights only, fresh optimizer), offload/onload incl. optimizer
  scisynth/rewards.py           changed  --recall_bonus (training reward only)
  scisynth/llm.py               changed  vLLM sleep mode flag; sleep()/wake_up() report success
  pbs/run_qwen3_8b_grpo.pbs     new      the job script (all paths and settings at the top)
  pbs/pipeline_v2.sh            new      stages: train_grpo, infer_trained, score, final_table
  tools/final_table.py          new      writes version_2/final_table.tex (+ final_table.txt)
  tests/test_v2_learner_cpu.py  new      CPU test of the new learner features
  README_v2.txt                 new      this file
  Unchanged copies from version_1: the rest of scisynth/, pbs/common.sh, tools/, tests/, requirements.txt

Setup on the cluster (once)
---------------------------
  cd /lstr/sahara/datalab-ml/limagents_update/sci_lit_synthesis
  mkdir -p version_2/pbs version_2/logs
  cp -r version_1/scisynth version_1/tools version_1/tests version_1/requirements.txt version_2/
  cp version_1/pbs/common.sh version_2/pbs/
  # then put the 9 new/changed files above into version_2 (same relative paths), or unzip version_2_code.zip there
  cd version_2
  python tests/test_v2_learner_cpu.py          # optional, 1 min on CPU

Run
---
  cd /lstr/sahara/datalab-ml/limagents_update/sci_lit_synthesis/version_2
  qsub pbs/run_qwen3_8b_grpo.pbs                # 2 GPUs: one for vLLM rollouts, one for training

  One GPU only (bf16 sleep mode, slower):
  qsub -l select=1:ncpus=16:ngpus=1:mem=128gb -v NUM_GPUS=1 pbs/run_qwen3_8b_grpo.pbs

  Also train the "GRPO from base model" ablation (about twice the time and cost):
  qsub -v GRPO_RUNS="dpo_init base_init" pbs/run_qwen3_8b_grpo.pbs

  If the job stops at MAX_HOURS (44 h), submit the same command again: training resumes from the last checkpoint,
  then evaluation and the table run automatically.
  Only evaluation + table after training:  qsub -v STAGES="infer_trained score final_table" pbs/run_qwen3_8b_grpo.pbs
  Only the table (CPU, any node):          python tools/final_table.py

Time and cost (estimates)
-------------------------
  150 steps x 64 rollouts: about 13-17 h on 2 GPUs (about 20 h with NUM_GPUS=1).
  gpt-4o-mini reward: about $1.2 per 1,000 rollouts -> about $12-14 per run incl. dev evaluations
  (version_1 GRPO: $8.49 for 7,200 rollouts). Test evaluation: about $0.25 per system.

Where to look
-------------
  job log            version_2/logs/<jobid>.OU   (one line per step: train reward, coverage, gold recall, dev score)
  training curve     version_2/outputs/df_updated_with_retrieval_upd_n1000/qwen3_8b/grpo/qa_pipeline3_process_turn_dpo_init/
                       train_log.jsonl      one JSON line per step (step 0 = the DPO starting point on dev)
                       trainer_state.json   best_step, best_dev, finished
                       ckpt/best            adapter evaluated on the test set
  test answers       version_2/outputs/.../qwen3_8b/gens/qa/pipeline3_grpo_dpo_init.jsonl
  test scores        version_2/outputs/.../qwen3_8b/scores/qa/pipeline3_grpo_dpo_init.eval.jsonl (+ .summary.json)
  final tables       version_2/final_table.tex  (main results, paired comparisons, headroom subset, GRPO runs)
                     version_2/final_table.txt  (same numbers as plain text)
  In the paper: \usepackage{booktabs,graphicx} (already loaded) and \input{final_table.tex}.

Rows in final_table.tex
-----------------------
  version_1 : zero-shot, CoT, RAG, ReAct, 3-step agent, +SFT, +DPO, oracle variants (and Task B rows)
  version_2 : + DPO -> GRPO (v2, ours)                 main new run
              + GRPO (v1: outcome reward, from base)  the finished version_1 GRPO run, now evaluated on test
              + GRPO (v2, from base)                  only if GRPO_RUNS includes base_init
