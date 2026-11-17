# SciSynth

**Citation-grounded scientific literature synthesis with retrieval and reinforcement learning.**

SciSynth searches scientific papers, reads relevant evidence, and generates cited answers or related-work sections. A single shared policy performs three roles—**Searcher → Reader → Synthesizer**—and is trained using Group Relative Policy Optimization (GRPO).

## Tasks

| Task | Input | Output |
|---|---|---|
| Research QA (`qa`) | A research question | An evidence-grounded answer with citations |
| Related-work generation (`rw`) | A paper abstract, optionally with its introduction | A cited related-work section |

Benchmark references come from papers cited by each seed paper, with key points extracted from those references. These citation sets serve as reference targets, not exhaustive lists of all relevant literature.

## Approach

- **Retrieval:** BM25, dense retrieval, or hybrid reciprocal-rank fusion.
- **Shared policy:** Qwen3-8B with LoRA generates rollouts in bf16 through vLLM and learns from those rollouts. Updated adapters are loaded after each training step.
- **Reward judge:** GPT-4o-mini assesses claim support, key-point coverage, and note support; it does not generate policy rollouts. Citation F1, format, and gold-paper recall are computed in code.
- **Training:** Optional DPO initialization, turn-level credit, a training-only recall bonus, and curriculum sampling based on disagreement among previously sampled answers.

## Rewards and Comparisons

The citation-based outcome score is:

```text
0.4 × coverage + 0.3 × support + 0.2 × citation F1 + 0.1 × format
```

| Reward mode | Credit assignment |
|---|---|
| `outcome` | One final-answer reward |
| `process_sum` | Process signals combined into one trajectory reward |
| `process_turn` | Process credit routed to the search, read, or synthesis turn that earned it |

Search rewards emphasize retrieval of reference papers; reading rewards assess supported notes and document selection; synthesis rewards assess support, citation validity, and format. The optional recall bonus changes training rewards only.

Baselines include zero-shot, chain-of-thought, RAG, ReAct, the untrained three-phase policy, SFT, DPO, and GRPO variants. Oracle evidence and evidence-removal/distractor settings provide additional comparisons.

## Setup

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Configure your OpenAI API key through `OPENAI_API_KEY`, and set dataset paths, model paths, environment activation, and GPU resources in the PBS scripts. Keep credentials outside the repository. GPU execution requires a compatible CUDA environment; cluster jobs use PBS.

Inspect the input data and run the benchmark pipeline:

```bash
python -m scisynth.cli inspect --input_path /path/to/data.csv
qsub pbs/run_qwen3_8b.pbs
```

Input parsing supports manuscript text and cited-paper fields, commonly `pdf_text` and `cited_in_abs_intro`.

## GRPO Training

After preparing the benchmark, retrieval index, and any initialization adapter:

```bash
# Two GPUs: separate rollout and training engines
qsub pbs/run_qwen3_8b_grpo.pbs

# One GPU: bf16 rollouts with sleep/offload scheduling
qsub -l select=1:ncpus=16:ngpus=1:mem=128gb -v NUM_GPUS=1 pbs/run_qwen3_8b_grpo.pbs

# Compare DPO initialization with base-model initialization
qsub -v GRPO_RUNS="dpo_init base_init" pbs/run_qwen3_8b_grpo.pbs
```

The default recipe uses 8 questions × 8 rollouts per step, two clipped updates per batch, and masking of truncated turns. Resubmitting the same job resumes checkpointed training.

## Evaluation and Outputs

Checkpoint selection uses the full development split, including the starting adapter at step 0. The selected checkpoint is evaluated on the test set once. Improvements over DPO are not guaranteed; the starting adapter can remain the selected checkpoint.

```bash
# Evaluate trained policies and generate tables
qsub -v STAGES="infer_trained score final_table" pbs/run_qwen3_8b_grpo.pbs

# Regenerate tables from existing scores
python tools/final_table.py
```

Generated answers, scores, adapters, and training logs are stored under `outputs/`. Results include citation support, key-point coverage, retrieval recall, and paired confidence intervals. Related-work evaluation also includes reference-point coverage and ROUGE-L. Final summaries are exported as `final_table.tex` and `final_table.txt`.

## Repository Layout

| Directory | Contents |
|---|---|
| `scisynth/` | Data preparation, retrieval, agents, rewards, training, and evaluation |
| `pbs/` | Cluster job scripts and pipeline configuration |
| `tools/` | Results aggregation and table generation |
| `tests/` | Synthetic data and learner checks |

