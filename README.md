# Aligning What LLMs Do and Say

Research code for **Aligning What LLMs Do and Say: Towards Self-Consistent Explanations**—measuring and improving alignment between what large language models use to answer and what they emphasize in post-hoc explanations, using feature attribution and preference-based fine-tuning.

## Overview

Post-hoc rationales can misrepresent the features that actually drove a model’s answer. This repository implements:

- **Post-hoc Self-Consistency Bank (PSCB)**-style pipelines: collect model decisions, multiple explanations per decision, and token-level attribution vectors (e.g. LIME, Layer Integrated Gradients) across QA settings.
- **Alignment metrics**: compare importance vectors for the **decision** vs. the **explanation** (e.g. cosine similarity, Spearman rank correlation on token scores).
- **Training**: supervised fine-tuning (SFT) and **Direct Preference Optimization (DPO)** on attribution-derived preferences (Unsloth + PEFT), plus evaluation scripts.

For details on the method, benchmarks, and results, see the paper: [arXiv:2506.07523](https://arxiv.org/abs/2506.07523).

The **repository root** is kept minimal: `README.md`, [`environment.yml`](environment.yml), [`LICENSE`](LICENSE), and the top-level trees below. Generated or machine-local folders (`data/`, `models/`, `wandb/`, `outputs/`, …) may appear after you run experiments but are not part of the tracked source tree.

## Repository layout

| Path | Role |
|------|------|
| [`src/collect_data/`](src/collect_data/) | End-to-end **data collection**: scenarios, generations, attributions, JSONL outputs under `data/collection_data/…` |
| [`src/llm_attribution/`](src/llm_attribution/) | **Attribution** utilities (e.g. `LLMAnalyzer`, Captum-backed helpers) |
| [`src/prepare_datasets/`](src/prepare_datasets/) | Load and format **ECQA**, **CODAH**, **Choice75**, **ARC** (easy/challenge) for the decider–explainer setup |
| [`src/analyze_data/`](src/analyze_data/) | Correlations, heatmaps, ranking analyses on collected data |
| [`src/pipeline_dpo/`](src/pipeline_dpo/) | **DPO**: dataset prep, `train_dpo_unsloth[_sweep].py` |
| [`src/pipeline_sft/`](src/pipeline_sft/) | **SFT** training sweeps |
| [`src/pipelline_ppo/`](src/pipelline_ppo/) | Experimental **PPO** scripts (legacy naming) |
| [`src/test_evaluations/`](src/test_evaluations/) | Evaluate **trained DPO/SFT** checkpoints |
| [`src/truthfulqa_eval/`](src/truthfulqa_eval/) | TruthfulQA-style evaluation helpers |
| [`scripts/run/`](scripts/run/) | Shell entrypoints for collect, train, eval (paths inside may need editing for your machine) |
| [`scripts/general/`](scripts/general/) | Splitting, indices, user-study extraction, correlation utilities |
| [`scripts/tools/`](scripts/tools/) | Ad hoc helpers (e.g. [`show_data.py`](scripts/tools/show_data.py) — print scenarios from JSONL; writes under `show_logs/`) |
| [`tests/`](tests/) | Smoke tests for GPU/env (optional) |
| [`notebooks/`](notebooks/) | Exploratory notebooks |

Plotting utilities for metric distributions live in [`src/analyze_data/visualization_of_data.py`](src/analyze_data/visualization_of_data.py) (run as a module from repo root after editing paths in `__main__`).

Heavy or machine-local directories (`data/`, `outputs/`, `models/`, `wandb/`, `datasets/`, `results/`, …) are listed in [.gitignore](.gitignore).

## Requirements

- **Python 3.11**
- **NVIDIA GPU** with a recent CUDA stack (examples in this repo use CUDA **12.4**-compatible PyTorch builds)
- Access to **Hugging Face** models you reference (e.g. Llama, Mistral, Qwen)—accept licenses and log in as needed (`huggingface-cli login`).

## Environment setup

Create the conda environment from the single spec in the repo root:

```bash
conda env create -f environment.yml
conda activate ConstLLM
```

To refresh an existing env after `environment.yml` changes: `conda env update -f environment.yml --prune`.

### Verify

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pytest tests/test_env.py -q   # optional
```

## Running from this repository

Most modules assume the **repository root** is the current working directory and use `python -m src.…`:

```bash
cd /path/to/ConstLLM
python -m src.collect_data.run_collect_data --help
```

Shell scripts under [`scripts/run/`](scripts/run/) typically `conda activate ConstLLM` and call the same pattern—adjust `CUDA_VISIBLE_DEVICES`, `DATASET_PATH`, and `model_id` to match your setup.

To inspect a few JSONL scenarios and write logs: `python scripts/tools/show_data.py` (edit the path in the script’s `__main__` block first).

### Typical stages

1. **Prepare / cache datasets** — use loaders in `src/prepare_datasets/` so data land where `collect_data` expects them (layout depends on dataset; see each script’s docstrings).
2. **Collect PSCB-style records** — [`scripts/run/collect_data.sh`](scripts/run/collect_data.sh) wraps `run_collect_data.py` (LIG, LIME, etc.).
3. **Build preference / SFT splits** — `src/pipeline_dpo/prepare_dataset_to_dpo.py`, `src/pipeline_sft/prepare_dataset_to_sft.py` after you have JSONL collections.
4. **Train** — e.g. [`scripts/run/train_dpo_sweep.sh`](scripts/run/train_dpo_sweep.sh), [`scripts/run/train_sft_sweep.sh`](scripts/run/train_sft_sweep.sh).
5. **Evaluate** — [`scripts/run/eval_trained_dpo.sh`](scripts/run/eval_trained_dpo.sh), [`scripts/run/eval_trained_sft.sh`](scripts/run/eval_trained_sft.sh), or TruthfulQA scripts as needed.

Enable [Weights & Biases](https://wandb.ai) where scripts call `wandb.init` if you want run tracking.

## Data and artifacts

- Processed **collections**, **checkpoints**, and **W&B runs** are intentionally gitignored. Recreate them with the pipelines above or unpack any supplementary archives you maintain locally.
- A large local archive `sft_eval_files.tar.gz` (if present) is ignored by git—keep it outside version control or store it in your artifact store.

## Citation

If you use this code or the associated benchmark methodology, please cite:

```bibtex
@article{admoni2025constllm,
  title   = {Aligning What {LLMs} Do and Say: Towards Self-Consistent Explanations},
  author  = {Admoni, Sahar and Amir, Ofra and Ziser, Yftah and Hallak, Assaf},
  journal = {arXiv preprint arXiv:2506.07523},
  year    = {2025},
  url     = {https://arxiv.org/abs/2506.07523}
}
```

## Authors

- **Sahar Admoni**, **Ofra Amir** — Technion – IIT  
- **Yftah Ziser**, **Assaf Hallak** — Nvidia Research  

## License

This project is licensed under the [MIT License](LICENSE).
