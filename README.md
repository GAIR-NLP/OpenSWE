<div align="center">

# OpenSWE: Efficient SWE Environment Synthesis at Scale

<p align="center">
  <img src="asset/arxiv-logo.svg" alt="arXiv" width="16" height="16"> <a href="https://arxiv.org/abs/" target="_blank">Paper</a> &nbsp; | &nbsp;
  <a href="https://github.com/GAIR-NLP/OpenSWE" target="_blank">Code</a> &nbsp; | &nbsp;
  <a href="https://github.com/GAIR-NLP/OpenSWE" target="_blank">Environments & Scripts</a>
</p>

<p align="center"> <img src="asset/teaser.png" style="width: 93%;" id="title-icon"> </p>
</div>

OpenSWE is the largest fully transparent framework for SWE agent training in Python, comprising **45,320 executable Docker environments** spanning over **12.8k repositories**, with all Dockerfiles, evaluation scripts, and infrastructure fully open-sourced for reproducibility. OpenSWE is built through a multi-agent synthesis pipeline deployed across a 64-node distributed cluster, automating repository exploration, Dockerfile construction, evaluation script generation, and iterative test analysis. Beyond scale, we propose a quality-centric filtering pipeline that characterizes the inherent difficulty of each environment, filtering out instances that are either unsolvable or insufficiently challenging and retaining only those that maximize learning efficiency. With \$891K spent on environment construction and an additional \$576K on trajectory sampling and difficulty-aware curation, the project yields about **13,000 curated trajectories** from roughly **9,000 quality-guaranteed environments**.

This repository contains the official implementation of the OpenSWE pipeline—an extensible SWE-bench–like dataset generation framework that supports custom data schemas, parallel multi-machine building, and full evaluation integration with SWE-agent / SWE-bench-fork (with provided patches).

## Highlights

- **Unprecedented Scale with Full Transparency**: We release 45,320 executable environments from 12.8k repositories at a construction cost of \$891K, with complete infrastructure including all Dockerfiles, evaluation scripts, and the distributed synthesis pipeline, enabling reproducibility and community-driven improvements.

- **Quality-Centric Filtering via Difficulty-Aware Curation**: A filtering pipeline characterizes environment difficulty to filter out unsolvable and trivially simple instances (e.g., PR–Issue misalignment, triviality). With an additional \$576K investment in trajectory sampling and curation, we obtain about 13,000 curated trajectories from roughly 9,000 high-quality environments.

- **Strong Empirical Validation**: OpenSWE-32B and OpenSWE-72B achieve **62.4%** and **66.0%** on SWE-bench Verified, establishing SOTA among SFT-based methods in the Qwen2.5 series. Models trained on OpenSWE consistently outperform SWE-rebench across all scales and scaffolds, with a log-linear data scaling trend showing no saturation, and SWE-focused training yields substantial out-of-domain improvements (e.g., up to 12 points on MATH-500, 5+ on science benchmarks) without degrading factual recall.

<div align="center">
<p align="center"> <img src="../scaling_env/figures/framework.pdf" style="width: 95%;" id="framework-icon"> </p>
</div>

## News

- **Paper**: OpenSWE (daVinci-Env) introduces the largest fully transparent SWE environment synthesis framework, with multi-agent pipeline design and scaling/curation analysis.

- **SOTA**: OpenSWE-32B / OpenSWE-72B set new SOTA among Qwen2.5 SFT methods on SWE-bench Verified (62.4% / 66.0%).

## Performance

<div align="center">

**Environment scale comparison**

| Dataset | # Repos | # Images | # Tasks | Source |
|:--------|:-------:|:--------:|:-------:|:------:|
| R2E-Gym (Subset) | 10 | 2.4k | 4.6k | Synthetic |
| SWE-gym | 11 | 2.4k | 2.4k | Real |
| SWE-rebench | 3.5k | 21.3k | 21.3k | Real |
| SWE-rebench (filtered) | 3.3k | 18.8k | 18.8k | Real |
| **OpenSWE (ours)** | **12.8k** | **45.3k** | **45.3k** | **Real** |

**SWE-bench Verified (Pass@1)**

| Model | Backbone | Scaffold | Score |
|:------|:--------|:--------:|:-----:|
| SWE-Master-32B-RL | Qwen2.5-Coder-32B-Inst. | R2E-Gym | 61.4 |
| daVinci-Dev-32B | Qwen2.5-32B-Base | SWE-Agent | 56.1 |
| **OpenSWE-32B (Ours)** | Qwen2.5-32B-Base | OpenHands | 59.8 |
| **OpenSWE-32B (Ours)** | Qwen2.5-32B-Base | SWE-Agent | **62.4** |
| daVinci-Dev-72B | Qwen2.5-72B-Base | SWE-Agent | 58.5 |
| **OpenSWE-72B (Ours)** | Qwen2.5-72B-Base | OpenHands | 65.0 |
| **OpenSWE-72B (Ours)** | Qwen2.5-72B-Base | SWE-Agent | **66.0** |

**Impact of environment source (SWE-bench Verified Pass@1)**

| Training Data | SWE-Agent 32B | SWE-Agent 72B | CodeAct 32B | CodeAct 72B |
|:--------------|:-------------:|:-------------:|:-----------:|:-----------:|
| SWE-rebench | 50.2% | 63.4% | 51.4% | 62.4% |
| **OpenSWE** | **62.4%** | **66.0%** | **59.8%** | **65.0%** |
| SWE-rebench + OpenSWE | 61.4% | 68.0% | 60.3% | 65.5% |

</div>

Training on OpenSWE alone yields large improvements over SWE-rebench across all model sizes and scaffolds; combining with SWE-rebench further improves 72B (e.g., 68.0% SWE-Agent). Data scaling analysis shows log-linear improvement with no saturation (see paper for curves). General capability evaluation shows gains on code (e.g., HumanEval +29), math (e.g., MATH-500 +12.2 for 72B), and science benchmarks without degrading factual recall.

## Quick Start

### 1. Data schema

Collect your dataset in the following schema:

| Field | Type | Description |
|-------|------|-------------|
| `instance_id` | `str` | Unique identifier for the sample. |
| `repo` | `str` | Full GitHub repo name (e.g., `psf/requests`). |
| `base_commit` | `str` | SHA of the commit immediately before the PR's first change. |
| `end_commit` | `str` | SHA of the final commit in the PR. |
| `problem_statement` | `str` | Issue description or problem to solve. |
| `patch` | `str` | Diff of changes to functional (non-test) code. |
| `test_patch` | `str` | Diff of changes to the test suite. |
| `language` | `str` | Primary programming language of the repo. |

### 2. (Recommended) Prepare system

- Download all git repositories into a _repocache_ directory.
- Build base Docker images with `scripts/prepare_baseimg.py`.

### 3. Apply patches for SWE-bench evaluation

Before running evaluation, apply:

- **swe-agent.patch** — for [SWE-agent/SWE-agent](https://github.com/SWE-agent/SWE-agent): adds `skip_fetch` and OpenSWE instance fields.
- **swe-bench-fork.patch** — for [SWE-rebench/SWE-bench-fork](https://github.com/SWE-rebench/SWE-bench-fork): adds `eval_script` support and `OPENSWE_EXIT_CODE` grading.

Replace `/path/to/openswe` with your OpenSWE repo root. On conflicts use `git apply --reject` and resolve `.rej` files. Apply each patch once per repo.

### 4. Configure and run

Edit `examples/run.sh` (set `OPENSWE_ROOT`, `DATA_PATH`, `OUTPUT_DIR`, `SETUP_DIR`, `RESULT_DIR`, `DATA_PATH`, API keys, and `DOCKER_REPOSITORY`), then:

```bash
bash examples/run.sh
```

For multi-machine building, see [Parallel Task Execution System](./scripts/parallel).

## Troubleshooting

- **Dataset missing**: Ensure your dataset JSONL exists at the path set in `DATA_PATH`; check schema matches the table above.
- **Patch conflicts**: Resolve `.rej` files after `git apply --reject` for swe-agent and swe-bench-fork.

## Acknowledgement

OpenSWE is inspired by [SWE-Rebench](https://arxiv.org/abs/2505.20411) and [SWE-Factory](https://arxiv.org/abs/2506.10954). We thank these teams for their open-source contributions. 

## License

This project is licensed under AGPL-3.0. See [LICENSE](./LICENSE) for details.

## Citation

If you find OpenSWE useful, please cite:

```bibtex
@article{openswe2026,
  title={daVinci-Env: Open SWE Environment Synthesis at Scale},
  author={Dayuan Fu and Shenyu Wu and Yunze Wu and Zerui Peng and Yaxing Huang and Jie Sun and Ji Zeng and Mohan Jiang and Lin Zhang and Yukun Li and Jiarui Hu and Liming Liu and Jinlong Hou and Pengfei Liu},
  journal={arXiv preprint},
  year={2026}
}
```
