# CommitteeAudit

Official implementation of the paper
**[The Illusion of Specialization: Unveiling the Domain-Invariant "Standing Committee" in Mixture-of-Experts Models](https://arxiv.org/abs/2601.03425)**
by Yan Wang, Yitao Xu, Nanhan Shen, Jinyan Su, Jimin Huang, and Zining Zhu (2026).

> Mixture-of-Experts (MoE) models are widely assumed to achieve domain specialization through sparse routing. We challenge this assumption by introducing **COMMITTEEAUDIT**, a post-hoc framework that analyzes routing behavior at the level of **expert groups** rather than individual experts. Across three representative MoE models (OLMoE, Qwen3-30B-A3B, DeepSeek-V2-Lite) and the MMLU benchmark, we uncover a domain-invariant **Standing Committee**: a compact coalition of routed experts that consistently captures the majority of routing mass across domains, layers, and routing budgets, even when the architecture already includes shared experts.

## Key Findings

- **Existence.** A small, stable set of 2–5 experts per layer absorbs up to 70% of the total routing mass across diverse domains. This committee persists even in architectures with explicit *shared experts*, suggesting centralization is an emergent property of sparse routing rather than an architectural artifact.
- **Dynamics.** The committee size stays compact as model capacity (`E`) and routing budget (`k`) grow. Masking committee experts sharply degrades MMLU accuracy (e.g., 0.39 → 0.09 on DeepSeek-V2-Lite middle layers), confirming their functional importance.
- **Core–periphery organization.** Qualitative token-level analysis shows committee members anchor **reasoning and syntactic structure** (e.g., `What`, `Which`, `the`, `in`), while peripheral experts are recruited on demand to handle **domain-specific knowledge**.
- **Implication for training.** Standard load-balancing auxiliary losses that enforce uniform expert utilization may be working *against* this natural computational hierarchy, limiting training efficiency.

## Framework Overview

COMMITTEEAUDIT is a model-agnostic, three-stage auditing pipeline:

| Stage | Purpose | Key Quantity |
|-------|---------|--------------|
| **I. Task-conditioned routing profiles** | Extract per-domain routing signatures from a pre-trained MoE model. | Expert Contribution Index `ECI = E_{x ∈ D_τ}[G(x)_i]` |
| **II. Task-specificity scoring** | Filter out domains whose routing is not distinctive enough to support group-level analysis. | Silhouette score over cosine distances of routing vectors |
| **III. Standing Committee exploration** | For each layer, rank experts within every domain and select the Pareto-optimal set that is both high-ranking (low `μ`) and stable (low `σ²`). | Pareto set over `(μ_i, σ_i²)` among experts with presence ratio ≥ γ |

## Repository Structure

```
CommitteeAudit/
├── router_lens.py                     # (1) Extract per-sample router outputs via forward hooks
├── MOE_adjustment_example.py          # Template: how to modify a model's MoE gate for data collection
├── get_frequnecy_data.py              # (2a) Aggregate expert activation frequency per (domain, layer)
├── get_frequency_and_weight_data.py   # (2b) Aggregate frequency × normalized routing weight per (domain, layer)
├── Audit_Committee.py                 # (3) Run Standing Committee audit on the aggregated top-k table
└── README.md
```

## End-to-End Pipeline

The full reproduction pipeline has four steps:

1. **Modify the MoE gate** of your target model so its `forward` exposes `topk_idx` / `topk_weight` — see `MOE_adjustment_example.py`.
2. **Collect router outputs** per prompt with `router_lens.py` → saves `RW/*.pth`.
3. **Aggregate** routing info per `(domain, layer)` with `get_frequency_and_weight_data.py` (or `get_frequnecy_data.py` for frequency only) → produces a top-k Excel table.
4. **Run the Standing Committee audit** on the aggregated table with `Audit_Committee.py` → produces the final CSV.

> **Note.** `MOE_adjustment_example.py` targets **DeepSeek-V2-Lite**; porting to other models (e.g., OLMoE, Qwen3-MoE) requires adapting the modification to that model's gating implementation. The modified gate is intended for **offline data collection only** — do not use it for downstream inference or training.

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/The-FinAI/CommitteeAudit.git
cd CommitteeAudit
pip install -r requirements.txt
```

Minimum dependencies:

- `torch`, `transformers`, `tqdm` (for `router_lens.py`)
- `pandas`, `numpy`, `openpyxl` (for aggregation and `Audit_Committee.py`)

## Quick Start

The standalone audit script (`Audit_Committee.py`) accepts a top-k routing table and produces one Standing Committee per layer. You can skip stages (1)–(2) and try the audit directly on your own routing data, as long as it matches one of the supported schemas below.

### Step 1. Collect router activations (optional, for full pipeline)

1. Adapt your MoE gate's `forward` to expose `{topk_idx, topk_weight}` per layer, following `MOE_adjustment_example.py`.
2. Edit the model id, layer range, and dataset path in `router_lens.py`:
   ```python
   MODEL_ID = "deepseek-ai/DeepSeek-V2-Lite"
   STABLE_PROMPTS_DATASET_PATH = "MMLU.json"
   TARGET_LAYERS = range(1, 27)
   MOE_path = "model.layers.XXX.mlp.gate"
   ```
3. Run the extractor:
   ```bash
   python router_lens.py --st 0 --ed 5000
   ```
   Router weights are saved to `./RW/router_weight_*.pth`.

### Step 2. Build the per-(domain, layer) top-k table

```bash
python get_frequency_and_weight_data.py   # frequency × normalized weight
# or
python get_frequnecy_data.py              # activation frequency only
```

Both scripts output an Excel file with columns `idx, domain, topk_indices, topk_values`, which is exactly the input format expected by `Audit_Committee.py`.

### Step 3. Run the Standing Committee audit

```bash
python Audit_Committee.py \
  --input  path/to/topk_table.xlsx \
  --output path/to/standing_committee.csv
```

Full reference for `Audit_Committee.py` is below.

## `Audit_Committee.py` Reference

### Usage

```bash
python Audit_Committee.py --input INPUT_FILE --output OUTPUT_FILE
```

Example with an Excel-like input:

```bash
python Audit_Committee.py \
  --input  example/olmoe_top16_mean_norm_weight_multi_freq.xlsx \
  --output output/olmoe_top16_ratio.csv
```

Example with a long-format CSV input:

```bash
python Audit_Committee.py \
  --input  example/c4_with_domain.csv \
  --output standing_committee_audit.csv \
  --presence-ratio 0.8 \
  --agg-weight mean
```

Example with an explicit `k` for long-format input:

```bash
python Audit_Committee.py \
  --input  example/c4_with_domain.csv \
  --output standing_committee_audit.csv \
  --k 16
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input` | Path to the input file. Supported: `.csv`, `.xlsx`, `.xls`. | *required* |
| `--output` | Path to the output CSV file. | *required* |
| `--presence-ratio` | Minimum fraction of domains (`γ` in the paper) in which an expert must appear to qualify for committee selection. Must lie in `(0, 1]`. | `0.8` |
| `--k` | Top-`k` value for long-format input. If omitted, inferred automatically. | `None` |
| `--agg-weight` | Aggregation for repeated experts within the same `(layer, domain)` group in long-format input. One of `mean`, `sum`, `median`. | `mean` |

### Supported Input Formats

#### Format 1: Excel-like top-k table

Required columns:

- `idx` — layer index.
- `domain` — domain name or identifier.
- `topk_indices` — ordered list of expert indices.
- `topk_values` — ordered list of expert weights aligned with `topk_indices`.

Example:

```csv
idx,domain,topk_indices,topk_values
1,news,"[3, 18, 42, 7]","[0.31, 0.22, 0.19, 0.14]"
1,code,"[18, 3, 5, 42]","[0.28, 0.24, 0.21, 0.15]"
2,news,"[9, 12, 3, 40]","[0.35, 0.20, 0.17, 0.11]"
```

Notes:

- `topk_indices` and `topk_values` must have the same length within each row.
- The order defines expert rank and matters.
- List cells can be stored as Python-like strings such as `"[1, 2, 3]"`.

#### Format 2: Long-format table

Required columns:

- `layer` — layer index.
- `domain` — domain name or identifier.
- `expert_idx` — expert ID.
- `expert_weight` — router weight, contribution, or ECI for that expert.

Optional column:

- `topk_rank` — rank used for tie-breaking and automatic `k` inference.

Example:

```csv
layer,domain,expert_idx,expert_weight,topk_rank
1,news,3,0.31,0
1,news,18,0.22,1
1,news,42,0.19,2
1,code,18,0.28,0
1,code,3,0.24,1
1,code,5,0.21,2
```

Notes:

- When `topk_rank` is present, it is used to infer `k` and break ties between experts with equal aggregated weight.
- When `topk_rank` is absent, `k` is inferred from the largest number of unique experts in any `(layer, domain)` group unless `--k` is passed.
- Repeated `(layer, domain, expert_idx)` rows are aggregated using `--agg-weight`.

### What the Script Does

1. Reads the input file and auto-detects the schema.
2. If long-format, reconstructs one ordered top-`k` list per `(layer, domain)`.
3. For each layer, computes per-expert statistics:
   - presence count across domains,
   - mean rank `μ` (missing domains penalized as `k + 1`),
   - rank variance `σ²`,
   - average contribution across all domains.
4. Filters experts whose presence fraction is below `--presence-ratio`.
5. Selects Pareto-optimal experts (lower `μ` is better; lower `σ²` is better).
6. Exports one summary row per layer.

### Output

A CSV file with the following columns:

| Column | Meaning |
|--------|---------|
| `Layer` | Layer ID reported by the audit. |
| `Committee` | Comma-separated list of selected expert IDs. |
| `Size` | Number of experts in the committee (`|C|`). |
| `Avg_mu` | Average committee mean rank (`μ`, lower is better). |
| `Avg_sigma_sq` | Average committee rank variance (`σ²`, lower is better). |
| `Coverage` | Fraction of total layer contribution captured by the committee (ECI coverage). |
| `Ratio` | Average member contribution divided by average non-member contribution (Influence Density Ratio). |

Example:

```csv
Layer,Committee,Size,Avg_mu,Avg_sigma_sq,Coverage,Ratio
0,"3, 18, 42",3,1.67,0.89,0.7421,58.31
1,"9, 12",2,1.50,0.25,0.6814,67.08
```

### Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| `Input columns do not match a supported schema.` | Missing required columns — check the schema sections above. |
| `Unsupported file format: <ext>` | Extension is not `.csv`, `.xlsx`, or `.xls`. |
| Parse error on list cells | Invalid list syntax in `topk_indices` or `topk_values`. |
| `presence_ratio must be in the range (0, 1].` | `--presence-ratio` outside the valid range. |
| Empty output | Input became empty after preprocessing (e.g., all rows filtered out). |

## Citation

If you find this work useful, please cite:

```bibtex
@misc{wang2026standingcommittee,
  title        = {The Illusion of Specialization: Unveiling the Domain-Invariant "Standing Committee" in Mixture-of-Experts Models},
  author       = {Yan Wang and Yitao Xu and Nanhan Shen and Jinyan Su and Jimin Huang and Zining Zhu},
  year         = {2026},
  eprint       = {2601.03425},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2601.03425}
}
```

## License

This project is released under the **MIT License** — see [`LICENSE`](LICENSE) for details.
<!-- If you prefer a different license (Apache-2.0, BSD-3, etc.), replace this section and add the corresponding LICENSE file. -->

## Contact

For questions or issues, please open a [GitHub issue](https://github.com/The-FinAI/CommitteeAudit/issues) or contact the authors:

- Yan Wang — `wy2266336@gmail.com`
- Zining Zhu — `zzhu41@stevens.edu`
