# CommitteeAudit

# Audit_Committee.py

`Audit_Committee.py` runs a Standing Committee audit on routing data and exports one summary row per layer.

The script supports two input formats:

- An Excel-like top-k table with one row per `(layer, domain)`.
- A long-format CSV/Excel table with one row per `(layer, domain, expert)` observation.

## Requirements

- Python 3
- `pandas`
- `numpy`
- Excel input also requires an Excel reader supported by `pandas` such as `openpyxl`

## How To Run

Basic usage:

```bash
python Audit_Committee.py --input INPUT_FILE --output OUTPUT_FILE
```

Example with an Excel-like input:

```bash
python Audit_Committee.py \
  --input input/olmoe_topk/olmoe_top16_mean_norm_weight_multi_freq.xlsx \
  --output output/topk/olmoe/olmoe_top16_ratio.csv
```

Example with a long-format CSV input:

```bash
python Audit_Committee.py \
  --input input/c4_with_domain.csv \
  --output standing_committee_audit.csv \
  --presence-ratio 0.8 \
  --agg-weight mean
```

Example with an explicit `k` for long-format input:

```bash
python Audit_Committee.py \
  --input input/c4_with_domain.csv \
  --output standing_committee_audit.csv \
  --k 16
```

## Command Line Arguments

- `--input`: Path to the input file. Supported extensions are `.csv`, `.xlsx`, and `.xls`.
- `--output`: Path to the output CSV file.
- `--presence-ratio`: Minimum fraction of domains in which an expert must appear to qualify for committee selection. Default: `0.8`.
- `--k`: Top-k value used for long-format input. If omitted, the script tries to infer it automatically.
- `--agg-weight`: Aggregation used when the same expert appears multiple times within the same `(layer, domain)` group in long-format input. Allowed values: `mean`, `sum`, `median`. Default: `mean`.

## Supported Input Formats

### Format 1: Excel-Like Top-k Table

This format must contain the following columns:

- `idx`
- `domain`
- `topk_indices`
- `topk_values`

Description:

- `idx`: Layer index.
- `domain`: Domain name or domain identifier.
- `topk_indices`: Ordered list of expert indices.
- `topk_values`: Ordered list of expert weights aligned with `topk_indices`.

Example:

```csv
idx,domain,topk_indices,topk_values
1,news,"[3, 18, 42, 7]","[0.31, 0.22, 0.19, 0.14]"
1,code,"[18, 3, 5, 42]","[0.28, 0.24, 0.21, 0.15]"
2,news,"[9, 12, 3, 40]","[0.35, 0.20, 0.17, 0.11]"
```

Notes:

- `topk_indices` and `topk_values` must have the same length within each row.
- The order matters because it defines the expert rank.
- The list cells can be stored as Python-like strings such as `"[1, 2, 3]"`.

### Format 2: Long-Format Table

This format must contain the following columns:

- `layer`
- `domain`
- `expert_idx`
- `expert_weight`

Optional column:

- `topk_rank`

Description:

- `layer`: Layer index.
- `domain`: Domain name or domain identifier.
- `expert_idx`: Expert ID.
- `expert_weight`: Weight, contribution, or ECI value for that expert.
- `topk_rank`: Optional rank used for tie-breaking and automatic `k` inference.

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

- If `topk_rank` is provided, the script uses it to infer `k` and to break ties when experts have the same aggregated weight.
- If `topk_rank` is not provided, the script infers `k` from the maximum number of unique experts seen in a `(layer, domain)` group, unless `--k` is passed explicitly.
- If the same expert appears multiple times in the same `(layer, domain)` group, the script aggregates `expert_weight` using `--agg-weight`.

## What The Script Does

1. It reads the input file and detects the schema automatically.
2. If the input is long-format, it reconstructs one ordered top-k list per `(layer, domain)`.
3. For each layer, it computes per-expert statistics:
   - presence count across domains
   - mean rank with missing domains penalized as `k + 1`
   - rank variance
   - average contribution across all domains
4. It filters experts by the presence threshold.
5. It selects Pareto-optimal experts based on:
   - lower `mu` is better
   - lower `sigma_sq` is better
6. It exports one summary row per layer.

## Output File

The output is always a CSV file with these columns:

- `Layer`
- `Committee`
- `Size`
- `Avg_mu`
- `Avg_sigma_sq`
- `Coverage`
- `Ratio`

Column meanings:

- `Layer`: Layer ID reported by the audit.
- `Committee`: Comma-separated list of selected expert IDs.
- `Size`: Number of experts in the committee.
- `Avg_mu`: Average committee mean rank.
- `Avg_sigma_sq`: Average committee rank variance.
- `Coverage`: Fraction of total layer contribution captured by the committee.
- `Ratio`: Average member contribution divided by average non-member contribution.

Example output:

```csv
Layer,Committee,Size,Avg_mu,Avg_sigma_sq,Coverage,Ratio
0,"3, 18, 42",3,1.67,0.89,0.7421,58.31
1,"9, 12",2,1.50,0.25,0.6814,67.08
```

## Common Failure Cases

- Missing required columns.
- Unsupported file extension.
- Invalid list syntax in `topk_indices` or `topk_values`.
- Empty input after preprocessing.
- `--presence-ratio` outside the range `(0, 1]`.

## Citation

If you use this code, please cite:

```bibtex
@misc{wang2026illusionspecializationunveilingdomaininvariant,
  title={The Illusion of Specialization: Unveiling the Domain-Invariant "Standing Committee" in Mixture-of-Experts Models},
  author={Yan Wang and Yitao Xu and Nanhan Shen and Jinyan Su and Jimin Huang and Zining Zhu},
  year={2026},
  eprint={2601.03425},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2601.03425}
}
```

