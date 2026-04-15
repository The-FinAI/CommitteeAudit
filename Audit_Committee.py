import argparse
import ast
import math
from pathlib import Path

import numpy as np
import pandas as pd


EXCEL_LIKE_REQUIRED_COLUMNS = {"idx", "domain", "topk_indices", "topk_values"}
LONG_TABLE_REQUIRED_COLUMNS = {"layer", "domain", "expert_idx", "expert_weight"}
SUPPORTED_AGGREGATIONS = {"mean", "sum", "median"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the Standing Committee audit on either an Excel-like top-k table "
            "or a long-format CSV/Excel file."
        )
    )
    parser.add_argument("--input", required=True, help="Path to the input CSV or Excel file.")
    parser.add_argument("--output", required=True, help="Path to the output CSV file.")
    parser.add_argument(
        "--presence-ratio",
        type=float,
        default=0.8,
        help="Minimum fraction of domains in which an expert must appear to qualify.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Top-k value for long-format inputs. If omitted, the script will infer it.",
    )
    parser.add_argument(
        "--agg-weight",
        choices=sorted(SUPPORTED_AGGREGATIONS),
        default="mean",
        help="Aggregation applied to repeated expert weights in long-format inputs.",
    )
    return parser.parse_args()


def load_input_file(input_file):
    path = Path(input_file)
    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported file format: {suffix}")


def parse_list_cell(value, cast_type):
    if isinstance(value, list):
        return [cast_type(item) for item in value]

    if pd.isna(value):
        return []

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = [item.strip() for item in text.strip("[]").split(",") if item.strip()]
        if isinstance(parsed, (list, tuple, np.ndarray)):
            return [cast_type(item) for item in parsed]
        return [cast_type(parsed)]

    return [cast_type(value)]


def detect_input_format(df):
    columns = set(df.columns)

    if EXCEL_LIKE_REQUIRED_COLUMNS.issubset(columns):
        return "excel_like"
    if LONG_TABLE_REQUIRED_COLUMNS.issubset(columns):
        return "long_table"

    raise ValueError(
        "Input columns do not match a supported schema. "
        "Expected either Excel-like columns "
        f"{sorted(EXCEL_LIKE_REQUIRED_COLUMNS)} or long-table columns "
        f"{sorted(LONG_TABLE_REQUIRED_COLUMNS)}."
    )


def prepare_excel_like_table(df):
    table = df.copy()
    table["idx"] = table["idx"].astype(int)
    table["domain"] = table["domain"].astype(str)
    table["topk_values"] = table["topk_values"].apply(lambda value: parse_list_cell(value, float))
    table["topk_indices"] = table["topk_indices"].apply(lambda value: parse_list_cell(value, int))
    return table


def infer_k_from_long_table(df, has_rank):
    if has_rank:
        return int(df["topk_rank"].max()) + 1

    experts_per_group = df.groupby(["layer", "domain"])["expert_idx"].nunique()
    if experts_per_group.empty:
        raise ValueError("Unable to infer k because the input contains no expert rows.")
    return int(experts_per_group.max())


def aggregate_long_table(df, agg_weight):
    if agg_weight not in SUPPORTED_AGGREGATIONS:
        raise ValueError(f"Unsupported agg_weight: {agg_weight}")

    has_rank = "topk_rank" in df.columns
    table = df.copy()
    table["layer"] = table["layer"].astype(int)
    table["domain"] = table["domain"].astype(str)
    table["expert_idx"] = table["expert_idx"].astype(int)
    table["expert_weight"] = table["expert_weight"].astype(float)

    if has_rank:
        table["topk_rank"] = table["topk_rank"].astype(int)

    group_columns = ["layer", "domain", "expert_idx"]
    aggregation = {"expert_weight": agg_weight}

    if has_rank:
        expert_agg = (
            table.groupby(group_columns, as_index=False)
            .agg(mean_weight=("expert_weight", agg_weight), mean_rank=("topk_rank", "mean"))
        )
    else:
        expert_agg = (
            table.groupby(group_columns, as_index=False)
            .agg(mean_weight=("expert_weight", agg_weight))
        )
        expert_agg["mean_rank"] = float("inf")

    return expert_agg, has_rank


def reconstruct_excel_like_table(df, k=None, agg_weight="mean"):
    expert_agg, has_rank = aggregate_long_table(df, agg_weight)

    if k is None:
        k = infer_k_from_long_table(df, has_rank)
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    rows = []
    for (layer, domain), group in expert_agg.groupby(["layer", "domain"], sort=False):
        group = group.sort_values(["mean_weight", "mean_rank"], ascending=[False, True]).copy()
        top_group = group.head(k)

        topk_indices = top_group["expert_idx"].astype(int).tolist()
        topk_values = top_group["mean_weight"].astype(float).tolist()

        if len(topk_indices) < k:
            padding = k - len(topk_indices)
            topk_indices.extend([-1] * padding)
            topk_values.extend([0.0] * padding)

        rows.append(
            {
                "idx": int(layer),
                "domain": str(domain),
                "topk_indices": topk_indices,
                "topk_values": topk_values,
            }
        )

    return pd.DataFrame(rows)


def infer_expert_pool_size(df):
    all_indices = [
        expert_idx
        for index_list in df["topk_indices"]
        for expert_idx in index_list
        if expert_idx >= 0
    ]

    if not all_indices:
        raise ValueError("No valid expert indices were found in the input data.")

    max_idx = max(all_indices)
    if max_idx < 64:
        return 64
    if max_idx < 128:
        return 128
    if max_idx < 256:
        return 256
    return max_idx + 1


def is_pareto_optimal(row, others):
    for _, other in others.iterrows():
        if (other["mu"] <= row["mu"] and other["sigma_sq"] <= row["sigma_sq"]) and (
            other["mu"] < row["mu"] or other["sigma_sq"] < row["sigma_sq"]
        ):
            return False
    return True


def compute_layer_offset(layers):
    if not layers:
        return False
    return layers[0] != 1


def run_standing_committee_audit(df, presence_ratio=0.8):
    if not 0 < presence_ratio <= 1:
        raise ValueError("presence_ratio must be in the range (0, 1].")

    layers = sorted(df["idx"].unique())
    domains = df["domain"].astype(str).unique()
    num_domains = len(domains)

    if num_domains == 0:
        raise ValueError("No domains were found in the input data.")

    first_non_empty = next((indices for indices in df["topk_indices"] if len(indices) > 0), None)
    if first_non_empty is None:
        raise ValueError("No top-k index lists were found in the input data.")

    k = len(first_non_empty)
    expert_pool_size = infer_expert_pool_size(df)
    presence_threshold = math.ceil(num_domains * presence_ratio)
    keep_original_layer_id = compute_layer_offset(layers)

    audit_rows = []

    for layer in layers:
        layer_id = layer if keep_original_layer_id else layer - 1
        layer_df = df[df["idx"] == layer]
        expert_stats = {}

        for domain in domains:
            domain_row = layer_df[layer_df["domain"] == domain]
            if domain_row.empty:
                continue

            topk_values = domain_row.iloc[0]["topk_values"]
            topk_indices = domain_row.iloc[0]["topk_indices"]

            for rank_index, (value, expert_idx) in enumerate(zip(topk_values, topk_indices), start=1):
                if expert_idx < 0:
                    continue
                if expert_idx not in expert_stats:
                    expert_stats[expert_idx] = {"ranks": [], "ecis": []}
                expert_stats[expert_idx]["ranks"].append(rank_index)
                expert_stats[expert_idx]["ecis"].append(float(value))

        if not expert_stats:
            audit_rows.append(
                {
                    "Layer": layer_id,
                    "Committee": "",
                    "Size": 0,
                    "Avg_mu": None,
                    "Avg_sigma_sq": None,
                    "Coverage": 0.0,
                    "Ratio": 1.0,
                }
            )
            continue

        expert_data = []
        for expert_idx, stats in expert_stats.items():
            presence = len(stats["ranks"])
            average_eci = sum(stats["ecis"]) / num_domains
            full_ranks = stats["ranks"] + [k + 1] * (num_domains - presence)

            expert_data.append(
                {
                    "id": expert_idx,
                    "mu": float(np.mean(full_ranks)),
                    "sigma_sq": float(np.var(full_ranks)),
                    "presence": presence,
                    "avg_eci": average_eci,
                }
            )

        expert_df = pd.DataFrame(expert_data)
        candidates = expert_df[expert_df["presence"] >= presence_threshold].copy()

        if candidates.empty:
            candidates = expert_df[expert_df["presence"] == expert_df["presence"].max()].copy()

        pareto_mask = candidates.apply(lambda row: is_pareto_optimal(row, candidates), axis=1)
        committee = candidates[pareto_mask].sort_values("mu")
        committee_ids = committee["id"].astype(int).tolist()
        committee_size = len(committee_ids)

        total_layer_eci = layer_df["topk_values"].apply(lambda values: float(np.sum(values))).mean()
        coverage = (
            float(committee["avg_eci"].sum() / total_layer_eci) if total_layer_eci and total_layer_eci > 0 else 0.0
        )

        if 0 < committee_size < expert_pool_size:
            avg_member = coverage / committee_size
            remaining_experts = expert_pool_size - committee_size
            avg_non_member = (1 - coverage) / remaining_experts if remaining_experts > 0 else 0.0
            ratio = avg_member / avg_non_member if avg_non_member > 0 else float("inf")
        else:
            ratio = 1.0

        audit_rows.append(
            {
                "Layer": layer_id,
                "Committee": ", ".join(map(str, committee_ids)),
                "Size": committee_size,
                "Avg_mu": round(float(committee["mu"].mean()), 2) if committee_size > 0 else None,
                "Avg_sigma_sq": round(float(committee["sigma_sq"].mean()), 2)
                if committee_size > 0
                else None,
                "Coverage": round(coverage, 4),
                "Ratio": round(float(ratio), 2) if np.isfinite(ratio) else float("inf"),
            }
        )

    return pd.DataFrame(audit_rows)


def prepare_audit_input(input_file, k=None, agg_weight="mean"):
    raw_df = load_input_file(input_file)
    input_format = detect_input_format(raw_df)

    if input_format == "excel_like":
        return prepare_excel_like_table(raw_df)

    return reconstruct_excel_like_table(raw_df, k=k, agg_weight=agg_weight)


def main():
    args = parse_args()
    audit_input = prepare_audit_input(args.input, k=args.k, agg_weight=args.agg_weight)
    result_df = run_standing_committee_audit(audit_input, presence_ratio=args.presence_ratio)
    result_df.to_csv(args.output, index=False)
    print(f"Audit finished. Results saved to: {args.output}")


if __name__ == "__main__":
    main()
