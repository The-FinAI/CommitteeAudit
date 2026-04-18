import torch
from tqdm import tqdm
from typing import Dict
import pandas as pd


# This file aims to generate visualization results(e.g. Excel file) to demonstrate frequency of each expert


input_file_path = './RW/router_weight_info.pth'    # path to router weight file
num_layers = range(1,27)
num_all_experts = 64
num_topk = 6




def get_frequency_file():
    data = torch.load(input_file_path)
    final_outputs = {}
    final_outputs['all_tasks'] = {}
    for t in num_layers:
        final_outputs['all_tasks'][t] = torch.zeros(num_all_experts)
    final_outputs['all_tasks']['total_samples'] = 1

    for i in tqdm(range(len(data))):
        tmp_data = data[i]
        domain = tmp_data['domain']

        if domain not in final_outputs.keys():
            final_outputs[domain] = {}
            for t in num_layers:
                final_outputs[domain][t] = torch.zeros(num_all_experts)
            final_outputs[domain]['total_samples'] = 1
        else:
            final_outputs[domain]['total_samples'] += 1
        final_outputs['all_tasks']['total_samples'] += 1

        for t in num_layers:
            non_zero_indices = tmp_data[t]['topk_idx']
            final_outputs[domain][t][non_zero_indices] += 1
            final_outputs['all_tasks'][t][non_zero_indices] += 1

    return final_outputs



def frequency_to_excel(
    D: Dict[str, Dict[int, torch.Tensor]],
    k: int,
    out_path: str = "./RW/topk_stats.xlsx",
):

    rows = []
    for domain, inner in D.items():
        for idx, t in inner.items():
            if idx == 'total_samples':
                continue
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"D[{domain}][{idx}] is not a torch.Tensor")

            t = t.flatten()
            if t.numel() == 0:
                rows.append({
                    "domain": domain,
                    "idx": idx,
                    "topk_mean": float("nan"),
                    "topk_values": "",
                    "topk_indices": "",
                })
                continue

            kk = min(k, t.numel())
            topk = torch.topk(t, kk)

            values = topk.values.tolist()
            values = [round(x, 4) for x in values]
            indices = topk.indices.tolist()

            rows.append({
                "domain": domain,
                "idx": idx,
                "topk_mean": float(torch.mean(topk.values)),
                "topk_values": ", ".join(map(str, values)),
                "topk_indices": ", ".join(map(str, indices)),
            })

    df = pd.DataFrame(rows).sort_values(["idx","domain"]).reset_index(drop=True)
    df.to_excel(out_path, index=False, engine="openpyxl")
    return df



data = get_frequency_file()
for domain in data.keys():
    total_samples = data[domain]['total_samples']
    for t in num_layers:
        data[domain][t] /= total_samples
frequency_to_excel(data, num_topk)




