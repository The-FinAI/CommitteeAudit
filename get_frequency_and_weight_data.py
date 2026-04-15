import torch
from tqdm import tqdm
from typing import Dict
import pandas as pd


input_file_path = './RW/router_weight_info.pth'
num_layers = range(1,27)
num_all_experts = 64
num_topk = 6



def min_max_and_mean(x):
    assert x.dim() == 1  #N, number of piece od data in domain task
    x = (x - x.min()) / (x.max() - x.min()+1e-6)
    return x.mean()


def compute_frequency_and_mean_weight(input_file_path):
    data = torch.load(input_file_path)

    activation_times = {}
    expert_weight = {}
    mean_normalized_weight = {}

    for i in tqdm(range(len(data))):
        tmp_data = data[i]
        domain = tmp_data['domain']
        if domain not in activation_times.keys():
            activation_times[domain] = {}
            expert_weight[domain] = {}
            mean_normalized_weight[domain] = {}

            for t in num_layers:
                activation_times[domain][t] = torch.zeros(num_all_experts)
                expert_weight[domain][t] = []
                mean_normalized_weight[domain][t] = torch.zeros(num_all_experts)
            activation_times[domain]['total_samples'] = 1
        else:
            activation_times[domain]['total_samples'] += 1

        for t in num_layers:
            non_zero_indices = tmp_data[t]['topk_idx']
            activation_times[domain][t][non_zero_indices] += 1
            tmp_weight_feature = torch.zeros(num_all_experts)
            tmp_weight_feature[non_zero_indices] = tmp_data[t]['topk_weight']
            expert_weight[domain][t].append(tmp_weight_feature)


    ###############################################

    for domain in activation_times.keys():
        total_samples = activation_times[domain]['total_samples']
        for t in num_layers:
            activation_times[domain][t] /= total_samples
            tmp = torch.stack(expert_weight[domain][t])  # [N,64]

            for expert_idx in range(num_all_experts):
                mean_normalized_weight[domain][t][expert_idx] = min_max_and_mean(tmp[:, expert_idx])


    ###################################################

    for domain in expert_weight.keys():
        for t in num_layers:
            mean_normalized_weight[domain][t] = mean_normalized_weight[domain][t] * activation_times[domain][t]

    return mean_normalized_weight


def draw_excel(
    D: Dict[str, Dict[int, torch.Tensor]],
    k: int,
    out_path: str = "./RW/mean_norm_weight_multi_freq.xlsx",
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

t = compute_frequency_and_mean_weight()
draw_excel(t, num_topk)