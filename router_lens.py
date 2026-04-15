import torch
from functools import partial
from typing import List, Dict, Any
from tqdm import tqdm
import json
import argparse


# This file aims to get router weight and activated experts in MOE
# The following part is for adjust

#####################################################################################################
#                    Adjust Parameters
#####################################################################################################
MODEL_ID = "deepseek-ai/DeepSeek-V2-Lite"                 # The name of the LLM model
STABLE_PROMPTS_DATASET_PATH = "MMLU.json"                 # Processed MMLU dataset
TARGET_LAYERS = range(1, 27)                              # Layers you want to get
MOE_path = "model.layers.XXX.mlp.gate"                    # This changed with LLM structure, point to MOE block (XXX means layer index)


# The following function should be instance code(A example is shown below, like huggingface)
def get_model_and_tokenizer():
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map='auto')
    model.generation_config = GenerationConfig.from_pretrained(MODEL_ID)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    return model, tokenizer


#####################################################################################################
#                    End
#####################################################################################################


class DS2Analyzer():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.target_layers = TARGET_LAYERS
        self.hook_handles = []
        self._hook_data = {}

    def hf_func(self, module, input_tensor, output_tensor, layer_idx: int):
        topk_idx = output_tensor[0]['topk_idx']
        topk_weight = output_tensor[0]['topk_weight']
        self._hook_data[layer_idx] = {'topk_idx':topk_idx,'topk_weight':topk_weight}

    def _register_hooks(self):
        """Registers hooks on the MoE modules of the target layers."""
        self.remove_hooks()
        for layer_idx in self.target_layers:
            module_name = MOE_path.replace('XXX',str(layer_idx))
            module = dict(self.model.named_modules())[module_name]
            handle = module.register_forward_hook(
                partial(self.hf_func, layer_idx=layer_idx)
            )
            self.hook_handles.append(handle)

        if not self.hook_handles:
            raise RuntimeError("Failed to register any hooks. Please check the TARGET_LAYERS configuration.")

    def remove_hooks(self):
        """Removes all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        self._hook_data = {}

    def format_prompt(self, prompt_data: Dict[str, Any]) -> str:
        """Formats the stable prompt data into a complete prompt."""
        question = prompt_data['text']
        choices = prompt_data['choices']
        choices_str = "".join([f"{chr(65 + i)}. {choice}\n" for i, choice in enumerate(choices)])
        return f"Question: {question}\nChoices:\n{choices_str}Answer:"

    def load_stable_prompts(self) -> List[Dict[str, Any]]:
        with open(STABLE_PROMPTS_DATASET_PATH, 'r', encoding='utf-8') as f:
            prompts_data = json.load(f)
        return prompts_data

    def analyze(self, prompts_data, st, ed):
        all_result = []
        i = 0
        for k in prompts_data.keys():
            tmp_data = prompts_data[k]
            for ids in tmp_data.keys():

                if i<st+1:
                    i += 1
                    continue

                prompt_data = tmp_data[ids]
                formatted_prompt = self.format_prompt(prompt_data)
                formatted_prompt = [{'role':'user',"content":formatted_prompt}]

                inputs = self.tokenizer.apply_chat_template(
                    formatted_prompt,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self.model.device)

                self._register_hooks()
                with torch.no_grad():
                    self.model(**inputs, max_length=40)

                self._hook_data['subject'] = prompt_data['subject']
                self._hook_data['id'] = prompt_data['id']
                self._hook_data['domain'] = prompt_data['domain']
                self._hook_data['cognitive'] = prompt_data['cognitive']
                self._hook_data['difficulty'] = prompt_data['difficulty']
                all_result.append(self._hook_data)


                if i % 500 == 0:
                    save_name = './RW/router_weight_' + str(i) + '.pth'
                    torch.save(all_result, save_name)
                    all_result = []

                    if i == ed:
                        exit()

                i += 1
                if i % 100 == 0:
                    print(i, ' is finished!')

        save_name = './RW/router_weight_last.pth'
        torch.save(all_result, save_name)



def main(st,ed):
    model, tokenizer = get_model_and_tokenizer()
    analyzer = DS2Analyzer(model, tokenizer)
    stable_prompts = analyzer.load_stable_prompts()
    analyzer.analyze(stable_prompts, st, ed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script")
    parser.add_argument("--st", type=int, default=-1, help="value of start index")
    parser.add_argument("--ed", type=int, default=999999, help="value of start index")
    args = parser.parse_args()
    print(args.st,' ',args.ed)
    main(args.st, args.ed)
