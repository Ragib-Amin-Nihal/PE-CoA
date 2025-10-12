import common
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import ATTACK_TEMP, TARGET_TEMP, ATTACK_TOP_P, TARGET_TOP_P
from collections import defaultdict
import time
from language_models import (
    GPT, Claude, Gemini, PaLM, DeepSeek, OpenSourceModelAPI, 
    CommercialAPI, HuggingFace
)

def load_attack_and_target_models(args):
    attackLM = AttackLM(model_name=args.attack_model,
                        max_n_tokens=args.attack_max_n_tokens,
                        max_n_attack_attempts=args.max_n_attack_attempts,
                        temperature=ATTACK_TEMP,
                        top_p=ATTACK_TOP_P,
                        )
    preloaded_model = None
    if args.attack_model == args.target_model:
        print("Using same attack and target model. Using previously loaded model.")
        preloaded_model = attackLM.model
    targetLM = TargetLM(model_name=args.target_model,
                        max_n_tokens=args.target_max_n_tokens,
                        temperature=TARGET_TEMP,
                        top_p=TARGET_TOP_P,
                        preloaded_model=preloaded_model,
                        )
    return attackLM, targetLM

class AttackLM():

    def __init__(self,
                 model_name: str,
                 max_n_tokens: int,
                 max_n_attack_attempts: int,
                 temperature: float,
                 top_p: float):

        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.max_n_attack_attempts = max_n_attack_attempts
        self.top_p = top_p
        self.model, self.template = load_indiv_model(model_name)

        if ("vicuna" in model_name or "llama" in model_name) and "api" not in model_name:
            self.model.extend_eos_tokens()

    def get_attack_mr_init_chain(self, convs_list, prompt_list):

        assert len(convs_list) == len(
            prompt_list), "Mismatch between number of conversations and prompts. Conv List:" + str(len(convs_list)) + " Prompt List:" + str(len(prompt_list))

        batchsize = len(convs_list)
        indices_to_regenerate = list(range(batchsize))
        valid_outputs = [None] * batchsize

        full_prompts = []
        for conv, prompt in zip(convs_list, prompt_list):
            conv = conv.copy()
            conv.append_message(conv.roles[0], prompt)
            if any(x in self.model_name.lower() for x in ["gpt", "deepseek"]):
                full_prompts.append(conv.to_openai_api_messages())
            elif "text-davinci" in self.model_name:
                conv.append_message(conv.roles[1], None)
                full_prompts.append(conv.get_prompt())
            elif "api" in self.model_name:
                conv.append_message(conv.roles[1], None)
                full_prompts.append(conv.get_prompt())
            elif "chatglm" in self.model_name:
                full_prompts.append(conv)
            else:
                conv.append_message(conv.roles[1], None)
                full_prompts.append(conv.get_prompt())

        for attempt in range(self.max_n_attack_attempts):
            full_prompts_subset = [full_prompts[i]
                                   for i in indices_to_regenerate]

            outputs_list, _ = self.model.batched_generate_by_thread(full_prompts_subset,
                                                                 max_n_tokens=self.max_n_tokens,
                                                                 temperature=self.temperature,
                                                                 top_p=self.top_p
                                                                 )
            new_indices_to_regenerate = []
            for i, full_output in enumerate(outputs_list):
                orig_index = indices_to_regenerate[i]

                if full_output == "" or full_output is None:
                    new_indices_to_regenerate.append(orig_index)
                    continue

                mr_conv, evaluation, json_str = common.extract_json_for_mr_init_chain(
                    full_output)

                if mr_conv is not None and evaluation is not None:
                    valid_outputs[orig_index] = {
                        "mr_conv": mr_conv, "evaluation": evaluation}
                else:
                    new_indices_to_regenerate.append(orig_index)

            indices_to_regenerate = new_indices_to_regenerate

            if len(valid_outputs) > 0:
                break

        if any([output for output in valid_outputs if output is None]):
            print(
                f"Failed to generate output after {self.max_n_attack_attempts} attempts. Terminating.")

        return valid_outputs

    def get_attack(self, convs_list, prompts_list, pre_prompt_sem, sem_judger, target):

        assert len(convs_list) == len(
            prompts_list), "Mismatch between number of conversations and prompts."

        batchsize = len(convs_list)
        indices_to_regenerate = list(range(batchsize))
        valid_outputs = [None] * batchsize
        candidate_outputs = [None] * batchsize
        candidate_max_sem = [0] * batchsize
        theta = 0.1

        if len(convs_list[0].messages) == 0:
            init_message = """{\"improvement\": \"\",\"prompt\": \""""
        else:
            init_message = """{\"improvement\": \""""

        full_prompts = []

        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            if any(x in self.model_name.lower() for x in ["gpt", "deepseek"]):
                full_prompts.append(conv.to_openai_api_messages())
            elif "text-davinci" in self.model_name:
                conv.append_message(conv.roles[1], None)
                full_prompts.append(conv.get_prompt())
            elif "api" in self.model_name:
                conv.append_message(conv.roles[1], None)
                full_prompts.append(conv.get_prompt())
            elif "zhipu" in self.model_name:
                full_prompts.append(conv)
            else:
                conv.append_message(conv.roles[1], init_message)
                full_prompts.append(conv.get_prompt())

        for attempt in range(self.max_n_attack_attempts):
            full_prompts_subset = [full_prompts[i]
                                   for i in indices_to_regenerate]

            outputs_list, _ = self.model.batched_generate_by_thread(full_prompts_subset,
                                                       max_n_tokens=self.max_n_tokens,
                                                       temperature=self.temperature,
                                                       top_p=self.top_p
                                                       )

            if outputs_list is None:
                print("Error in generating output.")
                indices_to_regenerate = [indices_to_regenerate[0]]
                continue

            new_indices_to_regenerate = []
            for i, full_output in enumerate(outputs_list):
                orig_index = indices_to_regenerate[i]

                if full_output == "" or full_output is None:
                    new_indices_to_regenerate.append(orig_index)
                    continue

                time.sleep(1)
                attack_dict, json_str = common.extract_json(full_output)

                if attack_dict is not None and attack_dict["prompt"] != "[new prompt]" and attack_dict["prompt"] != "[新提示]":
                    prompt_sem = sem_judger.batched_compute_similarity(target, [attack_dict["prompt"]])
                    if prompt_sem[0] > candidate_max_sem[orig_index]:
                        candidate_outputs[orig_index] = attack_dict
                        candidate_max_sem[orig_index] = prompt_sem[0]

                    if prompt_sem[0] < pre_prompt_sem[orig_index] * (1 - theta):
                        new_indices_to_regenerate.append(orig_index)
                    else:
                        valid_outputs[orig_index] = attack_dict
                        convs_list[orig_index].update_last_message(json_str)
                else:
                    new_indices_to_regenerate.append(orig_index)

            indices_to_regenerate = new_indices_to_regenerate

            if len(indices_to_regenerate) == 0:
                break
        
        for i in range(len(valid_outputs)):
            if valid_outputs[i] is None:
                valid_outputs[i] = candidate_outputs[i]

        if any([output for output in valid_outputs if output is None]):
            print(
                f"Failed to generate output after {self.max_n_attack_attempts} attempts. Terminating.")

        valid_outputs = [
            output for output in valid_outputs if output is not None]

        return valid_outputs

class TargetLM():

    def __init__(self,
                 model_name: str,
                 max_n_tokens: int,
                 temperature: float,
                 top_p: float,
                 preloaded_model: object = None):

        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.top_p = top_p
        if preloaded_model is None:
            self.model, self.template = load_indiv_model(model_name)
        else:
            self.model = preloaded_model
            _, self.template = get_model_path_and_template(model_name)

    def get_response(self, convs_list, is_get_attention=False):
        full_prompts = convs_list
        retry_attempts = 5

        indices_to_regenerate = list(range(len(full_prompts)))
        valid_outputs = [None] * len(full_prompts)
        valid_attentions = [None] * len(full_prompts)
        attention_list = [None] * len(full_prompts)

        for attempt in range(retry_attempts):
            full_prompts_subset = [full_prompts[i] for i in indices_to_regenerate]
            
            try:
                outputs_list, attention_list = self.model.batched_generate_by_thread(
                    full_prompts_subset,
                    max_n_tokens=self.max_n_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    is_get_attention=is_get_attention
                )
            except Exception as e:
                print(f"Error in generating output (attempt {attempt + 1}): {e}")
                if attempt == retry_attempts - 1:
                    for idx in indices_to_regenerate:
                        valid_outputs[idx] = "$ERROR$"
                    break
                continue
            
            if outputs_list is None:
                print("Error in generating output.")
                if len(indices_to_regenerate) > 0:
                    indices_to_regenerate = [indices_to_regenerate[0]]
                continue

            new_indices_to_regenerate = []

            for i, full_output in enumerate(outputs_list):
                orig_index = indices_to_regenerate[i]

                if full_output is not None:
                    valid_outputs[orig_index] = full_output
                    if is_get_attention and attention_list:
                        valid_attentions[orig_index] = attention_list[i]
                else:
                    new_indices_to_regenerate.append(orig_index)

            indices_to_regenerate = new_indices_to_regenerate

            if len(indices_to_regenerate) == 0:
                break

        if any([output for output in valid_outputs if output is None]):
            print(f"Failed to generate output after {retry_attempts} attempts. Some outputs may be incomplete.")
            
        return valid_outputs, attention_list

def load_indiv_model(model_name, device=None):
    model_path, template = get_model_path_and_template(model_name)
    model_name_lower = model_name.lower()
    
    if any(pattern in model_name_lower for pattern in ['gpt-3.5', 'gpt-4', 'gpt-4.1', 'gpt-4o', 'text-davinci', 'o1']):
        lm = GPT(model_name)
    
    elif "deepseek" in model_name_lower:
        lm = DeepSeek(model_name)
    
    elif any(pattern in model_name_lower for pattern in ["claude-2", "claude-instant", "claude-3", "claude"]):
        lm = Claude(model_name)
    
    elif any(pattern in model_name_lower for pattern in ["gemini", "bard"]):
        lm = Gemini(model_name)
    
    elif "palm" in model_name_lower:
        lm = PaLM(model_name)
    
    elif model_name.endswith("-api"):
        lm = OpenSourceModelAPI(model_name)
    
    elif model_name in ["zhipu", "douyin", "wenxinyiyan", "kuaishou", "baichuan"]:
        lm = CommercialAPI(model_name)
    
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True, 
            device_map="auto"
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False
        )

        if 'llama-2' in model_path.lower():
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.padding_side = 'left'
        if 'vicuna' in model_path.lower():
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        lm = HuggingFace(model_name, model, tokenizer)

    return lm, template

def get_model_path_and_template(model_name):
    full_model_dict = {
        "vicuna-api": {
            "path": "vicuna_v1.1",
            "template": "vicuna_v1.1"
        },
        "llama2-api": {
            "path": "llama-2",
            "template": "llama-2"
        },
        "llama3-api": {
            "path": "llama-3",
            "template": "llama-3"
        },
        "mistral-api": {
            "path": "mistral",
            "template": "mistral"
        },
        "chatglm-api": {
            "path": "chatglm",
            "template": "chatglm"
        },
        "chatglm2-api": {
            "path": "chatglm-2",
            "template": "chatglm-2"
        },
        "phi2-api": {
            "path": "phi2",
            "template": "phi2"
        },
        "zephyr-api": {
            "path": "zephyr",
            "template": "zephyr"
        },
        "baichuan-api": {
            "path": "baichuan2-chat",
            "template": "baichuan2-chat"
        },
        "one-shot": {
            "path": "one_shot",
            "template": "one_shot"
        },
        "zhipu": {
            "path": "zhipu",
            "template": "zhipu"
        },
        "douyin": {
            "path": "douyin",
            "template": "douyin"
        },
        "wenxinyiyan": {
            "path": "wenxinyiyan",
            "template": "wenxinyiyan"
        },
        "kuaishou": {
            "path": "kuaishou",
            "template": "kuaishou"
        },
        "baichuan": {
            "path": "baichuan",
            "template": "baichuan"
        },
        "zero-shot": {
            "path": "zero_shot",
            "template": "zero_shot"
        },
        "airoboros-1": {
            "path": "airoboros_v1",
            "template": "airoboros_v1"
        },
        "airoboros-2": {
            "path": "airoboros_v2",
            "template": "airoboros_v2"
        },
        "airoboros-3": {
            "path": "airoboros_v3",
            "template": "airoboros_v3"
        },
        "koala-1": {
            "path": "koala_v1",
            "template": "koala_v1"
        },
        "alpaca": {
            "path": "alpaca",
            "template": "alpaca"
        },
        "chatglm": {
            "path": "chatglm",
            "template": "chatglm"
        },
        "chatglm-2": {
            "path": "chatglm-2",
            "template": "chatglm-2"
        },
        "dolly-v2": {
            "path": "dolly_v2",
            "template": "dolly_v2"
        },
        "oasst-pythia": {
            "path": "oasst_pythia",
            "template": "oasst_pythia"
        },
        "oasst-llama": {
            "path": "oasst_llama",
            "template": "oasst_llama"
        },
        "tulu": {
            "path": "tulu",
            "template": "tulu"
        },
        "stablelm": {
            "path": "stablelm",
            "template": "stablelm"
        },
        "baize": {
            "path": "baize",
            "template": "baize"
        },
        "chatgpt": {
            "path": "chatgpt",
            "template": "chatgpt"
        },
        "bard": {
            "path": "bard",
            "template": "bard"
        },
        "falcon": {
            "path": "falcon",
            "template": "falcon"
        },
        "baichuan-chat": {
            "path": "baichuan_chat",
            "template": "baichuan_chat"
        },
        "baichuan2-chat": {
            "path": "baichuan2_chat",
            "template": "baichuan2_chat"
        },
        "falcon-chat": {
            "path": "falcon_chat",
            "template": "falcon_chat"
        },
        "gpt-4": {
            "path": "gpt-4",
            "template": "gpt-4"
        },
        "gpt-4-turbo": {
            "path": "gpt-4-turbo",
            "template": "gpt-4-turbo"
        },
        "gpt-3.5-turbo": {
            "path": "gpt-3.5-turbo",
            "template": "gpt-3.5-turbo"
        },
        "text-davinci-003": {
            "path": "text-davinci-003",
            "template": "text-davinci-003"
        },
        "gpt-3.5-turbo-instruct": {
            "path": "gpt-3.5-turbo-instruct",
            "template": "gpt-3.5-turbo-instruct"
        },
        "vicuna": {
            "path": "vicuna-api",
            "template": "vicuna_v1.1"
        },
        "llama2-chinese": {
            "path": "llama2_chinese",
            "template": "llama2_chinese"
        },
        "llama-2": {
            "path": "llama2-api",
            "template": "llama-2"
        },
        "llama-3": {
            "path": "llama3-api",
            "template": "llama-2"
        },
        "claude-instant-1": {
            "path": "claude-instant-1",
            "template": "claude-instant-1"
        },
        "claude-2": {
            "path": "claude-2",
            "template": "claude-2"
        },
        "palm-2": {
            "path": "palm-2",
            "template": "palm-2"
        },
        "gemini-pro": {
            "path": "gemini-pro",
            "template": "gemini-pro"
        },
        "gemini-1.5-pro": {
            "path": "gemini-1.5-pro-latest",
            "template": "gemini-1.5-pro"
        },
        "gemini-1.5-flash": {
            "path": "gemini-1.5-flash-latest", 
            "template": "gemini-1.5-flash"
        },
        "gemini-2.0-flash": {
            "path": "gemini-2.0-flash-latest", 
            "template": "gemini-2.0-flash"
        },
        "gemini-2.5-pro": {
            "path": "gemini-2.5-pro-latest",
            "template": "gemini-2.5-pro"
        },
        "gemini-2.5-flash": {
            "path": "gemini-2.5-flash-latest", 
            "template": "gemini-2.5-flash"
        },
        "gemini-1.0-pro": {
            "path": "gemini-1.0-pro-latest",
            "template": "gemini-1.0-pro"
        },
        "gpt-4.1-mini": {
            "path": "gpt-4.1-mini-2025-04-14",
            "template": "gpt-4"
        },
        "gpt-4.1-nano": {
            "path": "gpt-4.1-nano-2025-04-14", 
            "template": "gpt-4"
        },
        "gpt-4o-mini": {
            "path": "gpt-4o-mini-2024-07-18",
            "template": "gpt-4"
        },
        "deepseek-chat": {
            "path": "deepseek-chat",
            "template": "deepseek-chat"
        },
        "deepseek-coder": {
            "path": "deepseek-coder", 
            "template": "deepseek-chat"
        },
        "deepseek-v3": {
            "path": "deepseek-v3",
            "template": "deepseek-chat"
        }
    }
    path, template = full_model_dict[model_name]["path"], full_model_dict[model_name]["template"]
    return path, template