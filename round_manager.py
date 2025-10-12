from collections import defaultdict
from common import conv_template
import random
from conversers import get_model_path_and_template

class Message():
    def __init__(self, target, prompt, response, max_round,
                 action=None, now_round=None,
                 now_index=None, prompt_toxic=0, response_toxic=0,
                 prompt_sem=0, response_sem=0, dataset_name=None,
                 judge_score=0, judge_explanations=None, pattern_stage=None):
        self.target = target
        self.prompt = prompt
        self.response = response
        self.now_round = now_round
        self.now_index = now_index
        self.prompt_toxic = prompt_toxic
        self.response_toxic = response_toxic
        self.prompt_sem = prompt_sem
        self.response_sem = response_sem
        self.dataset_name = dataset_name
        self.action = action
        self.judge_score = judge_score
        self.judge_explanations = judge_explanations
        self.max_round = max_round
        self.base_response = ""
        self.base_response_sem = 0
        self.base_response_toxic = 0
        self.posted_response = ""
        self.posted_response_sem = 0
        self.posted_response_toxic = 0
        self.fixed_score = 0
        self.pattern_stage = pattern_stage

    def set_fixed_score(self, score):
        self.fixed_score = score

    def set_base_response(self, response):
        self.base_response = response

    def set_response(self, response):
        self.response = response

    def set_action(self, action):
        self.action = action

    def set_prompt(self, prompt):
        self.prompt = prompt

    def set_index(self, index):
        self.now_index = index

    def set_prompt_toxic(self, prompt_toxic):
        self.prompt_toxic = float(prompt_toxic)

    def set_response_toxic(self, response_toxic):
        self.response_toxic = float(response_toxic)

    def set_prompt_sem(self, prompt_sem):
        self.prompt_sem = float(prompt_sem)

    def set_response_sem(self, response_sem):
        self.response_sem = float(response_sem)

    def set_dataset_name(self, dataset_name):
        self.dataset_name = dataset_name

    def set_judge_score(self, score):
        self.judge_score = score

    def set_judge_explanations(self, explanation):
        self.judge_explanations = explanation

    def set_now_round(self, round):
        self.now_round = round

    def set_base_response_sem(self, response_sem):
        self.base_response_sem = float(response_sem)
    
    def set_base_response_toxic(self, response_toxic):
        self.base_response_toxic = float(response_toxic)

    def set_posted_response(self, response):
        self.posted_response = response

    def set_posted_response_sem(self, response_sem):
        self.posted_response_sem = float(response_sem)

    def set_posted_response_toxic(self, response_toxic):
        self.posted_response_toxic = float(response_toxic)

class Task():

    def __init__(self):
        self.task_messages = []

    def add_messages(self, message: Message):
        self.task_messages.append(message)

    def get_prompts(self):
        prompts = []
        for message in self.task_messages:
            prompts.append(message.prompt)
        return prompts

    def get_responses(self):
        responses = []
        for message in self.task_messages:
            responses.append(message.response)
        return responses

    def get_now_rounds(self):
        rounds = []
        for message in self.task_messages:
            rounds.append(message.now_round)
        return rounds

    def get_judge_scores(self):
        scores = []
        for message in self.task_messages:
            scores.append(message.judge_score)
        return scores
    
    def get_response_toxic(self):
        scores = []
        for message in self.task_messages:
            scores.append(message.response_toxic)
        return scores
    
    def get_prompt_toxic(self):
        scores = []
        for message in self.task_messages:
            scores.append(message.prompt_toxic)
        return scores
    
    def get_now_index(self):
        indexes = []
        for message in self.task_messages:
            indexes.append(message.now_index)
        return indexes

    def get_max_rounds(self):
        rounds = []
        for message in self.task_messages:
            rounds.append(message.max_round)
        return rounds
    
    def get_actions(self):
        actions = []
        for message in self.task_messages:
            actions.append(message.action)
        return actions
    
    def get_datasets_name(self):
        names = []
        for message in self.task_messages:
            names.append(message.dataset_name)
        return names
    
    def get_prompt_sem(self):
        scores = []
        for message in self.task_messages:
            scores.append(message.prompt_sem)
        return scores
    
    def get_response_sem(self):
        scores = []
        for message in self.task_messages:
            scores.append(message.response_sem)
        return scores

    def get_base_responses_sem(self):
        responses = []
        for message in self.task_messages:
            responses.append(message.base_response_sem)
        return responses
    
    def get_base_responses_toxic(self):
        responses = []
        for message in self.task_messages:
            responses.append(message.base_response_toxic)
        return responses
    
    def get_posted_responses_sem(self):
        responses = []
        for message in self.task_messages:
            responses.append(message.posted_response_sem)
        return responses
    
    def get_posted_responses_toxic(self):
        responses = []
        for message in self.task_messages:
            responses.append(message.posted_response_toxic)
        return responses
    
    def get_posted_responses(self):
        responses = []
        for message in self.task_messages:
            responses.append(message.posted_response)
        return responses
    
    def set_posted_responses(self, responses):
        for i in range(len(responses)):
            self.task_messages[i].set_posted_response(responses[i])

    def set_posted_responses_sem(self, responses_sem):
        for i in range(len(responses_sem)):
            self.task_messages[i].set_posted_response_sem(responses_sem[i])

    def set_posted_responses_toxic(self, responses_toxic):
        for i in range(len(responses_toxic)):
            self.task_messages[i].set_posted_response_toxic(responses_toxic[i])
    
    def set_posted_responses_info(self, rd_manager):
        responses = []
        responses_sem = []
        responses_toxic = []
        for batch in range(rd_manager.batchsize):
            posted_round = max(1, rd_manager.now_round[batch] - 1)
            if posted_round == 1:
                response = ""
                response_sem = 0 
                response_toxic = 0
            else:
                response = rd_manager.historys[batch][posted_round][-1].response
                response_sem = rd_manager.historys[batch][posted_round][-1].response_sem
                response_toxic = rd_manager.historys[batch][posted_round][-1].response_toxic
            
            responses.append(response)
            responses_sem.append(response_sem)
            responses_toxic.append(response_toxic)

        self.set_posted_responses(responses)
        self.set_posted_responses_sem(responses_sem)
        self.set_posted_responses_toxic(responses_toxic)
    
    def set_base_responses_sem(self, responses_sem):
        for i in range(len(responses_sem)):
            self.task_messages[i].set_base_response_sem(responses_sem[i])
    
    def set_base_responses_toxic(self, responses_toxic):
        for i in range(len(responses_toxic)):
            self.task_messages[i].set_base_response_toxic(responses_toxic[i])

    def set_responses(self, responses):
        for i in range(len(responses)):
            self.task_messages[i].set_response(responses[i])

    def set_base_responses(self, responses):
        for i in range(len(responses)):
            self.task_messages[i].set_base_response(responses[i])
    
    def set_prompts(self, prompts):
        for i in range(len(prompts)):
            self.task_messages[i].set_prompt(prompts[i])
    
    def set_prompt_sem(self, prompt_sem):
        for i in range(len(prompt_sem)):
            self.task_messages[i].set_prompt_sem(prompt_sem[i])

    def set_response_sem(self, response_sem):
        for i in range(len(response_sem)):
            self.task_messages[i].set_response_sem(response_sem[i])

    def set_prompt_toxic(self, prompt_toxic):
        for i in range(len(prompt_toxic)):
            self.task_messages[i].set_prompt_toxic(prompt_toxic[i])

    def set_response_toxic(self, response_toxic):
        for i in range(len(response_toxic)):
            self.task_messages[i].set_response_toxic(response_toxic[i])

    def set_judge_scores(self, judge_scores):
        for i in range(len(judge_scores)):
            self.task_messages[i].set_judge_score(judge_scores[i])

    def set_judge_explanations(self, judge_explanations):
        for i in range(len(judge_explanations)):
            self.task_messages[i].set_judge_explanations(judge_explanations[i])

    def set_judge_info(self, judge_scores, judge_explanations):
        self.set_judge_scores(judge_scores)
        self.set_judge_explanations(judge_explanations)

    def set_actions(self, actions):
        for i in range(len(actions)):
            self.task_messages[i].set_action(actions[i])

    def set_index(self, indexes):
        for i in range(len(indexes)):
            self.task_messages[i].set_index(indexes[i])

    def set_now_round(self, rounds):
        for i in range(len(rounds)):
            self.task_messages[i].set_now_round(rounds[i])
    
    def set_fixed_scores(self, scores):
        for i in range(len(scores)):
            self.task_messages[i].set_fixed_score(scores[i])

    def get_base_responses(self):
        responses = []
        for message in self.task_messages:
            responses.append(message.base_response)
        return responses

    def get_fixed_scores(self):
        scores = []
        for message in self.task_messages:
            scores.append(message.fixed_score)
        return scores
        
    def clear(self):
        self.task_messages = []

    def display_task_info(self, color="green"):
        if color == "green":
            cl_str = "\033[92m"
        elif color == "red":
            cl_str = "\033[91m"
        else:
            raise NotImplementedError

        for message in self.task_messages:
            print("======="*10)
            print("> {}[PROMPT]\033[0m:\t{}".format(cl_str, message.prompt))
            if color == "red":
                print("> {}[RESPONSE]\033[0m:\t{}".format(
                cl_str, message.response))
            else:
                print("> {}[RESPONSE]\033[0m:\t{}...".format(
                    cl_str, message.response[:500]))
            print("> {}[TOXICITY PROMPT]\033[0m:\t{}".format(
                cl_str, message.prompt_toxic))
            print("> {}[TOXICITY RESPONSE]\033[0m:\t{}".format(
                cl_str, message.response_toxic))
            print("> {}[JUDGE Explanations]\033[0m:\t{}".format(
                cl_str, message.judge_explanations))
            print("> {}[JUDGE SCORES]\033[0m:\t{}".format(
                cl_str, message.judge_score))
            print("> {}[FIXED SCORES]\033[0m:\t{}".format(
                cl_str, message.fixed_score))
            print("> {}[SEM SCORES PROMPT]\033[0m:\t{}".format(
                cl_str, message.prompt_sem))
            print("> {}[SEM SCORES RESPONSE]\033[0m:\t{}".format(
                cl_str, message.response_sem))
            print("> {}[BASE RESPONSE]\033[0m:\t{}...".format(
                cl_str, message.base_response[:500]))
            print("> {}[BASE RESPONSE SEM]\033[0m:\t{}".format(
                cl_str, message.base_response_sem))
            print("> {}[BASE RESPONSE TOXIC]\033[0m:\t{}".format(
                cl_str, message.base_response_toxic))
            print("> {}[DATASET NAME]\033[0m:\t{}".format(
                cl_str, message.dataset_name))
            if message.action != None:
                print("> {}[ACTION]\033[0m:\t{}\n".format(
                    cl_str, message.action.name))

    def select_best_task(self, method, rd_manager, topk=1):
        if method == "JFTS":
            self.task_messages.sort(key=lambda x: (x.judge_score, x.response_toxic), reverse=True)
        elif method == "TFJS":
            self.task_messages.sort(key=lambda x: (x.response_toxic, x.judge_score), reverse=True)
        elif method == "JPT":
            self.task_messages.sort(key=lambda x: (x.judge_score + x.response_toxic), reverse=True)
        elif method == "SPT":
            self.task_messages.sort(key=lambda x: (x.response_sem + x.response_toxic), reverse=True)
        elif method == "TFSS":
            self.task_messages.sort(key=lambda x: (x.response_toxic, x.response_sem), reverse=True)
        elif method == "SFTS":
            self.task_messages.sort(key=lambda x: (x.response_sem, x.response_toxic), reverse=True)
        elif method == "JFSSTS":
            self.task_messages.sort(key=lambda x: (x.judge_score, x.response_sem, x.response_toxic), reverse=True)
        elif method == "X": 
            process_messages = []
            rd_info = rd_manager.get_round_info()
            for message in self.task_messages:
                if message.response_sem > rd_info["sem_response"] and message.response_toxic > rd_info["toxic_response"]:
                    process_messages.append(message)
            if len(process_messages) > 0:
                self.task_messages = process_messages
            self.task_messages.sort(key=lambda x: (x.response_sem + x.response_toxic), reverse=True)
        elif method == "dp":
            topk = len(self.task_messages)
        else:
            raise NotImplementedError
        
        self.task_messages = self.task_messages[:topk]

class Action():
    ACTIONS = ["next", "restart", "regen", "back", "init", "exit"]

    def __init__(self, name, round_num):
        if name not in self.ACTIONS:
            raise NotImplementedError
        self.name = name
        self.round_num = round_num

class RoundManager():
    BACK_WALK_RATE = 0.3
    REGEN_WALK_RATE = 0.2

    def __init__(self, model_name, target, max_round, judgeLM, semLM, toxicLM, 
                 batchsize, mr_init_chain=[], pattern=None):
        self.model_name = model_name
        self.max_round = max_round
        self.target = target
        self.now_round = [1] * batchsize
        self.now_index = 1
        self.pattern = pattern
        self.judgeLM = judgeLM
        self.semLM = semLM
        self.toxicLM = toxicLM
        self.batchsize = batchsize
        
        self.current_stage_indices = [0] * batchsize
        self.pattern_stages = []
        if pattern:
            self.pattern_stages = pattern.stages

        if not mr_init_chain or not isinstance(mr_init_chain, list):
            self.mr_init_chain = self.create_default_chain()
        else:
            self.mr_init_chain = mr_init_chain

        self.historys = [defaultdict(list) for _ in range(batchsize)]
        path, templates = get_model_path_and_template(self.model_name)
        self.base_template = conv_template(templates)
        self.set_init_prompt()

    def create_default_chain(self):
        default_chain = []
        
        for batch in range(self.batchsize):
            chain = {"mr_conv": []}
            
            for round_num in range(self.max_round):
                stage = None
                if self.pattern and round_num < len(self.pattern_stages):
                    stage = self.pattern_stages[round_num]
                    
                round_data = {
                    "improvement": f"Round {round_num + 1}/{self.max_round}",
                    "prompt": stage.example_prompt if stage else f"Default prompt for round {round_num + 1}",
                    "sem_score": 0.0,
                    "toxicity": 0.0
                }
                chain["mr_conv"].append(round_data)
                
            chain["evaluation"] = [{
                "explain": "Default chain based on pattern",
                "score": 5
            }]
            default_chain.append(chain)
            
        return default_chain

    def set_init_prompt(self):
        self.now_index = 1
        
        for batch in range(self.batchsize):
            tmp_defaultdict = defaultdict(list)
            
            for index in range(self.max_round):
                current_stage = None
                stage_weight = 1.0
                if self.pattern and index < len(self.pattern_stages):
                    current_stage = self.pattern_stages[index]
                    stage_weight = current_stage.semantic_weight
                    self.current_stage_indices[batch] = index
                
                try:
                    item = self.mr_init_chain[batch]["mr_conv"][index]
                except (IndexError, KeyError):
                    item = {
                        "prompt": f"Default prompt for round {index + 1}",
                        "sem_score": 0.0,
                        "toxicity": 0.0
                    }
                
                new_message = Message(
                    self.target,
                    item["prompt"],
                    None,
                    self.max_round,
                    action=Action("init", self.now_round),
                    now_round=self.now_round,
                    now_index=-1,
                    prompt_sem=item.get("sem_score", 0.0) * stage_weight,
                    prompt_toxic=item.get("toxicity", 0.0),
                    pattern_stage=current_stage
                )
                
                tmp_defaultdict[index + 1].append(new_message)
            
            self.historys[batch] = tmp_defaultdict

    def get_preset_prompt(self):
        return [self.historys[batch][self.now_round[batch]][0].prompt for batch in range(self.batchsize)]

    def get_now_action(self):
        return [self.historys[batch][self.now_round[batch]][-1].action for batch in range(self.batchsize)]

    def get_pre_prompt_sem(self):
        pre_round = [max(self.now_round[batch] - 1, 1) for batch in range(self.batchsize)]
        return [self.historys[batch][pre_round[batch]][-1].prompt_sem for batch in range(self.batchsize)]

    def get_round_info(self):
        result = []
        for batch in range(self.batchsize):
            pre_round = max(self.now_round[batch] - 1, 1)

            sem_response = self.historys[batch][pre_round][-1].response_sem
            sem_prompt = self.historys[batch][pre_round][-1].prompt_sem
            toxic_response = self.historys[batch][pre_round][-1].response_toxic
            toxic_prompt = self.historys[batch][pre_round][-1].prompt_toxic
            data = {
                "sem_prompt": 0 if sem_prompt == None else sem_prompt,
                "sem_response": 0 if sem_response == None else sem_response,
                "toxic_prompt": 0 if toxic_prompt == None else toxic_prompt,
                "toxic_response": 0 if toxic_response == None else toxic_response
            }
            result.append(data)

        return result

    def get_base_conv(self, new_prompt):
        prompts = []
        for batch in range(self.batchsize):
            prompt = None

            conv = self.base_template.copy()

            if new_prompt[batch] != None:
                conv.append_message(conv.roles[0], new_prompt[batch])
            else:
                conv.append_message(
                    conv.roles[0], self.historys[batch][self.now_round[batch]][-1].prompt)

            if "gpt" in self.model_name:
                conv.append_message(conv.roles[1], None)
                prompt = conv.to_openai_api_messages()
            elif "api" in self.model_name:
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
            else:
                prompt = conv

            prompts.append(prompt)

        return prompts

    def get_next_round(self, new_prompt):
        prompts = []
        for batch in range(self.batchsize):
            conv = self.get_conv(batch)
            current_stage = None
            if self.pattern:
                current_stage = self.pattern.get_stage(self.current_stage_indices[batch])
                
            if new_prompt[batch] != None:
                prompt_to_use = new_prompt[batch]
                if current_stage:
                    prompt_to_use = self.apply_pattern_guidance(
                        prompt_to_use, 
                        current_stage
                    )
            else:
                prompt_to_use = self.historys[batch][self.now_round[batch]][-1].prompt
            
            conv.append_message(conv.roles[0], prompt_to_use)

            if "gpt" in self.model_name:
                conv.append_message(conv.roles[1], None)
                prompt = conv.to_openai_api_messages()
            elif "api" in self.model_name:
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
            else:
                prompt = conv

            prompts.append(prompt)

        return prompts

    def apply_pattern_guidance(self, prompt, stage):
        return prompt
    
    def update_pattern_stages(self, actions):
        for batch, action in enumerate(actions):
            if action.name == "next":
                if self.pattern:
                    self.current_stage_indices[batch] = (
                        self.current_stage_indices[batch] + 1
                    ) % len(self.pattern.stages)
            elif action.name == "back":
                if self.pattern:
                    self.current_stage_indices[batch] = max(
                        0,
                        self.current_stage_indices[batch] - action.round_num
                    )
    
    def get_update_data(self):
        preset_prompts = self.get_preset_prompt()
        
        responses = []
        for batch in range(self.batchsize):
            if self.historys[batch][self.now_round[batch]][-1].action.name == "next":
                round = max(self.now_round[batch] - 1, 1)
                response = self.historys[batch][round][-1].response
            elif self.historys[batch][self.now_round[batch]][-1].action.name == "regen":
                round = self.now_round[batch]
                tmp = self.historys[batch][self.now_round[batch]].copy()
                tmp.sort(key=lambda x: x.response_sem)
                response = tmp[-1].response
            elif self.historys[batch][self.now_round[batch]][-1].action.name == "back":
                round = max(self.now_round[batch] - 1, 1)
                response = self.historys[batch][round][-1].response
            elif self.historys[batch][self.now_round[batch]][-1].action.name == "init":
                round = max(self.now_round[batch] - 1, 1)
                response = self.historys[batch][round][-1].response
            elif self.historys[batch][self.now_round[batch]][-1].action.name == "exit":
                response = self.historys[batch][self.now_round[batch]][-1].response
            else:
                raise NotImplementedError
            
            responses.append(response)

        return preset_prompts, responses

    def add(self, task: Task):
        for batch in range(self.batchsize):
            message = task.task_messages[batch]
            action = message.action
            
            current_stage = None
            if self.pattern:
                if self.current_stage_indices[batch] < len(self.pattern_stages):
                    current_stage = self.pattern_stages[self.current_stage_indices[batch]]
            
            if current_stage:
                message.pattern_stage = current_stage
            
            if self.now_index == 1:
                self.historys[batch][self.now_round[batch]][-1] = message
            else:
                self.historys[batch][self.now_round[batch]].append(message)
            
            if action.name == "next":
                self.now_round[batch] += action.round_num
                if self.pattern:
                    self.current_stage_indices[batch] = (
                        self.current_stage_indices[batch] + 1
                    ) % len(self.pattern_stages)
            elif action.name == "back":
                self.now_round[batch] -= action.round_num
                if self.pattern:
                    self.current_stage_indices[batch] = max(
                        0,
                        self.current_stage_indices[batch] - action.round_num
                    )
            elif action.name == "restart":
                self.now_round[batch] = action.round_num
                self.current_stage_indices[batch] = 0
                self.set_init_prompt()
        
        self.now_index += 1

    def get_action(self, methods="dp", task=None):
        actions = []
        
        for batch in range(self.batchsize):
            if methods == "dp":
                message = task.task_messages[batch]
                
                pattern_score = 1.0
                if self.pattern:
                    current_stage = None
                    if self.current_stage_indices[batch] < len(self.pattern_stages):
                        current_stage = self.pattern_stages[self.current_stage_indices[batch]]
                        pattern_score = self.evaluate_pattern_adherence(message, current_stage)
                
                effective_score = message.response_sem * pattern_score
                
                if (effective_score >= message.base_response_sem and 
                    effective_score >= message.posted_response_sem):
                    actions.append(self.next_walk(batch))
                elif effective_score < min(message.base_response_sem, 
                                        message.posted_response_sem):
                    actions.append(self.back_walk(batch, 1))
                else:
                    actions.append(self.regen_walk())
            else:
                actions.append(self.default_action(batch, task))
                
        return actions

    def evaluate_pattern_adherence(self, message, stage):
        if not stage or not message:
            return 1.0
            
        adherence_score = 1.0
        
        if hasattr(stage, 'semantic_weight'):
            adherence_score *= stage.semantic_weight
            
        if hasattr(stage, 'example_prompt') and message.prompt:
            if len(message.prompt.split()) >= len(stage.example_prompt.split()) / 2:
                adherence_score *= 0.8
                
        if hasattr(message, 'now_round') and hasattr(stage, 'stage_number'):
            if message.now_round != stage.stage_number:
                adherence_score *= 0.7
                
        return adherence_score

    def next_walk(self, batch):
        if self.now_round[batch] >= self.max_round:
            if random.random() > self.REGEN_WALK_RATE:
                return self.regen_walk()
            else:
                step = 1 if random.random() > self.BACK_WALK_RATE else 2

                if self.now_round[batch] - step < 1:
                    step = max(self.now_round[batch] - 1, 0)

                if step == 0:
                    return self.regen_walk()
                else:
                    return self.back_walk(batch, step)

        return Action("next", 1)

    def sequence_walk(self, batch):
        if self.now_round[batch] == self.max_round:
            return Action("exit", 0)
        return Action("next", 1)

    def back_walk(self, batch, step):
        if self.now_round[batch] - step < 1:
            step = max(self.now_round[batch] - 1, 1)
        return Action("back", step)

    def regen_walk(self):
        return Action("regen", 0)

    def end_walk(self):
        return Action("exit", 0)

    def display_attack_history(self, is_print=True):
        for batch in range(self.batchsize):
            print("======="*10)
            print(f"Attack history (Batch {batch}):")
            print(f"Current round: {self.now_round[batch]}")
            print(f"Current index: {self.now_index}")
            
            if self.pattern and self.current_stage_indices[batch] < len(self.pattern_stages):
                stage = self.pattern_stages[self.current_stage_indices[batch]]
                print(f"Current pattern stage: {stage.name}")
                print(f"Stage description: {stage.description}")
            
            for round_num in range(1, self.max_round + 1):
                print(f"\nRound {round_num}:")
                messages = self.historys[batch][round_num]
                
                for msg in messages:
                    if not is_print:
                        continue
                        
                    pattern_info = ""
                    if hasattr(msg, 'pattern_stage') and msg.pattern_stage:
                        pattern_info = f" [Pattern: {msg.pattern_stage.name}]"
                    
                    response_preview = msg.response[:50] + "..." if msg.response else "None"
                    
                    print(
                        f"  Message {msg.now_index} ({msg.action.name}){pattern_info}\n"
                        f"  Prompt: {msg.prompt[:50]}...\n"
                        f"  Response: {response_preview}\n"
                        f"  Scores - Sem:{msg.response_sem:.2f} "
                        f"Toxic:{msg.response_toxic:.2f} "
                        f"Judge:{msg.judge_score}"
                    )
            
            print("======="*10)

    def get_conv(self, batch):
        conv = self.base_template.copy()

        for i in range(1, self.now_round[batch]):
            conv.append_message(conv.roles[0], self.historys[batch][i][-1].prompt)
            if "api" in self.model_name or "gpt" in self.model_name:
                conv.append_message(
                    conv.roles[1], self.historys[batch][i][-1].response)
            else:
                if self.historys[batch][i][-1].response != None:
                    conv.append_message(
                        conv.roles[1], self.historys[batch][i][-1].response)

        return conv

    def add_new_prompt(self, prompts):
        for batch in range(self.batchsize):
            self.historys[batch][self.now_round[batch]][-1].set_prompt(prompts[batch]) 

    def add_new_response(self, responses):
        for batch in range(self.batchsize):
            self.historys[batch][self.now_round[batch]][-1].set_response(responses[batch])