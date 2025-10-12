import openai
import anthropic
import os
import time
import torch
import gc
from typing import Dict, List
import google.generativeai as palm
import config
from concurrent.futures import ThreadPoolExecutor
import requests
import json
from conv_builder import ConvBuilder
import threading
from conversation_template import get_commercial_api_template
from threading import Lock
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LanguageModel():
    def __init__(self, model_name):
        self.model_name = model_name

    def batched_generate(self, prompts_list: List, max_n_tokens: int, temperature: float):
        raise NotImplementedError

    def batched_generate_by_thread(self,
                                   convs_list: List[List[Dict]],
                                   max_n_tokens: int,
                                   temperature: float,
                                   top_p: float,
                                   is_get_attention: bool = False):
        raise NotImplementedError

class HuggingFace(LanguageModel):
    def __init__(self, model_name, model, tokenizer):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_ids = [self.tokenizer.eos_token_id]

    def batched_generate(self,
                         full_prompts_list,
                         max_n_tokens: int,
                         temperature: float,
                         top_p: float = 1.0,):
        inputs = self.tokenizer(
            full_prompts_list, return_tensors='pt', padding=True)
        inputs = {k: v.to(self.model.device.index) for k, v in inputs.items()}

        if temperature > 0:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_n_tokens,
                do_sample=True,
                temperature=temperature,
                eos_token_id=self.eos_token_ids,
                top_p=top_p,
            )
        else:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_n_tokens,
                do_sample=False,
                eos_token_id=self.eos_token_ids,
                top_p=1,
                temperature=1,
            )

        if not self.model.config.is_encoder_decoder:
            output_ids = output_ids[:, inputs["input_ids"].shape[1]:]

        outputs_list = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True)

        for key in inputs:
            inputs[key].to('cpu')
        output_ids.to('cpu')
        del inputs, output_ids
        gc.collect()
        torch.cuda.empty_cache()

        return outputs_list

    def extend_eos_tokens(self):
        self.eos_token_ids.extend([
            self.tokenizer.encode("}")[1],
            29913,
            9092,
            16675])

class DeepSeek(LanguageModel):
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name)
        
        if hasattr(config, 'DEEPSEEK_MODELS'):
            self.model_name = config.DEEPSEEK_MODELS.get(model_name, model_name)
        else:
            self.model_name = model_name
            
        self.api_key = getattr(config, 'DEEPSEEK_API_KEY', None)
        self.api_base = getattr(config, 'DEEPSEEK_API_BASE', "https://api.deepseek.com")
        self.timeout = kwargs.get('timeout', getattr(config, 'API_TIMEOUT', 60))
        self.max_retries = kwargs.get('max_retries', getattr(config, 'RETRY_ATTEMPTS', 3))
        
        if not self.api_key or self.api_key == "YOUR_DEEPSEEK_API_KEY":
            raise ValueError("DeepSeek API key not configured")
        
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )
    
    def generate(self, conv, max_n_tokens: int, temperature: float, top_p: float = 1.0):
        
        for attempt in range(self.max_retries):
            try:
                messages = self._convert_conversation_to_messages(conv)
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_n_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
                content = response.choices[0].message.content
                
                if hasattr(self, 'update_stats'):
                    if hasattr(response, 'usage'):
                        self.update_stats(tokens_used=response.usage.total_tokens)
                    else:
                        self.update_stats()
                
                return content
                
            except Exception as e:
                logger.error(f"DeepSeek API error for {self.model_name} (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    if hasattr(self, 'update_stats'):
                        self.update_stats(error=True)
                    return "$ERROR$"
                time.sleep(getattr(config, 'RETRY_DELAY', 1) * (attempt + 1))
        
        return "$ERROR$"
    
    def _convert_conversation_to_messages(self, conv):
        
        if hasattr(conv, 'get_prompt'):
            return [{"role": "user", "content": conv.get_prompt()}]
        
        elif hasattr(conv, 'messages'):
            messages = []
            for msg in conv.messages:
                if len(msg) >= 2:
                    role = "user" if msg[0].lower() in ['user', 'human'] else "assistant"
                    content = msg[1] if msg[1] is not None else ""
                    messages.append({"role": role, "content": content})
            return messages
        
        elif isinstance(conv, list) and len(conv) > 0 and isinstance(conv[0], dict):
            return conv
        
        elif isinstance(conv, list):
            return [{"role": "user", "content": str(item)} for item in conv]
        
        elif isinstance(conv, str):
            return [{"role": "user", "content": conv}]
        
        else:
            return [{"role": "user", "content": str(conv)}]
    
    def batched_generate(self, convs_list, max_n_tokens: int,
                        temperature: float, top_p: float = 1.0, is_get_attention: bool = False):
        outputs = []
        for conv in convs_list:
            output = self.generate(conv, max_n_tokens, temperature, top_p)
            outputs.append(output)
            time.sleep(0.1)
        
        return outputs, [None] * len(outputs)

    def batched_generate_by_thread(self, convs_list, max_n_tokens: int,
                                  temperature: float, top_p: float = 1.0, is_get_attention: bool = False):
        
        def generate_with_delay(conv, delay):
            time.sleep(delay)
            return self.generate(conv, max_n_tokens, temperature, top_p)
        
        with ThreadPoolExecutor(max_workers=min(4, len(convs_list))) as executor:
            futures = []
            for i, conv in enumerate(convs_list):
                delay = i * 0.2
                future = executor.submit(generate_with_delay, conv, delay)
                futures.append(future)
            
            outputs = [future.result() for future in futures]
        
        return outputs, [None] * len(outputs)

class GPT(LanguageModel):
    
    def __init__(self, model_name: str, api_key: str = None, api_base: str = None, **kwargs):
        super().__init__(model_name)
        
        if hasattr(config, 'OPENAI_MODELS'):
            self.model_name = config.OPENAI_MODELS.get(model_name, model_name)
        else:
            self.model_name = model_name
            
        self.api_key = api_key or config.OPENAI_API_KEY
        self.api_base = api_base or getattr(config, 'OPENAI_API_BASE', "https://api.openai.com/v1")
        self.timeout = kwargs.get('timeout', getattr(config, 'API_TIMEOUT', 60))
        self.max_retries = kwargs.get('max_retries', getattr(config, 'RETRY_ATTEMPTS', 3))
        
        if self.api_key == "YOUR_OPENAI_API_KEY" or not self.api_key:
            raise ValueError("OpenAI API key not configured")
        
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )
    
    def generate(self, conv: List[Dict], max_n_tokens: int, temperature: float, top_p: float = 1.0):
        
        for attempt in range(self.max_retries):
            try:
                messages = []
                for msg in conv:
                    if isinstance(msg, dict):
                        messages.append({
                            "role": msg.get("role", "user"),
                            "content": msg.get("content", "")
                        })
                    else:
                        messages.append({"role": "user", "content": str(msg)})
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_n_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
                content = response.choices[0].message.content
                
                if hasattr(self, 'update_stats'):
                    if hasattr(response, 'usage'):
                        self.update_stats(tokens_used=response.usage.total_tokens)
                    else:
                        self.update_stats()
                
                return content
                
            except Exception as e:
                logger.error(f"OpenAI API error for {self.model_name} (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    if hasattr(self, 'update_stats'):
                        self.update_stats(error=True)
                    return "$ERROR$"
                time.sleep(getattr(config, 'RETRY_DELAY', 1) * (attempt + 1))
        
        return "$ERROR$"
    
    def batched_generate(self, convs_list: List, max_n_tokens: int,
                        temperature: float, top_p: float = 1.0, is_get_attention: bool = False):
        outputs = []
        for conv in convs_list:
            if isinstance(conv, str):
                conv_dict = [{"role": "user", "content": conv}]
            else:
                conv_dict = conv
            
            output = self.generate(conv_dict, max_n_tokens, temperature, top_p)
            outputs.append(output)
            time.sleep(0.1)
        
        return outputs, [None] * len(outputs)

    def batched_generate_by_thread(self, convs_list: List, max_n_tokens: int,
                                  temperature: float, top_p: float = 1.0, is_get_attention: bool = False):
        
        def generate_with_delay(conv, delay):
            time.sleep(delay)
            if isinstance(conv, str):
                conv_dict = [{"role": "user", "content": conv}]
            else:
                conv_dict = conv
            return self.generate(conv_dict, max_n_tokens, temperature, top_p)
        
        with ThreadPoolExecutor(max_workers=min(4, len(convs_list))) as executor:
            futures = []
            for i, conv in enumerate(convs_list):
                delay = i * 0.2
                future = executor.submit(generate_with_delay, conv, delay)
                futures.append(future)
            
            outputs = [future.result() for future in futures]
        
        return outputs, [None] * len(outputs)

class Claude():
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 500
    _last_request_time = 0
    _request_lock = Lock()
    MIN_REQUEST_INTERVAL = 1.0 

    def __init__(self, model_name, api_key=config.ANTHROPIC_API_KEY) -> None:
        if hasattr(config, 'ANTHROPIC_MODELS'):
            self.model_name = config.ANTHROPIC_MODELS.get(model_name, model_name)
        else:
            self.model_name = model_name
            
        self.api_key = api_key

        import anthropic
        
        if hasattr(config, 'IS_USE_PROXY_OPENAI') and config.IS_USE_PROXY_OPENAI:
            self.model = anthropic.Anthropic(
                api_key=self.api_key,
                proxies=config.PROXY
            )
        else:
            self.model = anthropic.Anthropic(
                api_key=self.api_key,
            )

    def generate(self, conv, max_n_tokens: int, temperature: float, top_p: float):
        with self._request_lock:
            current_time = time.time()
            time_since_last = current_time - Claude._last_request_time
            if time_since_last < self.MIN_REQUEST_INTERVAL:
                sleep_time = self.MIN_REQUEST_INTERVAL - time_since_last
                time.sleep(sleep_time)
            Claude._last_request_time = time.time()
            output = self.API_ERROR_OUTPUT
            
            messages = []
            system_message = ""
            
            if hasattr(conv, 'messages'):
                for msg in conv.messages:
                    if len(msg) >= 2:
                        role = "user" if msg[0].lower() in ['user', 'human'] else "assistant"
                        content = msg[1] if msg[1] is not None else ""
                        messages.append({"role": role, "content": content})
            elif isinstance(conv, list):
                if len(conv) > 0 and isinstance(conv[0], dict):
                    for msg in conv:
                        role = msg.get('role', 'user')
                        content = msg.get('content', '')
                        if role.lower() == 'system':
                            system_message = content
                        else:
                            if role.lower() in ['human', 'user']:
                                role = "user"
                            elif role.lower() in ['assistant', 'claude']:
                                role = "assistant"
                            else:
                                role = "user"
                            messages.append({"role": role, "content": content})
                else:
                    for item in conv:
                        messages.append({"role": "user", "content": str(item)})
            elif isinstance(conv, str):
                messages = [{"role": "user", "content": conv}]
            else:
                messages = [{"role": "user", "content": str(conv)}]
            
            if not messages:
                messages = [{"role": "user", "content": "Hello"}]
            
            for _ in range(self.API_MAX_RETRY):
                try:
                    kwargs = {
                        "model": self.model_name,
                        "max_tokens": max_n_tokens,
                        "messages": messages,
                        "temperature": temperature,
                        "top_p": top_p
                    }
                    
                    if system_message:
                        kwargs["system"] = system_message
                    
                    response = self.model.messages.create(**kwargs)
                    output = response.content[0].text
                    break
                except Exception as e:
                    print(type(e), e)
                    time.sleep(self.API_RETRY_SLEEP)

                time.sleep(self.API_QUERY_SLEEP)
            return output

    def batched_generate(self,
                         convs_list: List[List[Dict]],
                         max_n_tokens: int,
                         temperature: float,
                         top_p: float = 1.0,):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]

    def batched_generate_by_thread(self,
                                   convs_list: List[List[Dict]],
                                   max_n_tokens: int,
                                   temperature: float,
                                   top_p: float = 1.0,
                                   is_get_attention: bool = False):
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = executor.map(self.generate, convs_list, [
                                   max_n_tokens]*len(convs_list), [temperature]*len(convs_list), [top_p]*len(convs_list))
        return list(results), []*len(convs_list)

class Gemini(LanguageModel):
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name)
        
        if hasattr(config, 'GOOGLE_MODELS'):
            self.model_name = config.GOOGLE_MODELS.get(model_name, model_name)
        else:
            self.model_name = model_name
            
        self.timeout = kwargs.get('timeout', getattr(config, 'API_TIMEOUT', 60))
        self.max_retries = kwargs.get('max_retries', getattr(config, 'RETRY_ATTEMPTS', 3))
        
        if not hasattr(self, 'request_count'):
            self.request_count = 0
        if not hasattr(self, 'error_count'):
            self.error_count = 0
        if not hasattr(self, 'total_tokens'):
            self.total_tokens = 0
        
        self._initialize_client()
        self.safety_settings = self._configure_safety_settings()
    
    def update_stats(self, tokens_used: int = 0, error: bool = False):
        self.request_count += 1
        self.total_tokens += tokens_used
        if error:
            self.error_count += 1
    
    def get_stats(self):
        return {
            "requests": self.request_count,
            "errors": self.error_count,
            "total_tokens": self.total_tokens,
            "error_rate": self.error_count / max(1, self.request_count)
        }
    
    def _configure_safety_settings(self):
        try:
            import google.generativeai as genai
            
            try:
                safety_settings = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]
                return safety_settings
            except:
                try:
                    safety_settings = []
                    categories = [
                        getattr(genai.types.HarmCategory, 'HARM_CATEGORY_HARASSMENT', None),
                        getattr(genai.types.HarmCategory, 'HARM_CATEGORY_HATE_SPEECH', None),
                        getattr(genai.types.HarmCategory, 'HARM_CATEGORY_SEXUALLY_EXPLICIT', None),
                        getattr(genai.types.HarmCategory, 'HARM_CATEGORY_DANGEROUS_CONTENT', None),
                    ]
                    
                    threshold = getattr(genai.types.HarmBlockThreshold, 'BLOCK_NONE', None)
                    
                    for category in categories:
                        if category is not None and threshold is not None:
                            safety_settings.append({
                                "category": category,
                                "threshold": threshold
                            })
                    
                    if safety_settings:
                        return safety_settings
                except Exception as e:
                    pass
            
            return []
            
        except Exception as e:
            return []
    
    def _initialize_client(self):
        import os
        try:
            import google.generativeai as genai
            
            api_key = None
            
            api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
            
            if not api_key and hasattr(config, 'GOOGLE_API_KEY'):
                api_key = config.GOOGLE_API_KEY
                
            if api_key and api_key in ["YOUR_GOOGLE_API_KEY", "your-google-api-key-here", "your_actual_api_key_here", ""]:
                api_key = None
            
            if not api_key:
                raise ValueError("Google API key not found")
            
            genai.configure(api_key=api_key)
            self.client = genai
            
        except ImportError:
            raise ImportError("Google Generative AI library not installed")
        except Exception as e:
            raise ValueError(f"Failed to initialize Gemini client: {e}")
    
    def generate(self, conv, max_n_tokens: int, temperature: float, top_p: float = 1.0):
        
        for attempt in range(self.max_retries):
            try:
                prompt = self._convert_conversation_to_prompt(conv)
                
                model = self.client.GenerativeModel(self.model_name)
                
                try:
                    generation_config = self.client.GenerationConfig(
                        max_output_tokens=max_n_tokens,
                        temperature=min(max(temperature, 0.0), 2.0),
                        top_p=min(max(top_p, 0.0), 1.0),
                    )
                except:
                    generation_config = {
                        "max_output_tokens": max_n_tokens,
                        "temperature": min(max(temperature, 0.0), 2.0),
                        "top_p": min(max(top_p, 0.0), 1.0),
                    }
                
                try:
                    response = model.generate_content(
                        prompt,
                        generation_config=generation_config,
                        safety_settings=self.safety_settings
                    )
                except Exception as gen_error:
                    if self.safety_settings and attempt == 0:
                        response = model.generate_content(
                            prompt,
                            generation_config=generation_config
                        )
                    else:
                        raise gen_error
                
                content = self._extract_content_robust(response)
                
                if content is None:
                    block_info = self._check_blocking_robust(response)
                    if block_info:
                        logger.warning(f"Gemini blocked content for {self.model_name} (attempt {attempt + 1}): {block_info}")
                        if attempt == self.max_retries - 1:
                            self.update_stats(error=True)
                            return f"$BLOCKED${block_info}"
                        time.sleep(getattr(config, 'RETRY_DELAY', 1) * (attempt + 1))
                        continue
                    else:
                        logger.warning(f"Gemini returned no content for {self.model_name} (attempt {attempt + 1})")
                        if attempt == self.max_retries - 1:
                            self.update_stats(error=True)
                            return "$EMPTY_RESPONSE$"
                        time.sleep(getattr(config, 'RETRY_DELAY', 1) * (attempt + 1))
                        continue
                
                self.update_stats(tokens_used=len(content.split()))
                return content
                
            except Exception as e:
                logger.error(f"Gemini API error for {self.model_name} (attempt {attempt + 1}): {e}")
                if "quota" in str(e).lower() or "rate" in str(e).lower():
                    time.sleep(getattr(config, 'RETRY_DELAY', 2) * (attempt + 1) * 2)
                    if attempt == self.max_retries - 1:
                        self.update_stats(error=True)
                        return "$RATE_LIMITED$"
                else:
                    if attempt == self.max_retries - 1:
                        self.update_stats(error=True)
                        return "$ERROR$"
                    time.sleep(getattr(config, 'RETRY_DELAY', 1) * (attempt + 1))
        
        return "$ERROR$"
    
    def _extract_content_robust(self, response):
        try:
            if hasattr(response, 'text') and response.text:
                return response.text.strip()
        except:
            pass
        
        try:
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        if hasattr(candidate.content.parts[0], 'text'):
                            text = candidate.content.parts[0].text
                            if text and text.strip():
                                return text.strip()
        except:
            pass
        
        try:
            if hasattr(response, 'parts') and response.parts:
                if hasattr(response.parts[0], 'text'):
                    text = response.parts[0].text
                    if text and text.strip():
                        return text.strip()
        except:
            pass
        
        return None
    
    def _check_blocking_robust(self, response):
        block_reasons = []
        
        try:
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                if hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
                    block_reasons.append(f"PROMPT_BLOCKED:{response.prompt_feedback.block_reason}")
        except:
            pass
        
        try:
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                    finish_reason_str = str(candidate.finish_reason)
                    if "SAFETY" in finish_reason_str or "BLOCK" in finish_reason_str:
                        block_reasons.append(f"FINISH_REASON:{finish_reason_str}")
                
                if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                    for rating in candidate.safety_ratings:
                        try:
                            category = str(rating.category) if hasattr(rating, 'category') else "UNKNOWN"
                            probability = str(rating.probability) if hasattr(rating, 'probability') else "UNKNOWN"
                            if "HIGH" in probability or "MEDIUM" in probability:
                                block_reasons.append(f"SAFETY:{category}:{probability}")
                        except:
                            block_reasons.append("SAFETY:PARSE_ERROR")
        except:
            pass
        
        return ";".join(block_reasons) if block_reasons else None
    
    def _convert_conversation_to_prompt(self, conv):
        
        if hasattr(conv, 'get_prompt'):
            return conv.get_prompt()
        
        elif hasattr(conv, 'messages'):
            messages = []
            system_instruction = ""
            
            for msg in conv.messages:
                if len(msg) >= 2:
                    role = msg[0]
                    content = msg[1] if msg[1] is not None else ""
                    
                    if role == "system":
                        system_instruction = content
                    elif role.lower() in ["user", "human"]:
                        messages.append(f"User: {content}")
                    elif role.lower() in ["assistant", "model"]:
                        messages.append(f"Assistant: {content}")
                    else:
                        messages.append(f"User: {content}")
            
            prompt = "\n".join(messages)
            if system_instruction:
                prompt = f"System: {system_instruction}\n\n{prompt}"
            return prompt
        
        elif isinstance(conv, list):
            messages = []
            system_instruction = ""
            
            for msg in conv:
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    
                    if role == "system":
                        system_instruction = content
                    elif role.lower() in ["user", "human"]:
                        messages.append(f"User: {content}")
                    elif role.lower() in ["assistant", "model"]:
                        messages.append(f"Assistant: {content}")
                    else:
                        messages.append(f"User: {content}")
                else:
                    messages.append(f"User: {str(msg)}")
            
            prompt = "\n".join(messages)
            if system_instruction:
                prompt = f"System: {system_instruction}\n\n{prompt}"
            return prompt
        
        elif isinstance(conv, str):
            return conv
        
        else:
            return str(conv)
    
    def batched_generate(self, convs_list, max_n_tokens: int,
                        temperature: float, top_p: float = 1.0, is_get_attention: bool = False):
        outputs = []
        for conv in convs_list:
            output = self.generate(conv, max_n_tokens, temperature, top_p)
            outputs.append(output)
            time.sleep(0.3)
        
        return outputs, [None] * len(outputs)

    def batched_generate_by_thread(self, convs_list, max_n_tokens: int,
                                  temperature: float, top_p: float = 1.0, is_get_attention: bool = False):
        
        def generate_with_delay(conv, delay):
            time.sleep(delay)
            return self.generate(conv, max_n_tokens, temperature, top_p)
        
        with ThreadPoolExecutor(max_workers=min(2, len(convs_list))) as executor:
            futures = []
            for i, conv in enumerate(convs_list):
                delay = i * 0.5
                future = executor.submit(generate_with_delay, conv, delay)
                futures.append(future)
            
            outputs = [future.result() for future in futures]
        
        return outputs, [None] * len(outputs)

class PaLM():
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    default_output = "I'm sorry, but I cannot assist with that request."
    API_KEY = os.getenv("PALM_API_KEY")

    def __init__(self, model_name) -> None:
        self.model_name = model_name
        palm.configure(api_key=self.API_KEY)

    def generate(self, conv: List,
                 max_n_tokens: int,
                 temperature: float,
                 top_p: float):
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                completion = palm.chat(
                    messages=conv,
                    temperature=temperature,
                    top_p=top_p
                )
                output = completion.last

                if output is None:
                    output = self.default_output
                else:
                    output = output[:(max_n_tokens*4)]
                break
            except Exception as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)

            time.sleep(1)
        return output

    def batched_generate(self,
                         convs_list: List[List[Dict]],
                         max_n_tokens: int,
                         temperature: float,
                         top_p: float = 1.0,):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]

class OpenSourceModelAPI(LanguageModel):
    API = config.OPEN_SOURCE_MODEL_API

    def __init__(self, model_name):
        self.model_name = model_name
        self.API = self.API + '/' + self.model_name.split("-")[0]

    def batched_generate(self, conv: List,
                         max_n_tokens: int,
                         temperature: float,
                         top_p: float,
                         is_get_attention: bool = False
                         ):
        response = requests.post(self.API, json={
            "full_prompts_list": conv,
            "max_tokens": max_n_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "is_get_attention": is_get_attention,
        })

        if response.status_code != 200:
            print('Request failed with status code:', response.status_code)
            return []

        return response.json()["output_list"], response.json()["token2attn"]

    def batched_generate_by_thread(self,
                                   convs_list: List[List[Dict]],
                                   max_n_tokens: int,
                                   temperature: float,
                                   top_p: float = 1.0,
                                   is_get_attention: bool = False):

        return self.batched_generate(convs_list, max_n_tokens, temperature, top_p, is_get_attention)

class CommercialAPI(LanguageModel):
    def __init__(self, model_name):
        self.model_name = model_name
        self.base_url = config.COMMERCIALAPI[model_name.upper()]["BASE_URL"]            
        self.api_key = config.COMMERCIALAPI[model_name.upper()]["API_KEY"]
        if "MODEL_ID" in config.COMMERCIALAPI[model_name.upper()].keys():
            self.model_id = config.COMMERCIALAPI[model_name.upper()]["MODEL_ID"]
        else:
            self.model_id = None

    def batched_generate(self, conv: List,
                            max_n_tokens: int,
                            temperature: float,
                            top_p: float,
                            is_get_attention: bool = False):
            api_key = self.api_key
    
            builder = ConvBuilder(self.model_name)
    
            request_body = builder.convert_conv(conv, temperature, max_n_tokens)
            
            if self.model_id is not None:
                request_body["model"] = self.model_id

            headers = {
                "Authorization": f"Bearer {api_key}",
                'Content-Type': 'application/json',
            }

            retry = 0
            while retry < 5:
                try:
                    response = requests.post(self.base_url,
                                            json=request_body,
                                            headers=headers,
                                            timeout=500,
                                        )
                    
                    if response.status_code != 200:
                        print('Request failed with status code:', response.status_code)
                        retry += 1
                        time.sleep(5)
                        continue
    
                    if not response.text:
                        print("empty response")
                        time.sleep(5)
                        retry += 1
                        continue
    
                    response = json.loads(response.text)
                    break
                except Exception as e:
                    retry += 1
                    time.sleep(5)
                    print("retrying... {}".format(e))
                    continue
            
            if response['status'] != "success":
                print('Request failed with status:', response)
                return "error"

            return response["content"]

    def batched_generate_by_thread(self,
                                      convs_list: List[List[Dict]],
                                      max_n_tokens: int,
                                      temperature: float,
                                      top_p: float = 1.0,
                                      is_get_attention: bool = False):
          threads = []
          results = []
          attentions = []
    
          def thread_func(conv, max_n_tokens, temperature, top_p):
                result = self.batched_generate(
                 conv, max_n_tokens, temperature, top_p)
                results.append(result)
    
          for conv in convs_list:
                thread = threading.Thread(target=thread_func, args=(
                 conv, max_n_tokens, temperature, top_p))
                
                time.sleep(1)
                threads.append(thread)
                thread.start()
    
          for thread in threads:
                thread.join()
    
          return results, attentions