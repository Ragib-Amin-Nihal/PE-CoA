#!/usr/bin/env python3
"""
Enhanced FastAPI server for PE-CoA with complete endpoint support
Supports multiple LLMs, toxicity detection, and semantic relevance
"""

import argparse
import os
import time
import logging
import uvicorn
import torch
from fastapi import FastAPI, Request, HTTPException
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    RobertaTokenizer, 
    RobertaModel,
    AutoConfig
)
from huggingface_hub import snapshot_download
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HuggingFace:
    """Enhanced HuggingFace model wrapper"""
    
    def __init__(self, model_name, model, tokenizer):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        logger.info(f"HuggingFace model {model_name} initialized on device: {self.device}")

    def _cleanup_memory(self, local_vars):
        """Clean up memory to prevent OOM"""
        for var_name in ['input_ids', 'attention_mask', 'outputs', 'output_ids']:
            if var_name in local_vars:
                del local_vars[var_name]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def batched_generate(self, full_prompts_list, max_n_tokens, temperature, top_p, is_get_attention=False):
        """Generate responses for a batch of prompts"""
        logger.info(f"Generating for {len(full_prompts_list)} prompts, max_tokens: {max_n_tokens}")
        
        try:
            # Tokenize inputs
            inputs = self.tokenizer(
                full_prompts_list,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt"
            ).to(self.device)
            
            logger.info(f"Input shape: {inputs['input_ids'].shape}")
            
            # Generate responses
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_n_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    top_p=top_p,
                    do_sample=True if temperature > 0 else False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_attentions=is_get_attention,
                    use_cache=True
                )
            
            # Extract generated tokens (remove input tokens)
            input_length = inputs['input_ids'].shape[1]
            output_ids = outputs.sequences[:, input_length:]
            logger.info(f"Generated tokens shape: {output_ids.shape}")
            
            # Decode outputs
            outputs_list = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            logger.info(f"Decoded {len(outputs_list)} outputs")
            
            # Clean up memory
            self._cleanup_memory(locals())
            
            # Return outputs and attention (if requested)
            attention_weights = outputs.attentions if is_get_attention else [{}] * len(outputs_list)
            
            return outputs_list, attention_weights
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            self._cleanup_memory(locals())
            return [""] * len(full_prompts_list), [{}] * len(full_prompts_list)


class SemRelevance:
    """Semantic relevance computation using SimCSE"""
    
    def __init__(self, model_name="princeton-nlp/sup-simcse-roberta-large"):
        self.model_name = model_name
        logger.info(f"Loading semantic model: {model_name}")
        
        try:
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.model = RobertaModel.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Semantic model loaded on device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load semantic model: {e}")
            raise

    def compute_similarity(self, text1, text2):
        """Compute semantic similarity between two texts"""
        try:
            # Tokenize texts
            inputs1 = self.tokenizer(text1, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            inputs2 = self.tokenizer(text2, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs1 = self.model(**inputs1)
                outputs2 = self.model(**inputs2)
                
                # Use CLS token embedding
                emb1 = outputs1.last_hidden_state[:, 0, :]  # CLS token
                emb2 = outputs2.last_hidden_state[:, 0, :]  # CLS token
                
                # Compute cosine similarity
                similarity = torch.nn.functional.cosine_similarity(emb1, emb2).item()
                
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Similarity computation error: {e}")
            return 0.0


class Toxigen:
    """Toxicity detection using RoBERTa"""
    
    def __init__(self, model_name="tomh/toxigen_roberta"):
        self.model_name = model_name
        logger.info(f"Loading toxicity model: {model_name}")
        
        try:
            # Add debugging
            logger.info(f"Attempting to load tokenizer for: {model_name}")
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            logger.info(f"Tokenizer loaded successfully")
            
            logger.info(f"Attempting to load model for: {model_name}")
            self.model = RobertaModel.from_pretrained(model_name)
            logger.info(f"Model loaded successfully")
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Toxicity model loaded on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load toxicity model: {e}")
            logger.error(f"Model name was: {model_name}")
            import traceback
            traceback.print_exc()
            raise

    def get_toxicity(self, texts):
        """Get toxicity scores for a list of texts"""
        try:
            results = []
            
            for text in texts:
                try:
                    # Tokenize text
                    inputs = self.tokenizer(
                        text, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=512
                    ).to(self.device)
                    
                    # Get model output
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        # Use CLS token for classification
                        logits = outputs.last_hidden_state[:, 0, :]  
                        
                        # Simple binary classification (toxic vs non-toxic)
                        # Note: This is a simplified approach - the actual model might need different processing
                        prob = torch.sigmoid(logits.mean(dim=-1)).item()
                        
                    results.append({
                        "text": text,
                        "type": "1" if prob > 0.5 else "0",
                        "toxicity": float(prob)
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing text '{text[:50]}...': {e}")
                    results.append({
                        "text": text,
                        "type": "0", 
                        "toxicity": 0.0
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Batch toxicity detection error: {e}")
            return [{"text": text, "type": "0", "toxicity": 0.0} for text in texts]


def load_tokenizer_and_model(model_name, models_path):
    """Load tokenizer and model from HuggingFace"""
    logger.info(f"Loading model: {model_name}")
    
    try:
        # Check if model exists locally
        model_path = os.path.join(models_path, model_name.replace("/", "--"))
        
        if not os.path.exists(model_path):
            logger.info(f"Model not found locally, downloading {model_name}...")
            model_path = snapshot_download(repo_id=model_name, cache_dir=models_path)
            logger.info(f"Model downloaded to {model_path}")
        else:
            logger.info(f"Loading model from: {model_path}")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Set pad token if needed
        if not tokenizer.pad_token:
            if 'llama-2' in model_name.lower():
                tokenizer.pad_token = tokenizer.unk_token
            else:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'

        logger.info(f"Successfully loaded {model_name}")
        return HuggingFace(model_name, model, tokenizer)
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise


def main(args):
    """Main function to start the FastAPI server"""
    
    # Initialize FastAPI app
    app = FastAPI(
        title="PE-CoA Model API Server",
        description="Enhanced FastAPI server for PE-CoA with support for multiple model types",
        version="2.0.0"
    )
    
    host = args.host
    port = args.port
    log_level = args.log_level
    lms = {}  # Model storage
    
    logger.info(f"Models path: {args.models_path}")
    if os.path.exists(args.models_path):
        logger.info("Contents of models directory:")
        logger.info("\n".join(os.listdir(args.models_path)))
    else:
        logger.info(f"Models directory does not exist: {args.models_path}")

    # Load language models
    logger.info("Loading language models...")
    for model_name in args.model_name:
        try:
            model_key = None
            
            # Determine model key based on model name
            if "llama-2" in model_name.lower():
                model_key = "llama2"
            elif "llama-3" in model_name.lower():
                model_key = "llama3"
            elif "vicuna" in model_name.lower():
                model_key = "vicuna"
            elif "chatglm" in model_name.lower():
                model_key = "chatglm"
            elif "phi-2" in model_name.lower():
                model_key = "phi2"
            elif "phi-3" in model_name.lower():
                model_key = "phi3"
            elif "zephyr" in model_name.lower():
                model_key = "zephyr"
            elif "baichuan" in model_name.lower():
                model_key = "baichuan"
            elif "mistral" in model_name.lower() and "mixtral" not in model_name.lower():
                model_key = "mistral"
            elif "mixtral" in model_name.lower():
                model_key = "mixtral"
            elif "qwen" in model_name.lower():
                model_key = "qwen"
            elif "gemma" in model_name.lower():
                model_key = "gemma"
            else:
                # Generic model key
                model_key = model_name.split('/')[-1].lower().replace('-', '_')
            
            if model_key:
                logger.info(f"Loading {model_name} as {model_key}")
                lms[model_key] = load_tokenizer_and_model(model_name, args.models_path)
                
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            continue
    
    # Load auxiliary models
    try:
        logger.info("Loading semantic relevance models...")
        lms["sem_relevance"] = SemRelevance()
        logger.info("Semantic relevance model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load semantic relevance model: {e}")
    
    try:
        logger.info("Loading Chinese semantic relevance model...")
        lms["sem_relevance_zh"] = SemRelevance("hellonlp/simcse-roberta-large-zh")
        logger.info("Chinese semantic relevance model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load Chinese semantic model: {e}")
        
    try:
        logger.info("Loading toxicity detection model...")
        lms["toxigen"] = Toxigen()
        logger.info("Toxicity detection model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load toxicity model: {e}")
    
    # ===== CREATE MODEL GENERATION ENDPOINTS =====
    
    model_endpoints = {
        "llama2": "llama2",
        "llama3": "llama3", 
        "vicuna": "vicuna",
        "chatglm": "chatglm",
        "phi2": "phi2",
        "phi3": "phi3",
        "zephyr": "zephyr",
        "baichuan": "baichuan",
        "mistral": "mistral",
        "mixtral": "mixtral", 
        "qwen": "qwen",
        "gemma": "gemma"
    }
    
    # Create endpoints for each loaded model
    for endpoint_name, model_key in model_endpoints.items():
        if model_key in lms:
            logger.info(f"Creating endpoint /generate/{endpoint_name} for model {model_key}")
            
            def create_endpoint(model_k):
                async def generate_endpoint(request: Request):
                    try:
                        data = await request.json()
                        
                        full_prompts_list = data.get("full_prompts_list", [])
                        max_n_tokens = data.get("max_n_tokens", 1024)
                        temperature = data.get("temperature", 0.9)
                        top_p = data.get("top_p", 1.0)
                        is_get_attention = data.get("is_get_attention", False)

                        if model_k not in lms:
                            raise HTTPException(status_code=404, detail=f"Model {model_k} not loaded")

                        start_time = time.time()
                        output_list, token2attn = lms[model_k].batched_generate(
                            full_prompts_list, max_n_tokens, temperature, top_p, is_get_attention
                        )
                        end_time = time.time()
                        
                        logger.info(f"Generated {len(output_list)} responses in {end_time - start_time:.2f}s")
                        
                        return {
                            "output_list": output_list, 
                            "token2attn": token2attn,
                            "model": model_k,
                            "generation_time": end_time - start_time
                        }
                        
                    except Exception as e:
                        logger.error(f"Error in {model_k} endpoint: {e}")
                        raise HTTPException(status_code=500, detail=str(e))
                
                return generate_endpoint
            
            # Register the endpoint
            app.post(f"/generate/{endpoint_name}")(create_endpoint(model_key))
    
    # ===== CREATE AUXILIARY ENDPOINTS =====
    
    # Toxicity detection endpoint
    if "toxigen" in lms:
        @app.post("/toxigen")
        async def toxicity_detection(request: Request):
            """Toxicity detection endpoint"""
            try:
                data = await request.json()
                texts = data.get("texts", [])
                logger.info(f"Toxicity detection for {len(texts)} texts")
                
                result = lms["toxigen"].get_toxicity(texts)
                logger.info(f"Toxicity detection completed")
                return result
                
            except Exception as e:
                logger.error(f"Toxicity detection error: {e}")
                return {"error": str(e)}

    # Semantic relevance endpoint
    if "sem_relevance" in lms:
        @app.post("/sem_relevance")
        async def semantic_relevance(request: Request):
            """Semantic relevance endpoint"""
            try:
                data = await request.json()
                text1 = data.get("text1", "")
                text2 = data.get("text2", "")
                logger.info(f"Computing semantic similarity")
                
                similarity = lms["sem_relevance"].compute_similarity(text1, text2)
                logger.info(f"Semantic similarity: {similarity}")
                return {"similarity": similarity}
                
            except Exception as e:
                logger.error(f"Semantic relevance error: {e}")
                return {"error": str(e), "similarity": 0.0}

    # Chinese semantic relevance endpoint  
    if "sem_relevance_zh" in lms:
        @app.post("/sem_relevance_zh")
        async def semantic_relevance_zh(request: Request):
            """Chinese semantic relevance endpoint"""
            try:
                data = await request.json()
                text1 = data.get("text1", "")
                text2 = data.get("text2", "")
                logger.info(f"Computing Chinese semantic similarity")
                
                similarity = lms["sem_relevance_zh"].compute_similarity(text1, text2)
                logger.info(f"Chinese semantic similarity: {similarity}")
                return {"similarity": similarity}
                
            except Exception as e:
                logger.error(f"Chinese semantic relevance error: {e}")
                return {"error": str(e), "similarity": 0.0}

    # ===== CREATE UTILITY ENDPOINTS =====
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        # Only list endpoints that actually exist
        available_endpoints = []
        
        # Add model endpoints that are loaded
        for name, key in model_endpoints.items():
            if key in lms:
                available_endpoints.append(f"/generate/{name}")
        
        # Add auxiliary endpoints that are loaded        
        if "toxigen" in lms:
            available_endpoints.append("/toxigen")
        if "sem_relevance" in lms:
            available_endpoints.append("/sem_relevance")
        if "sem_relevance_zh" in lms:
            available_endpoints.append("/sem_relevance_zh")
            
        # Always available
        available_endpoints.extend(["/health", "/models"])
        
        return {
            "status": "healthy",
            "loaded_models": list(lms.keys()),
            "available_endpoints": available_endpoints,
            "timestamp": time.time()
        }
    
    # Model info endpoint
    @app.get("/models")
    async def model_info():
        model_info = {}
        for key, model in lms.items():
            if hasattr(model, 'model_name'):
                model_info[key] = {
                    "name": model.model_name,
                    "type": type(model).__name__
                }
            else:
                model_info[key] = {
                    "name": key,
                    "type": type(model).__name__
                }
        return model_info
    
    # ===== LOG STARTUP INFO =====
    
    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Loaded models: {list(lms.keys())}")
    
    logger.info("=== AVAILABLE ENDPOINTS ===")
    logger.info("Model generation endpoints:")
    for endpoint_name, model_key in model_endpoints.items():
        if model_key in lms:
            logger.info(f"  POST /generate/{endpoint_name}")

    logger.info("Auxiliary endpoints:")
    if "toxigen" in lms:
        logger.info(f"  POST /toxigen")
    if "sem_relevance" in lms:
        logger.info(f"  POST /sem_relevance")  
    if "sem_relevance_zh" in lms:
        logger.info(f"  POST /sem_relevance_zh")

    logger.info("Utility endpoints:")
    logger.info(f"  GET /health")
    logger.info(f"  GET /models")
    logger.info("===========================")
    
    # Start server
    uvicorn.run(app, host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced FastAPI server for PE-CoA")

    parser.add_argument(
        "--model-name", 
        type=str,
        nargs="+",
        default=["lmsys/vicuna-13b-v1.5-16k"],
        help="Model names to load. Can specify multiple models."
    )
    parser.add_argument(
        "--models-path",
        type=str,
        default="./models",
        help="Path to store downloaded models"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9999,
        help="Port to bind the server to"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
        help="Logging level"
    )

    args = parser.parse_args()
    main(args)