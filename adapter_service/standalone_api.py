#!/usr/bin/env python3
"""
Standalone FastAPI wrapper for Lightweight Adapter Training
Provides REST endpoints for training and inference
"""

import os
import sys
import uuid
import asyncio
import logging
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
import yaml
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Lightweight Adapter Training API",
    description="REST API for multilingual adapter training and inference",
    version="1.0.0"
)

# Job storage (in production, use Redis or database)
job_storage: Dict[str, Dict[str, Any]] = {}
LOG_BUFFER_LIMIT = 5000  # keep last N lines per job

# Pydantic models for request/response
class TrainRequest(BaseModel):
    """Request model for training endpoint"""
    source: str = "multilingual_corpus"
    max_samples: int = 200
    config_path: str = "adapter_config.yaml"
    mcp_config_path: str = "mcp_connectors.yml"

class GenerateRequest(BaseModel):
    """Request model for generation endpoint"""
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(100, ge=1, le=512)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    adapter_path: Optional[str] = Field(None)
    base_model: str = Field("gpt2", min_length=1)
    do_sample: bool = True
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=0)
    repetition_penalty: float = Field(1.1, ge=0.0, le=10.0)

    @field_validator("prompt")
    @classmethod
    def prompt_not_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt must not be empty or whitespace only")
        return v

class JobStatus(BaseModel):
    """Response model for job status"""
    job_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: float  # 0.0 to 1.0
    message: str
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None

# Background task functions
async def run_training_job(job_id: str, train_request: TrainRequest):
    """Background task to run training job"""
    try:
        logger.info(f"Starting training job {job_id}")
        
        # Update job status
        job_storage[job_id]["status"] = "running"
        job_storage[job_id]["message"] = "Training started"
        job_storage[job_id].setdefault("logs", [])
        
        # Run training as a subprocess to avoid import-path issues
        import subprocess
        current_dir = os.path.dirname(os.path.dirname(__file__))
        # Use full trainer to allow large sample counts and full streaming config
        script_path = os.path.join(current_dir, "adapter_service", "train_adapt.py")
        cmd = [
            sys.executable,
            script_path,
            "--config", train_request.config_path,
            "--mcp-config", train_request.mcp_config_path,
            "--source", train_request.source,
            "--max-samples", str(train_request.max_samples)
        ]
        logger.info(f"Launching subprocess: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            cwd=current_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # Stream stdout line-by-line and update progress/logs
        total_expected = max(1, int(train_request.max_samples))
        sample_count = 0
        logs_ref = job_storage[job_id]["logs"]
        for raw_line in proc.stdout:  # type: ignore[attr-defined]
            line = raw_line.rstrip("\n")
            # buffer logs
            logs_ref.append(line)
            if len(logs_ref) > LOG_BUFFER_LIMIT:
                del logs_ref[: len(logs_ref) - LOG_BUFFER_LIMIT]
            # parse progress from known messages
            # e.g., "Collected 300 samples" or "Streamed 300/5000 samples (local/hf)"
            try:
                if "Collected" in line and "samples" in line:
                    # Collected 300 samples
                    parts = line.split()
                    for i, tok in enumerate(parts):
                        if tok.isdigit() and i + 1 < len(parts) and parts[i + 1].startswith("samples"):
                            sample_count = max(sample_count, int(tok))
                            break
                elif "Streamed" in line and "/" in line and "samples" in line:
                    # Streamed 300/5000 samples (...)
                    try:
                        seg = line.split("Streamed", 1)[1].strip().split()[0]
                        cur, tot = seg.split("/")
                        cur_i = int(cur)
                        tot_i = int(tot)
                        total_expected = max(total_expected, tot_i)
                        sample_count = max(sample_count, cur_i)
                    except Exception:
                        pass
                elif line.startswith("TRAIN_DONE"):
                    # TRAIN_DONE samples=N
                    try:
                        sample_count = int(line.split("samples=")[-1])
                    except Exception:
                        pass
                # update progress
                progress = min(1.0, float(sample_count) / float(total_expected))
                job_storage[job_id]["progress"] = progress
                job_storage[job_id]["message"] = f"Training running: {sample_count}/{total_expected} samples"
            except Exception:
                # ignore parse errors
                pass

        ret = proc.wait()
        if ret != 0:
            # include last lines for context
            snippet = " | ".join(logs_ref[-20:])
            raise RuntimeError(f"Subprocess failed (code {ret}). Logs: {snippet}")
        # If nothing parsed, fail the job with stdout tail for debugging
        if sample_count == 0:
            snippet = " | ".join(logs_ref[-20:])
            raise RuntimeError("No samples trained. Logs: " + snippet)
        
        # Update job status
        job_storage[job_id]["status"] = "completed"
        job_storage[job_id]["progress"] = 1.0
        job_storage[job_id]["message"] = f"Training completed successfully! Processed {sample_count} samples"
        job_storage[job_id]["completed_at"] = datetime.now().isoformat()
        
        logger.info(f"Training job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Training job {job_id} failed: {e}")
        job_storage[job_id]["status"] = "failed"
        job_storage[job_id]["error"] = str(e)
        job_storage[job_id]["message"] = f"Training failed: {str(e)}"

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Lightweight Adapter Training API",
        "version": "1.0.0",
        "endpoints": {
            "train": "POST /adapter/train-lite",
            "generate": "POST /generate-lite", 
            "status": "GET /adapter/status/{job_id}",
            "logs": "GET /adapter/logs/{job_id}",
            "logs_tail": "GET /adapter/logs/{job_id}/tail?n=200"
        }
    }

@app.post("/adapter/train-lite")
async def start_training(train_request: TrainRequest, background_tasks: BackgroundTasks):
    """Start a lightweight adapter training job"""
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Create job record
        job_storage[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "progress": 0.0,
            "message": "Job queued for training",
            "created_at": datetime.now().isoformat(),
            "request": train_request.dict(),
            "logs": []
        }
        
        # Start background training task
        background_tasks.add_task(run_training_job, job_id, train_request)
        
        logger.info(f"Started training job {job_id}")
        
        return {
            "job_id": job_id,
            "status": "pending",
            "message": "Training job started",
            "created_at": job_storage[job_id]["created_at"]
        }
        
    except Exception as e:
        logger.error(f"Failed to start training job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

@app.get("/adapter/status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a training job"""
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_storage[job_id]
    return JobStatus(**job)

@app.get("/adapter/logs/{job_id}")
async def get_job_logs(job_id: str, start: int = 0, limit: int = 500):
    """Fetch logs for a job with pagination (start-based)."""
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    logs = job_storage[job_id].get("logs", [])
    start_idx = max(0, int(start))
    end_idx = min(len(logs), start_idx + max(1, int(limit)))
    return {
        "job_id": job_id,
        "start": start_idx,
        "end": end_idx,
        "total": len(logs),
        "lines": logs[start_idx:end_idx],
    }

@app.get("/adapter/logs/{job_id}/tail")
async def get_job_logs_tail(job_id: str, n: int = 200):
    """Fetch the last N log lines for a job."""
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    logs = job_storage[job_id].get("logs", [])
    n = max(1, min(int(n), LOG_BUFFER_LIMIT))
    return {
        "job_id": job_id,
        "lines": logs[-n:],
        "total": len(logs),
    }

@app.post("/generate-lite")
async def generate_text(generate_request: GenerateRequest):
    """Generate text using trained adapters"""
    try:
        logger.info(f"Generating text with prompt: {generate_request.prompt[:50]}...")
        
        # Resolve adapter path (optional)
        adapter_path = None
        if generate_request.adapter_path and str(generate_request.adapter_path).lower() not in {"none", ""}:
            p = Path(generate_request.adapter_path)
            if not p.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Adapter not found at {generate_request.adapter_path}. Set adapter_path='none' to skip."
                )
            adapter_path = p
        # Lazy-import heavy libs inside endpoint
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        import torch

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(generate_request.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Safer for causal models (BLOOM) during batched/padded generation
        try:
            tokenizer.padding_side = "left"
        except Exception:
            pass

        # Resolve adapter directory; prefer latest checkpoint if present
        effective_adapter = adapter_path
        if adapter_path is not None:
            try:
                if adapter_path.is_dir():
                    checkpoints = [
                        (int(p.name.split("-")[-1]), p)
                        for p in adapter_path.iterdir()
                        if p.is_dir() and p.name.startswith("checkpoint-") and (p/"adapter_model.safetensors").exists()
                    ]
                    if checkpoints:
                        checkpoints.sort(key=lambda x: x[0])
                        effective_adapter = checkpoints[-1][1]
                        logger.info(f"Using adapter checkpoint: {effective_adapter}")
            except Exception as e:
                logger.warning(f"Could not resolve adapter checkpoint, falling back to root: {e}")

        # Prefer CPU for BLOOM (stability) and for small/base-only runs (GPT-2 or no adapter)
        base_lower = generate_request.base_model.lower()
        is_bloom = "bloom" in base_lower
        is_small_cpu_model = base_lower in {"gpt2", "distilgpt2"}
        no_adapter = effective_adapter is None
        prefer_cpu = is_bloom or is_small_cpu_model or no_adapter
        if prefer_cpu:
            model = AutoModelForCausalLM.from_pretrained(
                generate_request.base_model,
                load_in_8bit=False,
                device_map=None
            )
            if effective_adapter is not None:
                model = PeftModel.from_pretrained(model, str(effective_adapter))
            model = model.to("cpu")
        else:
            # Load base model in 8-bit to fit on 4050 and attach adapter
            model = AutoModelForCausalLM.from_pretrained(
                generate_request.base_model,
                load_in_8bit=True,
                device_map="auto"
            )
            if effective_adapter is not None:
                model = PeftModel.from_pretrained(model, str(effective_adapter))
        # Ensure pad token id is defined on the model config
        try:
            if getattr(model.config, "pad_token_id", None) is None:
                model.config.pad_token_id = tokenizer.pad_token_id
        except Exception:
            pass
        model.eval()
        if effective_adapter is not None:
            try:
                logger.info(f"PEFT config loaded: {getattr(model, 'peft_config', None)}")
            except Exception:
                pass

        # Prepare inputs
        inputs = tokenizer(
            generate_request.prompt,
            return_tensors="pt",
            padding=True
        )
        # Move tensors to model device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate with CUDA-safe retry fallback to CPU
        def _do_generate(active_model, force_greedy: bool = False):
            # For BLOOM on small VRAM, force greedy and a small min_new_tokens
            is_bloom = "bloom" in generate_request.base_model.lower()
            effective_do_sample = False if (force_greedy or is_bloom) else bool(generate_request.do_sample)
            # Encourage continuation for BLOOM to avoid echoing the prompt
            min_new = max(16, int(generate_request.max_new_tokens)) if is_bloom else min(8, int(generate_request.max_new_tokens))
            return active_model.generate(
                **inputs,
                max_new_tokens=int(generate_request.max_new_tokens),
                min_new_tokens=min_new,
                temperature=float(generate_request.temperature),
                do_sample=effective_do_sample,
                top_p=float(generate_request.top_p),
                top_k=int(generate_request.top_k),
                renormalize_logits=True,
                repetition_penalty=float(generate_request.repetition_penalty),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
                early_stopping=False
            )

        with torch.inference_mode():
            try:
                try:
                    output_ids = _do_generate(model)
                except ValueError as ve:
                    # e.g., probability tensor contains inf/nan; retry greedy
                    logger.warning(f"Sampling failed, retrying greedy: {ve}")
                    output_ids = _do_generate(model, force_greedy=True)
            except Exception as cuda_err:
                # Heuristic: if it's a CUDA/device-side assert, retry on CPU for a safe result
                if "CUDA" in str(cuda_err) or "device-side assert" in str(cuda_err):
                    logger.warning(f"CUDA generation failed, retrying on CPU: {cuda_err}")
                    try:
                        import gc
                        del model
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    # Reload base and adapter on CPU
                    cpu_base = AutoModelForCausalLM.from_pretrained(generate_request.base_model, device_map=None)
                    cpu_model = PeftModel.from_pretrained(cpu_base, str(adapter_path))
                    cpu_model.eval()
                    # Move inputs to CPU
                    inputs = {k: v.to("cpu") for k, v in inputs.items()}
                    try:
                        output_ids = _do_generate(cpu_model)
                    except ValueError as ve:
                        logger.warning(f"CPU sampling failed, retrying greedy: {ve}")
                        output_ids = _do_generate(cpu_model, force_greedy=True)
                else:
                    raise

        # Decode only the newly generated tokens to avoid prompt-stripping empty outputs
        try:
            input_len = inputs["input_ids"].shape[1]
            gen_only_ids = output_ids[0][input_len:]
            generated_text = tokenizer.decode(gen_only_ids, skip_special_tokens=True).strip()
            if not generated_text:
                # Fallback to whole decode if slicing produced empty
                generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        except Exception:
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        # Sanitize: remove non-printable characters and normalize whitespace
        def _sanitize(text: str) -> str:
            cleaned = ''.join(ch for ch in text if ch.isprintable())
            cleaned = cleaned.replace('\r', ' ').replace('\n', ' ')
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            return cleaned

        generated_text = _sanitize(generated_text)

        # Final fallback: if nothing generated, try base model without adapter once
        if not generated_text:
            try:
                from transformers import AutoModelForCausalLM
                base_only = AutoModelForCausalLM.from_pretrained(
                    generate_request.base_model,
                    load_in_8bit=False,
                    device_map=None
                ).to("cpu")
                base_inputs = tokenizer(generate_request.prompt, return_tensors="pt", padding=True)
                with torch.inference_mode():
                    base_ids = base_only.generate(
                        **{k: v.to("cpu") for k, v in base_inputs.items()},
                        max_new_tokens=int(generate_request.max_new_tokens),
                        min_new_tokens=8,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        early_stopping=False
                    )
                try:
                    input_len = base_inputs["input_ids"].shape[1]
                    gen_ids = base_ids[0][input_len:]
                    generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                except Exception:
                    generated_text = tokenizer.decode(base_ids[0], skip_special_tokens=True).strip()
                generated_text = _sanitize(generated_text)
            except Exception:
                pass

        return {
            "prompt": generate_request.prompt,
            "generated_text": generated_text,
            "adapter_path": generate_request.adapter_path,
            "parameters": {
                "max_new_tokens": generate_request.max_new_tokens,
                "temperature": generate_request.temperature,
                "base_model": generate_request.base_model
            }
        }
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/adapter/list")
async def list_adapters():
    """List available adapter directories under adapters/.
    A directory is considered an adapter if it contains adapter_model.safetensors or adapter_config.json.
    """
    root = Path("adapters")
    items = []
    if root.exists():
        for child in root.iterdir():
            if child.is_dir():
                has_files = any((child / fname).exists() for fname in [
                    "adapter_model.safetensors", "adapter_config.json"
                ])
                if has_files:
                    items.append({
                        "name": child.name,
                        "path": str(child),
                    })
    return {"adapters": items}

@app.post("/rl/collect")
async def collect_episode(request: Dict[str, Any]):
    """Collect RL episode and log to cloud/NAS"""
    try:
        # Extract episode data
        prompt = request.get("prompt", "")
        output = request.get("output", "")
        reward = request.get("reward", 0.0)
        run_id = request.get("run_id", str(uuid.uuid4()))
        env_name = request.get("env_name", "api-collection")
        
        # Create episode record
        episode = {
            "run_id": run_id,
            "episode_index": 0,  # Single episode
            "timestamp": time.time(),
            "env_name": env_name,
            "prompt": prompt,
            "output": output,
            "reward": reward,
            "latency_s": 0.0,  # Not measured in API
            "meta": {
                "source": "api",
                "model_type": "bloomz-560m",
                "language": "multilingual"
            }
        }
        
        # Log to local file
        output_path = Path("rl_runs/api_episodes.jsonl")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(episode, ensure_ascii=False) + "\n")
        
        logger.info("RL episode collected: reward=%.3f len=%d", reward, len(output))
        
        return {
            "status": "success",
            "episode_id": run_id,
            "reward": reward,
            "output_path": str(output_path)
        }
        
    except Exception as e:
        logger.error("RL collection error: %s", e)
        raise HTTPException(status_code=500, detail=f"RL collection failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_jobs": len([job for job in job_storage.values() if job["status"] == "running"])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
