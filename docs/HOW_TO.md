## Lightweight Online Adapter + RL Pipeline (How-To)

### 1) Environment
- Windows (cmd):
  - venv\Scripts\activate
  - pip install -r adapter_service/requirements-api.txt

### 2) Start the Adapter API
- Start server:
  - python -m uvicorn adapter_service.standalone_api:app --host 127.0.0.1 --port 8100 --reload
- Health check:
  - curl.exe http://127.0.0.1:8100/health

### 3) Train a Lite Adapter (via API)
- Start a short job (50 samples fast check):
  - python test_api.py
- Or directly (CMD-safe):
  - curl.exe -X POST http://127.0.0.1:8100/adapter/train-lite -H "Content-Type: application/json" -d "{\"source\":\"multilingual_corpus\",\"max_samples\":50}"
- Check status:
  - curl.exe http://127.0.0.1:8100/adapter/status/<job_id>
- Output is saved to `adapters/gurukul_lite`.

### 4) Generate with Adapter
- Example (CMD-safe JSON):
  - curl.exe -X POST http://127.0.0.1:8100/generate-lite -H "Content-Type: application/json" -d "{\"prompt\":\"Translate to Hindi. Output only Hindi: Hello friend.\",\"max_new_tokens\":64,\"temperature\":0.8,\"adapter_path\":\"adapters/gurukul_lite\",\"base_model\":\"gpt2\"}"

Notes:
- For tiny test adapters (50 samples) outputs may be weak; quality improves with 2k+ samples.
- If VRAM tight on RTX 4050, keep max_new_tokens ≤ 64.

### 5) RL Episode Collection (Scaffold)
- Local JSONL logging:
  - python rl/collect.py --out rl_runs/episodes.jsonl --env_name mcp-lite --max_episodes 3 \
    --prompt "Translate to Hindi. Output only Hindi: Hello friend." --adapter_path adapters/gurukul_lite
- Output: `rl_runs/episodes.jsonl` (one JSON per line with prompt, output, reward, latency).

### 6) Upload Episodes to Cloud (Pre-signed URL)
- After local collection, optionally upload the JSONL file:
  - python rl/collect.py --out rl_runs/episodes.jsonl --upload --upload_url "https://<presigned-put-url>"

### 7) Smoke Tests
- Runs 10 multilingual prompts and writes results:
  - python scripts/run_smoke_tests.py
- Output: `docs/smoke_results.md`

### 8) Full 2k-sample Adapter (Recommended Quality Pass)
- Edit `adapter_config.yaml` if needed (e.g., `training.max_train_samples: 2000`, `training.max_steps`).
- Start training via API with higher `max_samples` (and let the script internally respect training config):
  - curl.exe -X POST http://127.0.0.1:8100/adapter/train-lite -H "Content-Type: application/json" -d "{\"source\":\"multilingual_corpus\",\"max_samples\":2000}"

### 9) Troubleshooting
- "uvicorn not recognized":
  - Use: `python -m uvicorn ...` with the venv activated.
- Emoji or odd console characters on Windows:
  - Use plain ASCII logs and avoid emojis.
- JSON errors in CMD:
  - Use `curl.exe` and the exact escaped JSON shown above.

### 10) Files of Interest
- API: `adapter_service/standalone_api.py`
- Trainer (minimal): `adapter_service/minimal_train.py`
- Configs: `adapter_config.yaml`, `mcp_connectors.yml`
- RL scaffold: `rl/collect.py`
- Smoke: `scripts/run_smoke_tests.py`, results in `docs/smoke_results.md`


