# Sovereign LM → KSML Smoke Test Results (2025-10-29)

## Environment
- **OS**: Windows 10
- **Python**: 3.11.2 (venv)
- **Servers**:
  - LM Core: `127.0.0.1:8000` (`lm_core_adapter.app:app`)
  - Sovereign KSML API: `127.0.0.1:8116` (`sovereign_core.api:app`)
- **Config**:
  - **GROQ**: configured (live)
  - **MongoDB Atlas**: configured (auth OK)
  - **Qdrant Cloud**: reachable; collection `documents` exists; count = 0
  - **Vaani**: not configured (text-only path expected)

---

## Test 1 — Bhavesh `/compose.final_text` (top_k = 0)
- What we’re testing: Direct LM Core call with retrieval disabled.
- INPUT
```http
POST http://127.0.0.1:8000/compose.final_text
Content-Type: application/json

{"query":"Hello","language":"en","top_k":0,"context":[]}
```
- OUTPUT (200)
```json
{"final_text":"Hello","vaani_audio":{"error":"Vaani credentials not configured in environment (VAANI_USERNAME/VAANI_PASSWORD)"}}
```
- Result: **PASS**

---

## Test 2 — Bhavesh `/compose.final_text` (top_k = 3)
- What we’re testing: Direct LM Core call with retrieval enabled.
- INPUT
```http
POST http://127.0.0.1:8000/compose.final_text
Content-Type: application/json

{"query":"Explain gravity in simple words","language":"en","top_k":3,"context":[]}
```
- OUTPUT (200, snippet)
```json
{"final_text":"Gravity is a force that pulls everything towards each other. It's what keeps you on the ground and what makes things fall down. The Earth is pulling on you, and you are pulling on the Earth, but the Earth is much heavier, so you don't feel like you're pulling it. That's why you don't float off into space when you're standing on the Earth. Gravity is also what keeps the planets in our solar system moving around the sun, and what holds the atmosphere in place around the Earth.","vaani_audio":{"error":"Vaani credentials not configured in environment (VAANI_USERNAME/VAANI_PASSWORD)"}}
```
- Result: **PASS** (GROQ live; Qdrant reachable; no retrieved chunks yet because count = 0)

---

## Test 3 — Sovereign `/bridge.reason` (LM → KSML)
- What we’re testing: Our end-to-end pipeline (LM Core → KSML alignment) via our public API.
- INPUT
```http
POST http://127.0.0.1:8116/bridge.reason
Content-Type: application/json

{"text":"Explain gravity in simple words","user_id":"u1","session_id":"s1","include_audio":false}
```
- OUTPUTS (200)
  - Run A — processing_time ≈ 2.18s
  ```json
  {"aligned_text":"Gravity is a force that pulls everything towards each other. ...","ksml_metadata":{"intent":"question","source_lang":"en","target_lang":"en","karma_state":"sattva","semantic_roots":[],"confidence":0.7},"speech_ready":null,"processing_time":2.1831,"trace_id":"bridge_..."}
  ```
  - Run B — processing_time ≈ 1.11s
  ```json
  {"aligned_text":"Gravity is a force that pulls everything towards each other. ...","ksml_metadata":{"intent":"question","source_lang":"en","target_lang":"en","karma_state":"sattva","semantic_roots":[],"confidence":0.7},"speech_ready":null,"processing_time":1.1106,"trace_id":"bridge_..."}
  ```
  - Run C — processing_time ≈ 1.08s
  ```json
  {"aligned_text":"Gravity is a force that pulls everything towards each other. ...","ksml_metadata":{"intent":"question","source_lang":"en","target_lang":"en","karma_state":"sattva","semantic_roots":[],"confidence":0.7},"speech_ready":null,"processing_time":1.0773,"trace_id":"bridge_..."}
  ```
- Result: **PASS**

---

## Qdrant Health
- What we tested: Cloud availability and point count.
- INPUT
```http
POST https://<cluster>/collections/documents/points/count
Content-Type: application/json

{"filter":null,"exact":true}
```
- OUTPUT
```json
{"result":{"count":0},"status":"ok"}
```
- Impact: Retrieval path functional; returns no context until data is ingested.

## MongoDB Health
- Atlas URI valid; telemetry writes are tolerant and non-blocking.

## Vaani Status
- Not configured; responses include `vaani_audio.error`; text path unaffected.

---

## Additional Live Samples (2025-10-29 21:14 IST)

### LM Core — `/compose.final_text`
- INPUT top_k=0, query: `Hello`
```json
{"final_text":"Hello", "vaani_audio":{"error":"Vaani credentials not configured in environment (VAANI_USERNAME/VAANI_PASSWORD)"}}
```
- INPUT top_k=0, query: `Translate 'Namaste' to English`
```json
{"final_text":"'Namaste' translates to \"I bow to you\" or \"Greetings\" or \"Hello\" or \"Respect to you\" in English.","vaani_audio":{"error":"Vaani credentials not configured in environment (VAANI_USERNAME/VAANI_PASSWORD)"}}
```
- INPUT top_k=0, query: `What is photosynthesis?`
```json
{"final_text":"Photosynthesis is the process by which plants, algae, and some bacteria convert light energy ...","vaani_audio":{"error":"Vaani credentials not configured in environment (VAANI_USERNAME/VAANI_PASSWORD)"}}
```
- INPUT top_k=3, query: `Explain gravity in simple words`
```json
{"final_text":"Gravity is a force that pulls everything towards each other. It's what keeps you on the ground ...","vaani_audio":{"error":"Vaani credentials not configured in environment (VAANI_USERNAME/VAANI_PASSWORD)"}}
```
- INPUT top_k=3, query: `What is photosynthesis?`
```json
{"final_text":"Photosynthesis is the process by which plants, algae, and some bacteria convert light energy ... 6CO2 + 6H2O + light energy → C6H12O6 (glucose) + 6O2","vaani_audio":{"error":"Vaani credentials not configured in environment (VAANI_USERNAME/VAANI_PASSWORD)"}}
```

### KSML Bridge — `/bridge.reason`
- INPUT: `Explain gravity in simple words`
```json
{"aligned_text":"Gravity is a force that pulls everything towards each other. ...","ksml_metadata":{"intent":"question","source_lang":"en","target_lang":"en","karma_state":"sattva","semantic_roots":[],"confidence":0.7},"speech_ready":null,"processing_time":1.2983}
```
- INPUT: `What is photosynthesis?`
```json
{"aligned_text":"Photosynthesis is the process by which plants, algae, and some bacteria convert light energy ...","ksml_metadata":{"intent":"question","source_lang":"en","target_lang":"en","karma_state":"rajas","semantic_roots":[],"confidence":0.8},"speech_ready":null,"processing_time":0.6698}
```

---

## Summary
- LM Core → KSML pipeline is live and stable across multiple prompts.
- End-to-end latency for `/bridge.reason`: **~0.67s to 2.18s**.
- Next steps:
  1) Ingest documents into Qdrant to enable retrieval with `top_k > 0`.
  2) Optionally add Vaani credentials for audio.
