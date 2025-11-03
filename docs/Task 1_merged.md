**Soham** **Kotkar** **–** **Multilingual** **Tokenization** **&**
**Model** **Integration** **(56** **hours)**

Goal:

Build and integrate a multilingual tokenizer and inferenceAPI for Hindi,
Sanskrit, Marathi, and English, ensuring smooth plug-in with the Black
Hole DB and Vaani (Blackhole TTS) systems. By Day 7, the language model
must process user input in any of the four languages and output
grammatically correct text in the chosen language, without any external
knowledge baked in.

**Day-by-Day** **Plan**

**Day** **1** **–** **Understanding** **&** **Setup** **(8** **hrs)**

> · Review Hugging Face transformers docs for tokenizer customization
> and multilingual model integration.
>
> · Study how tokenization impacts different scripts (Devanagari vs
> Latin).
>
> · Learning Resources:
>
> ◦ [<u>Hugging Face Tokenizers
> Course</u>](https://huggingface.co/course/chapter6)
>
> ◦ YouTube: “Custom Tokenizers with HuggingFace” – by AssemblyAI (15
> min)
>
> ◦ Wikipedia: Devanagari script rules for Hindi, Sanskrit, Marathi.
>
> · Install and test tokenizers, sentencepiece, transformers.

**Day** **2** **–** **Multilingual** **Tokenizer** **Creation** **(8**
**hrs)**

> · Create a SentencePiece tokenizer that supports all 4 languages with
> proper handling of ligatures.
>
> · Ensure proper Unicode normalization for Devanagari.
>
> · Train tokenizer on provided parallel corpora (clean text in 4
> languages).
>
> · Save vocab and merge files in a sharable format.
>
> · Learning Resource: [<u>Google SentencePiece
> GitHub</u>](https://github.com/google/sentencepiece)

**Day** **3** **–** **Base** **Model** **Integration** **(8** **hrs)**

> · Select a light, open-source decoder-only model (e.g., GPT-NeoX,
> BLOOM-560M) with no domain knowledge.
>
> · Replace its tokenizer with the custom multilingual tokenizer.
>
> · Test with dummy sentences in all 4 languages.
>
> · Verify that tokenization + detokenization is lossless.

**Day** **4** **–** **Language** **Identification** **&** **Routing**
**(8** **hrs)**

> · Implement a simple language detection module (fastText or
> langdetect).
>
> · Route detected language to correct tokenizer encoding flow.
>
> · Begin designing an inference wrapper to acceptAPI calls from KB.
>
> · Learning Resource: [<u>fastText Language
> Identification</u>](https://fasttext.cc/docs/en/language-identification.html)

**Day** **5** **–** **API** **Development** **&** **Integration**
**Hooks** **(8** **hrs)**

> · Build RESTAPI endpoints for:
>
> ◦ /tokenize – returns token IDs.
>
> ◦ /generate – returns generated text.
>
> ◦ /language-detect – returns detected language.
>
> · Ensure stateless design so KB can call it anytime.
>
> · Test with Gurukul sample queries in all 4 languages.

**Day** **6** **–** **Fine-Tuning** **&** **RLHF** **Prep** **(8**
**hrs)**

> · Apply small fine-tuning to improve grammar and sentence structure
> using bilingual corpora.
>
> · Prepare model for RLHF phase (hand over to Abhishek for reward model
> connection).
>
> · Test switching between languages mid-conversation.

**Day** **7** **–** **Full** **Integration** **&** **QA** **(8**
**hrs)**

> · ConnectAPI with Alpha/Core knowledge base mock endpoint.
>
> · Test multilingual Q&A loop:
>
> ◦ User → Multilingual LM → KB → LM → Response in user language.
>
> · Fix tokenization errors, latency issues.
>
> · Deliver full code + integration docs.

**Deliverables** **by** **Day** **7**

> · Multilingual tokenizer (Hindi, Sanskrit, Marathi, English).
>
> · Integrated base LM (decoder-only) with tokenizer.
>
> · RESTAPI for inference and tokenization.
>
> · Language detection and routing module.
>
> · Tested integration with Gurukul/Uniguru KB endpoints.

Contact Details for Integration with other members of this task will be
provided on day 4 .

**Learning** **&** **References**

> · ModelArchitecture Basics: [<u>YouTube – The Illustrated
> Transformer</u>](https://www.youtube.com/watch?v=4Bdc55j80l8)
>
> · Fine-tuning Hugging Face models: [<u>HF Course – Fine
> Tuning</u>](https://huggingface.co/course/chapter3)
>
> · Indic NLP Library:
> [<u>GitHub</u>](https://github.com/anoopkunchukuttan/indic_nlp_library)
>
> · AI4Bharat Corpora: [<u>ai4bharat.org</u>](https://ai4bharat.org/)
>
> · Language Detection: [<u>fastText Language
> ID</u>](https://fasttext.cc/docs/en/language-identification.html)
>
> · Dockerizing NLP Models: [<u>Docker + FastAPI
> Guide</u>](https://fastapi.tiangolo.com/deployment/docker/)

(Note: if links are broken please GPT the terms for more links.)

**Review** **of** **Soham** **Kotkar’s** **Task** **Submission**

What’s done well:

> · Tokenizer Creation: SentencePiece tokenizer handles Hindi, Sanskrit,
> Marathi, and English; Unicode normalization for Devanagari applied.
>
> · Model Integration: Light decoder-only model (GPT-NeoX/BLOOM-560M)
> successfully swapped with custom multilingual tokenizer.
>
> · Language Routing: Basic language detection + routing in place
> (fastText/langdetect).
>
> · API Endpoints: /tokenize, /generate, /language-detect implemented
> and tested on sample Gurukul queries.
>
> · Stateless & Modular Design: Good for KB integration and future
> scaling.

Gaps /Areas for Improvement:

> · Corpus Limitation: Currently limited to 4 languages;AI4Bharat
> corpora not loaded → need more robust Indic datasets for real-world
> usability.
>
> · Evaluation Metrics Missing: No BLEU/ROUGE/MOS-proxy or human
> evaluation reported for grammar or tokenization quality.
>
> · RLHF Prep: Ready, but no real reward model connection yet.
>
> · Integration Hooks: Endpoint not yet integrated with full KB / Vaani
> / TTV pipeline.

Score: 8/10 – solid technical base, missing scale & evaluation for
production-level deployment.

**Next** **Step** **/** **TaskAssignment**

Task Name: Indic Multilingual Expansion & MCPTraining

Objective: Extend Soham’s tokenizer + LM to cover at least 20 Indian
languages, make the LM robust for real datasets, and prepare it for
large-scale Gurukul integration.

Task Breakdown:

> 1\. Language Expansion
>
> ◦ Add 16+ additional Indian languages (Tamil, Telugu, Kannada,
> Bengali, Gujarati, Punjabi, Odia, Malayalam, Assamese, Marathi
> dialects, etc.).
>
> ◦ Ensure proper Unicode normalization & sentencepiece merges.
>
> ◦ Collect clean corpora: Wikipedia dumps, Indic NLP corpora, open
> datasets from AI4Bharat, HindMono, CC-100, OSCAR, or other public
> sources.
>
> 2\. MCP (Multi-Corpus Preprocessing) Training
>
> ◦ Preprocess corpora for consistent tokenization across scripts.
>
> ◦ Train multilingual tokenizer on combined dataset (SentencePiece).
>
> ◦ Save vocab & merges in sharable format.
>
> 3\. LM Fine-tuning on Real Datasets
>
> ◦ Fine-tune decoder-only LM (BLOOM/NeoX) using MCP datasets.
>
> ◦ Validate grammar + sentence fluency for each language.
>
> ◦ Ensure language switching mid-conversation works reliably.
>
> 4\. Integration Prep
>
> ◦ EnsureAPI endpoints accept 20+ languages.
>
> ◦ Prepare wrapper for /generate and /tokenize to feed Vaani TTS.
>
> ◦ Make output fully compatible with Indigenous NLP composer (Nisarg)
> and Vaani TTS (Karthikeya).
>
> 5\. Evaluation & QA
>
> ◦ Automatic evaluation: BLEU/ROUGE, perplexity, tokenization accuracy.
>
> ◦ Manual checks for fluency in 5–10 prompts per language.
>
> ◦ Latency checks to ensureAPI scales for multiple requests
> concurrently.

Deliverables:

> · Multilingual tokenizer for ≥20 Indian languages.
>
> · Fine-tuned LM compatible with new tokenizer.
>
> · RESTAPI for tokenization, generation, and language detection.
>
> · Integration guide for linking with Indigenous NLP + Vaani TTS.
>
> · Evaluation report (automatic + manual).

**Recommendations** **for** **Robust** **System** **Design**

> 1\. Scalability:
>
> ◦ Dockerize tokenizer + LM for cloud deployment; scale horizontally
> using FastAPI + GPU allocation.
>
> ◦ Integrate caching for repeated tokenization/generation to reduce
> latency.
>
> 2\. Extensibility:
>
> ◦ Modular architecture → new languages added easily without retraining
> full LM.
>
> ◦ Separate preprocessing, tokenization, and inference modules.
>
> 3\. Data Pipeline Robustness:
>
> ◦ Ensure cleaning of scripts, remove noise, handle transliterations,
> diacritics, and ligatures.
>
> ◦ Automate corpora ingestion from open datasets + periodic updates.
>
> 4\. User Experience:
>
> ◦ Integrate LM output seamlessly with Vaani TTS → voice-native
> experience.
>
> ◦ Ensure multi-turn context memory + language switching mid-dialogue.
>
> 5\. Evaluation Metrics:
>
> ◦ Use BLEU/ROUGE/Perplexity for text quality.
>
> ◦ Track inference latency, memory usage, error logs.
>
> 6\. Integration Path:
>
> ◦ Provide structured API for Indigenous NLP composer → TTS → Gurukul
> UI.
>
> ◦ Allow other teams (Nisarg, Karthikeya, Shashank) to plug in
> audio/video outputs easily.

Perfect. Here’s a concise, structured, high-value learning kit for Soham
to master MCP (Multi-Corpus Preprocessing) and efficiently implement it
for our 20+ Indian language LM expansion:

**MCP** **Learning** **Material** **for** **Soham**

**1.** **Conceptual** **Overview**

Goal: Combine multiple corpora (parallel, monolingual, multilingual)
into a unified, clean, tokenizable dataset for LM training while
preserving grammar, semantics, and script integrity.

Core Concepts:

> · Unicode Normalization: Ensure consistency across Devanagari, Tamil,
> Telugu, Bengali, etc.
>
> · Script-specific token handling: Ligatures, diacritics, compound
> characters.
>
> · Sentence Segmentation: Use language-specific rules to avoid
> splitting wrong boundaries.
>
> · Deduplication & Noise Removal: Remove duplicates, irrelevant tokens,
> HTML tags, encoding errors.
>
> · Tokenization Prep: SentencePiece or BPE-friendly formatting.

Reference Reads:

> · Google’s SentencePiece Paper:
> [<u>https://arxiv.org/abs/1804.10959</u>](https://arxiv.org/abs/1804.10959)
>
> · AI4Bharat: Indic NLP preprocessing best practices:
> [<u>https://github.com/AI4Bharat/indicnlp</u>](https://github.com/AI4Bharat/indicnlp)

**2.** **Step-by-Step** **MCPWorkflow**

> 1\. Corpus Collection
>
> ◦ Wikipedia dumps (Indic languages)
>
> ◦ OSCAR/CC-100 multilingual corpora
>
> ◦ AI4Bharat Indic corpora
>
> ◦ Gurukul-specific curated text (lessons, scripts)
>
> 2\. Cleaning & Normalization
>
> ◦ Remove control characters, HTML tags, boilerplate text
>
> ◦ Unicode normalization (NFC/NFKC)
>
> ◦ Strip punctuation/noise for tokenization
>
> ◦ Optional transliteration consistenc
>
> 3\. Sentence Segmentation
>
> ◦ Use indic-nlp-library for Indian languages
>
> ◦ Tools: nltk, sacremoses, or custom regex-based split
>
> 4\. Deduplication
>
> ◦ Hash each sentence → remove duplicates
>
> ◦ Optional fuzzy matching for near-duplicates
>
> 5\. Tokenization Preparation
>
> ◦ Normalize whitespace & punctuation
>
> ◦ Add language tags for multilingual corpora: \<lang:hi\> ...
> \</lang\>
>
> ◦ Format ready for SentencePiece/BPE training
>
> 6\. Training Tokenizer
>
> ◦ Use sentencepiece.SentencePieceTrainer.train()
>
> ◦ Merge vocab across languages to handle multilingual LM
>
> 7\. Integration
>
> ◦ Save vocab + merges → plug into LM
>
> ◦ Test tokenization → detokenization loop for all language
>
> **3.** **Recommended** **Tools** **&** **Libraries**

||
||
||
||
||
||
||
||
||
||

> **4.** **Sample** **MCP** **Pipeline** **Snippet** **(Python)**
>
> import sentencepiece as spm
>
> from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
>
> import re
>
> \# 1. Normalize Indic text
>
> factory = IndicNormalizerFactory()
>
> normalizer = factory.get_normalizer("hi") \# for Hindi

text = "नम्त, यह एक उदाहरण ह।" normalized = normalizer.normalize(text)

\# 2. Clean noise

clean_text = re.sub(r'\[^0-9a-zA-Z\u0900-\u097F\s\]', '', normalized)

\# 3. Save corpus for SentencePiece

with open("corpus.txt", "w", encoding="utf-8") as f:
f.write(clean_text + "\n")

\# 4. Train SentencePiece tokenizer spm.SentencePieceTrainer.train(

'--input=corpus.txt --model_prefix=indic_sp --vocab_size=32000
--character_coverage=1.0 --model_type=bpe' )

**5.** **Learning** **Resources** **/** **References**

> 1\. Hugging Face: Training Tokenizers
>
> ◦ https://huggingface.co/docs/tokenizers/python/latest/
>
> 2\. AI4Bharat: Indic NLP Preprocessing
>
> ◦ https://github.com/AI4Bharat/indicnlp
>
> 3\. SentencePiece Guide
>
> ◦ https://github.com/google/sentencepiece
>
> 4\. Unicode & Devanagari Handling
>
> ◦ https://unicode.org/
>
> ◦ https://www.unicode.org/reports/tr29/
>
> 5\. Python Text Processing Tutorials
>
> ◦ re, unicodedata, ftfy for cleaning

**6.** **Tips** **for** **Fast,** **Optimized** **MCP** **Usage**

> · Batch Processing: Read/write corpora in chunks to avoid memory
> issues.
>
> · Parallelization: Use multiprocessing or joblib for preprocessing
> large corpora.
>
> · Language Tags: Prepend language codes (\<hi\>, \<ta\>, \<bn\>) to
> help LM distinguish scripts.
>
> · Cache Intermediate Steps: Save normalized & cleaned files for
> reproducibility.
>
> · Validation: Random sample checks for tokenization quality per
> language.

**Note** **to** **Soham**

Hi Soham,

Good progress on the multilingual LM base — the tokenizer and initial LM
integration for Hindi, Sanskrit, Marathi, and English is solid.

Next, we need to expand this to at least 20 Indian languages using MCP
pipelines and real-world datasets, so our system can serve a truly
pan-Indian audience. Focus on clean preprocessing, robust tokenization,
and API readiness for Indigenous NLP + Vaani TTS integration.

Once completed, this will become the backbone of our multilingual
capabilities, making Gurukul scalable for millions of users. Keep
logging evaluation metrics and integration notes so handoff is smooth.

**Task** **Review** **—** **Soham** **Kotkar**

Repo:
[<u>Multilingual-Tokenization-Model-Integration</u>](https://github.com/Soham20030/Multilingual-Tokenization-Model-Integration)

Task: Multilingual Tokenization & Model Integration (Hindi, Sanskrit,
Marathi, English)

**What’s** **Done** **Well**

> · Tokenizer Creation: SentencePiece tokenizer correctly supports
> Hindi, Sanskrit, Marathi, and English with Unicode normalization for
> Devanagari.
>
> · Model Integration: Successful swap of GPT-NeoX/BLOOM-560M tokenizer
> with the custom multilingual tokenizer.
>
> · Routing & Language Detection: Basic routing with fastText/langdetect
> implemented and functional.
>
> · API Endpoints: /tokenize, /generate, and /language-detect endpoints
> tested with sample Gurukul queries.
>
> · Architecture: Stateless, modular design with room for scaling and
> easy KB (Knowledge Base) integration.

**Gaps** **/Areas** **for** **Improvement**

> · Corpus Scale: Currently limited to 4 languages — missing AI4Bharat
> and other large Indic corpora.
>
> · Evaluation Metrics: No BLEU/ROUGE or perplexity metrics for
> tokenization or generation accuracy.
>
> · RLHF/Reward Model: Framework ready but no reward signal integration
> yet.
>
> · Integration Hooks: Needs formal connection with KB, Indigenous NLP
> Composer (Nisarg), and Vaani TTS (Karthikeya).

**Score:**

**8.5** **/** **10**

Strong technical execution and clean modularity. The next challenge is
scale — expanding to 16+ additional Indic languages with real datasets
and ensuring full pipeline readiness.

**Next** **Task:** **Indic** **MCP** **Expansion** **&** **Lightweight**
**LM** **Integration**

Objective:

Scale Soham’s multilingual system to support at least 20 Indian
languages using Multi-Corpus Preprocessing (MCP), fine-tune the LM on
real datasets, and prepare it for seamless integration with the Gurukul
BHIV Core (NLP → LM → TTS).

**3-Day** **Deliverable** **Plan**

**Day** **1** **–** **MCP** **Setup** **&** **Corpus** **Expansion**
**(8–10** **hrs)**

> · Load new datasets:AI4Bharat Indic Corpora, CC-100, OSCAR, Wikipedia
> dumps.
>
> · Add 16+ languages: Tamil, Telugu, Kannada, Bengali, Gujarati,
> Punjabi, Odia, Malayalam, Assamese, Konkani, Bhojpuri, Maithili,
> Sindhi, Nepali, Manipuri, Santali.
>
> · Apply Unicode normalization, ligature handling, and sentence
> segmentation per script.
>
> · Implement data cleaning pipeline (noise removal, deduplication,
> transliteration).

**Day** **2** **–** **Tokenizer** **+** **LM** **Fine-Tuning** **(8–10**
**hrs)**

> · Train SentencePiece multilingual tokenizer with merged vocab
> (~32K–50K tokens).
>
> · Fine-tune BLOOM-560M or GPT-NeoX on cleaned MCP datasets.
>
> · Validate tokenization → detokenization loop per language.
>
> · Log token coverage and loss metrics.

**Day** **3** **–** **Integration** **&** **Evaluation** **(8–10**
**hrs)**

> · Extend API endpoints to handle 20+ languages.
>
> · Connect /generate output to Indigenous NLP + Vaani TTS for a demo
> chain.
>
> · Evaluate: BLEU, perplexity, generation fluency for 10 random
> prompts/language.
>
> · Document results + create handover README for the next integration
> layer.

**MCP** **Learning** **Kit** **(Attached** **for** **Soham)**

1\. Conceptual Overview

> · Multi-Corpus Preprocessing unifies multi-script data for
> multilingual LM training.
>
> · Key: Unicode normalization, sentence segmentation, and shared token
> space.
>
> · Reference:[<u>AI4Bharat Indic NLP
> Preprocessing</u>,](https://github.com/AI4Bharat/indicnlp)
> [<u>SentencePiece Paper</u>](https://arxiv.org/abs/1804.10959)

2\. MCPWorkflow Steps

> · Collect: Wikipedia, OSCAR, AI4Bharat, CC-100 corpora.
>
> · Normalize: IndicNLP Unicode normalization, diacritics cleanup.
>
> · Deduplicate: Hash-based and fuzzy duplicate removal.
>
> · Tokenize: SentencePiece training with \<lang\> tags.
>
> · Validate: Random sample detokenization checks.

3\. Tools & Libraries

> · indic-nlp-library, sentencepiece, fasttext, regex, transformers,
> unicodedata, ftfy.

4\. Code Snippet Example

import sentencepiece as spm

from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

langs = \["hi", "bn", "ta", "te", "kn"\] for lang in langs:

> factory = IndicNormalizerFactory() norm = factory.get_normalizer(lang)
>
> with open(f"data/{lang}\_corpus.txt", "r") as f: text = f.read()
>
> normalized = norm.normalize(text) open(f"clean/{lang}\_norm.txt",
> "w").write(normalized)

\# Train unified tokenizer spm.SentencePieceTrainer.train(

"--input=clean/\*.txt --model_prefix=indic_sp --vocab_size=50000
--character_coverage=1.0 --model_type=bpe" )

5\. Optimization Tips

> · Use joblib or multiprocessing for batch cleaning.
>
> · Prepend \<lang:xx\> tags to preserve multilingual context.
>
> · Cache cleaned corpora for reproducibility.
>
> · Test throughput with 1k-line samples before full runs.

**Deliverables** **for** **Review** **(after** **3** **days)**

MCP pipeline trained on ≥20 languages.

Fine-tuned LM + tokenizer weights.

RESTAPI for /tokenize, /generate, /language-detect (multilingual).

Integration-ready outputs for Indigenous NLP + Vaani.

Evaluation logs (BLEU, perplexity, latency).

Handover README + short demo video.

**Review** **Expectation** **at** **Handover**

Soham will be assessed on:

> 1\. Integration quality with NLP and TTS.
>
> 2\. Tokenization quality and grammar correctness.
>
> 3\. Dataset handling and efficiency (PC load optimization).
>
> 4\. Quality of evaluation metrics and documentation.

Professional Note to Soham:

Soham, you’ve built a strong multilingual foundation — now it’s time to
scale. This next step will make Gurukul linguistically universal. Focus
on clean corpus ingestion and MCP optimization so that all 20+ Indian
languages flow smoothly through the NLP → LM → TTS pipeline. This will
mark a huge leap toward real Bharat-scale inclusivity for Gurukul.

**Task** **name**

Soham Kotkar— Lightweight OnlineAdapter + RL Pipeline (MCP-enabled)

**One-line** **goal**

Enable fast, incremental multilingual quality improvements without big
local downloads: stream corpora from remote MCP connectors, train tiny
adapters/LoRA on the 4050 (8-bit/FP16 + gradient-accum), and run RL
policy updates in cloud; expose simple inference + control endpoints.

**Deliverables** **(3** **days,** **lightweight)**

> 1\. adapter_service/ with scripts to train/apply LoRA-style adapters
> using streaming datasets (no full dataset download).
>
> 2\. REST endpoints:
>
> ◦ POST /adapter/train-lite — starts a small local adapter update job
> (uses streaming subset).
>
> ◦ POST /generate-lite — inference endpoint using base LM + adapter.
>
> ◦ GET /adapter/status/{job_id} — job progress + metrics.
>
> 3\. Config: mcp_connectors.yml (S3/http/Qdrant stream sources) and
> adapter_config.yaml.
>
> 4\. Lightweight RL hook scaffold: rl/collect.py logs episodes to
> NAS/cloud for remote trainer.
>
> 5\. Smoke results: run 10 multilingual prompts (selected from MCP
> stream) and commit smoke_results.md.
>
> 6\. Short how-to: commands to run locally and how to trigger cloud RL
> job.

**Acceptance** **criteria**

> · Adapter fine-tune runs on 4050 with a small batch (batching + 8-bit)
> and completes within a few hours (not days).
>
> · generate-lite returns sensible, language-correct output for 10 test
> prompts across languages present in MCP.
>
> · No local corpus \>100MB is required; streaming works.
>
> · RL logs are being pushed to NAS / S3 for cloud trainer to consume.

**High-level** **approach** **&** **technical** **choices**

> · Use PEFT / LoRA + bitsandbytes 8-bit (low VRAM) and accelerate for
> training.
>
> · Stream corpora via Hugging Face datasets with streaming=True, or via
> MCP connectors that yield lines/samples. No full download.
>
> · Use small adapter updates (few epochs, low batch) + gradient
> accumulation to fit 4050 VRAM.
>
> · Inference uses the base decoder model + adapter merged at runtime.
>
> · RL: local agent collects (prompt, output, auto-metric reward) and
> uploads episodes for cloud PPO/PPO2 trainer (runs on Yotta/office
> GPUs). Local 4050 only records and optionally runs tiny policy updates
> (bandit/Q-table) for immediate improvements.
>
> · Tokenizer: reuse Soham’s SentencePiece; allow on-the-fly
> tokenization viaAPI.

**Exact** **commands** **&** **snippets** **(copy-pasteable)**

> 1\. Create branch

git checkout -b task_adapter_mcp

git commit --allow-empty -m "start: lightweight adapter + MCP streaming"

git push -u origin task_adapter_mcp

> 2\. Install (local 4050)

python -m venv .venv && source .venv/bin/activate pip install -r
requirements-lite.txt

\# requirements-lite.txt should include: accelerate, transformers,
bitsandbytes, peft, datasets

> 3\. Run a tiny adapter training (streaming subset)

\# example: train_adapt.py reads mcp_connectors.yml and streams N
samples

python adapter_service/train_adapt.py \\

> --model_name_or_path gpt-small-base \\ --output_dir
> adapters/gurukul_lite \\ --num_epochs 3 \\
>
> --per_device_train_batch_size 1 \\ --gradient_accumulation_steps 8 \\
> --use_8bit True \\
>
> --streaming_source "hf:your_remote_dataset" \\ --max_train_samples
> 2000
>
> 4\. Launch inferenceAPI (FastAPI)

uvicorn adapter_service.api:app --host 0.0.0.0 --port 8100 --reload

\# POST /generate-lite -\> {"prompt":"...", "lang":"hi"}

> 5\. Trigger cloud RL trainer (manual)

\# upload episodes to S3/NAS

python rl/collect.py --upload-path s3://gurukul-rl/episodes/ \# then
trigger cloud job (Vijay/Yotta) via provided script: bash
rl/trigger_cloud_trainer.sh --episodes s3://gurukul-rl/ episodes/

**Minimal** **file** **plan** **(what** **Soham** **should** **push)**

adapter_service/

> train_adapt.py \# streaming LoRA trainer

api.py \# FastAPI wrapper (generate-lite, train-lite trigger)

> model_utils.py \# load base model + adapter merge
> requirements-lite.txt

mcp_connectors.yml \# remote data sources adapter_config.yaml

rl/ collect.py

upload_helper.py test_prompts/

> prompts_10.json

smoke_results.md README.md

**Who** **to** **coordinate** **with** **(quick)**

> · Vijay — grant access to Yotta for cloud RL trainer and NAS path for
> episodes.
>
> · Nisarg — ensure /compose and BHIV trace_id can call generate-lite.
>
> · Karthikeya — confirm language tag / TTS compatibility (audio
> format).
>
> · Nipun — Qdrant/MCP connector endpoints for streaming KB chunks.

**Timeline** **(aggressive,** **start** **now)**

> · Day 0 (today, 2–4 hrs): branch + repo scaffold, mcp_connectors.yml,
> requirements-lite.
>
> · Day 1 (6–8 hrs): implement train_adapt.py streaming LoRA flow, local
> run on 4050 with small sample.
>
> · Day 2 (4–6 hrs): FastAPI wrapper + generate-lite + smoke tests (10
> prompts), push smoke_results.md.
>
> · Day 3 (optional): RL collect + cloud trigger template + docs and PR.

**Quick** **operational** **tips** **for** **Soham**

> · Use --use_8bit True (bitsandbytes), --gradient_accumulation_steps to
> emulate larger batch.
>
> · Limit max_train_samples to a few thousand for quick adapter updates;
> iterate often.
>
> · Use streaming splits with max_train_samples instead of full dataset.
>
> · Persist adapters to NAS so others can pull & test.

**Task:**

**Soham** **Kotkar** **—** **Sovereign** **LM** **Bridge** **+**
**Multilingual** **KSML** **Core** **(MCP** **+** **RL** **+**
**Vaani-ready)**

Duration: Oct 28 – Nov 2

Goal: Build the sovereign multilingual reasoning bridge that connects
Bhavesh’s LM Core, Vaani TTS, and Gurukul/Uniguru front-end — fully KSML
aligned, RL-updatable, and MCP-streaming ready.

**One-line** **Objective**

Create a live multilingual reasoning core that listens to Bhavesh’s LM
responses, refines them via RL-based language alignment, and streams
KSML-tagged results + speech-ready text to Karthikeya’s Vaani system.

**Core** **Deliverables**

**KSML** **SemanticAlignment** **Engine**

> · Implement /align.ksml service (FastAPI).
>
> · Accepts raw LM text (from Bhavesh’s system) and adds:

{

> "intent": "...",
>
> "source_lang": "target_lang":
>
> "karma_state":

"hi", "en",

"sattva/rajas/tamas",

"semantic_roots": \["dhātu", "artha", "bhava"\] }

> · Lightweight Sanskrit-root tagging via predefined lookup JSON
> (ksml_roots.json).

**MCP-Driven** **Feedback** **Stream**

> · Integrate with MCP connectors to pull live examples (user prompts +
> corrections).
>
> · Auto-store into /data/feedback_stream.jsonl.
>
> · Every feedback cycle updates a small in-memory policy (Q-table or
> bandit style).

**RL** **Self-Improvement** **Loop**

> · Add /rl.feedback endpoint:
>
> Accepts { prompt, output, reward } → updates local adapter delta or
> policy table.
>
> · Run periodic reward-based adjustments (no full retraining).
>
> · Sync logs to s3://bhiv/rl_feedback/sovereign_core/.

**Vaani** **Compatibility** **Layer**

> · Create /compose.speech_ready endpoint:
>
> Converts aligned text → prosody-optimized JSON for Karthikeya’s TTS
> engine.

{

> "text": "The answer is...",
>
> "tone":
>
> "lang":

"calm",

"en",

"prosody_hint": "gentle_low" }

> · Confirm with Karthikeya that tone + prosody_hint fields map
> correctly.

**Multilingual** **Reasoning** **Bridge**

> · Add connector to Bhavesh’s /compose.final_textAPI.
>
> · Automatically run alignment + feedback + prosody preparation in one
> flow.
>
> · Expose /bridge.reason endpoint → gives unified output (text + KSML +
> prosody).

**System** **Integration** **+** **Logging**

> · Store everything under /logs/ksml_bridge.jsonl with timestamps,
> source trace_id.
>
> · Maintain latency under 2s (end-to-end pipeline).
>
> · Use \<4GB VRAM, run smoothly on RTX 4050.

**File** **&** **Folder** **Plan**

sovereign_core/

> ├── api.py \# FastAPI endpoints

(align.ksml, rl.feedback, ├── ksml/

│ ├── aligner.py tagging

compose.speech_ready)

> \# intent + karma + root
>
> │ ├── ksml_roots.json ├── rl/
>
> │ ├── policy.py

\# Sanskrit roots +

\# simple RL/bandit

meanings

for reward

> learning
>
> │ ├── feedback_logger.py \# ├── bridge/
>
> │ ├── bhavesh_connector.py \#

logs reward

connects to

updates

/

> compose.final_text
>
> │ ├── vaani_adapter.py \# maps tone/prosody for speech-ready output
>
> ├── mcp/
>
> │ ├── stream_client.py \# fetch feedback samples │ ├── config.yml
>
> ├── logs/
>
> │ └── ksml_bridge.jsonl ├── requirements.txt
>
> └── README.md
>
> **Coordination**

||
||
||
||
||
||
||
||

> **Timeline** **(5** **Days)**

||
||
||
||
||
||
||

||
||
||
||

> **Acceptance** **Criteria**

||
||
||
||
||
||
||
||
||
||

> **After** **Completion**
>
> This task completes Layer 2 of the Gurukul Sovereign LM Stack —
>
> Layer 1: Bhavesh’s LM Core
>
> Layer 2: Soham’s Multilingual Reasoning Bridge
>
> Layer 3 (Next): Karthikeya’s Vaani Expressive RL-TTS
>
> All three connect under the BHIV Central Cognitive Mesh (managed by
> Vinayak).
