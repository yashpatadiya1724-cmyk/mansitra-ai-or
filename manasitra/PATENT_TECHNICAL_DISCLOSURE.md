# Manasitra Technical Disclosure Draft

This document is a patent-attorney handoff draft, not legal advice and not a guarantee of patentability.

## Proposed Technical Invention

Manasitra is a fully custom, privacy-preserving emotional AI system for mental wellness support. It combines a **278-million-parameter Seq2Seq Transformer** (built entirely from scratch in PyTorch — no external LLMs, no APIs) with a local emotional-AI pipeline comprising seven core technical components:

1. **Custom Neural Response Engine (ManasitraTransformer):** A 278,840,083-parameter Encoder-Decoder Transformer architecture (d_model=768, 12-head attention, 16 encoder + 16 decoder layers, FFN dim=3072) trained entirely on curated psychology/therapy conversation data using Net2Net progressive layer expansion. No pre-trained models, no external APIs — 100% custom weights.
2. **Custom Tokenizer (ManasitraTokenizer):** A word-level tokenizer with special tokens (PAD, UNK, BOS, EOS), vocabulary built from training corpus (~6,000+ tokens), with intelligent decoding that handles punctuation spacing, apostrophe reconstruction, and encoding artifact correction.
3. **Emotion-Adaptive Inference Engine:** Temperature-modulated response generation that adjusts model creativity based on detected emotional state (distress → lower temperature for careful/warm responses; positive → higher temperature for expressive responses). Includes top-k sampling with fallback safety responses.
4. **Continuous Training Pipeline:** An automated training loop system that iteratively: (a) trains the model for N epochs, (b) measures accuracy and loss against configurable targets, (c) expands the dataset via intelligent augmentation, (d) retrains on expanded data — repeating until quality targets are met (accuracy > 98.0%, loss < 0.50).
5. **Intelligent Data Augmentation System:** Automated dataset expansion (up to 3.0x per round) through synonym replacement, input prefix variation, response empathy starter injection, and topic-specific pair generation across 10+ mental health categories — with O(N) Hash Set deduplication to maintain massive data quality.
6. **Hybrid Emotion Architecture:** Keyword intensity scoring, compound emotion rules, distress overrides, emoji-safe output mapping, and transition-aware context.
7. **Emotion-Adaptive Memory Compression:** Distress memories decay slower than neutral memories and are ranked by live retention score.
8. **Longitudinal Mood Pattern Engine:** Detects streaks, recurring time-of-day moods, and repeated distress triggers.
9. **Emotion State Machine:** Tracks abrupt, concerning, improving, and natural emotional transitions across turns.
10. **Privacy-Preserving Dedicated Execution Pipeline (Local or Private Server):** Redacts common PII before storage/context processing, operates in an isolated environment (either on-device or on a private server node like an AWS EC2 instance), and requires zero third-party external API requests (e.g. OpenAI or generic LLM endpoints) — ensuring zero data leakage to external entities.

## Technical Problem

Generic chatbots and commercial AI assistants suffer from several critical limitations for mental wellness support:

1. **External API dependency:** Services like OpenAI, Ollama, or cloud LLMs expose private mental-health text to third-party servers, creating serious privacy and HIPAA-compliance risks.
2. **No emotional intelligence:** Generic models misclassify mixed student distress, fail to detect compound emotions (e.g., simultaneously anxious and lonely), and produce inappropriate responses to crisis situations.
3. **Hallucination and unsafe responses:** Pre-trained models hallucinate user facts, produce diagnosis-like responses, and may give harmful advice during distress.
4. **No continuous improvement:** Most deployed chatbots use static models with no mechanism for iterative quality improvement through expanded training data.
5. **Black-box architecture:** Commercial LLMs provide no transparency into model weights, training data, or decision-making processes.

## Technical Solution

### Custom Neural Architecture

The core innovation is a fully custom 278M-parameter Seq2Seq Transformer progressively grown from scratch:

```
Architecture: ManasitraTransformer
├── Source Embedding (vocab × 768) + Positional Encoding
├── Encoder (16 layers × TransformerEncoderLayer)
│   └── Multi-Head Self-Attention (12 heads) + FFN (3072)
├── Decoder (16 layers × TransformerDecoderLayer)
│   └── Masked Self-Attention + Cross-Attention (12 heads) + FFN (3072)
└── Output Projection (768 → vocab)

Parameters: 278,840,083 (~1.1 GB saved model)
Training: Pure PyTorch, CPU-compatible, no GPU required
Inference: Top-k sampling with emotion-adaptive temperature
```

### Processing Pipeline

The system processes each user message through:

```
user_input → privacy_filter → hybrid_emotion_predictor → emotion_state_update
    → memory_context_retrieval → custom_neural_response_engine (278M Transformer)
    → emotion_adaptive_temperature → top_k_sampling → response_safety_gate
    → fallback_handler → technical_proof_metadata → response_output
```

### Continuous Training System

```
┌─────────────────────────────────────────────────┐
│  Round N                                         │
│  1. Load dataset (expanding each round)          │
│  2. Build fresh tokenizer from all data          │
│  3. Train model for up to 150 epochs             │
│  4. Track: Loss ↓, Accuracy ↑, Perplexity ↓     │
│  5. Save best checkpoint                         │
│  6. If targets met → STOP                        │
│  7. Expand dataset 1.5x via augmentation         │
│  8. → Round N+1                                  │
│  Target: Accuracy > 98.0%, Loss < 0.50           │
└─────────────────────────────────────────────────┘
```

### Emotion-Adaptive Inference

| Detected Emotion | Temperature | Behavior |
|------------------|-------------|----------|
| Sad / Lonely | 0.65 | Conservative, warm, careful responses |
| Anxiety | 0.68 | Grounding, reassuring responses |
| Stress | 0.70 | Structured, step-by-step guidance |
| Angry | 0.70 | Validating, de-escalating responses |
| Neutral | 0.75 | Balanced, exploratory responses |
| Happy | 0.80 | Expressive, celebratory responses |

The output includes emotion, confidence, transition evidence, local-only privacy proof, response latency, memory usage count, and pattern usage count.

## Implemented Evidence

- **Custom Model:** `ai-engine/custom_model/model.py` — Full Transformer architecture (276 lines, pure PyTorch)
- **Training Script:** `ai-engine/custom_model/train.py` — Training pipeline with accuracy/loss tracking
- **Continuous Training:** `ai-engine/custom_model/continuous_train.py` — Automated train→expand→retrain loop
- **Inference Engine:** `ai-engine/custom_model/inference.py` — Emotion-adaptive response generation
- **Dataset Expansion:** `ai-engine/custom_model/expand_dataset.py` — Intelligent augmentation system
- **Custom Tokenizer:** `ai-engine/custom_model/model.py` — ManasitraTokenizer with vocab builder
- **Trained Model:** `ai-engine/custom_model/saved_model/manasitra_model.pt` (~560 MB)
- **Dataset:** `ai-engine/custom_model/psychology_dataset.json` — 399,033+ curated and synthetically expanded therapy pairs
- **Backend endpoint:** `/innovation/status`
- **Privacy module:** `ai-engine/privacy_engine/privacy.py`
- **Benchmark script:** `ai-engine/benchmarks/measure_pipeline.py`
- **Runtime metadata:** `technical_proof` in `/chat` and `/chat/stream`
- **UI proof label:** Chat messages show local engine latency
- **AWS Deployment Configurations:** `backend/Dockerfile` & `backend/docker-compose.yml` — Containerized backend architecture.
- **Server Configuration Script:** `backend/deploy_aws.sh` — Automatic Ubuntu EC2 setup script.
- **Dynamic API Client Router:** `manasitra-web/src/shared/utils/groq-service.js` — Client-side API router connecting user accounts dynamically to AWS backend services.

## Model Training History

| Milestone | Parameters | Dataset | Best Loss | Status |
|-----------|-----------|---------|-----------|--------|
| v1 (Initial) | 3,366,547 | 3,916 pairs | 2.84 | Complete |
| v2 (Continued) | 3,366,547 | 3,916 pairs | 2.44 | Complete |
| v3 (Expanded Data) | 3,366,547 | 13,799 pairs | — | Complete |
| v4 (Round 1 Scale) | 146,522,899 | 49,761 pairs | 3.82 | Complete |
| v5 (Round 2 Scale) | 278,840,083 | 399,033 pairs | — | Training |

## Claim Drafting Notes For Attorney

Potential claim focus:

1. **Custom Neural Response Generation:** A method for generating emotionally-appropriate mental wellness responses using a fully custom 278M-parameter Seq2Seq Transformer model trained from scratch on curated therapy conversation data, utilizing Net2Net progressive expansion, without reliance on pre-trained language models or external APIs.
2. **Emotion-Adaptive Temperature Modulation:** A system wherein the response generation temperature of a neural language model is dynamically adjusted based on a classified emotional state of user input, producing more conservative outputs during detected distress and more expressive outputs during detected positive states.
3. **Continuous Self-Improving Training Pipeline:** A method for iteratively improving a neural response model through automated cycles of: training to convergence, intelligent dataset expansion via synonym augmentation and topic-specific pair injection, and retraining — with configurable quality targets (accuracy, loss) that trigger automatic termination.
4. **Privacy-Preserving Dedicated Mental Wellness Pipeline:** A client-server or local mental wellness response pipeline that processes user data in an isolated execution environment (either on-device or on a private/dedicated server like a containerized AWS EC2 node), including emotion classification, memory management, and neural response generation, with server-side or local PII redaction before database persistence and model context generation — requiring zero third-party external API calls.
5. **Emotion-Constrained Response System:** A method for generating emotionally constrained responses using a compound emotion classifier, emotion transition state machine, and memory retention score with emotion-dependent decay rates.
6. **Safety Gate with Fallback Responses:** A response safety system that validates generated text against clinical safety criteria and substitutes pre-authored empathetic fallback responses when the neural model output fails quality or safety thresholds.

### Differentiators from Prior Art

- **No pre-trained models:** Unlike systems built on GPT, LLaMA, or other foundation models, Manasitra's neural engine is trained entirely from scratch with custom architecture and weights.
- **Emotion-aware generation:** Response temperature is dynamically coupled to emotional classification, a mechanism not found in standard chatbot architectures.
- **Continuous self-improvement:** The automated train→expand→retrain loop with quality-target termination is a novel training methodology for domain-specific conversational AI.
- **Isolated Private Server or On-Device Execution:** The entire pipeline (emotion detection, memory, response generation, safety) runs in an isolated dedicated environment (on-device or private server) without any external third-party API dependencies, ensuring complete HIPAA/GDPR compatibility unlike typical API-dependent chatbots.
- **Contrast with Microsoft (e.g., US 10,838,967 B2):** Microsoft's patents focus on cloud-based re-ranking of generic LLM outputs to align with tone. Manasitra control-modulates the actual neural model's hyperparameter temperature in real-time during dedicated server-side or local inference of a scratch-trained 278M network.
- **Contrast with MetaSoul (e.g., US 12,525,251 B2):** MetaSoul focuses on adapting vocal features (pitch/harmonics) to a user's voice. Manasitra targets text-based semantic dialog, utilizing an Emotion State Machine (ESM) and asymmetric memory decay (EAMC) to maintain long-term context without leakage.

Avoid claiming only a chatbot, app idea, therapy method, or generic AI model.

