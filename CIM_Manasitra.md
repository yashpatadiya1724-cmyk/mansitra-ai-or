# Confidential Information Memorandum (CIM)
## Project Manasitra: Privacy-Preserving On-Device Emotional AI System

**Date:** July 2026  
**Asking Price:** $25,000  
**Listing Status:** Pre-revenue / Fully Functional Technology Asset & IP Transfer  

---

## 1. Executive Summary
Manasitra is a state-of-the-art, fully custom, privacy-preserving emotional AI companion system designed for mental wellness support. Built entirely from scratch in PyTorch, the core innovation lies in its ability to process, analyze, and respond to emotional distress completely offline (on-device or on a private server). 

By eliminating all external LLM API dependencies (such as OpenAI, Anthropic, or LLaMA), Manasitra solves the critical data-privacy and compliance issues (HIPAA & GDPR) that plague generic cloud-based chatbots. It is offered as a complete business asset transfer, including frontend, backend, fully trained custom model weights, a 77,000+ counseling conversation dataset, and a patent disclosure draft.

---

## 2. Technical Architecture & Product Overview
Unlike standard wrapper apps, Manasitra is built on a proprietary, locally runnable model architecture:

* **ManasitraTransformer (212M Parameters):** A PyTorch-based Sequence-to-Sequence (Seq2Seq) Encoder-Decoder model (d_model=768, 12-head attention, 12 encoder + 12 decoder layers, FFN dim=3072) trained entirely from scratch.
* **Custom Word-Level Tokenizer:** Built specifically on a therapy-oriented vocabulary to optimize tokenization and prevent common translation artifacts.
* **Emotion-Adaptive Inference:** Dynamically adjusts response temperature and sampling based on detected emotional intensity (e.g., lower temperature for distress to provide warm/grounded support; higher temperature for positive states).
* **Local Memory Compression:** Features an emotional decay algorithm where critical distress memories persist longer than neutral conversational facts.

---

## 3. The Proprietary Dataset
The acquisition includes a highly clean, structured, and domain-specific dataset of **77,934+ counseling and emotional support dialogue pairs** (`psychology_dataset.json`). The dataset was created through curated psychology logs and expanded using an intelligent data augmentation system (synonym replacement, empathy starters, and topic variation) with O(N) Hash Set deduplication to ensure maximum data quality.

---

## 4. Financial & Operational Margins
Because the AI engine runs locally on standard CPU/mobile hardware, the business benefits from **100% Gross Margins**:
* **Active Revenue:** $0 (Pre-revenue / Pre-launch technology transfer)
* **Monthly API Cost:** $0 (No OpenAI or cloud processing bills)
* **Hosting & Operational Expenses:** ~$10/year (Domain registration only)

---

## 5. Market Opportunity & Growth Plan
The buyer can immediately monetize and scale the system using two primary strategies:
1. **B2C Mobile App & Web Subscriptions:** Launching the system as a premium private subscription app ($9.99/month) for individuals seeking secure, private emotional coaching.
2. **B2B Licensing:** Licensing the offline, HIPAA-compliant chat SDK to universities, employee wellness platforms, and clinical practices for an annual license fee ($2k–$10k/year).

---

## 6. Assets Included in the Sale
* **Proprietary Source Code:** Complete full-stack React (frontend) and FastAPI (backend) code.
* **Trained Weights:** The completed `manasitra_model.pt` model weights file (~560 MB).
* **Proprietary Dataset:** The 77k+ therapy dialogue JSON dataset.
* **Patent Disclosure Draft:** Ready-to-file technical documentation for the unique offline emotional AI system.
* **Domain Name:** Ownership of `mansitra.in` domain and related branding.

---

## 7. Risks & Mitigation
* **Market Competition:** While other chatbots exist (e.g., Wysa, Woebot), they route private chat data through external servers and incur high API costs. Manasitra’s on-device architecture completely mitigates this vulnerability, making it uniquely secure.
