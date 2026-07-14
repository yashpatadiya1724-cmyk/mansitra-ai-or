# Manasitra Emotional AI

A fully custom emotional AI assistant — no external APIs.

## Features
- Emotion Detection (DistilBERT)
- Memory Engine (SQLite)
- Conversation Brain (local LLM)
- Voice Assistant (Whisper + Coqui TTS)
- React Frontend with Chat UI + Mood Dashboard

## Stack
- **Frontend**: React + Vite + Tailwind CSS
- **Backend**: Python FastAPI
- **AI**: PyTorch + Transformers
- **Database**: SQLite (dev) / PostgreSQL (prod)

## Quick Start

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Prepare Dataset
```bash
cd ai-engine
python custom_model/prepare_dataset.py
```

### Train Emotion Model
```bash
cd ai-engine
pip install -r requirements.txt
python emotion_model/train.py
```

