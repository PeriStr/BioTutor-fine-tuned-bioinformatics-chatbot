# 🧬 BioTutor — a fine-tuned bioinformatics tutor (LLM chatbot)

A small language model, **fine-tuned to specialize in bioinformatics**, that acts as an
in-app tutor. It explains the concepts behind DNA sequence analysis — GC content, ORFs,
reading frames, codons, translation, mutations (SNPs), transitions vs transversions,
synonymous / missense / nonsense effects, sequence alignment, k-mers, and machine-learning
classification — and answers general bioinformatics questions in plain language.

It is the companion assistant to the
[Sequence Toolkit](https://sequence-toolkit.onrender.com) web app.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TinyLlama](https://img.shields.io/badge/base-TinyLlama--1.1B-purple)
![LoRA](https://img.shields.io/badge/fine--tune-LoRA-ff69b4)
![FastAPI](https://img.shields.io/badge/serving-FastAPI-009688)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## What it is

BioTutor is built by **fine-tuning** the open `TinyLlama-1.1B-Chat` model on a curated
bioinformatics question/answer dataset, using **LoRA** (small trainable adapters). The
fine-tuned model is served behind a **FastAPI** backend with a dark, animated **chat UI**,
and conversations are saved to a small **SQLite** database.

The whole pipeline is here: build the dataset → fine-tune on a free GPU → serve the model
locally (or in a container).

---

## Features

- **Domain-specialized chatbot** — trained on ~150 curated Q&A pairs covering the core of
  sequence analysis, mutations, alignment and ML.
- **Dark, animated chat interface** — message bubbles, typing indicator, suggestion chips,
  glowing accents.
- **Persistent chat history** — each browser keeps its own conversation (SQLite), restored
  on reload, with a one-click clear button.
- **Hardware auto-detection** — runs on GPU (fp16) if available, else CPU (fp32).
- **Runs anywhere** — a single Docker image serves both the API and the UI.

---

## How it works

```
data/bioinfo_qa.jsonl          training/train_qlora_colab.ipynb        backend/ + frontend/
   curated Q&A          ─────▶   LoRA fine-tune (TinyLlama, Colab)  ─────▶  FastAPI serves the
   (build_dataset.py)            → biotutor-lora adapter                    model + chat UI
```

1. **Data** — `data/build_dataset.py` validates the Q&A pairs and formats them into the
   TinyLlama chat template, with a train/validation split.
2. **Training** — `training/train_qlora_colab.ipynb` fine-tunes TinyLlama with LoRA on a
   free Colab GPU, training only on the assistant's answers so the model learns to reply
   and then stop. It produces a small adapter (`biotutor-lora`).
3. **Serving** — `backend/main.py` loads the base model + adapter and exposes a `/api/chat`
   endpoint; `frontend/` is the chat UI it serves.

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Base model | [TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) |
| Fine-tuning | Hugging Face `transformers`, `peft` (LoRA), `trl` |
| Serving | FastAPI + Uvicorn |
| Storage | SQLite (standard library) |
| Frontend | Vanilla HTML / CSS / JavaScript |
| Packaging | Docker |

---

## Project structure

```
bioinfo_tutor_llm/
├── README.md · LICENSE · Dockerfile · .dockerignore · .gitignore · render.yaml
├── data/
│   ├── bioinfo_qa.jsonl        # curated Q&A dataset (the knowledge)
│   └── build_dataset.py        # validate + format + train/val split
├── training/
│   └── train_qlora_colab.ipynb # LoRA fine-tuning notebook (run on Colab)
├── hosting/
│   └── launch_gradio_colab.ipynb  # optional: public demo link via Gradio share
├── backend/
│   ├── main.py                 # FastAPI: loads model + adapter, chat + history endpoints
│   ├── db.py                   # SQLite chat history
│   ├── requirements.txt
│   └── biotutor-lora/          # the trained adapter (NOT in git; add it here to serve)
└── frontend/
    ├── index.html · style.css · app.js   # the dark chat UI
```

> The trained adapter (`backend/biotutor-lora/`) is git-ignored because it is large. To
> serve the fine-tuned model, place the adapter there (unzip `biotutor-lora.zip` from the
> training notebook). Without it, the backend runs the base model.

---

## Run locally with Docker (recommended)

Prerequisite: Docker Desktop installed and running. From the project folder:

```bash
# 1. build the image
docker build -t biotutor .

# 2. run it (the volume caches the downloaded base model between runs)
docker run -p 8000:8000 -v biotutor_cache:/home/user/.cache/huggingface biotutor
```

Then open **http://localhost:8000**. The first run downloads the base model (~2 GB) once.

---

## Run without Docker (development)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 (or http://localhost:8000/docs for the API).

---

## API reference

| Method | Endpoint | Description |
|-------:|----------|-------------|
| `GET`  | `/` | The chat UI |
| `GET`  | `/api/health` | Model status + device |
| `POST` | `/api/chat` | `{ "message": "...", "session_id": "..." }` → `{ "reply": "..." }` |
| `GET`  | `/api/history?session_id=...` | This browser's saved conversation |
| `POST` | `/api/clear` | Clear this browser's conversation |

---

## Limitations

- **Small model.** TinyLlama-1.1B is tiny by LLM standards; answers can be imperfect or
  occasionally wrong. This is a learning/portfolio project, not a production assistant.
- **Ephemeral history.** On containers/free hosts the SQLite file resets on restart.
- **Hosting.** A 1.1B model needs memory; free static hosts cannot run it. Use local
  Docker, a paid instance, or a temporary Colab share link (`hosting/`).

---

## License

MIT — see [LICENSE](LICENSE).

*Built with 🧬 and Python, as a companion to the Sequence Toolkit.*
