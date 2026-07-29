# 🧬 BioTutor — a fine-tuned bioinformatics tutor (LLM chatbot)

A language model, **fine-tuned to specialize in bioinformatics**, that acts as an in-app
tutor. It explains the concepts behind DNA sequence analysis — GC content, ORFs, reading
frames, codons, translation, mutations (SNPs), transitions vs transversions, synonymous /
missense / nonsense effects, sequence alignment, k-mers, and machine-learning classification
— and answers general bioinformatics questions in plain language.

It is the companion assistant to the
[Sequence Toolkit](https://sequence-toolkit.onrender.com) web app.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Qwen3-4B](https://img.shields.io/badge/base-Qwen3--4B-purple)
![Unsloth](https://img.shields.io/badge/training-Unsloth%20%2B%20LoRA-ff69b4)
![FastAPI](https://img.shields.io/badge/serving-FastAPI-009688)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## What it is

BioTutor is built by **fine-tuning** an open instruction model on a curated bioinformatics
question/answer dataset, using **LoRA** (small trainable adapters). The fine-tuned model is
served behind a **FastAPI** backend with a dark, animated **chat UI** that has a
conversation-history sidebar, and conversations are saved to a small **SQLite** database.

The whole pipeline is here: build the dataset → fine-tune on a free GPU → serve the model
locally (or in a container).

### Model

The current model fine-tunes **[Qwen3-4B-Instruct](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)**
with **[Unsloth](https://github.com/unslothai/unsloth)** — a big step up from the original
1.1B model: more knowledge, better reasoning, far fewer made-up answers, and good Greek too.
Unsloth fits the 4B model on a **free Colab T4** (4-bit) and removes the earlier bitsandbytes
issues. The original TinyLlama-1.1B notebook is kept as a lightweight/legacy option.

---

## Features

- **Domain-specialized chatbot** — trained on curated Q&A pairs covering the core of
  sequence analysis, mutations, alignment and ML.
- **LLM-style chat interface** — a **conversation-history sidebar** (multiple chats, each
  saved separately), *New chat*, per-conversation delete, **Markdown answers** (bold, lists,
  code blocks) and a **Copy** button on every reply.
- **Dark, animated theme** — glowing accents, typing indicator, suggestion chips; collapses
  to a hamburger menu on mobile.
- **Persistent chat history** — each conversation keeps its own thread (SQLite), restored on
  reload.
- **Hardware auto-detection** — runs on GPU (fp16) if available, else CPU (bfloat16).
- **Runs anywhere** — a single Docker image serves both the API and the UI.

---

## How it works

```
data/bioinfo_qa.jsonl      training/train_qwen3_unsloth_colab.ipynb       backend/ + frontend/
   curated Q&A        ─────▶   LoRA fine-tune (Qwen3-4B, Unsloth, Colab) ─────▶  FastAPI serves
                              → biotutor-qwen3-lora adapter                       the model + UI
```

1. **Data** — `data/bioinfo_qa.jsonl` holds the curated Q&A pairs. The Unsloth notebook
   validates them, drops duplicates, and wraps each pair in Qwen's chat format.
   (`data/build_dataset.py` is the legacy formatter used by the TinyLlama notebook.)
2. **Training** — `training/train_qwen3_unsloth_colab.ipynb` fine-tunes Qwen3-4B with LoRA on
   a free Colab GPU, training only on the assistant's answers (`train_on_responses_only`) so
   the model learns to reply and then stop. It produces a small adapter
   (`biotutor-qwen3-lora`) and can also export a **GGUF Q4** for fast CPU serving.
3. **Serving** — `backend/main.py` loads the base model + adapter, builds prompts with the
   model's own chat template, and exposes a `/api/chat` endpoint; `frontend/` is the UI.

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Base model | [Qwen3-4B-Instruct](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) |
| Fine-tuning | [Unsloth](https://github.com/unslothai/unsloth) + `peft` (LoRA), `trl` |
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
│   ├── bioinfo_qa.jsonl                 # curated Q&A dataset (the knowledge)
│   ├── build_dataset.py                 # legacy formatter (TinyLlama path)
│   └── train.jsonl · val.jsonl          # generated split (git-ignored)
├── training/
│   ├── train_qwen3_unsloth_colab.ipynb  # ⭐ current: Qwen3-4B + Unsloth
│   └── train_qlora_colab.ipynb          # legacy: TinyLlama-1.1B
├── backend/
│   ├── main.py                          # FastAPI: loads model + adapter, chat/history API
│   ├── db.py                            # SQLite chat history
│   ├── requirements.txt
│   └── biotutor-qwen3-lora/             # the trained adapter (NOT in git; add it here)
└── frontend/
    ├── index.html · style.css · app.js  # the dark, LLM-style chat UI
```

> The trained adapter (`backend/biotutor-qwen3-lora/`) is git-ignored because it is large.
> To serve the fine-tuned model, unzip `biotutor-qwen3-lora.zip` from the training notebook
> into that folder. Without it, the backend runs the base model.

---

## Train the model (free Colab)

1. Open `training/train_qwen3_unsloth_colab.ipynb` in Google Colab.
2. **Runtime → Change runtime type → GPU (T4)**.
3. Run the cells: install Unsloth → load Qwen3-4B → attach LoRA → upload `bioinfo_qa.jsonl`
   → train → test → download `biotutor-qwen3-lora.zip` (and optionally a GGUF Q4).
4. Unzip the adapter into `backend/biotutor-qwen3-lora/`.

---

## Run locally with Docker (recommended)

Prerequisite: Docker Desktop installed and running. From the project folder:

```bash
docker build -t biotutor .
docker run -p 8000:8000 -v biotutor_cache:/home/user/.cache/huggingface biotutor
```

Then open **http://localhost:8000**. The first run downloads the base model once.

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
| `GET`  | `/api/history?session_id=...` | A conversation's saved messages |
| `POST` | `/api/clear` | Delete a conversation |

---

## Deployment notes

- **CPU serving.** A 4B model in plain PyTorch on CPU needs ~8 GB RAM and is slow. For a
  snappy CPU server, export the **GGUF Q4** from the training notebook and serve it with
  llama.cpp / Ollama instead of the PyTorch backend.
- **GPU.** The FastAPI backend runs fast on any CUDA GPU (fp16), including a Colab session.
- **Free static hosts can't run it** — the model needs real memory/compute. Use local
  Docker, a GPU instance, or an always-on VM (e.g. Oracle Cloud Always Free).
- `render.yaml` is included for a Render deployment.

---

## Limitations

- **Model size.** Qwen3-4B is strong for its size but still a small model; answers can be
  imperfect. This is a learning/portfolio project, not a production assistant.
- **Ephemeral history.** On containers/free hosts the SQLite file resets on restart.

---

## License

MIT — see [LICENSE](LICENSE).

*Built with 🧬 and Python, as a companion to the Sequence Toolkit.*
