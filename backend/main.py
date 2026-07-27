"""
main.py — BioTutor backend (FastAPI)

Loads TinyLlama-1.1B-Chat plus our fine-tuned LoRA adapter and serves a chat endpoint.

Hardware auto-detection:
  - If a CUDA GPU is available -> load in float16 on the GPU (fast).
  - Otherwise -> load in float32 on the CPU (works everywhere, just slower).

Where the adapter goes:
  Unzip `biotutor-lora.zip` (downloaded from the training notebook) into this folder,
  so that `backend/biotutor-lora/` contains the adapter files. If it is missing, the
  server still starts using the base model, so you can test the API without the adapter.

Run:
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8000
    -> open http://localhost:8000
"""

from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import db

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
HERE = Path(__file__).resolve().parent
ADAPTER_DIR = HERE / "biotutor-lora"          # unzip biotutor-lora.zip here
FRONTEND_DIR = HERE.parent / "frontend"       # chat UI (added in the next step)

# Must match the system prompt used during training.
SYSTEM_PROMPT = (
    "You are BioTutor, a knowledgeable and friendly assistant that explains "
    "bioinformatics concepts clearly and concisely."
)

MAX_INPUT_CHARS = 2000
MAX_NEW_TOKENS = 220

# ------------------------------------------------------------------
# Hardware auto-detection
# ------------------------------------------------------------------
USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"
DTYPE = torch.float16 if USE_GPU else torch.float32   # fp16 needs a GPU; CPU uses fp32
print(f"[BioTutor] device={DEVICE}, dtype={DTYPE}")


# ------------------------------------------------------------------
# Load the model once, at startup
# ------------------------------------------------------------------
def load_model():
    """Load tokenizer + base model, and merge the LoRA adapter if present."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=DTYPE)

    adapter_loaded = False
    if ADAPTER_DIR.exists() and any(ADAPTER_DIR.iterdir()):
        # Load our fine-tuned adapter and fold it into the base weights for speed.
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
        model = model.merge_and_unload()
        adapter_loaded = True
        print(f"[BioTutor] LoRA adapter loaded from {ADAPTER_DIR}")
    else:
        print(f"[BioTutor] WARNING: no adapter at {ADAPTER_DIR}; serving the BASE model only.")

    model.to(DEVICE)
    model.eval()
    return tokenizer, model, adapter_loaded


tokenizer, model, ADAPTER_LOADED = load_model()


# ------------------------------------------------------------------
# Text generation
# ------------------------------------------------------------------
@torch.inference_mode()
def generate_answer(question: str) -> str:
    """Format the prompt, generate greedily, and clean up the output."""
    prompt = (
        f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
        f"<|user|>\n{question}</s>\n"
        f"<|assistant|>\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,                       # greedy: focused, stops on EOS
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Keep only the newly generated tokens (drop the prompt).
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.split("<|")[0].strip()         # safety guard against leftover markers


# ------------------------------------------------------------------
# API
# ------------------------------------------------------------------
app = FastAPI(title="BioTutor API")


db.init_db()   # make sure the chats table exists


class ChatIn(BaseModel):
    message: str
    session_id: str = "default"   # per-browser id, so histories stay separate


class SessionIn(BaseModel):
    session_id: str = "default"


@app.get("/api/health")
def health():
    """Simple status endpoint — handy to confirm the model loaded."""
    return {"status": "ok", "device": DEVICE, "adapter_loaded": ADAPTER_LOADED}


@app.post("/api/chat")
def chat(data: ChatIn):
    question = (data.message or "").strip()
    if not question:
        return {"error": "Please type a question."}
    if len(question) > MAX_INPUT_CHARS:
        return {"error": f"Message too long (max {MAX_INPUT_CHARS} characters)."}
    try:
        reply = generate_answer(question)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Generation failed: {e}"})
    db.save_chat(data.session_id, question, reply)   # persist the exchange
    return {"reply": reply}


@app.get("/api/history")
def history(session_id: str = "default"):
    """Return this browser's saved conversation, oldest first."""
    return {"history": db.recent_chats(session_id)}


@app.post("/api/clear")
def clear(data: SessionIn):
    """Delete this browser's conversation."""
    db.clear_chats(data.session_id)
    return {"cleared": True}


# ------------------------------------------------------------------
# Serve the frontend (index.html / style.css / app.js are added next step)
# ------------------------------------------------------------------
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def home():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return index.read_text(encoding="utf-8")
    return ("<h1>BioTutor API</h1>"
            "<p>The chat UI is not built yet. Try the API docs at "
            "<a href='/docs'>/docs</a>.</p>")
