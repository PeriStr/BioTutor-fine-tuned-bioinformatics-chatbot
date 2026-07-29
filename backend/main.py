"""
main.py — BioTutor backend (FastAPI)

Loads Qwen3-4B-Instruct plus our fine-tuned LoRA adapter and serves a chat endpoint.

Hardware auto-detection:
  - If a CUDA GPU is available -> load in float16 on the GPU (fast).
  - Otherwise -> load in bfloat16 on the CPU (works, but a 4B model is slow on CPU).

Where the adapter goes:
  Unzip `biotutor-qwen3-lora.zip` (downloaded from the training notebook) into this folder,
  so that `backend/biotutor-qwen3-lora/` contains the adapter files. If it is missing, the
  server still starts using the base model, so you can test the API without the adapter.

Note on CPU deployment (e.g. Oracle free tier): a 4B model in PyTorch on CPU needs ~8GB RAM
and is slow. For a snappy CPU server, use the GGUF Q4 export from the notebook with
llama.cpp / Ollama instead of this PyTorch backend.

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
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
HERE = Path(__file__).resolve().parent
ADAPTER_DIR = HERE / "biotutor-qwen3-lora"    # unzip biotutor-qwen3-lora.zip here
FRONTEND_DIR = HERE.parent / "frontend"       # chat UI (added in the next step)

# Must match the system prompt used during training.
SYSTEM_PROMPT = (
    "You are BioTutor, an expert tutor in bioinformatics, biology, and machine learning. "
    "You give thorough, detailed explanations with concrete examples, and you stay detailed "
    "even when the question is short."
)

MAX_INPUT_CHARS = 2000
MAX_NEW_TOKENS = 400

# ------------------------------------------------------------------
# Hardware auto-detection
# ------------------------------------------------------------------
USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"
# fp16 on GPU; bfloat16 on CPU (halves RAM vs fp32 and avoids the fp16-on-CPU op errors).
DTYPE = torch.float16 if USE_GPU else torch.bfloat16
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
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
        # On GPU we merge for speed; on CPU we DON'T merge, to avoid a memory spike
        # (merging briefly needs a second full copy of the 4B weights -> MemoryError).
        if USE_GPU:
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
    """Build the prompt with the model's own chat template, generate, and clean up."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]
    # apply_chat_template inserts the exact special tokens Qwen expects and the
    # trailing assistant marker. return_dict=True -> a BatchEncoding with input_ids
    # and attention_mask, which we expand into generate() with **inputs.
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(DEVICE)
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,                       # greedy: focused, stops on EOS
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Keep only the newly generated tokens (drop the prompt).
    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()


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
        import traceback
        traceback.print_exc()   # full trace in the uvicorn console
        msg = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
        return JSONResponse(status_code=500, content={"error": f"Generation failed: {msg}"})
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
