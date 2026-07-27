# BioTutor — container image for the chatbot backend + frontend.
#
# Works locally, on Render, and on Hugging Face Spaces. Runs as a non-root user with a
# writable Hugging Face cache so the base TinyLlama model can be downloaded at startup.
# If a fine-tuned adapter exists at backend/biotutor-lora/ it is used; otherwise the
# container runs the base model.

FROM python:3.11-slim

# Run as a non-root user (required by Hugging Face Spaces).
RUN useradd -m -u 1000 user
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    PATH=/home/user/.local/bin:$PATH
USER user
# Pre-create the HF cache dir as the non-root user, so a mounted named volume
# inherits user ownership (otherwise the volume mountpoint is owned by root).
RUN mkdir -p /home/user/.cache/huggingface
WORKDIR /home/user/app

# Install Python dependencies first (better layer caching).
COPY --chown=user backend/requirements.txt ./backend/requirements.txt
RUN pip install --user -r backend/requirements.txt

# Copy the rest of the project (backend code, frontend, and adapter if present).
COPY --chown=user . .

WORKDIR /home/user/app/backend

EXPOSE 8000

# Use $PORT if the host provides one, else default to 8000 (matches app_port on Spaces).
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
