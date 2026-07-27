"""
build_dataset.py — prepare the raw Q&A file for fine-tuning.

What this script does:
  1) Loads and VALIDATES the raw pairs in `bioinfo_qa.jsonl`
     (every record must have a non-empty 'instruction' and 'response'; duplicates dropped).
  2) FORMATS each pair into the exact chat format that TinyLlama-1.1B-Chat expects,
     with a fixed system prompt that gives the model its "tutor" persona.
  3) SPLITS the data into a training set and a small validation set, and writes
     `train.jsonl` and `val.jsonl` next to this script.

Run it locally (no GPU needed):
    python build_dataset.py

The training notebook then just loads train.jsonl / val.jsonl and reads the "text" field.
"""

import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "bioinfo_qa.jsonl"
SEED = 42
VAL_FRACTION = 0.10   # 10% of the data is held out for validation

# The persona the fine-tuned model should adopt. Kept identical for every example so the
# model learns a consistent voice. (This must match the system prompt used at inference.)
SYSTEM_PROMPT = (
    "You are BioTutor, a knowledgeable and friendly assistant that explains "
    "bioinformatics concepts clearly and concisely."
)


def format_chat(instruction: str, response: str) -> str:
    """Build the full training string in TinyLlama-Chat's (Zephyr-style) chat format.

    The model was pretrained to read exactly this layout, so matching it makes
    fine-tuning far more effective:

        <|system|>
        ...</s>
        <|user|>
        ...</s>
        <|assistant|>
        ...</s>
    """
    return (
        f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
        f"<|user|>\n{instruction}</s>\n"
        f"<|assistant|>\n{response}</s>\n"
    )


def load_and_validate(path: Path):
    """Read the raw JSONL, keep only valid, unique records."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run author step first.")

    rows, seen, skipped = [], set(), 0
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            instr = str(rec.get("instruction", "")).strip()
            resp = str(rec.get("response", "")).strip()
            if not instr or not resp:            # both fields required
                skipped += 1
                continue
            key = instr.lower()
            if key in seen:                      # drop duplicate questions
                continue
            seen.add(key)
            rows.append({"instruction": instr, "response": resp})
    return rows, skipped


def main():
    rows, skipped = load_and_validate(RAW)

    # Add the formatted training text to every record.
    for r in rows:
        r["text"] = format_chat(r["instruction"], r["response"])

    # Shuffle deterministically, then split into train / validation.
    random.seed(SEED)
    random.shuffle(rows)
    n_val = max(1, int(len(rows) * VAL_FRACTION))
    val, train = rows[:n_val], rows[n_val:]

    for name, data in [("train.jsonl", train), ("val.jsonl", val)]:
        with open(HERE / name, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Valid records: {len(rows)}  (skipped {skipped})")
    print(f"Train: {len(train)}  |  Validation: {len(val)}")
    print("\nExample of a formatted training string:\n")
    print(train[0]["text"])


if __name__ == "__main__":
    main()
