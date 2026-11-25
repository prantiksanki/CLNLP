"""
CLNLP Experiment 10 - Low-Resource Neural Machine Translation
==============================================================

Dataset : WMT 2026 Shared Task on Low-Resource Indic Language Translation
          https://www2.statmt.org/wmt26/indic-mt-task.html
Model   : NLLB-200 (facebook/nllb-200-distilled-600M), a multilingual Transformer
          encoder-decoder, fine-tuned on the WMT26 training data (transfer learning).

One bidirectional model is trained per language pair (English -> X and X -> English
in the same model). It is compared with the zero-shot (not fine-tuned) NLLB-200
baseline using BLEU, chrF++ and TER (and COMET, optional).

This module holds all the NMT logic. The website (app.py) imports it.

Command line usage
------------------
    python nmt.py languages                          # list language pairs
    python nmt.py download  --lang as                # download WMT26 training data
    python nmt.py train     --lang as --max-train-samples 20000 --epochs 1
    python nmt.py evaluate  --lang as --model finetuned     # or --model base
    python nmt.py translate --lang as --direction en2x "How are you?"

Training needs a GPU (e.g. Google Colab T4). Translation also works on a CPU.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import threading
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths & global configuration
# --------------------------------------------------------------------------------------
BASE_DIR = Path(os.environ.get("NMT_WORKDIR", Path(__file__).resolve().parent))
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

BASE_MODEL = os.environ.get("NMT_BASE_MODEL", "facebook/nllb-200-distilled-600M")
ENGLISH_CODE = "eng_Latn"
SEED = 42
DRIVE_FOLDER = "https://drive.google.com/drive/folders/1TsaINvXwb5GbiEbEkj2OgcK-1wfrWESG"


@dataclass(frozen=True)
class Language:
    key: str          # short code used on the command line / website
    name: str
    category: str     # WMT26 category (I = moderate data, II = very limited data)
    drive_id: str     # Google Drive file id of the WMT26 training file
    file_ext: str
    nllb_code: str    # NLLB-200 language tag used for this language
    native: bool      # True if NLLB-200 was pre-trained on this language


# NLLB-200 already knows Assamese, Mizo and Manipuri (Bengali script), so a zero-shot
# baseline exists for them. The other languages are not in NLLB-200: for them the tag of a
# related language written in the same script is re-used, and the model must be fine-tuned
# before it can translate them.
# English-Meitei Mayek is not included: the NLLB vocabulary has no Meitei Mayek characters
# (about half of all tokens become <unk>), so it would need a new tokenizer.
LANGUAGES: dict[str, Language] = {
    lang.key: lang for lang in [
        Language("as",  "Assamese",              "I",  "1W4AE95P7zWWyZ6crkdK2i7pR9_XPKbLZ", ".csv",  "asm_Beng", True),
        Language("lus", "Mizo",                  "I",  "16PlP4LMkJ1jgseHC81Dq4_7wXGe8uKFb", ".xlsx", "lus_Latn", True),
        Language("mni", "Manipuri (Bengali script)", "I", "1a9z-1vS7HVRpU7FtOvCDyujAhi8OHPan", ".xlsx", "mni_Beng", True),
        Language("kha", "Khasi",                 "I",  "1-7-tioDOZovflKQ7_mXMVJ8kqEZunwih", ".xlsx", "lus_Latn", False),
        Language("njz", "Nyishi",                "I",  "1Zl3jGN-1j6qtsQnn-MVfQh8g44AavT4A", ".xlsx", "lus_Latn", False),
        Language("brx", "Bodo",                  "II", "11-UK1B2crlayzTgsvHNkEZFedEd7XrLc", ".xlsx", "hin_Deva", False),
        Language("trp", "Kokborok",              "II", "1OwITtMhmh1uRP3qJ62WxYkfeFSfMwpB-", ".xlsx", "lus_Latn", False),
        Language("mjw", "Karbi",                 "II", "1x8USEAEj54jlduobmZmQQyvEzzuYSp72", ".xlsx", "lus_Latn", False),
        Language("nag", "Nagamese",              "II", "1bK4dwZFGs4HpGwLIE6Xk6wgk24Rr2WEJ", ".xlsx", "lus_Latn", False),
        Language("tgj", "Tagin",                 "II", "1g8uUoYwwU1XVHS0kbLlJk5QYo3Xiq6kb", ".xlsx", "lus_Latn", False),
    ]
}
DIRECTIONS = ("en2x", "x2en")


class ModelNotAvailable(RuntimeError):
    """Raised when no usable model exists for the requested language."""


def get_language(key: str) -> Language:
    if key not in LANGUAGES:
        raise KeyError(f"Unknown language '{key}'. Choose from: {', '.join(LANGUAGES)}")
    return LANGUAGES[key]


def lang_codes(lang: Language, direction: str) -> tuple[str, str]:
    """(source NLLB code, target NLLB code) for a direction."""
    if direction == "en2x":
        return ENGLISH_CODE, lang.nllb_code
    if direction == "x2en":
        return lang.nllb_code, ENGLISH_CODE
    raise ValueError("direction must be 'en2x' or 'x2en'")


def direction_label(lang: Language, direction: str) -> str:
    return f"English → {lang.name}" if direction == "en2x" else f"{lang.name} → English"


# --------------------------------------------------------------------------------------
# Data: download, load, clean, split
# --------------------------------------------------------------------------------------
def raw_data_path(lang: Language) -> Path:
    return DATA_DIR / lang.key / f"train_2026{lang.file_ext}"


def download_data(lang: Language, force: bool = False) -> Path:
    """Download the WMT26 training file of a language from the public Google Drive folder."""
    path = raw_data_path(lang)
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    import gdown

    print(f"Downloading English-{lang.name} WMT26 training data ...")
    out = gdown.download(id=lang.drive_id, output=str(path), quiet=False)
    if out is None or not path.exists():
        raise RuntimeError(
            f"Download failed. Download the English-{lang.name} file manually from\n  {DRIVE_FOLDER}\n"
            f"and save it as\n  {path}")
    return path


_ENGLISH_HINTS = re.compile(r"\b(the|of|and|is|to|in|a|was|for|that|it|with|are|this)\b", re.I)


def _english_score(texts) -> float:
    texts = [t for t in texts if isinstance(t, str)]
    return sum(len(_ENGLISH_HINTS.findall(t)) for t in texts) / max(1, sum(len(t.split()) for t in texts))


def _read_table(path: Path):
    """Return a 2-column DataFrame (no header handling yet) from the first usable sheet."""
    import pandas as pd

    if path.suffix == ".csv":
        sheets = [pd.read_csv(path, header=None, dtype=str, keep_default_na=False)]
    else:
        sheets = list(pd.read_excel(path, sheet_name=None, header=None, dtype=str).values())
    for df in sheets:
        df = df.replace("", None).dropna(how="all").dropna(axis=1, how="all")
        if df.empty:
            continue
        df = df.loc[:, df.notna().mean() > 0.5]          # drop almost-empty columns
        if df.shape[1] >= 2:
            return df.iloc[:, :2].reset_index(drop=True)
    raise ValueError(f"No sheet with two text columns found in {path}")


def clean_text(text) -> str:
    text = unicodedata.normalize("NFC", str(text)).replace("\xa0", " ").replace("​", "")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,;:!?।)\]’”])", r"\1", text)    # "soil ." -> "soil."
    text = re.sub(r"([(\[‘“])\s+", r"\1", text)
    return text


def load_parallel_data(lang: Language) -> list[tuple[str, str]]:
    """Load (english, other) sentence pairs, whatever the column order / header of the file."""
    df = _read_table(download_data(lang))

    # 1) which column is English? (the Tagin file, for example, has English second)
    sample = df.iloc[1:300]
    en_col = 0 if _english_score(sample.iloc[:, 0]) >= _english_score(sample.iloc[:, 1]) else 1
    # 2) drop a header row such as "English | Assamese" or "en | as"
    first = [str(c).strip().lower() for c in df.iloc[0]]
    header_words = {"en", "eng", "english", lang.key, lang.name.lower().split(" ")[0]}
    if any(c in header_words for c in first):
        df = df.iloc[1:]

    pairs, seen = [], set()
    for en, xx in zip(df.iloc[:, en_col], df.iloc[:, 1 - en_col]):
        if en is None or xx is None:
            continue
        en, xx = clean_text(en), clean_text(xx)
        if not en or not xx or en.lower() == "nan" or xx.lower() == "nan":
            continue
        n_en, n_xx = len(en.split()), len(xx.split())
        if n_en > 150 or n_xx > 150:                          # very long lines
            continue
        if min(n_en, n_xx) > 3 and max(n_en, n_xx) / min(n_en, n_xx) > 3.5:   # misaligned pairs
            continue
        if (en, xx) in seen:
            continue
        seen.add((en, xx))
        pairs.append((en, xx))
    return pairs


def _write_tsv(path: Path, pairs):
    with open(path, "w", encoding="utf-8") as f:
        for en, xx in pairs:
            f.write(f"{en}\t{xx}\n")


def _read_tsv(path: Path):
    with open(path, encoding="utf-8") as f:
        return [tuple(line.rstrip("\n").split("\t")) for line in f if "\t" in line]


def get_splits(lang: Language, test_size: int = 500, refresh: bool = False):
    """Fixed, reproducible train/test split (saved to data/<lang>/train.tsv & test.tsv)."""
    folder = DATA_DIR / lang.key
    train_p, test_p = folder / "train.tsv", folder / "test.tsv"
    if train_p.exists() and test_p.exists() and not refresh:
        return _read_tsv(train_p), _read_tsv(test_p)
    pairs = load_parallel_data(lang)
    random.Random(SEED).shuffle(pairs)
    n_test = min(test_size, max(50, len(pairs) // 10))       # at most 10% for tiny datasets
    test, train = pairs[:n_test], pairs[n_test:]
    folder.mkdir(parents=True, exist_ok=True)
    _write_tsv(train_p, train)
    _write_tsv(test_p, test)
    print(f"English-{lang.name}: {len(pairs)} clean pairs -> {len(train)} train / {len(test)} test")
    return train, test


# --------------------------------------------------------------------------------------
# Models: loading & translation
# --------------------------------------------------------------------------------------
def finetuned_model_path(lang: Language) -> Path:
    return MODEL_DIR / f"nllb-{lang.key}"


def has_finetuned_model(lang: Language) -> bool:
    return (finetuned_model_path(lang) / "config.json").exists()


def resolve_model(lang: Language, choice: str = "auto") -> tuple[str, str]:
    """Return (model name/path, label). choice: 'auto' | 'finetuned' | 'base'."""
    if choice in ("auto", "finetuned") and has_finetuned_model(lang):
        folder = finetuned_model_path(lang)
        weights = ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin")
        if not any((folder / w).exists() for w in weights):
            found = ", ".join(sorted(p.name for p in folder.glob("*.safetensors"))) or "none"
            raise ModelNotAvailable(
                f"The weights file in {folder} must be named 'model.safetensors' (found: {found}). "
                "Google Drive downloads can rename it, e.g. to 'model-001.safetensors' - rename it back.")
        return str(folder), "fine-tuned NLLB-200"
    if choice == "finetuned":
        raise ModelNotAvailable(
            f"No fine-tuned model for {lang.name} yet. Train it with: python nmt.py train --lang {lang.key}")
    if not lang.native:
        raise ModelNotAvailable(
            f"NLLB-200 was not pre-trained on {lang.name}, so it must be fine-tuned first: "
            f"python nmt.py train --lang {lang.key}")
    return BASE_MODEL, "zero-shot NLLB-200 (not fine-tuned)"


class Translator:
    """Wraps a (fine-tuned or base) NLLB model for batched translation."""

    def __init__(self, model_name_or_path: str, device: str | None = None):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        if self.device == "cuda":
            self.model = self.model.half()
        self.model.to(self.device).eval()
        self._lock = threading.Lock()        # the tokenizer's src_lang is shared state

    def translate(self, texts: list[str], src_code: str, tgt_code: str, num_beams: int = 4,
                  batch_size: int = 16, max_input_len: int = 256, max_new_tokens: int = 256,
                  verbose: bool = False) -> list[str]:
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))     # length-sorted batches
        outputs = [""] * len(texts)
        tgt_id = self.tokenizer.convert_tokens_to_ids(tgt_code)
        with self._lock:
            self.tokenizer.src_lang = src_code
            for b in range(0, len(order), batch_size):
                idx = order[b:b + batch_size]
                enc = self.tokenizer([texts[i] for i in idx], return_tensors="pt", padding=True,
                                     truncation=True, max_length=max_input_len).to(self.device)
                with self.torch.no_grad():
                    gen = self.model.generate(**enc, forced_bos_token_id=tgt_id, num_beams=num_beams,
                                              max_length=max_new_tokens)
                for i, out in zip(idx, self.tokenizer.batch_decode(gen, skip_special_tokens=True)):
                    outputs[i] = out.strip()
                if verbose and (b // batch_size) % 10 == 0:
                    print(f"    translated {min(b + batch_size, len(order))}/{len(order)}")
        return outputs


_translator_cache: dict[str, Translator] = {}
_cache_lock = threading.Lock()


def get_translator(model_name_or_path: str) -> Translator:
    with _cache_lock:
        if model_name_or_path not in _translator_cache:
            print(f"Loading model: {model_name_or_path}")
            _translator_cache[model_name_or_path] = Translator(model_name_or_path)
        return _translator_cache[model_name_or_path]


def split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?।])\s+", text.strip()) if s]


def translate_text(lang_key: str, direction: str, text: str, model_choice: str = "auto",
                   num_beams: int = 4) -> dict:
    """Translate free text (keeps line breaks, translates sentence by sentence)."""
    lang = get_language(lang_key)
    model_path, model_label = resolve_model(lang, model_choice)
    translator = get_translator(model_path)
    src, tgt = lang_codes(lang, direction)

    lines = [line.strip() for line in text.strip().split("\n")]
    sentences = [s for line in lines for s in split_sentences(line)]
    t0 = time.time()
    translated = iter(translator.translate(sentences, src, tgt, num_beams=num_beams)) if sentences else iter([])
    out_lines = [" ".join(next(translated) for _ in split_sentences(line)) for line in lines]
    return {"translation": "\n".join(out_lines), "model": model_label,
            "direction": direction_label(lang, direction), "seconds": round(time.time() - t0, 2)}


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
def compute_metrics(hypotheses, references, sources=None, use_comet: bool = False) -> dict:
    import sacrebleu

    scores = {
        "BLEU": round(sacrebleu.corpus_bleu(hypotheses, [references]).score, 2),
        "chrF++": round(sacrebleu.corpus_chrf(hypotheses, [references], word_order=2).score, 2),
        "TER": round(sacrebleu.corpus_ter(hypotheses, [references]).score, 2),
    }
    if use_comet:
        try:
            import torch
            from comet import download_model, load_from_checkpoint

            comet = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
            data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hypotheses, references)]
            out = comet.predict(data, batch_size=32, gpus=1 if torch.cuda.is_available() else 0)
            scores["COMET"] = round(100 * out.system_score, 2)
        except ImportError:
            print("COMET skipped: install it with  pip install unbabel-comet")
    return scores


def results_path(lang: Language) -> Path:
    return RESULTS_DIR / f"{lang.key}.json"


def load_results(lang: Language) -> dict:
    p = results_path(lang)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_results(lang: Language, key: str, value: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res = load_results(lang)
    res.update({"language": lang.name, "key": lang.key, key: value})
    results_path(lang).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")


def evaluate(lang: Language, model_choice: str = "finetuned", max_samples: int | None = None,
             test_size: int = 500, num_beams: int = 4, use_comet: bool = False) -> dict:
    """Evaluate a model on the held-out test split in both directions and save the scores."""
    _, test = get_splits(lang, test_size)
    if max_samples:
        test = test[:max_samples]
    model_path, label = resolve_model(lang, model_choice)
    translator = get_translator(model_path)
    english, other = [p[0] for p in test], [p[1] for p in test]

    report = {"model": label, "model_path": model_path, "test_sentences": len(test), "scores": {}, "samples": []}
    hyps = {}
    for direction in DIRECTIONS:
        src_texts, refs = (english, other) if direction == "en2x" else (other, english)
        src, tgt = lang_codes(lang, direction)
        print(f"  [{label}] {direction_label(lang, direction)} on {len(test)} test sentences ...")
        t0 = time.time()
        hyps[direction] = translator.translate(src_texts, src, tgt, num_beams=num_beams, verbose=True)
        scores = compute_metrics(hyps[direction], refs, src_texts, use_comet)
        scores["seconds"] = round(time.time() - t0, 1)
        report["scores"][direction] = scores
        print(f"    {scores}")
    for i in range(min(8, len(test))):
        report["samples"].append({"english": english[i], "other": other[i],
                                  "en2x": hyps["en2x"][i], "x2en": hyps["x2en"][i]})
    is_finetuned = model_path == str(finetuned_model_path(lang))
    save_results(lang, "finetuned" if is_finetuned else "baseline", report)
    return report


# --------------------------------------------------------------------------------------
# Training (fine-tuning NLLB-200)
# --------------------------------------------------------------------------------------
def build_bidirectional_dataset(tokenizer, pairs, lang: Language, max_len: int):
    """English->X and X->English examples, each tagged with NLLB language codes."""
    from datasets import Dataset

    records = []
    english, other = [p[0] for p in pairs], [p[1] for p in pairs]
    for direction in DIRECTIONS:
        src_texts, tgt_texts = (english, other) if direction == "en2x" else (other, english)
        tokenizer.src_lang, tokenizer.tgt_lang = lang_codes(lang, direction)
        enc = tokenizer(src_texts, text_target=tgt_texts, max_length=max_len, truncation=True)
        records += [{"input_ids": a, "attention_mask": m, "labels": l}
                    for a, m, l in zip(enc["input_ids"], enc["attention_mask"], enc["labels"])]
    random.Random(SEED).shuffle(records)
    return Dataset.from_list(records)


def train(lang: Language, epochs: float = 1.0, max_train_samples: int | None = None,
          batch_size: int = 16, grad_accum: int = 1, lr: float = 1e-4, max_len: int = 128,
          test_size: int = 500, eval_samples: int | None = None, skip_baseline: bool = False,
          use_comet: bool = False, base_model: str | None = None) -> dict:
    import torch
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed)

    set_seed(SEED)
    base_model = base_model or BASE_MODEL
    train_pairs, _ = get_splits(lang, test_size)
    if max_train_samples:
        train_pairs = train_pairs[:max_train_samples]

    # 1) zero-shot baseline (only for languages NLLB-200 was pre-trained on)
    if lang.native and not skip_baseline and base_model == BASE_MODEL:
        print("\n=== Zero-shot baseline ===")
        evaluate(lang, "base", eval_samples, test_size, use_comet=use_comet)
        _translator_cache.pop(BASE_MODEL, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 2) fine-tuning
    print(f"\n=== Fine-tuning {base_model} on {len(train_pairs)} English-{lang.name} pairs (both directions) ===")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    train_ds = build_bidirectional_dataset(tokenizer, train_pairs, lang, max_len)

    total_steps = math.ceil(len(train_ds) / (batch_size * grad_accum)) * epochs
    args = Seq2SeqTrainingArguments(
        output_dir=str(BASE_DIR / "checkpoints" / lang.key),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=epochs,
        warmup_steps=max(1, int(0.05 * total_steps)),
        weight_decay=0.01,
        optim="adafactor",                       # low memory: fits NLLB-600M on a 16 GB T4
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        seed=SEED,
    )
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=train_ds,
                             data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
                             processing_class=tokenizer)
    t0 = time.time()
    trainer.train()
    minutes = round((time.time() - t0) / 60, 1)

    out_dir = finetuned_model_path(lang)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"Fine-tuned model saved to {out_dir} ({minutes} min)")

    info = {"base_model": base_model, "train_pairs": len(train_pairs), "train_examples": len(train_ds),
            "epochs": epochs, "batch_size": batch_size * grad_accum, "learning_rate": lr,
            "minutes": minutes,
            "loss_curve": [(h["step"], round(h["loss"], 4)) for h in trainer.state.log_history if "loss" in h]}
    save_results(lang, "training", info)
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3) evaluate the fine-tuned model
    print("\n=== Fine-tuned model ===")
    _translator_cache.pop(str(out_dir), None)
    report = evaluate(lang, "finetuned", eval_samples, test_size, use_comet=use_comet)
    print_comparison(lang)
    return report


def print_comparison(lang: Language):
    res = load_results(lang)
    print(f"\nResults for English-{lang.name} (held-out test set)")
    print(f"{'model':<40s}{'direction':<12s}{'BLEU':>8s}{'chrF++':>8s}{'TER':>8s}")
    for key in ("baseline", "finetuned"):
        if key not in res:
            continue
        for d, s in res[key]["scores"].items():
            print(f"{res[key]['model']:<40s}{d:<12s}{s['BLEU']:>8.2f}{s['chrF++']:>8.2f}{s['TER']:>8.2f}")


# --------------------------------------------------------------------------------------
# Helpers used by the website
# --------------------------------------------------------------------------------------
def language_status() -> list[dict]:
    status = []
    for lang in LANGUAGES.values():
        status.append({
            "key": lang.key, "name": lang.name, "category": lang.category, "native": lang.native,
            "finetuned": has_finetuned_model(lang),
            "usable": lang.native or has_finetuned_model(lang),
            "results": load_results(lang),
        })
    return status


# --------------------------------------------------------------------------------------
# Command line interface
# --------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Low-resource NMT on the WMT26 Indic MT data (NLLB-200)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("languages", help="list the available language pairs")

    p = sub.add_parser("download", help="download and split the training data")
    p.add_argument("--lang", required=True)
    p.add_argument("--test-size", type=int, default=500)

    p = sub.add_parser("train", help="zero-shot baseline + fine-tuning + evaluation")
    p.add_argument("--lang", required=True)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-train-samples", type=int, default=None, help="use only N training pairs")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--test-size", type=int, default=500)
    p.add_argument("--eval-samples", type=int, default=None, help="evaluate on only N test sentences")
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--comet", action="store_true", help="also compute COMET (needs unbabel-comet)")
    p.add_argument("--base-model", default=None)

    p = sub.add_parser("evaluate", help="evaluate a model on the test split")
    p.add_argument("--lang", required=True)
    p.add_argument("--model", choices=["finetuned", "base"], default="finetuned")
    p.add_argument("--eval-samples", type=int, default=None)
    p.add_argument("--test-size", type=int, default=500)
    p.add_argument("--comet", action="store_true")

    p = sub.add_parser("translate", help="translate a sentence")
    p.add_argument("--lang", required=True)
    p.add_argument("--direction", choices=DIRECTIONS, default="en2x")
    p.add_argument("--model", choices=["auto", "finetuned", "base"], default="auto")
    p.add_argument("text")

    args = parser.parse_args()
    if args.command == "languages":
        for s in language_status():
            note = "pre-trained in NLLB-200" if s["native"] else "needs fine-tuning"
            print(f"{s['key']:<5s}{s['name']:<28s}Category {s['category']:<4s}{note:<26s}"
                  f"{'fine-tuned model found' if s['finetuned'] else ''}")
        return
    lang = get_language(args.lang)
    if args.command == "download":
        get_splits(lang, args.test_size, refresh=True)
    elif args.command == "train":
        train(lang, args.epochs, args.max_train_samples, args.batch_size, args.grad_accum, args.lr,
              args.max_len, args.test_size, args.eval_samples, args.skip_baseline, args.comet, args.base_model)
    elif args.command == "evaluate":
        evaluate(lang, args.model, args.eval_samples, args.test_size, use_comet=args.comet)
        print_comparison(lang)
    elif args.command == "translate":
        r = translate_text(lang.key, args.direction, args.text, args.model)
        print(f"[{r['direction']} | {r['model']} | {r['seconds']}s]\n{r['translation']}")


if __name__ == "__main__":
    main()
