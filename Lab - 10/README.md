# CLNLP Experiment 10 – Low-Resource Neural Machine Translation

**Aim:** Build and evaluate a neural machine translation (NMT) system for low-resource Indic languages using the
[WMT 2026 Shared Task on Low-Resource Indic Language Translation](https://www2.statmt.org/wmt26/indic-mt-task.html) training data (2026 release).

**Approach (transfer learning):** The multilingual Transformer **NLLB-200** (`facebook/nllb-200-distilled-600M`) is fine-tuned on one
English ⇄ X language pair. One model learns **both directions**. It is compared with the **zero-shot** (not fine-tuned)
NLLB-200 using **BLEU, chrF++ and TER** (COMET optional) on a held-out test split.

## Files

| File | Purpose |
|---|---|
| `nmt.py` | Main NMT code: data download and cleaning, train/test split, fine-tuning, evaluation, translation, command line |
| `app.py` | Flask website. It imports `nmt.py` and calls its functions |
| `templates/index.html`, `static/` | Web page (translator + evaluation results) |
| `requirements.txt` | Python packages |
| `data/`, `models/`, `results/` | Created automatically: dataset, fine-tuned models, scores |

## Languages (WMT26)

| Key | Language | Category | In NLLB-200 pre-training? |
|---|---|---|---|
| `as` | Assamese | I | yes (zero-shot works) |
| `lus` | Mizo | I | yes |
| `mni` | Manipuri (Bengali script) | I | yes |
| `kha`, `njz` | Khasi, Nyishi | I | no, must be fine-tuned |
| `brx`, `trp`, `mjw`, `nag`, `tgj` | Bodo, Kokborok, Karbi, Nagamese, Tagin | II | no, must be fine-tuned |

English–Meitei Mayek is not included because the NLLB vocabulary has no Meitei Mayek characters.

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Train (on a GPU – Google Colab T4 recommended)

Upload the `Experiment-10` folder to Colab (or Google Drive), then:

```bash
!pip install -q -r requirements.txt
!python nmt.py train --lang as --max-train-samples 20000 --epochs 1
```

This command:
1. downloads the WMT26 English–Assamese training file and makes a fixed 500-sentence test split
2. evaluates the zero-shot NLLB-200 baseline
3. fine-tunes NLLB-200 in both directions (with 20,000 pairs, roughly 20–30 min on a T4 including evaluation; an estimate, not measured)
4. evaluates the fine-tuned model and saves the scores to `results/as.json`

Then download `models/nllb-as/` and `results/as.json` and put them in the same place in your local `Experiment-10` folder.
Remove `--max-train-samples` to use all the data. Add `--comet` (after `pip install unbabel-comet`) for COMET scores.

Other commands:

```bash
python nmt.py languages
python nmt.py evaluate  --lang as --model finetuned
python nmt.py translate --lang as --direction en2x "Education is the key to a better future."
```

## 3. Run the website

```bash
python app.py
```

Open <http://127.0.0.1:5000>. Choose the language pair, type the text, and click **Translate**. Use ⇄ to switch the direction.
The page also shows the evaluation results table and sample translations from `results/`.

- With no fine-tuned model, Assamese, Mizo and Manipuri still work using zero-shot NLLB-200.
- The first translation downloads the model (about 2.4 GB) and loads it. After that, translations take a few seconds on a CPU.
- `python app.py --preload as` loads the model at start-up.
