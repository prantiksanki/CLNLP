"""
CLNLP Experiment 10 - website for the low-resource NMT system.

All translation logic lives in nmt.py; this file only serves the web interface.

Run:
    python app.py                 # then open http://127.0.0.1:5000
    python app.py --preload as    # load the Assamese model at start-up
"""

import argparse
import threading

from flask import Flask, jsonify, render_template, request

import nmt

app = Flask(__name__)
MAX_CHARS = 3000


@app.route("/")
def index():
    return render_template("index.html", languages=nmt.language_status(), base_model=nmt.BASE_MODEL,
                           max_chars=MAX_CHARS)


@app.get("/api/languages")
def api_languages():
    return jsonify(nmt.language_status())


@app.post("/api/translate")
def api_translate():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    lang = data.get("lang", "as")
    direction = data.get("direction", "en2x")
    model_choice = data.get("model", "auto")

    if not text:
        return jsonify(error="Please enter some text to translate."), 400
    if len(text) > MAX_CHARS:
        return jsonify(error=f"Text is too long (maximum {MAX_CHARS} characters)."), 400
    if lang not in nmt.LANGUAGES or direction not in nmt.DIRECTIONS or model_choice not in ("auto", "finetuned", "base"):
        return jsonify(error="Invalid language, direction or model."), 400
    try:
        return jsonify(nmt.translate_text(lang, direction, text, model_choice))
    except nmt.ModelNotAvailable as e:
        return jsonify(error=str(e)), 409
    except Exception as e:                       # e.g. model download failed / out of memory
        app.logger.exception("Translation failed")
        return jsonify(error=f"Translation failed: {e}"), 500


def preload(lang_key):
    try:
        path, _ = nmt.resolve_model(nmt.get_language(lang_key))
        nmt.get_translator(path)
        print(f"Model for '{lang_key}' is loaded.")
    except Exception as e:
        print(f"Could not preload '{lang_key}': {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--preload", default=None, help="language key whose model is loaded at start-up")
    args = parser.parse_args()
    if args.preload:
        threading.Thread(target=preload, args=(args.preload,), daemon=True).start()
    print(f"Open http://{args.host}:{args.port} in your browser")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
