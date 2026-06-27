"""
EduAI - assistente educacional com Flask + Groq estável.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
import requests

from flask import Flask, jsonify, render_template, request, session
from openai import OpenAI
from tavily import TavilyClient

# =========================
# FLASK
# =========================

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "eduai-dev-secret")
app.config["JSON_AS_ASCII"] = False

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("eduai")

# =========================
# GROQ
# =========================

API_KEY = os.getenv("GROQ_API_KEY", "")
BASE_URL = "https://api.groq.com/openai/v1"
MODEL = os.getenv("MODEL", "llama3-8b-8192")

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

# =========================
# TAVILY
# =========================

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", ""))

# =========================
# PROMPT
# =========================

SYSTEM_PROMPT = """
Você é a EduAI, uma IA educacional clara e objetiva.
Responda em português do Brasil.
""".strip()

# =========================
# UTILS
# =========================

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def get_history():
    if "history" not in session:
        session["history"] = []
    return session["history"]

# =========================
# HOME (NÃO QUEBRA MAIS)
# =========================

@app.route("/")
def index():
    try:
        return render_template("index.html")
    except Exception:
        return "EduAI rodando ✔", 200

# =========================
# CHAT
# =========================

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = clean_text(data.get("message", ""))

    if not message:
        return jsonify({"error": "Mensagem vazia"}), 400

    history = get_history()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # histórico seguro
    for m in history[-10:]:
        if isinstance(m, dict) and m.get("role") and m.get("content"):
            messages.append({
                "role": m["role"],
                "content": str(m["content"])
            })

    messages.append({"role": "user", "content": message})

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
        )

        answer = resp.choices[0].message.content

    except Exception as e:
        logger.error(f"Erro IA: {e}")
        return jsonify({
            "error": "Erro na IA",
            "details": str(e)
        }), 500

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    session["history"] = history

    return jsonify({
        "answer": answer
    })

# =========================
# CLEAR
# =========================

@app.route("/api/clear", methods=["POST"])
def clear():
    session["history"] = []
    return jsonify({"ok": True})

# =========================
# HEALTH
# =========================

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL
    })

@app.route("/ping")
def ping():
    return "pong", 200

# =========================
# KEEP ALIVE
# =========================

URL = os.getenv("RENDER_EXTERNAL_URL", "")

def keep_alive():
    if not URL:
        return

    while True:
        try:
            requests.get(URL + "/ping", timeout=10)
        except:
            pass
        time.sleep(300)

threading.Thread(target=keep_alive, daemon=True).start()

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
