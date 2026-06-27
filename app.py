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
# FLASK CONFIG
# =========================

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "eduai-dev-secret")
app.config["JSON_AS_ASCII"] = False

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("eduai")

# =========================
# GROQ CONFIG FIXADO
# =========================

API_KEY = os.getenv("GROQ_API_KEY", "")

BASE_URL = "https://api.groq.com/openai/v1"
MODEL = os.getenv("MODEL", "llama-3.1-8b-instant")

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
Use contexto de pesquisa quando disponível.
""".strip()

RECENT_KEYWORDS = [
    "atual", "agora", "hoje", "ontem", "recente", "notícia",
    "2025", "2026", "preço", "valor", "resultado", "lançamento"
]

# =========================
# UTILS
# =========================

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def get_history():
    if "history" not in session:
        session["history"] = []
    return session["history"]


def should_search_web(message: str) -> bool:
    msg = message.lower()
    return any(k in msg for k in RECENT_KEYWORDS)

# =========================
# WEB SEARCH
# =========================

def search_web(query: str, max_results: int = 5):
    try:
        res = tavily_client.search(
            query=query,
            search_depth="basic",
            max_results=max_results
        )

        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", "")
            }
            for r in res.get("results", [])
        ]
    except Exception as e:
        logger.warning(f"Tavily error: {e}")
        return []

# =========================
# ROUTE CHAT (FIXADO GROQ)
# =========================

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = clean_text(data.get("message", ""))

    if not message:
        return jsonify({"error": "Mensagem vazia"}), 400

    history = get_history()

    web_results = []
    web_context = ""

    if should_search_web(message):
        web_results = search_web(message)

    if web_results:
        web_context = "\n".join(
            f"{r['title']} - {r['snippet']}" for r in web_results
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if web_context:
        messages.append({"role": "system", "content": web_context})

    # history seguro (SEM quebrar Groq)
    for m in history[-10:]:
        if isinstance(m, dict) and "role" in m and "content" in m:
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
            timeout=30,
        )

        answer = resp.choices[0].message.content

    except Exception as e:
        logger.error(f"Groq error: {e}")
        return jsonify({
            "error": "Erro na IA",
            "details": str(e)
        }), 500

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    session["history"] = history

    return jsonify({
        "answer": answer,
        "sources": web_results
    })


@app.route("/api/clear", methods=["POST"])
def clear():
    session["history"] = []
    return jsonify({"ok": True})


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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
