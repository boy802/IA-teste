"""
EduAI - assistente educacional com Flask e Groq/OpenAI compatível.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
import requests
from typing import Any

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
# API CONFIG (GROQ / OPENAI / OPENROUTER COMPAT)
# =========================

API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("OPENROUTER_API_KEY") or ""
BASE_URL = os.getenv("BASE_URL", "https://api.groq.com/openai/v1")
MODEL = os.getenv("MODEL", "llama-3.1-8b-instant")

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

# =========================
# TAVILY (INTERNET)
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
    "atual", "agora", "hoje", "ontem", "recente", "recentes",
    "última", "últimas", "último", "últimos",
    "notícia", "notícias",
    "2025", "2026",
    "preço", "cotação", "valor",
    "placar", "resultado",
    "lançamento",
    "dólar", "dolar",
    "euro",
    "bolsa",
    "quem é o presidente",
]

# =========================
# ERROR CLASS
# =========================

class SearchError(RuntimeError):
    pass

# =========================
# UTILS
# =========================

def get_history() -> list[dict[str, str]]:
    if "history" not in session:
        session["history"] = []
    return session["history"]


def should_search_web(message: str) -> bool:
    text = message.lower()
    return any(k in text for k in RECENT_KEYWORDS)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

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
        raise SearchError(f"Falha na busca web: {e}")

# =========================
# CONTEXT BUILDER
# =========================

def build_web_context(results: list[dict[str, str]]) -> str:
    if not results:
        return ""

    lines = ["Contexto de pesquisa web recente:"]

    for i, r in enumerate(results, 1):
        lines.append(
            f"Fonte {i}: {r['title']}\nURL: {r['url']}\nResumo: {r['snippet']}"
        )

    return "\n\n".join(lines)

# =========================
# ROUTES
# =========================

@app.route("/")
def index():
    return render_template(
        "index.html",
        history=get_history(),
        message_count=len(get_history()),
    )


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = clean_text(data.get("message", ""))

    if not message:
        return jsonify({"error": "Mensagem vazia"}), 400

    history = get_history()

    web_results = []
    web_context = ""

    search_performed = should_search_web(message)

    if search_performed:
        try:
            web_results = search_web(message)
            web_context = build_web_context(web_results)
        except SearchError as e:
            logger.warning("Web falhou: %s", e)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if web_context:
        messages.append({"role": "system", "content": web_context})

    messages.extend(history[-10:])
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
        return jsonify({"error": str(e)}), 500

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    session["history"] = history
    session.modified = True

    return jsonify({
        "answer": answer,
        "sources": web_results,
        "search_performed": search_performed,
        "message_count": len(history),
    })


@app.route("/api/clear", methods=["POST"])
def clear():
    session["history"] = []
    return jsonify({"ok": True})


@app.route("/test-search")
def test_search():
    q = clean_text(request.args.get("q", "preço do dólar hoje"))

    try:
        results = search_web(q)
        return jsonify({"ok": True, "results": results})
    except SearchError as e:
        return jsonify({"ok": False, "error": str(e)}), 502


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL,
    })


@app.route("/ping")
def ping():
    return "pong", 200


# =========================
# KEEP ALIVE (SAFE VERSION)
# =========================

URL = os.getenv("RENDER_EXTERNAL_URL", "")

def keep_alive():
    if not URL:
        return

    while True:
        try:
            r = requests.get(URL + "/ping", timeout=10)
            print(f"[KEEP-ALIVE] /ping -> {r.status_code}")
        except Exception as e:
            print(f"[KEEP-ALIVE] erro: {e}")

        time.sleep(300)

threading.Thread(target=keep_alive, daemon=True).start()


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
