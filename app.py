"""
EduAI - assistente educacional com Flask e Groq.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import requests
from bs4 import BeautifulSoup, __version__ as BS4_VERSION
from flask import Flask, jsonify, render_template, request, session
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

# =========================
# FLASK CONFIG
# =========================

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "eduai-dev-secret")
app.config["JSON_AS_ASCII"] = False

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("eduai")

# =========================
# GROQ CONFIG
# =========================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

client = OpenAI(
    api_key=GROQ_API_KEY or "missing-groq-key",
    base_url=GROQ_BASE_URL,
)

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


def normalize_duckduckgo_url(url: str) -> str:
    if not url:
        return ""
    if url.startswith("//"):
        url = "https:" + url

    parsed = urlparse(url)
    target = parse_qs(parsed.query).get("uddg", [None])[0]
    return unquote(target) if target else url

# =========================
# SEARCH ENGINE
# =========================

def search_web(query: str, max_results: int = 5) -> list[dict[str, str]]:
    cleaned = clean_text(query)
    if not cleaned:
        raise SearchError("Consulta vazia")

    url = f"https://html.duckduckgo.com/html/?q={quote_plus(cleaned)}"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
        ),
        "Accept-Language": "pt-BR,pt;q=0.9",
    }

    logger.info("Search: %s", cleaned)

    try:
        res = requests.get(url, headers=headers, timeout=(5, 15))
        res.raise_for_status()
    except Exception as e:
        raise SearchError(str(e))

    soup = BeautifulSoup(res.text, "html.parser")
    results: list[dict[str, str]] = []

    for item in soup.select(".result, .web-result, div.result.results_links"):
        title_el = item.select_one(".result__title a, a.result__a")
        snippet_el = item.select_one(".result__snippet, .result__body")

        if not title_el:
            continue

        title = clean_text(title_el.get_text(" "))
        link = normalize_duckduckgo_url(title_el.get("href", ""))
        snippet = clean_text(snippet_el.get_text(" ") if snippet_el else "")

        if title and link:
            results.append({
                "title": title,
                "url": link,
                "snippet": snippet
            })

        if len(results) >= max_results:
            break

    logger.info("Results: %s", len(results))
    return results


def build_web_context(results: list[dict[str, str]]) -> str:
    if not results:
        return ""

    lines = [
        "PESQUISA WEB REALIZADA. Use as fontes abaixo e cite URLs:"
    ]

    for i, r in enumerate(results, 1):
        lines.append(
            f"Fonte {i}: {r['title']}\nURL: {r['url']}\nResumo: {r['snippet']}"
        )

    return "\n\n".join(lines)

# =========================
# ERROR HANDLER
# =========================

def friendly_error(error: Exception) -> tuple[str, int]:
    if not GROQ_API_KEY:
        return "Falta GROQ_API_KEY", 503
    return str(error), 500

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
            return jsonify({
                "error": str(e),
                "search_performed": True,
                "sources": []
            }), 502

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if web_context:
        messages.append({"role": "system", "content": web_context})

    messages.extend(history[-10:])
    messages.append({"role": "user", "content": message})

    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
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
        "model": GROQ_MODEL,
        "requests": requests.__version__,
        "bs4": BS4_VERSION,
    })


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
