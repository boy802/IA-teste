"""EduAI - assistente educacional com Flask e Groq.

Este arquivo concentra a aplicação web, a integração com a API da Groq
(compatível com o cliente `OpenAI`) e uma pesquisa web gratuita baseada em
DuckDuckGo HTML. O código foi escrito de forma simples e comentada para ser
fácil de hospedar no Render e manter em projetos educacionais.
"""

from __future__ import annotations

import os
import re
from typing import Any
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, render_template, request, session
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError


# Configuração básica do Flask. A chave de sessão pode ser configurada no Render
# por variável de ambiente; em desenvolvimento, uma chave local é usada.
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "eduai-dev-secret-change-in-render")
app.config["JSON_AS_ASCII"] = False

# Variáveis exigidas para a Groq. O provedor configurado é somente a Groq.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Cliente compatível com a API da Groq. A classe OpenAI é apenas o cliente SDK;
# o provedor configurado é exclusivamente a Groq por causa do base_url abaixo.
client = OpenAI(api_key=GROQ_API_KEY or "missing-groq-key", base_url=GROQ_BASE_URL)

SYSTEM_PROMPT = """
Você é a EduAI, uma IA educacional amigável, clara e objetiva.
Responda em português do Brasil, explique conceitos passo a passo quando útil
e nunca afirme que pesquisou na internet se nenhum contexto de pesquisa foi
fornecido. Quando houver fontes no contexto, use-as para responder e cite-as.
""".strip()

RECENT_KEYWORDS = [
    "atual", "atuais", "agora", "hoje", "ontem", "recente", "recentes",
    "última", "último", "notícia", "notícias", "2025", "2026", "preço",
    "cotação", "placar", "resultado", "lançamento", "quem é o presidente",
]


def get_history() -> list[dict[str, str]]:
    """Retorna o histórico salvo na sessão atual do navegador."""
    if "history" not in session:
        session["history"] = []
    return session["history"]


def should_search_web(message: str) -> bool:
    """Decide se a pergunta parece depender de informação recente/atual."""
    text = message.lower()
    return any(keyword in text for keyword in RECENT_KEYWORDS)


def clean_text(text: str) -> str:
    """Remove espaços repetidos para deixar resumos e snippets mais legíveis."""
    return re.sub(r"\s+", " ", text).strip()


def search_web(query: str, max_results: int = 4) -> list[dict[str, str]]:
    """Pesquisa gratuitamente no DuckDuckGo HTML e retorna resultados simples.

    A busca não exige chave de API. Caso o mecanismo esteja indisponível, a
    exceção é tratada no endpoint e a conversa continua sem derrubar a página.
    """
    url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
    headers = {"User-Agent": "Mozilla/5.0 EduAI/1.0"}
    response = requests.get(url, headers=headers, timeout=8)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    results: list[dict[str, str]] = []

    for item in soup.select(".result"):
        title_el = item.select_one(".result__title a")
        snippet_el = item.select_one(".result__snippet")
        if not title_el:
            continue

        title = clean_text(title_el.get_text(" "))
        link = title_el.get("href", "")
        snippet = clean_text(snippet_el.get_text(" ") if snippet_el else "")

        if title and link:
            results.append({"title": title, "url": link, "snippet": snippet})

        if len(results) >= max_results:
            break

    return results


def build_web_context(results: list[dict[str, str]]) -> str:
    """Transforma resultados web em contexto curto para enviar ao modelo."""
    if not results:
        return ""

    lines = ["Contexto de pesquisa web recente:"]
    for index, result in enumerate(results, start=1):
        lines.append(
            f"Fonte {index}: {result['title']}\nURL: {result['url']}\nResumo: {result['snippet']}"
        )
    return "\n\n".join(lines)


def friendly_error(error: Exception) -> tuple[str, int]:
    """Converte erros técnicos da Groq/rede em mensagens amigáveis."""
    if not GROQ_API_KEY:
        return (
            "A chave da Groq não foi configurada. Defina a variável GROQ_API_KEY no Render.",
            503,
        )
    if isinstance(error, RateLimitError):
        return "O limite da API da Groq foi atingido. Tente novamente em alguns instantes.", 429
    if isinstance(error, APITimeoutError):
        return "A Groq demorou para responder. Tente enviar a mensagem novamente.", 504
    if isinstance(error, APIConnectionError):
        return "Não foi possível conectar à Groq. Verifique sua conexão e tente novamente.", 503
    if isinstance(error, APIStatusError):
        if error.status_code in {401, 403}:
            return "A chave da Groq parece inválida ou sem permissão. Verifique GROQ_API_KEY.", 401
        return f"A Groq retornou um erro temporário ({error.status_code}). Tente novamente.", 502
    return "Ocorreu um erro inesperado. Tente novamente em alguns instantes.", 500


@app.route("/")
def index() -> str:
    """Renderiza a página principal do chat."""
    return render_template("index.html", history=get_history(), message_count=len(get_history()))


@app.route("/api/chat", methods=["POST"])
def chat() -> tuple[Any, int] | Any:
    """Recebe a mensagem do usuário, consulta a Groq e devolve a resposta."""
    payload = request.get_json(silent=True) or {}
    user_message = clean_text(payload.get("message", ""))

    if not user_message:
        return jsonify({"error": "Digite uma mensagem antes de enviar."}), 400

    history = get_history()
    web_results: list[dict[str, str]] = []
    web_context = ""

    # Pesquisa web apenas quando a pergunta parece exigir informação recente.
    if should_search_web(user_message):
        try:
            web_results = search_web(user_message)
            web_context = build_web_context(web_results)
        except (requests.RequestException, ValueError):
            web_context = "A pesquisa web gratuita falhou temporariamente; responda com cautela."

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if web_context:
        messages.append({"role": "system", "content": web_context})

    # Envia parte do histórico para manter contexto sem crescer indefinidamente.
    messages.extend(history[-12:])
    messages.append({"role": "user", "content": user_message})

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.7,
            timeout=30,
        )
        answer = completion.choices[0].message.content or "Não consegui gerar uma resposta."
    except Exception as error:  # A função friendly_error especializa os principais casos.
        message, status = friendly_error(error)
        return jsonify({"error": message}), status

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": answer})
    session["history"] = history
    session.modified = True

    return jsonify({"answer": answer, "sources": web_results, "message_count": len(history)})


@app.route("/api/clear", methods=["POST"])
def clear() -> Any:
    """Limpa o histórico da sessão atual."""
    session["history"] = []
    session.modified = True
    return jsonify({"ok": True, "message_count": 0})


@app.route("/health")
def health() -> Any:
    """Endpoint simples para verificação de saúde no Render."""
    return jsonify({"status": "ok", "provider": "Groq", "model": GROQ_MODEL})


if __name__ == "__main__":
    # Porta dinâmica para Render e porta 5000 para desenvolvimento local.
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
