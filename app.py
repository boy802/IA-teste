"""Aplicação Flask da EduAI.

Este arquivo concentra as rotas HTTP e delega a geração de respostas para
funções pequenas, facilitando o estudo e a manutenção do projeto.
"""

import os
import re
from datetime import datetime
from typing import Any, Dict, List

import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, render_template, request, session
from openai import OpenAI, OpenAIError

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "eduai-chave-local-de-desenvolvimento")
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024  # Evita payloads grandes demais.

AI_NAME = "EduAI"
MAX_MESSAGE_LENGTH = 1200
MAX_HISTORY_MESSAGES = 20
SEARCH_TIMEOUT = 8


def get_history() -> List[Dict[str, str]]:
    """Retorna o histórico salvo na sessão atual do navegador."""
    return session.setdefault("history", [])


def save_message(role: str, content: str) -> None:
    """Salva uma mensagem na sessão e mantém o histórico em tamanho controlado."""
    history = get_history()
    history.append({"role": role, "content": content})
    session["history"] = history[-MAX_HISTORY_MESSAGES:]
    session.modified = True


def should_search(message: str) -> bool:
    """Detecta se a pergunta provavelmente precisa de dados atuais da internet."""
    triggers = [
        "hoje", "agora", "atual", "atuais", "último", "ultima", "notícia",
        "noticias", "preço", "cotação", "clima", "2026", "internet", "pesquise",
        "procure", "fonte", "recentemente", "lançamento",
    ]
    text = message.lower()
    return any(trigger in text for trigger in triggers)


def search_internet(query: str) -> List[Dict[str, str]]:
    """Pesquisa na web usando DuckDuckGo HTML e devolve resultados resumidos.

    A função não exige chave de API, o que facilita rodar localmente e no Render.
    Em produção, você pode trocar esta função por uma API de busca dedicada.
    """
    try:
        response = requests.get(
            "https://duckduckgo.com/html/",
            params={"q": query, "kl": "br-pt"},
            headers={"User-Agent": "EduAI/1.0 (+https://render.com)"},
            timeout=SEARCH_TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    results: List[Dict[str, str]] = []
    for item in soup.select(".result")[:3]:
        title_el = item.select_one(".result__title a")
        snippet_el = item.select_one(".result__snippet")
        if not title_el:
            continue
        title = title_el.get_text(" ", strip=True)
        url = title_el.get("href", "")
        snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
        results.append({"title": title, "url": url, "snippet": snippet})
    return results


def build_prompt(user_message: str, search_results: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Monta as mensagens enviadas ao modelo mantendo o contexto da sessão."""
    system_prompt = (
        f"Você é {AI_NAME}, uma assistente educacional de IA. "
        "Responda sempre em português brasileiro, com clareza, cordialidade e objetividade. "
        "Use o histórico para manter contexto. Se houver resultados de busca, resuma-os e cite fontes."
    )
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    for item in get_history()[-12:]:
        messages.append({"role": item["role"], "content": item["content"]})

    if search_results:
        sources = "\n".join(
            f"- {result['title']}: {result['snippet']} Fonte: {result['url']}"
            for result in search_results
        )
        messages.append({
            "role": "system",
            "content": f"Resultados recentes encontrados na internet:\n{sources}",
        })

    messages.append({"role": "user", "content": user_message})
    return messages


def fallback_response(user_message: str, search_results: List[Dict[str, str]]) -> str:
    """Resposta local quando a chave da API não está configurada ou há falha externa."""
    if search_results:
        lines = [
            "Encontrei alguns resultados na internet e preparei um resumo inicial:",
            "",
        ]
        for index, result in enumerate(search_results, start=1):
            lines.append(f"{index}. {result['title']}: {result['snippet'] or 'Sem resumo disponível.'}")
            lines.append(f"   Fonte: {result['url']}")
        lines.append("\nConfigure OPENAI_API_KEY para eu transformar esses dados em uma resposta mais completa.")
        return "\n".join(lines)

    return (
        f"Olá! Eu sou a {AI_NAME}. Recebi sua mensagem: “{user_message}”. "
        "Para respostas inteligentes com modelo de linguagem, configure a variável de ambiente OPENAI_API_KEY. "
        "Enquanto isso, posso manter o histórico da sessão e demonstrar o fluxo do chat."
    )


def generate_ai_response(user_message: str) -> Dict[str, Any]:
    """Gera a resposta com OpenAI quando disponível e usa fallback seguro se necessário."""
    search_results = search_internet(user_message) if should_search(user_message) else []
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        return {"reply": fallback_response(user_message, search_results), "sources": search_results}

    try:
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=build_prompt(user_message, search_results),
            temperature=0.7,
            max_tokens=700,
        )
        reply = completion.choices[0].message.content or "Não consegui gerar uma resposta agora."
        return {"reply": reply, "sources": search_results}
    except (OpenAIError, requests.RequestException, KeyError, IndexError) as exc:
        app.logger.warning("Falha ao gerar resposta da IA: %s", exc)
        return {
            "reply": "Desculpe, tive um problema ao consultar a IA. Tente novamente em instantes.",
            "sources": search_results,
        }


@app.route("/")
def index():
    """Renderiza a página inicial do chat."""
    return render_template("index.html", ai_name=AI_NAME)


@app.post("/chat")
def chat():
    """Endpoint JSON que recebe uma mensagem e devolve a resposta da EduAI."""
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"error": "Envie uma mensagem válida."}), 400
    if len(message) > MAX_MESSAGE_LENGTH:
        return jsonify({"error": f"A mensagem deve ter até {MAX_MESSAGE_LENGTH} caracteres."}), 413
    if not re.search(r"\S", message):
        return jsonify({"error": "A mensagem não pode conter apenas espaços."}), 400

    save_message("user", message)
    result = generate_ai_response(message)
    save_message("assistant", result["reply"])

    return jsonify({
        "reply": result["reply"],
        "sources": result.get("sources", []),
        "message_count": len([m for m in get_history() if m["role"] == "user"]),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })


@app.post("/clear")
def clear_chat():
    """Limpa o histórico da conversa salvo na sessão."""
    session["history"] = []
    session.modified = True
    return jsonify({"ok": True, "message_count": 0})


@app.errorhandler(413)
def request_too_large(_error):
    """Resposta amigável para requisições maiores que o permitido."""
    return jsonify({"error": "Requisição muito grande."}), 413


@app.errorhandler(500)
def internal_error(_error):
    """Evita expor detalhes internos em caso de erro inesperado."""
    return jsonify({"error": "Erro interno. Tente novamente mais tarde."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
