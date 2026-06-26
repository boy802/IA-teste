# EduAI

EduAI é um chat educacional em Python + Flask com interface escura inspirada no ChatGPT, histórico por sessão e integração exclusiva com a API da Groq.

## Recursos

* Interface moderna, responsiva e em tema escuro.
* Histórico visual da conversa salvo por sessão do navegador.
* Indicador **"EduAI está digitando..."**.
* Botão para limpar conversa.
* Contador de mensagens.
* Pesquisa web gratuita via DuckDuckGo HTML para perguntas recentes/atuais, com logs detalhados e endpoint `/test-search`.
* Tratamento amigável para chave ausente, chave inválida, limite da API, conexão, timeout e erros inesperados.
* Deploy pronto para Render com `Procfile` e `render.yaml`.

## Estrutura

```text
app.py
templates/
  index.html
static/
  style.css
  script.js
requirements.txt
Procfile
render.yaml
README.md
```

## Variáveis de ambiente

Configure no Render:

| Variável           | Obrigatória | Valor sugerido                     |
| ------------------ | ----------- | ---------------------------------- |
| `GROQ_API_KEY`     | Sim         | Sua chave da Groq                  |
| `GROQ_MODEL`       | Não         | `llama-3.1-8b-instant`             |
| `FLASK_SECRET_KEY` | Recomendado | Gerado automaticamente pelo Render |

> Configure apenas as variáveis da Groq. O provedor deste projeto é exclusivamente a Groq.

## Como executar localmente

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY="sua-chave-da-groq"
export GROQ_MODEL="llama-3.1-8b-instant"
python app.py
```

Acesse `http://localhost:5000`.

## Deploy no Render

1. Faça push deste repositório para o GitHub.
2. No Render, crie um novo Web Service apontando para o repositório.
3. Use as configurações do `render.yaml` ou defina manualmente:

   * Build command: `pip install -r requirements.txt`
   * Start command: `gunicorn app:app`
4. Configure `GROQ_API_KEY`.
5. Publique o serviço.

## Endpoints

* `GET /` renderiza a interface.
* `POST /api/chat` envia mensagens para a EduAI.
* `POST /api/clear` limpa o histórico da sessão.
* `GET /health` retorna status básico do serviço e versões de dependências.
* `GET /test-search?q=preço do dólar hoje` retorna resultados brutos da pesquisa web para depuração.

## Validação

Execute:

```bash
python -m py_compile app.py
git diff --check

python - <<'PY'
import requests
from bs4 import BeautifulSoup

print(requests.__version__)
print(BeautifulSoup('<p>ok</p>', 'html.parser').get_text())
PY
```
