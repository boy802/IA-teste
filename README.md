# EduAI — Chatbot com Flask e Groq

EduAI é uma aplicação web de inteligência artificial inspirada em chatbots modernos. O projeto usa Python, Flask, HTML, CSS e JavaScript, mantém o histórico da conversa durante a sessão e consulta a internet quando a pergunta parece exigir informações atuais.

## Funcionalidades

- Interface moderna em tema escuro por padrão.
- Chat responsivo para celular e computador.
- Mensagens do usuário à direita e respostas da EduAI à esquerda.
- Indicador “EduAI está digitando...” enquanto a resposta é processada.
- Histórico de conversa salvo na sessão do navegador.
- Contador de mensagens enviadas.
- Botão para limpar a conversa.
- Endpoint `POST /chat` retornando JSON.
- Pesquisa web via DuckDuckGo HTML quando necessário.
- Integração com a API da Groq usando a biblioteca `openai` em modo compatível.
- Tratamento amigável para chave ausente, chave inválida, limite de uso e falhas de conexão.

## Estrutura

```text
app.py
templates/index.html
static/style.css
static/script.js
requirements.txt
Procfile
render.yaml
```

## Configuração local

1. Crie e ative um ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Configure as variáveis de ambiente:

```bash
export SECRET_KEY="troque-esta-chave-local"
export GROQ_API_KEY="gsk_sua_chave_da_groq"
export GROQ_MODEL="llama-3.1-8b-instant"
```

> A chave da Groq pode ser criada no console da Groq. A aplicação usa `https://api.groq.com/openai/v1` como URL base compatível com o cliente OpenAI.

4. Execute a aplicação:

```bash
python app.py
```

5. Abra no navegador:

```text
http://localhost:5000
```

## Deploy no Render

O projeto já inclui `Procfile` e `render.yaml`. No Render, configure a variável secreta:

- `GROQ_API_KEY`: sua chave da Groq com prefixo `gsk_`.

As demais variáveis já estão descritas no `render.yaml`:

- `SECRET_KEY`: gerada automaticamente pelo Render.
- `GROQ_MODEL`: `llama-3.1-8b-instant` por padrão.
- `PYTHON_VERSION`: `3.11.9`.

## Como a integração com Groq funciona

A Groq oferece compatibilidade com o formato da API da OpenAI. Por isso, o projeto utiliza a classe `OpenAI` apontando para a URL base da Groq:

```python
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)
```

O modelo padrão pode ser alterado pela variável `GROQ_MODEL` sem mudanças no código.

## Observações de segurança

- Nunca coloque a chave da Groq diretamente no código.
- Use variáveis de ambiente localmente e no Render.
- O backend limita mensagens a 1200 caracteres.
- Respostas de erro não expõem detalhes internos da aplicação ao usuário final.
