const form = document.querySelector('#chatForm');
const input = document.querySelector('#messageInput');
const messages = document.querySelector('#messages');
const typing = document.querySelector('#typing');
const clearButton = document.querySelector('#clearChat');
const messageCount = document.querySelector('#messageCount');

function scrollToBottom() {
  messages.scrollTop = messages.scrollHeight;
}

function createMessage(role, text) {
  const article = document.createElement('article');
  article.className = `message ${role === 'user' ? 'user-message' : 'ai-message'}`;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? 'Você' : 'E';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  const paragraph = document.createElement('p');
  paragraph.textContent = text;
  bubble.appendChild(paragraph);

  article.appendChild(avatar);
  article.appendChild(bubble);
  messages.appendChild(article);
  scrollToBottom();
}

function setLoading(isLoading) {
  typing.classList.toggle('hidden', !isLoading);
  form.querySelector('button').disabled = isLoading;
  input.disabled = isLoading;
  if (isLoading) scrollToBottom();
}

input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
});

input.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text) return;

  createMessage('user', text);
  input.value = '';
  input.style.height = 'auto';
  setLoading(true);

  try {
    const response = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Falha ao enviar mensagem.');
    createMessage('assistant', data.reply);
    messageCount.textContent = data.message_count ?? messageCount.textContent;
  } catch (error) {
    createMessage('assistant', `Ops! ${error.message}`);
  } finally {
    setLoading(false);
    input.focus();
  }
});

clearButton.addEventListener('click', async () => {
  try {
    await fetch('/clear', { method: 'POST' });
    messages.innerHTML = '';
    createMessage('assistant', 'Conversa limpa. Como posso ajudar agora?');
    messageCount.textContent = '0';
  } catch (_error) {
    createMessage('assistant', 'Não consegui limpar a conversa agora.');
  }
});

scrollToBottom();
