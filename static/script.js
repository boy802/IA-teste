// Controla a experiência do chat sem recarregar a página.
const form = document.querySelector('#chatForm');
const input = document.querySelector('#messageInput');
const historyBox = document.querySelector('#chatHistory');
const typing = document.querySelector('#typing');
const clearBtn = document.querySelector('#clearBtn');
const messageCount = document.querySelector('#messageCount');

function scrollToBottom() {
  historyBox.scrollTop = historyBox.scrollHeight;
}

function removeEmptyState() {
  const empty = historyBox.querySelector('.empty-state');
  if (empty) empty.remove();
}

function addMessage(role, content) {
  removeEmptyState();
  const article = document.createElement('article');
  article.className = `message ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? 'Você' : 'AI';

  const paragraph = document.createElement('p');
  paragraph.textContent = content;

  article.append(avatar, paragraph);
  historyBox.appendChild(article);
  scrollToBottom();
}

function setLoading(isLoading) {
  typing.hidden = !isLoading;
  form.querySelector('button').disabled = isLoading;
  input.disabled = isLoading;
}

input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = `${input.scrollHeight}px`;
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) return;

  addMessage('user', message);
  input.value = '';
  input.style.height = 'auto';
  setLoading(true);

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });
    const data = await response.json();

    if (!response.ok) {
      addMessage('assistant', data.error || 'Não foi possível responder agora.');
      return;
    }

    let answer = data.answer;
    if (data.sources && data.sources.length) {
      const sources = data.sources.map((source, index) => `${index + 1}. ${source.title} - ${source.url}`).join('\n');
      answer += `\n\nFontes utilizadas:\n${sources}`;
    }
    addMessage('assistant', answer);
    messageCount.textContent = data.message_count;
  } catch (error) {
    addMessage('assistant', 'Falha de conexão. Verifique sua internet e tente novamente.');
  } finally {
    setLoading(false);
    input.focus();
  }
});

clearBtn.addEventListener('click', async () => {
  await fetch('/api/clear', { method: 'POST' });
  historyBox.innerHTML = '<div class="empty-state"><h2>Conversa limpa.</h2><p>Envie uma nova mensagem para começar.</p></div>';
  messageCount.textContent = '0';
  input.focus();
});

scrollToBottom();
