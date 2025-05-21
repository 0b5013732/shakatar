const { useState } = React;

function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');

  const sendMessage = async () => {
    if (!input) return;
    const userMsg = { role: 'user', text: input };
    setMessages(prev => [...prev, userMsg]);
    const resp = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: input })
    });
    const data = await resp.json();
    setMessages(prev => [...prev, { role: 'bot', text: data.answer }]);
    setInput('');
  };

  return (
    <div id="chat">
      {messages.map((m, i) => (
        <div key={i} className={`msg ${m.role}`}>
          <span>{m.text}</span>
        </div>
      ))}
      <div className="input-row">
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && sendMessage()}
        />
        <button onClick={sendMessage}>Send</button>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<Chat />);
