const axios = require('axios');

// Query a locally running Llama-based model to generate an answer.
// The server URL and model name can be customised via the environment
// variables `LLAMA_ENDPOINT` and `LLAMA_MODEL`.
async function generateAnswer(question, context = []) {
  const endpoint = process.env.LLAMA_ENDPOINT ||
    'http://localhost:11434/v1/chat/completions';
  const model = process.env.LLAMA_MODEL || 'llama';

  const ctxText = context
    .map((doc) => doc.metadata?.text || doc.text)
    .join('\n');
  const systemPrompt = ctxText
    ? `Use the following context to answer the question:\n${ctxText}`
    : 'Answer the user';

  const payload = {
    model,
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: question }
    ],
    stream: false
  };

  try {
    const res = await axios.post(endpoint, payload, { timeout: 30000 });
    return (
      res.data?.choices?.[0]?.message?.content?.trim() ||
      res.data?.content?.trim() || ''
    );
  } catch (err) {
    throw new Error(`LLM request failed: ${err.message}`);
  }
}

module.exports = { generateAnswer };
