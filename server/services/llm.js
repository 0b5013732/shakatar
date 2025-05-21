const axios = require('axios');

// Query a locally running Llama-based model to generate an answer.
// The server URL and model name can be customised via the environment
// variables `LLAMA_ENDPOINT` and `LLAMA_MODEL`.
async function generateAnswer(question) {
  const endpoint = process.env.LLAMA_ENDPOINT ||
    'http://localhost:11434/v1/chat/completions';
  const model = process.env.LLAMA_MODEL || 'llama';
  const systemPrompt = 'Answer the user';

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
