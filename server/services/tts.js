const axios = require('axios');

async function textToSpeech(text) {
  const apiKey = process.env.ELEVENLABS_API_KEY;
  const voiceId = process.env.ELEVENLABS_VOICE_ID || '21m00Tcm4TlvDq8ikWAM';
  const url = `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`;

  const response = await axios({
    method: 'POST',
    url,
    data: { text },
    headers: {
      'xi-api-key': apiKey,
      'Content-Type': 'application/json'
    },
    responseType: 'arraybuffer'
  });

  return response.data; // Buffer containing mp3 audio
}

module.exports = { textToSpeech };
