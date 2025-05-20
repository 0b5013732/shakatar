const express = require('express');
const router = express.Router();
const tts = require('../services/tts');
const logger = require('../utils/logger');

router.post('/', async (req, res) => {
  const { text } = req.body;
  try {
    const audio = await tts.textToSpeech(text);
    logger.info(`Audio generated for text length ${text.length}`);
    res.set('Content-Type', 'audio/mpeg');
    res.send(audio);
  } catch (err) {
    logger.error(`Audio error: ${err}`);
    res.status(500).json({ error: 'Failed to generate audio' });
  }
});

module.exports = router;
