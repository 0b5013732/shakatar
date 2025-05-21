const express = require('express');
const router = express.Router();
const llm = require('../services/llm');
const logger = require('../utils/logger');

router.post('/', async (req, res) => {
  const { question } = req.body;
  try {
    const answer = await llm.generateAnswer(question);
    logger.info(`Q: ${question}\nA: ${answer}`);
    res.json({ answer });
  } catch (err) {
    logger.error(`Chat error: ${err}`);
    res.status(500).json({ error: 'Failed to generate answer' });
  }
});

module.exports = router;
