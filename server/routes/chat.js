const express = require('express');
const router = express.Router();
const llm = require('../services/llm');
const { queryRelevant } = require('../utils/pinecone');
const logger = require('../utils/logger');

router.post('/', async (req, res) => {
  const { question } = req.body;
  try {
    const context = await queryRelevant(question);
    const answer = await llm.generateAnswer(question, context);
    logger.info(`Q: ${question}\nA: ${answer}`);
    res.json({ answer });
  } catch (err) {
    logger.error(`Chat error: ${err}`);
    res.status(500).json({ error: 'Failed to generate answer' });
  }
});

module.exports = router;
