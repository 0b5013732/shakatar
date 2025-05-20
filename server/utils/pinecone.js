const { Pinecone } = require('@pinecone-database/pinecone');
const logger = require('./logger');

const pinecone = new Pinecone();
let index;
const indexName = process.env.PINECONE_INDEX || 'shaka';

async function init() {
  if (index) return;
  try {
    await pinecone.init({
      apiKey: process.env.PINECONE_API_KEY || '',
      environment: process.env.PINECONE_ENV || 'us-east1-gcp',
      host: process.env.PINECONE_HOST
    });
    index = pinecone.Index(indexName);
  } catch (err) {
    logger.error(`Pinecone init error: ${err}`);
  }
}

async function queryRelevant(text, topK = 5) {
  await init();
  // Placeholder embedding step - replace with your embedding function
  const vector = Array(1536).fill(0); // dummy vector
  try {
    const result = await index.query({ vector, topK });
    return result.matches || [];
  } catch (err) {
    logger.error(`Pinecone query error: ${err}`);
    return [];
  }
}

module.exports = { init, queryRelevant };
