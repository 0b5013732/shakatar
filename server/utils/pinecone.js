const { PineconeClient } = require('@pinecone-database/pinecone');
const logger = require('./logger');

const client = new PineconeClient();
client.init({
  apiKey: process.env.PINECONE_API_KEY || '',
  environment: process.env.PINECONE_ENV || 'us-east1-gcp',
  host: process.env.PINECONE_HOST

});

const indexName = process.env.PINECONE_INDEX || 'shaka';

async function embed(text) {
  // Placeholder embedding function; replace with real model
  return Array(1536).fill(0);
}

async function queryRelevant(text, topK = 5) {
  const vector = await embed(text);

  try {
    const result = await client.query({
      indexName,
      queryRequest: { vector, topK }
    });
    return result.matches || [];
  } catch (err) {
    logger.error(`Pinecone query error: ${err}`);
    return [];
  }
}

module.exports = { client, queryRelevant };
