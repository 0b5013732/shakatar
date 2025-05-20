const { PineconeClient } = require('@pinecone-database/pinecone');
const logger = require('./logger');

const client = new PineconeClient();
client.init({
  apiKey: process.env.PINECONE_API_KEY || '',

  environment: process.env.PINECONE_ENV || 'us-east1-gcp',
  host: process.env.PINECONE_HOST

});

const indexName = process.env.PINECONE_INDEX || 'shaka';
const embeddingModel = process.env.PINECONE_EMBEDDING_MODEL || 'text-embedding-3-large';

async function queryRelevant(text, topK = 5) {
  // Placeholder embedding step - replace with your embedding function
  const vector = Array(1536).fill(0); // dummy vector
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


module.exports = { client, queryRelevant, embeddingModel };