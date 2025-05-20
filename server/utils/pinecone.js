const { Pinecone } = require('@pinecone-database/pinecone');
const logger = require('./logger');

// Initialize Pinecone client. Version 6 uses the `Pinecone` class.
const client = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY || '',
  environment: process.env.PINECONE_ENV || 'us-east1-gcp',
  host: process.env.PINECONE_HOST // optional custom host
});

const indexName = process.env.PINECONE_INDEX || 'shaka';
// Embedding model name used when generating vectors
const embeddingModel = process.env.PINECONE_EMBEDDING_MODEL || 'text-embedding-ada-002';

const index = client.index(indexName);

async function queryRelevant(text, topK = 5) {
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

module.exports = { client, index, embeddingModel, queryRelevant };
