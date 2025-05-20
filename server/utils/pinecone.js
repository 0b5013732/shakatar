const logger = require('./logger');

let client;
let index;
const indexName = process.env.PINECONE_INDEX || 'shaka';

async function getClient() {
  if (!client) {
    const mod = await import('@pinecone-database/pinecone');
    const Pinecone = mod.PineconeClient || mod.Pinecone;
    client = new Pinecone();
    await client.init({
      apiKey: process.env.PINECONE_API_KEY || '',
      environment: process.env.PINECONE_ENV || 'us-east1-gcp',
      host: process.env.PINECONE_HOST || undefined
    });
    index = client.Index ? client.Index(indexName) : client.index(indexName);
  }
  return { client, index };
}

async function queryRelevant(text, topK = 5) {
  const vector = Array(1536).fill(0); // TODO: replace with real embeddings
  try {
    const { index } = await getClient();
    const result = await index.query({
      queryRequest: { vector, topK }
    });
    return result.matches || [];
  } catch (err) {
    logger.error(`Pinecone query error: ${err}`);
    return [];
  }
}

module.exports = { queryRelevant };
