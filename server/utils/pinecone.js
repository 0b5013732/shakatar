
const { Pinecone } = require('@pinecone-database/pinecone');
const logger = require('./logger');

// Initialise the Pinecone client using the API key and environment from the
// environment variables.  The `Pinecone` constructor is available in the
// latest versions of the SDK and replaces the older `PineconeClient` class.
// The latest Pinecone Node SDK expects only an apiKey and optional
// controllerHostUrl when instantiating the client.  Passing legacy
// `environment` or `host` properties results in a PineconeArgumentError.
const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY || '',
  controllerHostUrl: process.env.PINECONE_HOST
});

const indexName = process.env.PINECONE_INDEX || 'shaka';
const embeddingModel = process.env.PINECONE_EMBEDDING_MODEL || 'text-embedding-3-large';
const index = pinecone.index(indexName);


async function queryRelevant(text, topK = 5) {
  // Placeholder embedding step - replace with your embedding function.
  const vector = Array(1536).fill(0); // dummy vector
  try {


    const result = await index.query({ vector, topK });
    return result.matches || [];
  } catch (err) {
    logger.error(`Pinecone query error: ${err}`);
    return [];
  }
}



module.exports = { pinecone, index, queryRelevant, embeddingModel };
