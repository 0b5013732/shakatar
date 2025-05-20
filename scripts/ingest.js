// Simple ingestion script to read text files and produce a JSONL corpus
const fs = require('fs');
const path = require('path');

const sourceDir = process.argv[2] || path.join(__dirname, '../data/corpus');
const outFile = path.join(__dirname, '../data/processed/corpus.jsonl');

function readFiles(dir) {
  const files = fs.readdirSync(dir);
  const docs = [];
  for (const file of files) {
    const full = path.join(dir, file);
    if (fs.statSync(full).isDirectory()) continue;
    const text = fs.readFileSync(full, 'utf8');
    docs.push({ file, text });
  }
  return docs;
}

function preprocess(text) {
  return text.replace(/\r?\n+/g, ' ').trim();
}

function main() {
  const docs = readFiles(sourceDir);
  const stream = fs.createWriteStream(outFile);
  docs.forEach(doc => {
    const cleaned = preprocess(doc.text);
    stream.write(JSON.stringify({ text: cleaned }) + '\n');
  });
  stream.end();
  console.log(`Wrote ${docs.length} documents to ${outFile}`);
}

main();
