const express = require('express');
const cors = require('cors');
const path = require('path');
const dotenv = require('dotenv');
const logger = require('./utils/logger');

// Load environment variables from the repo root `.env` file, even if the server
// is started from within the `server/` directory.
dotenv.config({ path: path.join(__dirname, '../.env') });


const chatRouter = require('./routes/chat');
const audioRouter = require('./routes/audio');

const app = express();
app.use(cors());
app.use(express.json({ limit: '2mb' }));

app.use('/chat', chatRouter);
app.use('/audio', audioRouter);

// serve static client
app.use(express.static(path.join(__dirname, '../client')));

app.use((err, req, res, next) => {
  logger.error(err);
  res.status(500).json({ error: 'Internal server error' });
});

const port = process.env.PORT || 3434;
app.listen(port, () => logger.info(`Server running on port ${port}`));
