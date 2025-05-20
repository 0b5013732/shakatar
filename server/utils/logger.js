const { createLogger, format, transports } = require('winston');
const path = require('path');

const logger = createLogger({
  level: 'info',
  format: format.combine(
    format.timestamp(),
    format.printf(({ level, message, timestamp }) => `${timestamp} [${level}] ${message}`)
  ),
  transports: [
    new transports.File({ filename: path.join(__dirname, '../../logs/server.log') }),
    new transports.Console()
  ]
});

module.exports = logger;
