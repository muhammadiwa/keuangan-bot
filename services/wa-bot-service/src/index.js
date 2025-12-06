import express from 'express';
import dotenv from 'dotenv';
import pino from 'pino';
import axios from 'axios';
import makeWASocket, {
  useMultiFileAuthState,
  fetchLatestBaileysVersion,
  DisconnectReason,
  makeCacheableSignalKeyStore,
  downloadContentFromMessage,
} from '@whiskeysockets/baileys';
import NodeCache from 'node-cache';
import { Boom } from '@hapi/boom';
import crypto from 'crypto';
import path from 'path';
import fs from 'fs';
import qrcode from 'qrcode-terminal';

// Storage abstraction
import { createStorage, getStorageConfig } from './storage/index.js';

dotenv.config();

const app = express();
const logger = pino({ level: process.env.LOG_LEVEL || 'info' });
const PORT = process.env.PORT || 3000;
const BACKEND_API_URL = process.env.BACKEND_API_URL || 'http://localhost:8000';
const BACKEND_TIMEOUT_MS = (() => {
  const parsed = Number(process.env.BACKEND_TIMEOUT_MS);
  if (Number.isFinite(parsed) && parsed > 0) {
    return parsed;
  }
  return 60000;
})();
const SESSION_ID = process.env.SESSION_ID || 'default';

app.use(express.json({ limit: '10mb' }));

const sessions = new Map();
const qrCache = new NodeCache({ stdTTL: 120 });

// Initialize storage based on configuration
const storageConfig = getStorageConfig();
const storage = createStorage(storageConfig, logger);

// Serve local media files if using local storage
if (storageConfig.type === 'local') {
  const localBasePath = path.resolve(storageConfig.local.basePath);

  app.get('/media/*', (req, res) => {
    const relativePath = req.params[0];
    const filePath = path.join(localBasePath, relativePath);

    // Security: prevent directory traversal
    if (!filePath.startsWith(localBasePath)) {
      return res.status(403).json({ message: 'Forbidden' });
    }

    // Check if file exists
    if (!fs.existsSync(filePath)) {
      return res.status(404).json({ message: 'File not found' });
    }

    // Get content type
    const contentType = storage.getContentType(filePath);
    res.setHeader('Content-Type', contentType);

    // Stream file
    const stream = fs.createReadStream(filePath);
    stream.pipe(res);
  });

  logger.info({ basePath: localBasePath }, 'Local media serving enabled at /media/*');
}

app.get('/healthz', (_, res) => {
  res.json({
    status: 'ok',
    session: Array.from(sessions.keys()),
    storage: storage.getType(),
  });
});

app.get('/session/:id/qr', (req, res) => {
  const qr = qrCache.get(req.params.id);
  if (!qr) {
    return res.status(404).json({ message: 'QR not available' });
  }
  res.json({ sessionId: req.params.id, qr });
});

app.post('/deliver', async (req, res) => {
  const { to, message } = req.body;
  if (!to || !message) {
    return res.status(400).json({ message: 'Missing to or message' });
  }
  try {
    const session = sessions.get(SESSION_ID);
    if (!session) {
      return res.status(503).json({ message: 'Session not ready' });
    }
    await session.socket.sendMessage(`${to}@s.whatsapp.net`, { text: message });
    res.json({ status: 'sent' });
  } catch (error) {
    logger.error({ err: error }, 'Failed to send WhatsApp message');
    res.status(500).json({ message: 'Failed to send message' });
  }
});

app.post('/webhook/test', async (req, res) => {
  logger.info({ body: req.body }, 'Received webhook test payload');
  res.json({ status: 'ok' });
});

// Storage info endpoint
app.get('/storage/info', (_, res) => {
  res.json({
    type: storage.getType(),
    config: storageConfig.type === 'local'
      ? { basePath: storageConfig.local.basePath, baseUrl: storageConfig.local.baseUrl }
      : { endpoint: storageConfig.minio.endpoint, bucket: storageConfig.minio.bucket },
  });
});

const forwardToBackend = async (payload) => {
  try {
    const { data } = await axios.post(`${BACKEND_API_URL}/wa/incoming`, payload, {
      timeout: BACKEND_TIMEOUT_MS,
    });
    const reply = data?.reply;
    if (reply) {
      const sessionEntry = sessions.get(SESSION_ID);
      if (!sessionEntry?.socket) {
        logger.warn({ sessionId: SESSION_ID }, 'Cannot send reply, session not ready');
        return;
      }
      try {
        await sessionEntry.socket.sendMessage(`${payload.from_number}@s.whatsapp.net`, { text: reply });
        logger.info({ to: payload.from_number }, 'Delivered reply from backend');
      } catch (sendError) {
        logger.error({ err: sendError, to: payload.from_number }, 'Failed sending reply to WhatsApp');
      }
    }
  } catch (error) {
    if (axios.isAxiosError(error) && error.code === 'ECONNABORTED') {
      logger.error(
        { err: error, payload, timeoutMs: BACKEND_TIMEOUT_MS },
        'Backend request timed out before completing'
      );
    } else {
      logger.error({ err: error, payload }, 'Failed forwarding to backend');
    }
  }
};

/**
 * Store media using configured storage backend
 * @param {Buffer} buffer - Media buffer
 * @param {Object} options - Store options
 * @returns {Promise<string>} Media URL
 */
const storeMedia = async (buffer, { folder, contentType }) => {
  return storage.store(buffer, { folder, contentType });
};

const normalizeMessage = async (message) => {
  const { messageTimestamp, key, message: content } = message;
  if (!content || key.fromMe) return null;

  const remoteJid = key.remoteJid || '';
  const fromNumber = remoteJid.replace(/@s\.whatsapp\.net$/, '').replace(/@g\.us$/, '');
  const tsSeconds = messageTimestamp != null ? Number(messageTimestamp) : Date.now() / 1000;
  const timestamp = new Date(tsSeconds * 1000);

  if (content.conversation || content.extendedTextMessage) {
    const text = content.conversation || content.extendedTextMessage?.text || '';
    return {
      from_number: fromNumber,
      message_type: 'text',
      text,
      media_url: null,
      timestamp,
    };
  }

  if (content.audioMessage) {
    const stream = await downloadContentFromMessage(content.audioMessage, 'audio');
    const bufferArray = [];
    for await (const chunk of stream) {
      bufferArray.push(chunk);
    }
    const mediaBuffer = Buffer.concat(bufferArray);
    const mediaUrl = await storeMedia(mediaBuffer, { folder: 'audio', contentType: 'audio/ogg' });
    return {
      from_number: fromNumber,
      message_type: 'audio',
      text: null,
      media_url: mediaUrl,
      timestamp,
    };
  }

  if (content.imageMessage) {
    const stream = await downloadContentFromMessage(content.imageMessage, 'image');
    const bufferArray = [];
    for await (const chunk of stream) {
      bufferArray.push(chunk);
    }
    const mediaBuffer = Buffer.concat(bufferArray);
    const mimeType = content.imageMessage.mimetype || 'image/jpeg';
    const mediaUrl = await storeMedia(mediaBuffer, { folder: 'images', contentType: mimeType });
    return {
      from_number: fromNumber,
      message_type: 'image',
      text: content.imageMessage.caption || null,
      media_url: mediaUrl,
      timestamp,
    };
  }

  if (content.documentMessage) {
    const stream = await downloadContentFromMessage(content.documentMessage, 'document');
    const bufferArray = [];
    for await (const chunk of stream) {
      bufferArray.push(chunk);
    }
    const mediaBuffer = Buffer.concat(bufferArray);
    const mimeType = content.documentMessage.mimetype || 'application/octet-stream';
    const mediaUrl = await storeMedia(mediaBuffer, { folder: 'documents', contentType: mimeType });
    return {
      from_number: fromNumber,
      message_type: 'document',
      text: content.documentMessage.caption || content.documentMessage.fileName || null,
      media_url: mediaUrl,
      timestamp,
    };
  }

  return null;
};

const startSession = async (sessionId) => {
  const sessionPath = path.join(process.cwd(), 'sessions', sessionId);
  fs.mkdirSync(sessionPath, { recursive: true });
  const { state, saveCreds } = await useMultiFileAuthState(sessionPath);
  const { version } = await fetchLatestBaileysVersion();
  const socket = makeWASocket({
    version,
    auth: {
      creds: state.creds,
      keys: makeCacheableSignalKeyStore(state.keys, pino({ level: 'silent' })),
    },
    logger,
    printQRInTerminal: false,
    browser: ['KeuanganBot', 'Chrome', '1.0.0'],
  });

  socket.ev.on('creds.update', saveCreds);
  socket.ev.on('connection.update', (update) => {
    const { connection, lastDisconnect, qr } = update;
    if (qr) {
      const cachedQr = qrCache.get(sessionId);
      qrCache.set(sessionId, qr);
      if (cachedQr !== qr) {
        logger.info({ sessionId }, 'WhatsApp QR code generated, scan the code below');
        qrcode.generate(qr, { small: true });
      }
    }
    if (connection === 'close') {
      const shouldReconnect = (lastDisconnect?.error instanceof Boom && lastDisconnect.error.output.statusCode !== DisconnectReason.loggedOut);
      logger.warn({ sessionId, reason: lastDisconnect?.error }, 'WhatsApp connection closed');
      if (shouldReconnect) {
        startSession(sessionId).catch((err) => logger.error({ err }, 'Failed restarting session'));
      } else {
        sessions.delete(sessionId);
      }
    } else if (connection === 'open') {
      logger.info({ sessionId }, 'WhatsApp connection established');
    }
  });

  socket.ev.on('messages.upsert', async ({ messages }) => {
    for (const message of messages) {
      try {
        const payload = await normalizeMessage(message);
        if (payload) {
          logger.info({ payload }, 'Forwarding incoming message to backend');
          await forwardToBackend({
            ...payload,
            timestamp: payload.timestamp.toISOString(),
          });
        }
      } catch (error) {
        logger.error({ err: error }, 'Failed processing incoming message');
      }
    }
  });

  sessions.set(sessionId, { socket });
};

// Initialize storage and start application
const initialize = async () => {
  try {
    // Initialize storage
    await storage.initialize();
    logger.info({ storageType: storage.getType() }, 'Storage initialized successfully');

    // Start WhatsApp session
    await startSession(SESSION_ID);

    // Start HTTP server
    app.listen(PORT, () => {
      logger.info({ port: PORT, storageType: storage.getType() }, 'wa-bot-service started');
    });
  } catch (error) {
    logger.error({ err: error }, 'Failed to initialize application');
    process.exit(1);
  }
};

initialize();
