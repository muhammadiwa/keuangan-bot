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
import { S3Client, HeadBucketCommand, CreateBucketCommand } from '@aws-sdk/client-s3';
import { Upload } from '@aws-sdk/lib-storage';
import crypto from 'crypto';
import path from 'path';
import fs from 'fs';

dotenv.config();

const app = express();
const logger = pino({ level: process.env.LOG_LEVEL || 'info' });
const PORT = process.env.PORT || 3000;
const BACKEND_API_URL = process.env.BACKEND_API_URL || 'http://localhost:8000';
const SESSION_ID = process.env.SESSION_ID || 'default';
const MEDIA_BUCKET = process.env.MEDIA_BUCKET || 'wa-media';
const MINIO_ENDPOINT = process.env.MINIO_ENDPOINT || 'http://localhost:9000';
const MINIO_REGION = process.env.MINIO_REGION || 'us-east-1';
const MINIO_ACCESS_KEY = process.env.MINIO_ACCESS_KEY || 'minio';
const MINIO_SECRET_KEY = process.env.MINIO_SECRET_KEY || 'miniopass';
const MINIO_PUBLIC_URL = process.env.MINIO_PUBLIC_URL || `${MINIO_ENDPOINT}/${MEDIA_BUCKET}`;

app.use(express.json({ limit: '10mb' }));

const sessions = new Map();
const qrCache = new NodeCache({ stdTTL: 120 });

const s3Client = new S3Client({
  forcePathStyle: true,
  endpoint: MINIO_ENDPOINT,
  region: MINIO_REGION,
  credentials: {
    accessKeyId: MINIO_ACCESS_KEY,
    secretAccessKey: MINIO_SECRET_KEY,
  },
});

const ensureBucket = async () => {
  try {
    await s3Client.send(new HeadBucketCommand({ Bucket: MEDIA_BUCKET }));
  } catch (error) {
    if (error?.$metadata?.httpStatusCode === 404 || error?.name === 'NotFound') {
      try {
        await s3Client.send(new CreateBucketCommand({ Bucket: MEDIA_BUCKET }));
        logger.info({ bucket: MEDIA_BUCKET }, 'Created media bucket');
      } catch (createErr) {
        logger.error({ err: createErr }, 'Failed to create bucket');
      }
    } else {
      logger.warn({ err: error }, 'Bucket check failed');
    }
  }
};

app.get('/healthz', (_, res) => {
  res.json({ status: 'ok', session: Array.from(sessions.keys()) });
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

const forwardToBackend = async (payload) => {
  try {
    await axios.post(`${BACKEND_API_URL}/wa/incoming`, payload, { timeout: 10000 });
  } catch (error) {
    logger.error({ err: error, payload }, 'Failed forwarding to backend');
  }
};

const storeMedia = async (stream, { folder, contentType }) => {
  const timestamp = new Date().toISOString().replace(/[:]/g, '-');
  const fileName = `${folder}/${timestamp}-${crypto.randomUUID()}`;
  const upload = new Upload({
    client: s3Client,
    params: {
      Bucket: MEDIA_BUCKET,
      Key: fileName,
      Body: stream,
      ContentType: contentType,
    },
  });
  await upload.done();
  return `${MINIO_PUBLIC_URL}/${fileName}`;
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
      qrCache.set(sessionId, qr);
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

ensureBucket().catch((err) => logger.error({ err }, 'Failed ensuring media bucket'));
startSession(SESSION_ID).catch((err) => logger.error({ err }, 'Failed to start session'));

app.listen(PORT, () => {
  logger.info(`wa-bot-service listening on port ${PORT}`);
});
