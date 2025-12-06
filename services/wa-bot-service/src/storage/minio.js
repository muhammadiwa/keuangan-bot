/**
 * MinIO/S3 Storage Implementation
 */

import { S3Client, HeadBucketCommand, CreateBucketCommand, GetObjectCommand, DeleteObjectCommand } from '@aws-sdk/client-s3';
import { Upload } from '@aws-sdk/lib-storage';
import crypto from 'crypto';

export class MinioStorage {
    /**
     * @param {Object} config - MinIO configuration
     * @param {string} config.endpoint - MinIO endpoint URL
     * @param {string} config.region - AWS region
     * @param {string} config.accessKey - Access key
     * @param {string} config.secretKey - Secret key
     * @param {string} config.bucket - Bucket name
     * @param {string} config.publicUrl - Public URL for accessing files
     * @param {Object} logger - Logger instance
     */
    constructor(config, logger) {
        this.config = config;
        this.logger = logger;
        this.bucket = config.bucket;
        this.publicUrl = config.publicUrl;

        this.client = new S3Client({
            forcePathStyle: true,
            endpoint: config.endpoint,
            region: config.region,
            credentials: {
                accessKeyId: config.accessKey,
                secretAccessKey: config.secretKey,
            },
        });
    }

    /**
     * Get storage type name
     * @returns {string}
     */
    getType() {
        return 'minio';
    }

    /**
     * Initialize storage - ensure bucket exists
     * @returns {Promise<void>}
     */
    async initialize() {
        try {
            await this.client.send(new HeadBucketCommand({ Bucket: this.bucket }));
            this.logger.info({ bucket: this.bucket }, 'MinIO bucket exists');
        } catch (error) {
            if (error?.$metadata?.httpStatusCode === 404 || error?.name === 'NotFound') {
                try {
                    await this.client.send(new CreateBucketCommand({ Bucket: this.bucket }));
                    this.logger.info({ bucket: this.bucket }, 'Created MinIO bucket');
                } catch (createErr) {
                    this.logger.error({ err: createErr }, 'Failed to create bucket');
                    throw createErr;
                }
            } else {
                this.logger.warn({ err: error }, 'Bucket check failed');
                throw error;
            }
        }
    }

    /**
     * Store file to MinIO
     * @param {Buffer} buffer - File buffer
     * @param {Object} options - Store options
     * @param {string} options.folder - Folder/prefix
     * @param {string} options.contentType - MIME type
     * @returns {Promise<string>} Public URL of stored file
     */
    async store(buffer, { folder, contentType }) {
        const timestamp = new Date().toISOString().replace(/[:]/g, '-');
        const extension = this._getExtension(contentType);
        const fileName = `${folder}/${timestamp}-${crypto.randomUUID()}${extension}`;

        const upload = new Upload({
            client: this.client,
            params: {
                Bucket: this.bucket,
                Key: fileName,
                Body: buffer,
                ContentType: contentType,
            },
        });

        await upload.done();

        const url = `${this.publicUrl}/${fileName}`;
        this.logger.debug({ fileName, url }, 'Stored file to MinIO');

        return url;
    }

    /**
     * Retrieve file from MinIO
     * @param {string} url - File URL
     * @returns {Promise<Buffer>} File buffer
     */
    async retrieve(url) {
        const key = this._urlToKey(url);

        const response = await this.client.send(new GetObjectCommand({
            Bucket: this.bucket,
            Key: key,
        }));

        const chunks = [];
        for await (const chunk of response.Body) {
            chunks.push(chunk);
        }

        return Buffer.concat(chunks);
    }

    /**
     * Delete file from MinIO
     * @param {string} url - File URL
     * @returns {Promise<void>}
     */
    async delete(url) {
        const key = this._urlToKey(url);

        await this.client.send(new DeleteObjectCommand({
            Bucket: this.bucket,
            Key: key,
        }));

        this.logger.debug({ key }, 'Deleted file from MinIO');
    }

    /**
     * Convert URL to S3 key
     * @private
     */
    _urlToKey(url) {
        // Remove public URL prefix to get the key
        if (url.startsWith(this.publicUrl)) {
            return url.substring(this.publicUrl.length + 1);
        }
        // Try to extract key from URL path
        const urlObj = new URL(url);
        const pathParts = urlObj.pathname.split('/');
        // Remove bucket name from path if present
        if (pathParts[1] === this.bucket) {
            return pathParts.slice(2).join('/');
        }
        return pathParts.slice(1).join('/');
    }

    /**
     * Get file extension from content type
     * @private
     */
    _getExtension(contentType) {
        const map = {
            'image/jpeg': '.jpg',
            'image/png': '.png',
            'image/gif': '.gif',
            'image/webp': '.webp',
            'audio/ogg': '.ogg',
            'audio/mpeg': '.mp3',
            'audio/wav': '.wav',
            'audio/mp4': '.m4a',
            'video/mp4': '.mp4',
            'application/pdf': '.pdf',
        };
        return map[contentType] || '';
    }
}
