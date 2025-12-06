/**
 * Local Filesystem Storage Implementation
 */

import fs from 'fs/promises';
import fsSync from 'fs';
import path from 'path';
import crypto from 'crypto';

export class LocalStorage {
    /**
     * @param {Object} config - Local storage configuration
     * @param {string} config.basePath - Base path for storing files
     * @param {string} config.baseUrl - Base URL for accessing files
     * @param {Object} logger - Logger instance
     */
    constructor(config, logger) {
        this.config = config;
        this.logger = logger;
        this.basePath = path.resolve(config.basePath);
        this.baseUrl = config.baseUrl;
    }

    /**
     * Get storage type name
     * @returns {string}
     */
    getType() {
        return 'local';
    }

    /**
     * Initialize storage - ensure base directory exists
     * @returns {Promise<void>}
     */
    async initialize() {
        try {
            await fs.mkdir(this.basePath, { recursive: true });

            // Create subdirectories for different media types
            const subdirs = ['audio', 'images', 'documents'];
            for (const subdir of subdirs) {
                await fs.mkdir(path.join(this.basePath, subdir), { recursive: true });
            }

            this.logger.info({ basePath: this.basePath }, 'Local storage initialized');
        } catch (error) {
            this.logger.error({ err: error }, 'Failed to initialize local storage');
            throw error;
        }
    }

    /**
     * Store file to local filesystem
     * @param {Buffer} buffer - File buffer
     * @param {Object} options - Store options
     * @param {string} options.folder - Folder/prefix
     * @param {string} options.contentType - MIME type
     * @returns {Promise<string>} Public URL of stored file
     */
    async store(buffer, { folder, contentType }) {
        const timestamp = new Date().toISOString().replace(/[:]/g, '-');
        const extension = this._getExtension(contentType);
        const fileName = `${timestamp}-${crypto.randomUUID()}${extension}`;

        // Ensure folder exists
        const folderPath = path.join(this.basePath, folder);
        await fs.mkdir(folderPath, { recursive: true });

        // Write file
        const filePath = path.join(folderPath, fileName);
        await fs.writeFile(filePath, buffer);

        // Generate URL
        const url = `${this.baseUrl}/${folder}/${fileName}`;

        this.logger.debug({ filePath, url }, 'Stored file to local storage');

        return url;
    }

    /**
     * Retrieve file from local filesystem
     * @param {string} url - File URL
     * @returns {Promise<Buffer>} File buffer
     */
    async retrieve(url) {
        const filePath = this._urlToPath(url);

        try {
            const buffer = await fs.readFile(filePath);
            return buffer;
        } catch (error) {
            this.logger.error({ err: error, filePath }, 'Failed to retrieve file');
            throw error;
        }
    }

    /**
     * Delete file from local filesystem
     * @param {string} url - File URL
     * @returns {Promise<void>}
     */
    async delete(url) {
        const filePath = this._urlToPath(url);

        try {
            await fs.unlink(filePath);
            this.logger.debug({ filePath }, 'Deleted file from local storage');
        } catch (error) {
            if (error.code !== 'ENOENT') {
                this.logger.error({ err: error, filePath }, 'Failed to delete file');
                throw error;
            }
        }
    }

    /**
     * Get file path from URL
     * @param {string} url - File URL
     * @returns {string} Absolute file path
     */
    getFilePath(url) {
        return this._urlToPath(url);
    }

    /**
     * Check if file exists
     * @param {string} url - File URL
     * @returns {Promise<boolean>}
     */
    async exists(url) {
        const filePath = this._urlToPath(url);
        try {
            await fs.access(filePath);
            return true;
        } catch {
            return false;
        }
    }

    /**
     * Convert URL to file path
     * @private
     */
    _urlToPath(url) {
        // Remove base URL to get relative path
        let relativePath;

        if (url.startsWith(this.baseUrl)) {
            relativePath = url.substring(this.baseUrl.length);
        } else {
            // Try to extract path from URL
            try {
                const urlObj = new URL(url);
                relativePath = urlObj.pathname.replace(/^\/media/, '');
            } catch {
                relativePath = url;
            }
        }

        // Remove leading slash
        relativePath = relativePath.replace(/^\//, '');

        return path.join(this.basePath, relativePath);
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

    /**
     * Get content type from file extension
     * @param {string} filePath - File path
     * @returns {string} Content type
     */
    getContentType(filePath) {
        const ext = path.extname(filePath).toLowerCase();
        const map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.ogg': 'audio/ogg',
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.m4a': 'audio/mp4',
            '.mp4': 'video/mp4',
            '.pdf': 'application/pdf',
        };
        return map[ext] || 'application/octet-stream';
    }

    /**
     * Create read stream for file (useful for serving)
     * @param {string} url - File URL
     * @returns {ReadStream}
     */
    createReadStream(url) {
        const filePath = this._urlToPath(url);
        return fsSync.createReadStream(filePath);
    }
}
