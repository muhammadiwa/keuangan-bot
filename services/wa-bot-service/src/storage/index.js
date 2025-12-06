/**
 * Storage Abstraction Layer
 * 
 * Mendukung dua jenis storage:
 * - minio: S3-compatible object storage (MinIO)
 * - local: Local filesystem storage
 * 
 * Konfigurasi via environment variable STORAGE_TYPE
 */

import { MinioStorage } from './minio.js';
import { LocalStorage } from './local.js';

/**
 * @typedef {Object} StorageConfig
 * @property {string} type - 'minio' atau 'local'
 * @property {Object} minio - MinIO configuration
 * @property {Object} local - Local storage configuration
 */

/**
 * @typedef {Object} StoreOptions
 * @property {string} folder - Folder/prefix untuk file
 * @property {string} contentType - MIME type file
 */

/**
 * Storage interface
 * @interface IStorage
 * @method {Promise<void>} initialize() - Initialize storage
 * @method {Promise<string>} store(buffer, options) - Store file and return URL
 * @method {Promise<Buffer>} retrieve(url) - Retrieve file by URL
 * @method {Promise<void>} delete(url) - Delete file by URL
 * @method {string} getType() - Get storage type name
 */

/**
 * Create storage instance based on configuration
 * @param {Object} config - Storage configuration
 * @param {Object} logger - Logger instance
 * @returns {IStorage} Storage instance
 */
export function createStorage(config, logger) {
    const storageType = config.type || 'minio';

    logger.info({ storageType }, 'Initializing storage');

    switch (storageType.toLowerCase()) {
        case 'local':
            return new LocalStorage(config.local, logger);
        case 'minio':
        case 's3':
        default:
            return new MinioStorage(config.minio, logger);
    }
}

/**
 * Get storage configuration from environment variables
 * @returns {StorageConfig}
 */
export function getStorageConfig() {
    const storageType = process.env.STORAGE_TYPE || 'minio';

    return {
        type: storageType,
        minio: {
            endpoint: process.env.MINIO_ENDPOINT || 'http://localhost:9000',
            region: process.env.MINIO_REGION || 'us-east-1',
            accessKey: process.env.MINIO_ACCESS_KEY || 'minio',
            secretKey: process.env.MINIO_SECRET_KEY || 'miniopass',
            bucket: process.env.MEDIA_BUCKET || 'wa-media',
            publicUrl: process.env.MINIO_PUBLIC_URL || `${process.env.MINIO_ENDPOINT || 'http://localhost:9000'}/${process.env.MEDIA_BUCKET || 'wa-media'}`,
        },
        local: {
            basePath: process.env.LOCAL_STORAGE_PATH || './data/media',
            baseUrl: process.env.LOCAL_STORAGE_URL || `http://localhost:${process.env.PORT || 3000}/media`,
        },
    };
}

export { MinioStorage } from './minio.js';
export { LocalStorage } from './local.js';
