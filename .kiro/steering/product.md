# Product Overview

## Keuangan Bot - AI-Powered Personal Finance Tracker

Aplikasi pencatatan keuangan pribadi berbasis WhatsApp yang menggunakan AI untuk memproses input multi-modal (teks, suara, foto struk). Sistem ini memungkinkan pengguna mencatat transaksi, mengelola tabungan, dan mendapatkan laporan keuangan melalui percakapan WhatsApp yang natural.

## Core Features

- **Multi-modal Input**: Teks, voice notes, dan foto struk otomatis diproses menggunakan NLU, STT, dan OCR
- **Smart Transaction Parsing**: AI mengekstrak jumlah, kategori, dan deskripsi dari input natural bahasa Indonesia
- **Savings Management**: Sistem tabungan dengan target dan tracking progress
- **Automated Reports**: Laporan harian, mingguan, dan bulanan otomatis via WhatsApp
- **Web Dashboard**: Interface web untuk analisis mendalam dan export data
- **Multi-user Support**: Mendukung multiple users dengan isolasi data per nomor WhatsApp

## Target Users

Individu yang ingin mencatat keuangan pribadi dengan cara yang mudah dan natural melalui WhatsApp, tanpa perlu membuka aplikasi khusus atau mengingat format input yang kaku.

## Business Logic

- Semua transaksi dikategorikan otomatis berdasarkan deskripsi
- Sistem confidence scoring untuk validasi transaksi dari OCR
- Automatic user creation untuk nomor WhatsApp baru
- Timezone-aware reporting (default: Asia/Jakarta)