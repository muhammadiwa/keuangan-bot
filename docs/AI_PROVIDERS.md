# AI Providers Configuration Guide

Panduan konfigurasi untuk Multi-Provider AI pada Keuangan Bot. Sistem ini mendukung berbagai provider AI/LLM untuk Natural Language Understanding (NLU) dan Speech-to-Text (STT).

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Supported NLU Providers](#supported-nlu-providers)
  - [Ollama (Self-Hosted)](#ollama-self-hosted)
  - [OpenAI](#openai)
  - [MegaLLM](#megallm)
  - [Groq](#groq)
  - [Together AI](#together-ai)
  - [Deepseek](#deepseek)
  - [Qwen (Alibaba DashScope)](#qwen-alibaba-dashscope)
  - [Kimi / Moonshot](#kimi--moonshot)
  - [GLM (Zhipu AI / BigModel)](#glm-zhipu-ai--bigmodel)
  - [Anthropic Claude](#anthropic-claude)
  - [Google Gemini](#google-gemini)
- [Supported STT Providers](#supported-stt-providers)
  - [Local Whisper](#local-whisper)
  - [OpenAI Whisper API](#openai-whisper-api)
  - [Groq Whisper](#groq-whisper)
- [Fallback Configuration](#fallback-configuration)
- [Advanced Options](#advanced-options)
- [Cost Tracking](#cost-tracking)
- [Health Monitoring](#health-monitoring)
- [Troubleshooting](#troubleshooting)

## Overview

Sistem Multi-Provider AI memungkinkan Anda memilih provider AI yang sesuai dengan kebutuhan:

| Consideration | Recommended Provider |
|---------------|---------------------|
| Privacy & Self-hosted | Ollama |
| Cost-effective | Groq, Deepseek, Gemini |
| High Quality | Anthropic Claude, OpenAI GPT-4 |
| Indonesian Language | Qwen, GLM |
| Fast Response | Groq, MegaLLM |
| Long Context | Kimi/Moonshot |

## Quick Start

### Minimal Configuration (Ollama - Default)

```bash
# .env file
AI_PROVIDER=ollama
OLLAMA_URL=http://ollama:11434
OLLAMA_MODEL=qwen2.5:3b-instruct
```

### Cloud Provider Example (OpenAI)

```bash
# .env file
AI_PROVIDER=openai
AI_API_KEY=sk-your-api-key-here
AI_MODEL=gpt-3.5-turbo
```

## Supported NLU Providers

### Ollama (Self-Hosted)

Ollama adalah runtime LLM self-hosted yang mendukung berbagai model open-source.

**Kelebihan:**
- Gratis (self-hosted)
- Data tetap lokal (privacy)
- Tidak perlu API key

**Konfigurasi:**

```bash
AI_PROVIDER=ollama
AI_BASE_URL=http://ollama:11434    # Optional, default: http://ollama:11434
AI_MODEL=qwen2.5:3b-instruct       # Optional, default: qwen2.5:3b-instruct
```

**Model yang Direkomendasikan:**
- `qwen2.5:3b-instruct` - Balanced performance
- `qwen2.5:1.5b` - Faster, lower resource
- `llama3:8b` - Good quality
- `phi3:mini` - Lightweight

**API Endpoint:** `POST /api/generate`

---

### OpenAI

Provider resmi OpenAI dengan model GPT.

**Konfigurasi:**

```bash
AI_PROVIDER=openai
AI_API_KEY=sk-your-openai-api-key
AI_BASE_URL=https://api.openai.com    # Optional, uses default
AI_MODEL=gpt-3.5-turbo                # Optional
```

**Model yang Tersedia:**
- `gpt-4o` - Most capable
- `gpt-4o-mini` - Cost-effective
- `gpt-4-turbo` - Fast GPT-4
- `gpt-3.5-turbo` - Budget option

**API Endpoint:** `POST /v1/chat/completions`  
**Authentication:** Bearer token

---

### MegaLLM

Provider lokal Indonesia dengan latency rendah.

**Konfigurasi:**

```bash
AI_PROVIDER=megallm
AI_API_KEY=your-megallm-api-key
AI_BASE_URL=https://api.megallm.id/v1    # Optional, uses default
AI_MODEL=gpt-3.5-turbo                   # Optional
```

**API Endpoint:** `POST /v1/chat/completions`  
**Authentication:** Bearer token

---

### Groq

Provider dengan inference tercepat menggunakan LPU.

**Kelebihan:**
- Sangat cepat (< 500ms)
- Harga kompetitif
- Free tier tersedia

**Konfigurasi:**

```bash
AI_PROVIDER=groq
AI_API_KEY=gsk_your-groq-api-key
AI_BASE_URL=https://api.groq.com/openai    # Optional
AI_MODEL=llama-3.1-8b-instant              # Optional
```

**Model yang Tersedia:**
- `llama-3.1-70b-versatile` - Most capable
- `llama-3.1-8b-instant` - Fast & efficient
- `mixtral-8x7b-32768` - Good for long context
- `gemma2-9b-it` - Google's Gemma

**API Endpoint:** `POST /v1/chat/completions`  
**Authentication:** Bearer token

---

### Together AI

Platform untuk menjalankan berbagai model open-source.

**Konfigurasi:**

```bash
AI_PROVIDER=together
AI_API_KEY=your-together-api-key
AI_BASE_URL=https://api.together.xyz    # Optional
AI_MODEL=meta-llama/Llama-3-8b-chat-hf  # Optional
```

**Model yang Tersedia:**
- `meta-llama/Llama-3-70b-chat-hf`
- `meta-llama/Llama-3-8b-chat-hf`
- `mistralai/Mixtral-8x7B-Instruct-v0.1`
- `Qwen/Qwen2-72B-Instruct`

**API Endpoint:** `POST /v1/chat/completions`  
**Authentication:** Bearer token

---

### Deepseek

Provider dengan harga sangat kompetitif dan kualitas tinggi.

**Kelebihan:**
- Harga sangat murah
- Kualitas setara GPT-4
- Bagus untuk coding

**Konfigurasi:**

```bash
AI_PROVIDER=deepseek
AI_API_KEY=sk-your-deepseek-api-key
AI_BASE_URL=https://api.deepseek.com    # Optional
AI_MODEL=deepseek-chat                  # Optional
```

**Model yang Tersedia:**
- `deepseek-chat` - General purpose
- `deepseek-coder` - Optimized for code

**API Endpoint:** `POST /v1/chat/completions`  
**Authentication:** Bearer token

---

### Qwen (Alibaba DashScope)

Model dari Alibaba dengan kemampuan multilingual yang baik.

**Kelebihan:**
- Bagus untuk bahasa Indonesia
- Harga kompetitif
- Model bervariasi

**Konfigurasi:**

```bash
AI_PROVIDER=qwen
AI_API_KEY=sk-your-dashscope-api-key
AI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode    # Optional
AI_MODEL=qwen-turbo                                           # Optional
```

**Model yang Tersedia:**
- `qwen-max` - Most capable
- `qwen-plus` - Balanced
- `qwen-turbo` - Fast & cheap

**API Endpoint:** `POST /v1/chat/completions`  
**Authentication:** Bearer token

---

### Kimi / Moonshot

Provider dengan kemampuan long-context terbaik.

**Kelebihan:**
- Context window sangat besar (128K-200K tokens)
- Bagus untuk dokumen panjang

**Konfigurasi:**

```bash
AI_PROVIDER=kimi
# atau
AI_PROVIDER=moonshot

AI_API_KEY=your-moonshot-api-key
AI_BASE_URL=https://api.moonshot.cn    # Optional
AI_MODEL=moonshot-v1-8k                # Optional
```

**Model yang Tersedia:**
- `moonshot-v1-8k` - 8K context
- `moonshot-v1-32k` - 32K context
- `moonshot-v1-128k` - 128K context

**API Endpoint:** `POST /v1/chat/completions`  
**Authentication:** Bearer token

---

### GLM (Zhipu AI / BigModel)

Model GLM dari Zhipu AI dengan kemampuan Chinese dan multilingual.

**Kelebihan:**
- Bagus untuk Chinese & multilingual
- JWT authentication (lebih aman)

**Konfigurasi:**

```bash
AI_PROVIDER=glm
# atau
AI_PROVIDER=zhipu
# atau
AI_PROVIDER=bigmodel

AI_API_KEY=your-api-id.your-api-secret    # Format: {id}.{secret}
AI_BASE_URL=https://open.bigmodel.cn/api/paas    # Optional
AI_MODEL=glm-4-flash                             # Optional
```

> **Penting:** API key GLM harus dalam format `{id}.{secret}`. Sistem akan generate JWT token dari API key ini.

**Model yang Tersedia:**
- `glm-4` - Most capable
- `glm-4-flash` - Fast & efficient
- `glm-3-turbo` - Budget option

**API Endpoint:** `POST /v4/chat/completions`  
**Authentication:** JWT Bearer token (auto-generated)

---

### Anthropic Claude

Model Claude dari Anthropic dengan kualitas tinggi.

**Kelebihan:**
- Kualitas sangat tinggi
- Bagus untuk reasoning
- Safety-focused

**Konfigurasi:**

```bash
AI_PROVIDER=anthropic
AI_API_KEY=sk-ant-your-anthropic-api-key
AI_BASE_URL=https://api.anthropic.com    # Optional
AI_MODEL=claude-3-5-sonnet-20241022      # Optional
```

**Model yang Tersedia:**
- `claude-3-5-sonnet-20241022` - Best value
- `claude-3-5-haiku-20241022` - Fast & cheap
- `claude-3-opus-20240229` - Most capable

**API Endpoint:** `POST /v1/messages`  
**Authentication:** `x-api-key` header

---

### Google Gemini

Model Gemini dari Google dengan harga kompetitif.

**Kelebihan:**
- Free tier generous
- Multimodal capable
- Harga kompetitif

**Konfigurasi:**

```bash
AI_PROVIDER=gemini
AI_API_KEY=your-google-api-key
AI_BASE_URL=https://generativelanguage.googleapis.com    # Optional
AI_MODEL=gemini-1.5-flash                                # Optional
```

**Model yang Tersedia:**
- `gemini-1.5-pro` - Most capable
- `gemini-1.5-flash` - Fast & efficient
- `gemini-1.0-pro` - Legacy

**API Endpoint:** `POST /v1beta/models/{model}:generateContent`  
**Authentication:** API key in query parameter

---

## Supported STT Providers

### Local Whisper

Whisper model yang berjalan lokal menggunakan faster-whisper.

**Kelebihan:**
- Gratis
- Data tetap lokal
- Tidak perlu internet

**Konfigurasi:**

```bash
STT_PROVIDER=whisper
STT_MODEL=base    # Options: tiny, base, small, medium, large
```

**Model Options:**
| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| tiny | 39M | Fastest | Lower |
| base | 74M | Fast | Good |
| small | 244M | Medium | Better |
| medium | 769M | Slow | High |
| large | 1550M | Slowest | Highest |

---

### OpenAI Whisper API

Whisper API dari OpenAI.

**Konfigurasi:**

```bash
STT_PROVIDER=openai
STT_API_KEY=sk-your-openai-api-key    # Required
STT_MODEL=whisper-1                   # Fixed model
```

> **Note:** Jika `AI_PROVIDER=openai` dan `STT_API_KEY` tidak diset, sistem akan menggunakan `AI_API_KEY`.

---

### Groq Whisper

Whisper API dari Groq dengan kecepatan tinggi.

**Kelebihan:**
- Sangat cepat
- Harga kompetitif

**Konfigurasi:**

```bash
STT_PROVIDER=groq
STT_API_KEY=gsk_your-groq-api-key    # Required
STT_MODEL=whisper-large-v3           # Default
```

> **Note:** Jika `AI_PROVIDER=groq` dan `STT_API_KEY` tidak diset, sistem akan menggunakan `AI_API_KEY`.

---

## Fallback Configuration

Sistem mendukung fallback provider jika provider utama gagal.

```bash
# Primary Provider
AI_PROVIDER=openai
AI_API_KEY=sk-your-openai-key
AI_MODEL=gpt-3.5-turbo

# Fallback Provider
AI_FALLBACK_PROVIDER=ollama
AI_FALLBACK_BASE_URL=http://ollama:11434
AI_FALLBACK_MODEL=qwen2.5:3b-instruct
```

**Fallback Behavior:**
1. Request dikirim ke primary provider
2. Jika gagal (network error, rate limit, server error), retry ke fallback provider
3. Jika fallback juga gagal, gunakan heuristic parsing sebagai last resort

---

## Advanced Options

```bash
# Timeout untuk request AI (dalam detik)
AI_TIMEOUT=30

# Jumlah retry sebelum fallback
AI_MAX_RETRIES=2

# Enable debug logging (log full request/response)
AI_DEBUG_LOGGING=false
```

---

## Cost Tracking

Sistem mencatat penggunaan token untuk setiap request ke cloud provider.

```bash
# Enable/disable cost tracking
AI_COST_TRACKING_ENABLED=true

# Backend API URL untuk menyimpan data
BACKEND_API_URL=http://backend-api:8000
```

Data yang dicatat:
- Provider name
- Model used
- Input tokens
- Output tokens
- Estimated cost
- Timestamp

---

## Health Monitoring

Endpoint `/health` menyediakan status semua provider yang dikonfigurasi.

```bash
curl http://localhost:8500/health
```

Response example:
```json
{
  "status": "healthy",
  "ai_provider": {
    "status": "healthy",
    "provider": "openai",
    "latency_ms": 245.3,
    "model": "gpt-3.5-turbo"
  },
  "stt_provider": {
    "status": "healthy",
    "provider": "whisper",
    "model": "base"
  }
}
```

---

## Troubleshooting

### Provider tidak terkoneksi

1. Periksa `AI_API_KEY` sudah benar
2. Periksa `AI_BASE_URL` jika menggunakan custom endpoint
3. Cek logs: `docker compose logs -f ai-media-service`

### Rate limit error

1. Kurangi request frequency
2. Upgrade plan di provider
3. Konfigurasi fallback provider

### GLM JWT Error

Pastikan API key dalam format `{id}.{secret}`:
```bash
# Benar
AI_API_KEY=abc123.xyz789secret

# Salah
AI_API_KEY=abc123xyz789secret
```

### Ollama model not found

1. Pull model terlebih dahulu:
   ```bash
   docker compose exec ollama ollama pull qwen2.5:3b-instruct
   ```
2. Cek model tersedia:
   ```bash
   docker compose exec ollama ollama list
   ```

### STT tidak berfungsi

1. Untuk local Whisper, pastikan model sudah terdownload
2. Untuk cloud STT, pastikan `STT_API_KEY` sudah diset
3. Cek format audio yang dikirim (harus WAV/MP3/OGG)

---

## Environment Variables Reference

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `AI_PROVIDER` | Primary AI provider | `ollama` | No |
| `AI_API_KEY` | API key for primary provider | - | For cloud providers |
| `AI_BASE_URL` | Custom base URL | Provider default | No |
| `AI_MODEL` | Model to use | Provider default | No |
| `AI_FALLBACK_PROVIDER` | Fallback provider | - | No |
| `AI_FALLBACK_API_KEY` | Fallback API key | - | If fallback needs auth |
| `AI_FALLBACK_BASE_URL` | Fallback base URL | Provider default | No |
| `AI_FALLBACK_MODEL` | Fallback model | Provider default | No |
| `STT_PROVIDER` | STT provider | `whisper` | No |
| `STT_API_KEY` | STT API key | - | For cloud STT |
| `STT_MODEL` | STT model | `base` | No |
| `AI_TIMEOUT` | Request timeout (seconds) | `30` | No |
| `AI_MAX_RETRIES` | Max retries before fallback | `2` | No |
| `AI_DEBUG_LOGGING` | Enable debug logging | `false` | No |
| `AI_COST_TRACKING_ENABLED` | Enable cost tracking | `true` | No |
| `BACKEND_API_URL` | Backend API URL | `http://backend-api:8000` | No |
