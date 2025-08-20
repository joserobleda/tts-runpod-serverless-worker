# TTS RunPod Serverless Worker

This is a serverless worker for text-to-speech generation using RunPod infrastructure with Cloudflare R2 storage.

## Configuration

### Cloudflare R2 Setup

To upload generated audio files to Cloudflare R2, set the following environment variables:

```bash
# Cloudflare R2 Configuration
BUCKET_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
BUCKET_ACCESS_KEY_ID=<your-r2-access-key-id>
BUCKET_SECRET_ACCESS_KEY=<your-r2-secret-access-key>
```

### How to get Cloudflare R2 credentials:

1. **Account ID**: Found in your Cloudflare dashboard sidebar
2. **R2 Access Key ID & Secret**: 
   - Go to Cloudflare Dashboard → R2 Object Storage → Manage R2 API tokens
   - Create a new API token with R2 read/write permissions
   - Copy the Access Key ID and Secret Access Key

### Environment Variables

- `BUCKET_ENDPOINT_URL`: Your R2 endpoint URL in the format `https://<account-id>.r2.cloudflarestorage.com`
- `BUCKET_ACCESS_KEY_ID`: Your R2 access key ID
- `BUCKET_SECRET_ACCESS_KEY`: Your R2 secret access key
- `WORKER_MODEL_DIR`: Directory containing the TTS model (default: `/model`)

#### Example Configuration:
```bash
# Example environment variables for Cloudflare R2
export BUCKET_ENDPOINT_URL=https://abc123def456.r2.cloudflarestorage.com
export BUCKET_ACCESS_KEY_ID=1234567890abcdef1234567890abcdef
export BUCKET_SECRET_ACCESS_KEY=abcdef1234567890abcdef1234567890abcdef12
export WORKER_MODEL_DIR=/model
```

### Fallback Behavior

If R2 credentials are not configured, the worker will return audio files as base64-encoded strings instead of uploading to R2.

## Usage

The worker accepts text-to-speech generation requests and returns either:
- A URL to the uploaded audio file in R2 (if configured)
- Base64-encoded audio data (fallback)

## Model Configuration

The worker uses advanced quality parameters for optimal audio generation:
- Temperature: 0.7 (balanced creativity/stability)
- Repetition penalty: 5.0 (prevents repetition)
- Top-k: 50, Top-p: 0.8 (good diversity with nucleus sampling)
- Enhanced audio processing enabled by default

## RunPod Endpoint

This repository contains the worker for the xTTSv2 AI Endpoints.

## Docker Image

```bash
docker build .
```
 or

 ```bash
 docker pull devbes/tts-runpod-serverless-worker:latest
 ```

## Continuous Deployment
This worker follows a modified version of the [worker template](https://github.com/runpod-workers/worker-template) where the Docker build workflow contains additional SD models to be built and pushed.

## API (simplified)

Request body:
```json
{
  "input": {
    "language": "es",
    "voice": {
      "speaker_0": "https://..."
    },
    "text": [
      ["speaker_0", "¿Cómo estás?"],
      ["speaker_0", "¡Genial!", { "temperature": 0.95 }],
      { "speaker": "speaker_0", "text": "Seguimos.", "options": { "speed_multiplier": 0.95 } }
    ],
    "options": {
      "temperature": 0.75,
      "top_p": 0.9,
      "speed": 1.02,
      "crossfade_length_ms": 50.0,
      "silence_fade_length_ms": 25.0,
      "enhance_audio": true,
      "question_temperature_delta": 0.18,
      "question_top_p_delta": 0.10,
      "question_speed_multiplier": 1.07,
      "exclamation_temperature_delta": 0.22,
      "exclamation_top_p_delta": 0.12,
      "exclamation_speed_multiplier": 1.05
    }
  }
}
```

Notes:
- Per-segment `options` override global `options`; global overrides defaults.
- If a segment defines `temperature`/`top_p`/`speed` (o sus variantes relativas), prevalece sobre el ajuste automático por “?”/“!”.
- Si un segmento no define nada, se usan defaults + globales + ajuste “?”/“!”.
 - Defaults de ajuste automático:
   - Pregunta: `temperature +0.30`, `top_p +0.40`, `speed ×0.90` (equiv. −0.1)
   - Exclamación: `temperature +0.30`, `top_p +0.40`, `speed ×1.00` (equiv. 0.0)
